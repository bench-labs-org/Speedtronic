"""v2 optimizers: pure-PyTorch Muon, Muon+, cautious wrapping, and hybrid routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class ParameterRouting:
    """Deterministic parameter ownership for the hybrid optimizer."""

    muon: list[tuple[str, nn.Parameter]]
    adamw: list[tuple[str, nn.Parameter]]
    skipped: list[tuple[str, nn.Parameter]]


def _work_dtype(value: torch.Tensor) -> torch.dtype:
    if value.dtype == torch.float64:
        return torch.float64
    if value.device.type == "cuda" and value.dtype in {torch.float16, torch.bfloat16}:
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float32


def newton_schulz(
    matrix: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate the orthogonal polar factor with a quintic iteration.

    The implementation intentionally uses only PyTorch matrix operations.  It
    does not require a custom CUDA extension, SVD, or a device-specific package.
    """

    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalization requires a 2-D matrix")
    if not matrix.is_floating_point():
        raise TypeError("Newton-Schulz orthogonalization requires floating-point values")
    if steps < 1:
        raise ValueError("Newton-Schulz steps must be positive")
    if eps <= 0:
        raise ValueError("Newton-Schulz eps must be positive")

    dtype = _work_dtype(matrix)

    def iterate(value: torch.Tensor) -> torch.Tensor:
        transposed = value.shape[0] > value.shape[1]
        x = value.transpose(0, 1) if transposed else value
        norm = torch.linalg.vector_norm(x)
        x = x / (norm + eps)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(steps):
            gram = x @ x.transpose(0, 1)
            correction = b * gram + c * (gram @ gram)
            x = a * x + correction @ x
        return x.transpose(0, 1) if transposed else x

    x = matrix.detach().to(dtype=dtype)
    try:
        result = iterate(x)
    except torch.cuda.OutOfMemoryError:
        # Do not hide genuine allocation pressure with a second large matrix.
        raise
    except (RuntimeError, TypeError):
        # Some older CUDA devices do not implement every low-precision matmul.
        # Float32 is the portable fallback and does not change the API.
        if dtype == torch.float32:
            raise
        result = iterate(x.to(dtype=torch.float32))
    return result


def post_polar_normalize(
    update: torch.Tensor,
    *,
    mode: str = "row_col",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply Muon+'s inexpensive post-orthogonalization normalization."""

    if update.ndim != 2:
        raise ValueError("Muon+ normalization requires a 2-D update")
    if eps <= 0:
        raise ValueError("Muon+ normalization eps must be positive")
    mode = str(mode).lower()
    if mode not in {"none", "row", "col", "row_col", "col_row"}:
        raise ValueError("Muon+ normalization mode is invalid")

    if mode == "none":
        return update

    work = update.float() if update.dtype in {torch.float16, torch.bfloat16} else update
    if mode in {"row", "row_col"}:
        work = work / torch.sqrt(work.square().sum(dim=1, keepdim=True) + eps)
    if mode in {"col", "col_row", "row_col"}:
        work = work / torch.sqrt(work.square().sum(dim=0, keepdim=True) + eps)
    return work.to(dtype=update.dtype)


def _parameter_owners(model: nn.Module) -> Mapping[int, tuple[str, nn.Module, str]]:
    owners: dict[int, tuple[str, nn.Module, str]] = {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for local_name, parameter in module.named_parameters(recurse=False):
            owners.setdefault(id(parameter), (prefix + local_name, module, local_name))
    return owners


def _is_embedding_or_head(name: str, module: nn.Module) -> bool:
    lowered = name.lower()
    module_name = type(module).__name__.lower()
    if "embedding" in module_name or "norm" in module_name:
        return True
    parts = lowered.split(".")
    leaf_name = parts[-1]
    excluded = {
        "embed",
        "embedding",
        "wte",
        "wpe",
        "lm_head",
        "classifier",
        "output",
        "output_proj",
        "score",
        "score_proj",
        "head",
    }
    return leaf_name in excluded or any(part in excluded for part in parts[:-1])


def route_parameters(model: nn.Module) -> ParameterRouting:
    """Route trainable parameters to Muon or AdamW by role and dimensionality.

    Hidden ``nn.Linear`` matrices are eligible for Muon. Embeddings, heads,
    normalization parameters, biases, non-2D tensors, and frozen parameters are
    sent to AdamW or omitted.  A module can opt into a route with
    ``_speedtronic_optimizer_role = "muon"`` or ``"adamw"``.
    """

    owners = _parameter_owners(model)
    muon: list[tuple[str, nn.Parameter]] = []
    adamw: list[tuple[str, nn.Parameter]] = []
    skipped: list[tuple[str, nn.Parameter]] = []
    seen: set[int] = set()

    for name, parameter in model.named_parameters(remove_duplicate=False):
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        if not parameter.requires_grad:
            skipped.append((name, parameter))
            continue
        owner_name, owner_module, local_name = owners.get(
            identity, (name, model, name.rsplit(".", 1)[-1])
        )
        explicit = getattr(owner_module, "_speedtronic_optimizer_role", None)
        if explicit not in {None, "muon", "adamw"}:
            raise ValueError(f"invalid _speedtronic_optimizer_role on {owner_name!r}: {explicit!r}")
        if explicit == "adamw":
            adamw.append((name, parameter))
            continue
        if explicit == "muon":
            if (
                local_name.endswith("bias")
                or parameter.ndim != 2
                or not parameter.is_floating_point()
            ):
                adamw.append((name, parameter))
                continue
            muon.append((name, parameter))
            continue
        if (
            not isinstance(owner_module, nn.Linear)
            or parameter.ndim != 2
            or not parameter.is_floating_point()
            or local_name.endswith("bias")
            or _is_embedding_or_head(owner_name, owner_module)
        ):
            adamw.append((name, parameter))
            continue
        muon.append((name, parameter))

    return ParameterRouting(muon=muon, adamw=adamw, skipped=skipped)


class Muon(Optimizer):
    """A pure-PyTorch Muon optimizer for routed 2-D parameter groups."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        eps: float = 1e-7,
        norm_eps: float = 1e-8,
        muon_plus: bool = False,
    ) -> None:
        if lr <= 0:
            raise ValueError("Muon learning rate must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("Muon momentum must be in [0, 1)")
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "ns_steps": int(ns_steps),
            "weight_decay": float(weight_decay),
            "eps": float(eps),
            "norm_eps": float(norm_eps),
            "muon_plus": bool(muon_plus),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            weight_decay = float(group["weight_decay"])
            norm_eps = float(group["norm_eps"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("Muon only accepts 2-D parameters")
                gradient = parameter.grad
                if not gradient.is_sparse:
                    if gradient.is_complex():
                        raise TypeError("Muon does not support complex parameters")
                    state = self.state[parameter]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter, dtype=torch.float32)
                    state["step"] = int(state.get("step", 0)) + 1
                    buffer = state["momentum_buffer"]
                    buffer.mul_(momentum).add_(gradient.float(), alpha=1.0 - momentum)
                    if nesterov:
                        direction = torch.lerp(gradient.float(), buffer, momentum)
                    else:
                        direction = buffer
                    update = newton_schulz(direction, steps=ns_steps, eps=group["eps"])
                    if bool(group["muon_plus"]):
                        update = post_polar_normalize(update, eps=norm_eps)
                    scale = max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
                    if weight_decay:
                        parameter.mul_(1.0 - lr * weight_decay)
                    parameter.add_(update.to(dtype=parameter.dtype), alpha=-lr * scale)
                else:
                    raise RuntimeError("Muon does not support sparse gradients")
        return loss


class HybridOptimizer(Optimizer):
    """One optimizer facade combining Muon matrices and AdamW parameters."""

    def __init__(
        self,
        muon_params: Sequence[torch.Tensor],
        adamw_params: Sequence[torch.Tensor],
        *,
        muon_lr: float,
        adamw_lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        fused: bool = False,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        muon_norm_eps: float = 1e-8,
        muon_plus: bool = False,
    ) -> None:
        if not muon_params and not adamw_params:
            raise ValueError("hybrid optimizer requires at least one trainable parameter")
        groups: list[dict[str, Any]] = []
        if muon_params:
            groups.append(
                {
                    "params": list(muon_params),
                    "algorithm": "muon",
                    "lr": float(muon_lr),
                    "momentum": float(muon_momentum),
                    "ns_steps": int(muon_ns_steps),
                    "norm_eps": float(muon_norm_eps),
                    "muon_plus": bool(muon_plus),
                    "weight_decay": float(weight_decay),
                }
            )
        if adamw_params:
            groups.append(
                {
                    "params": list(adamw_params),
                    "algorithm": "adamw",
                    "lr": float(adamw_lr),
                    "betas": tuple(betas),
                    "eps": float(eps),
                    "weight_decay": float(weight_decay),
                    "fused": bool(fused),
                }
            )
        super().__init__(groups, {"algorithm": "hybrid"})
        self._muon = (
            Muon(
                list(muon_params),
                lr=muon_lr,
                momentum=muon_momentum,
                ns_steps=muon_ns_steps,
                norm_eps=muon_norm_eps,
                muon_plus=muon_plus,
                weight_decay=weight_decay,
            )
            if muon_params
            else None
        )
        adam_kwargs: dict[str, Any] = {
            "lr": float(adamw_lr),
            "betas": tuple(betas),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
        }
        if fused and adamw_params:
            adam_kwargs["fused"] = True
        try:
            self._adam = (
                torch.optim.AdamW(list(adamw_params), **adam_kwargs) if adamw_params else None
            )
        except (TypeError, RuntimeError, ValueError, NotImplementedError):
            adam_kwargs.pop("fused", None)
            self._adam = (
                torch.optim.AdamW(list(adamw_params), **adam_kwargs) if adamw_params else None
            )
        self._bind_children()

    def _bind_children(self) -> None:
        if self._muon is not None:
            self._muon.state = self.state
        if self._adam is not None:
            self._adam.state = self.state

    def _sync_children(self) -> None:
        if self._muon is not None:
            for outer, inner in zip(self.param_groups, self._muon.param_groups):
                for key in ("lr", "weight_decay", "momentum", "ns_steps", "norm_eps", "muon_plus"):
                    inner[key] = outer[key]
        if self._adam is not None:
            adam_group = next(
                (group for group in self.param_groups if group["algorithm"] == "adamw"),
                None,
            )
            if adam_group is not None:
                inner = self._adam.param_groups[0]
                for key in ("lr", "betas", "eps", "weight_decay"):
                    inner[key] = adam_group[key]

    def step(self, closure: Any | None = None) -> Any:
        self._sync_children()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._muon is not None:
            self._muon.step()
        if self._adam is not None:
            self._adam.step()
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._bind_children()
        self._sync_children()


class CautiousOptimizer(Optimizer):
    """Composable cautious wrapper around a Speedtronic/native optimizer.

    The wrapper snapshots parameters, lets the base optimizer calculate its
    update, and masks entries whose observed update does not align with the
    current gradient.  It intentionally supports arbitrary base optimizers;
    native AdamW and Muon remain available for exact direction-specific
    integrations.
    """

    def __init__(self, base: Optimizer) -> None:
        # Initialize the facade so generic optimizer tooling (schedulers,
        # GradScaler, repr, and step hooks) sees a complete Optimizer object.
        self.base = base
        self._cautious_initializing = True
        super().__init__(base.param_groups, base.defaults)
        self._cautious_initializing = False
        self.param_groups = base.param_groups
        self.defaults = base.defaults
        self.state = base.state
        # GradScaler must unscale this facade before the base step is called.
        self._step_supports_amp_scaling = False

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        before = {
            parameter: parameter.detach().clone()
            for group in self.param_groups
            for parameter in group["params"]
        }
        self.base.step()
        for parameter, old_value in before.items():
            if parameter.grad is None:
                continue
            direction = old_value - parameter
            mask = (direction * parameter.grad) > 0
            # This is the cautious update rule from the v2 addendum: keep
            # aligned entries and revert non-aligned entries without inventing
            # an extra learning-rate multiplier.
            parameter.copy_(old_value - direction * mask.to(dtype=direction.dtype))
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return self.base.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base.load_state_dict(state_dict)
        self.param_groups = self.base.param_groups
        self.defaults = self.base.defaults
        self.state = self.base.state
        self._step_supports_amp_scaling = False

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if getattr(self, "_cautious_initializing", False):
            Optimizer.add_param_group(self, param_group)
            return
        self.base.add_param_group(param_group)
        self.param_groups = self.base.param_groups


def build_v2_optimizer(
    model: nn.Module,
    config: Any,
    device: torch.device,
) -> Optimizer:
    """Build AdamW, hybrid Muon/AdamW, and cautious variants from config."""

    optimizer_config = config.optimizer
    name = str(getattr(optimizer_config, "name", "adamw")).lower()
    trainable = [
        parameter
        for _, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    ]
    trainable = list({id(parameter): parameter for parameter in trainable}.values())
    if not trainable:
        raise ValueError("model has no trainable parameters")
    routing = route_parameters(model)
    muon_params = [parameter for _, parameter in routing.muon]
    adamw_params = [parameter for _, parameter in routing.adamw]
    if muon_params:
        # Remove duplicate IDs that can arise from aliases and role heuristics.
        muon_params = list({id(parameter): parameter for parameter in muon_params}.values())
    if adamw_params:
        adamw_params = list({id(parameter): parameter for parameter in adamw_params}.values())
    if name == "adamw" and not adamw_params:
        # v1 AdamW must remain usable for models whose entire parameter set
        # consists of 2-D matrices; never construct an empty optimizer.
        adamw_params = trainable
        muon_params = []

    lr = float(optimizer_config.lr)
    betas = tuple(float(value) for value in optimizer_config.betas)
    eps = float(optimizer_config.eps)
    weight_decay = float(optimizer_config.weight_decay)
    fused_requested = optimizer_config.fused
    cautious = bool(getattr(optimizer_config, "cautious", False))
    if fused_requested is None:
        from .precision import supports_fused_adamw

        fused = bool(adamw_params) and supports_fused_adamw(device)
    else:
        fused = bool(fused_requested) and bool(adamw_params)
    # Cautious needs an observable base update; keep the portable non-fused
    # path so the wrapper can mask it consistently across devices.
    if cautious:
        fused = False

    if name == "adamw":
        kwargs: dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        if fused:
            try:
                optimizer: Optimizer = torch.optim.AdamW(adamw_params, fused=True, **kwargs)
            except (TypeError, RuntimeError, ValueError, NotImplementedError):
                optimizer = torch.optim.AdamW(adamw_params, **kwargs)
        else:
            optimizer = torch.optim.AdamW(adamw_params, **kwargs)
    else:
        optimizer = HybridOptimizer(
            muon_params,
            adamw_params,
            muon_lr=lr,
            adamw_lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=bool(fused),
            muon_momentum=float(getattr(optimizer_config, "muon_momentum", 0.95)),
            muon_ns_steps=int(getattr(optimizer_config, "muon_ns_steps", 5)),
            muon_norm_eps=float(getattr(optimizer_config, "muon_norm_eps", 1e-8)),
            muon_plus=bool(getattr(optimizer_config, "muon_plus", False)),
        )

    if cautious:
        optimizer = CautiousOptimizer(optimizer)
    return optimizer


__all__ = [
    "CautiousOptimizer",
    "HybridOptimizer",
    "Muon",
    "ParameterRouting",
    "build_v2_optimizer",
    "newton_schulz",
    "post_polar_normalize",
    "route_parameters",
]
