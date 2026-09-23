"""Hardware capability detection and mixed-precision helpers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrecisionPlan:
    mode: str
    dtype: Any
    use_scaler: bool
    autocast_enabled: bool
    device_type: str

    @property
    def is_mixed_precision(self) -> bool:
        return self.mode in {"bf16", "fp16"}


def resolve_device(requested: str | Any = "auto") -> Any:
    import torch

    if requested is None or requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
    return device


def resolve_precision(config: Any, device: Any) -> PrecisionPlan:
    """Resolve a precision config against the actual device.

    The explicit mode wins, except that an explicit unsupported mixed mode
    falls back to a safe mode rather than making a run unusable.
    """

    import torch

    device = torch.device(device)
    requested = str(getattr(config, "mode", "auto")).lower()
    explicit_dtype = getattr(config, "dtype", None)
    if explicit_dtype and requested == "auto":
        requested = str(explicit_dtype).lower()

    if requested == "auto":
        if device.type == "cuda":
            try:
                supported = bool(torch.cuda.is_bf16_supported())
            except Exception:
                supported = False
            requested = "bf16" if supported else "fp16"
        else:
            # CPU autocast can support bf16 on recent PyTorch, but fp32 is the
            # portable and predictable default for arbitrary custom modules.
            requested = "fp32"

    if requested == "bf16" and device.type == "cuda":
        try:
            if not torch.cuda.is_bf16_supported():
                requested = "fp32"
        except Exception:
            requested = "fp32"
    if requested in {"bf16", "fp16"} and device.type == "mps":
        # MPS autocast support varies substantially across PyTorch releases.
        # Keep the run correct and let users opt into a backend-specific path.
        requested = "fp32"
    if requested == "fp16" and device.type == "cpu":
        # CPU training has no CUDA-style GradScaler path; fp32 is the safe
        # correctness fallback for an explicit fp16 request.
        requested = "fp32"

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[requested]
    return PrecisionPlan(
        mode=requested,
        dtype=dtype,
        use_scaler=requested == "fp16" and device.type == "cuda",
        autocast_enabled=requested in {"bf16", "fp16"},
        device_type=device.type,
    )


def autocast_context(plan: PrecisionPlan, device: Any):
    import torch

    if not plan.autocast_enabled:
        return nullcontext()
    try:
        return torch.autocast(device_type=plan.device_type, dtype=plan.dtype)
    except TypeError:  # older torch used device_type only
        return torch.autocast(plan.device_type, dtype=plan.dtype)


def make_grad_scaler(plan: PrecisionPlan):
    import torch

    if not plan.use_scaler:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler()


def supports_fused_adamw(device: Any) -> bool:
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return False
    try:
        # A zero-step optimizer construction is the most reliable capability
        # check across CUDA/PyTorch combinations.
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1, device=device))], fused=True)
        return True
    except (TypeError, RuntimeError, ValueError):
        return False


def parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "PrecisionPlan",
    "autocast_context",
    "make_grad_scaler",
    "resolve_device",
    "resolve_precision",
    "supports_fused_adamw",
]
