"""Architecture-neutral training loop."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import torch
from torch.nn import functional as F

from .checkpoint import CheckpointManager, capture_rng_state, restore_rng_state
from .data import infinite_batches
from .precision import PrecisionPlan, autocast_context, make_grad_scaler, resolve_precision
from .profiling import MetricLogger


class SyncCoordinator(Protocol):
    def start(self) -> None: ...

    def after_optimizer_step(self, model: Any, step: int) -> bool | None: ...

    def stop(self) -> None: ...

    def state_dict(self) -> dict[str, Any] | None: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


@dataclass
class TrainResult:
    steps: int
    samples: int
    tokens: int
    final_loss: float | None
    elapsed_s: float
    metrics: list[dict[str, Any]]


class Trainer:
    """Train a user-provided module with optional local or DumbDiLoCo sync.

    The model receives a batch dictionary when the loader emits dictionaries;
    otherwise the first two tuple elements are passed as ``inputs, labels``.
    A model may return a loss directly, a mapping with ``loss``, or a tuple
    whose first element is the loss.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: Iterable[dict[str, torch.Tensor]],
        *,
        device: torch.device | str,
        config: Any | None = None,
        scheduler: Any | None = None,
        precision: PrecisionPlan | None = None,
        logger: MetricLogger | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        coordinator: SyncCoordinator | None = None,
        max_steps: int | None = None,
        start_step: int = 0,
    ) -> None:
        if isinstance(config, dict):
            from .config import SpeedtronicConfig

            config = SpeedtronicConfig.from_dict(config)
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.config = config
        self.scheduler = scheduler
        self.logger = logger or MetricLogger()
        self.checkpoint_manager = checkpoint_manager
        self.coordinator = coordinator
        self.precision = precision or resolve_precision(
            getattr(config, "precision", None) or _AutoPrecision(), self.device
        )
        self.max_steps = int(
            max_steps
            if max_steps is not None
            else getattr(getattr(config, "run", None), "max_steps", 1000)
        )
        self.step = int(start_step)
        self.samples = 0
        self.tokens = 0
        self._active_model = self.model
        self._compiled = False
        self._started = False
        self._deferred_coordinator_state: dict[str, Any] | None = None
        self._deferred_scaler_state: dict[str, Any] | None = None
        self._configure_features()

    @property
    def accumulation_steps(self) -> int:
        value = getattr(getattr(self.config, "data", None), "accumulation_steps", None)
        return max(1, int(value or 1))

    def _configure_features(self) -> None:
        checkpointing = bool(getattr(self.config, "gradient_checkpointing", False))
        if checkpointing:
            setter = getattr(self.model, "set_gradient_checkpointing", None)
            if callable(setter):
                try:
                    setter(True)
                except Exception as exc:
                    self.logger.warning(
                        "gradient checkpointing hook failed; continuing without it: %s",
                        exc,
                    )
            else:
                self.logger.warning(
                    "gradient checkpointing requested but model has no "
                    "set_gradient_checkpointing hook"
                )
        if bool(getattr(self.config, "compile", False)):
            self._enable_compile()

    def _enable_compile(self) -> None:
        try:
            self._active_model = torch.compile(self.model)
            self._compiled = True
            self.logger.emit("compile", {"enabled": True})
        except Exception as exc:
            self._active_model = self.model
            self._compiled = False
            self.logger.emit(
                "compile",
                {"enabled": False, "reason": str(exc)},
            )

    def _move_batch(self, batch: Any) -> Any:
        def move(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(self.device, non_blocking=True)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, list):
                return [move(item) for item in value]
            return value

        return move(batch)

    @staticmethod
    def _batch_labels(batch: Any) -> torch.Tensor | None:
        if isinstance(batch, dict):
            labels = batch.get("labels")
            return labels if isinstance(labels, torch.Tensor) else None
        if isinstance(batch, (tuple, list)) and len(batch) > 1:
            return batch[1] if isinstance(batch[1], torch.Tensor) else None
        return None

    @classmethod
    def _loss_from_logits(cls, output: Any, batch: Any) -> torch.Tensor | None:
        if isinstance(output, dict):
            logits = output.get("logits")
        elif isinstance(output, torch.Tensor):
            logits = output
        else:
            return None
        labels = cls._batch_labels(batch)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or labels is None:
            return None
        if labels.ndim != 2 or labels.shape[0] != logits.shape[0]:
            return None
        if labels.shape[1] == logits.shape[1]:
            labels = labels[:, 1:]
        if labels.shape[1] != logits.shape[1] - 1:
            return None
        if labels.numel() == 0 or not torch.any(labels != -100):
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

    def _forward(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        if isinstance(batch, dict):
            output = self._active_model(**batch)
        elif isinstance(batch, (tuple, list)):
            if len(batch) < 2:
                raise ValueError("tuple batches must contain inputs and labels")
            output = self._active_model(batch[0], batch[1])
        else:
            raise TypeError(f"unsupported batch type: {type(batch)!r}")
        if isinstance(output, dict):
            if "loss" in output:
                return output["loss"], output
            loss = self._loss_from_logits(output, batch)
            if loss is None:
                raise ValueError("model output mapping must contain 'loss' or causal 'logits'")
            return loss, {**output, "loss": loss}
        if isinstance(output, (tuple, list)) and output:
            if not isinstance(output[0], torch.Tensor):
                raise ValueError("model output tuple must start with a loss tensor")
            if output[0].ndim == 3:
                loss = self._loss_from_logits(output[0], batch)
                if loss is not None:
                    return loss, {"logits": output[0], "loss": loss}
            return output[0], {"loss": output[0]}
        if not isinstance(output, torch.Tensor):
            raise TypeError("model must return a loss tensor or a mapping with loss")
        return output, {"loss": output}

    def _forward_with_compile_fallback(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        try:
            return self._forward(batch)
        except Exception as exc:
            if not self._compiled:
                raise
            self._compiled = False
            self._active_model = self.model
            self.logger.emit(
                "compile",
                {"enabled": False, "reason": f"runtime failure; disabled: {exc}"},
            )
            return self._forward(batch)

    @staticmethod
    def _batch_size_and_tokens(batch: Any) -> tuple[int, int]:
        candidate: Any = batch
        if isinstance(batch, dict):
            candidate = batch.get("input_ids", batch.get("inputs"))
        elif isinstance(batch, (tuple, list)) and batch:
            candidate = batch[0]
        if isinstance(candidate, torch.Tensor) and candidate.ndim > 0:
            tokens = candidate.numel()
            if isinstance(batch, dict) and candidate.ndim > 1 and "attention_mask" in batch:
                mask = batch["attention_mask"]
                tokens = int(mask.sum().item())
            return int(candidate.shape[0]), tokens
        return 1, 1

    def _optimizer_step(
        self,
        loss: torch.Tensor,
        final_microbatch: bool,
        loss_scale: float = 1.0,
    ) -> float:
        scaler = getattr(self, "_scaler", None)
        loss_value = float(loss.detach().float().cpu())
        backward_loss = loss if loss_scale == 1.0 else loss * loss_scale
        if final_microbatch:
            if scaler is not None:
                scaler.scale(backward_loss).backward()
                grad_clip = getattr(getattr(self.config, "optimizer", None), "grad_clip", None)
                if grad_clip is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                backward_loss.backward()
                grad_clip = getattr(getattr(self.config, "optimizer", None), "grad_clip", None)
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            return loss_value
        if scaler is not None:
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()
        return loss_value

    def _reset_inner_optimizer_if_needed(self) -> None:
        reset = getattr(getattr(self.config, "distributed", None), "reset_inner_optimizer", False)
        if reset:
            self.optimizer.state.clear()

    def _checkpoint(self) -> None:
        if self.checkpoint_manager is None or not self.checkpoint_manager.enabled:
            return
        if not self.checkpoint_manager.should_save(self.step):
            return
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self._scaler.state_dict()
            if getattr(self, "_scaler", None) is not None
            else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "step": self.step,
            "samples": self.samples,
            "tokens": self.tokens,
            "config": _checkpoint_config(self.config),
            "rng": capture_rng_state(),
            "coordinator": self.coordinator.state_dict() if self.coordinator is not None else None,
            "precision": {
                "mode": self.precision.mode,
                "autocast_enabled": self.precision.autocast_enabled,
                "use_scaler": self.precision.use_scaler,
            },
        }
        path = self.checkpoint_manager.save(self.step, state)
        self.logger.emit("checkpoint", {"step": self.step, "path": str(path)})

    def resume(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        if state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            self._deferred_scaler_state = state["scaler"]
        self.step = int(state.get("step", self.step))
        self.samples = int(state.get("samples", self.samples))
        self.tokens = int(state.get("tokens", self.tokens))
        if self.coordinator is not None and state.get("coordinator") is not None:
            # The coordinator may not have contacted Hub yet.  Defer applying
            # its state until fit() has started it, otherwise startup could
            # overwrite a resumed baseline with a fresh remote download.
            self._deferred_coordinator_state = state["coordinator"]
        restore_rng_state(state.get("rng"))

    def fit(self, max_steps: int | None = None) -> TrainResult:
        step_limit = self.max_steps if max_steps is None else int(max_steps)
        if step_limit <= 0:
            raise ValueError("max_steps must be positive")
        self.model.to(self.device)
        self._active_model = self._compiled_model() if self._compiled else self.model
        self._scaler = make_grad_scaler(self.precision)
        if self._scaler is not None and self._deferred_scaler_state is not None:
            self._scaler.load_state_dict(self._deferred_scaler_state)
            self._deferred_scaler_state = None
        self.optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        start_step = self.step
        self._start_samples = self.samples
        self._start_tokens = self.tokens
        final_loss: float | None = None
        metrics: list[dict[str, Any]] = []
        # ``max_steps`` is a global optimizer-step target, which makes a
        # resumed run idempotent instead of accidentally adding a second full
        # schedule on every invocation.
        target_step = max(self.step, step_limit)
        batches = infinite_batches(self.dataloader)
        if self.step < target_step and self.coordinator is not None and not self._started:
            self.coordinator.start()
            self._started = True
            if self._deferred_coordinator_state is not None:
                self.coordinator.load_state_dict(self._deferred_coordinator_state)
                self._deferred_coordinator_state = None
        self.logger.emit(
            "train_start",
            {
                "start_step": self.step,
                "target_step": target_step,
                "device": str(self.device),
                "precision": self.precision.mode,
                "accumulation_steps": self.accumulation_steps,
                "parameters": sum(p.numel() for p in self.model.parameters()),
            },
        )
        try:
            while self.step < target_step:
                loss_total = 0.0
                batch_count = 0
                for micro_index in range(self.accumulation_steps):
                    batch = self._move_batch(next(batches))
                    size, tokens = self._batch_size_and_tokens(batch)
                    self.samples += size
                    self.tokens += tokens
                    with autocast_context(self.precision, self.device):
                        loss, _ = self._forward_with_compile_fallback(batch)
                        if loss.ndim != 0:
                            loss = loss.mean()
                    loss_total += self._optimizer_step(
                        loss,
                        micro_index == self.accumulation_steps - 1,
                        loss_scale=1.0 / self.accumulation_steps,
                    )
                    batch_count += 1
                self.step += 1
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.coordinator is not None:
                    weights_replaced = self.coordinator.after_optimizer_step(self.model, self.step)
                    if weights_replaced:
                        self._reset_inner_optimizer_if_needed()
                current_lr = self.optimizer.param_groups[0].get("lr", 0.0)
                final_loss = loss_total / self.accumulation_steps
                metric = {
                    "step": self.step,
                    "loss": final_loss,
                    "lr": float(current_lr),
                    "steps_per_sec": (self.step - start_step)
                    / max(time.perf_counter() - started, 1e-9),
                    "samples_per_sec": (self.samples - self._start_samples)
                    / max(time.perf_counter() - started, 1e-9),
                    "tokens_per_sec": (self.tokens - self._start_tokens)
                    / max(time.perf_counter() - started, 1e-9),
                    "samples": self.samples,
                    "tokens": self.tokens,
                    "micro_batches": batch_count,
                }
                self.logger.emit("train_step", metric)
                metrics.append(metric)
                self._checkpoint()
        finally:
            if self.coordinator is not None:
                self.coordinator.stop()
        elapsed = time.perf_counter() - started
        result = TrainResult(
            steps=self.step,
            samples=self.samples,
            tokens=self.tokens,
            final_loss=final_loss,
            elapsed_s=elapsed,
            metrics=metrics,
        )
        self.logger.emit(
            "train_end", {"steps": self.step, "elapsed_s": elapsed, "loss": final_loss}
        )
        return result

    train = fit

    def _compiled_model(self) -> torch.nn.Module:
        return self._active_model


def _checkpoint_config(config: Any) -> Any:
    if not hasattr(config, "to_dict"):
        return None
    try:
        return config.to_dict(redact_secrets=True)
    except TypeError:
        return config.to_dict()


class _AutoPrecision:
    mode = "auto"
    dtype = None


TrainingEngine = Trainer
SpeedtronicTrainer = Trainer


__all__ = ["SpeedtronicTrainer", "SyncCoordinator", "TrainResult", "Trainer", "TrainingEngine"]
