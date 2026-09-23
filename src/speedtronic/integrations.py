"""Optional logging integrations loaded only when explicitly requested."""

from __future__ import annotations

from typing import Any


class WandbHook:
    def __init__(self, project: str | None = None, **settings: Any) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install speedtronic[logging] to use the W&B hook") from exc
        self._wandb = wandb
        self._run = wandb.init(project=project, **settings)

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        self._wandb.log({**payload, "event": event})


class TensorboardHook:
    def __init__(self, log_dir: str = "runs") -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install speedtronic[logging] to use the TensorBoard hook") from exc
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{event}/{key}", value)

    def close(self) -> None:
        self.writer.close()


__all__ = ["TensorboardHook", "WandbHook"]
