"""Lightweight stdout/file metrics and pluggable training hooks."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TextIO

Metric = dict[str, Any]
Hook = Callable[[str, dict[str, Any]], None]


class TrainingHook(Protocol):
    def on_event(self, event: str, payload: dict[str, Any]) -> None: ...


def memory_usage_bytes() -> int | None:
    """Return allocated accelerator memory when the backend exposes it."""

    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
        mps = getattr(torch, "mps", None)
        if mps is not None and mps.is_available():
            return int(mps.current_allocated_memory())
    except Exception:
        return None
    return None


@dataclass
class MetricLogger:
    level: str = "INFO"
    file: str | None = None
    json_file: str | None = None
    every_steps: int = 1
    stream: TextIO | None = None
    hooks: list[Hook] | None = None

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(f"speedtronic.metrics.{id(self)}")
        self._logger.setLevel(getattr(logging, self.level.upper(), logging.INFO))
        self._logger.propagate = False
        if not any(getattr(handler, "_speedtronic", False) for handler in self._logger.handlers):
            handler: logging.Handler
            if self.file:
                Path(self.file).parent.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(self.file, encoding="utf-8")
            else:
                handler = logging.StreamHandler(self.stream or sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            handler._speedtronic = True  # type: ignore[attr-defined]
            self._logger.addHandler(handler)
        if self.json_file:
            Path(self.json_file).parent.mkdir(parents=True, exist_ok=True)
        self.hooks = self.hooks or []
        self._start = time.perf_counter()
        self._emit_lock = threading.RLock()

    def emit(self, event: str, payload: Metric) -> None:
        with self._emit_lock:
            self._emit_locked(event, payload)

    def _emit_locked(self, event: str, payload: Metric) -> None:
        safe = dict(payload)
        safe.setdefault("elapsed_s", time.perf_counter() - self._start)
        memory = safe.get("memory_bytes")
        if memory is None:
            memory = memory_usage_bytes()
            if memory is not None:
                safe["memory_bytes"] = memory
        if event == "train_step" and self.every_steps > 1:
            step = int(safe.get("step", 0))
            if step % self.every_steps:
                return
        self._logger.info("%s %s", event, _format_metrics(safe))
        if self.json_file:
            with Path(self.json_file).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": event, **safe}, default=str) + "\n")
        for hook in self.hooks or []:
            try:
                callback = getattr(hook, "on_event", None)
                if callable(callback):
                    callback(event, safe)
                else:
                    hook(event, safe)
            except Exception as exc:  # hooks must not kill training
                self._logger.warning("metric hook failed: %s", exc)

    def log(self, level: int, message: str) -> None:
        self._logger.log(level, message)

    def debug(self, message: str, *args: Any) -> None:
        self._logger.debug(message, *args)

    def info(self, message: str, *args: Any) -> None:
        self._logger.info(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        self._logger.warning(message, *args)

    def error(self, message: str, *args: Any) -> None:
        self._logger.error(message, *args)

    def close(self) -> None:
        with self._emit_lock:
            for handler in list(self._logger.handlers):
                if getattr(handler, "_speedtronic", False):
                    handler.flush()
                    if isinstance(handler, logging.FileHandler):
                        handler.close()
                    self._logger.removeHandler(handler)


def _format_metrics(metrics: Metric) -> str:
    chunks: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, float):
            chunks.append(f"{key}={value:.6g}")
        else:
            chunks.append(f"{key}={value}")
    return " ".join(chunks)


class CallbackList:
    """Fan-out adapter for optional W&B/TensorBoard-style callbacks."""

    def __init__(self, callbacks: list[Any] | None = None) -> None:
        self.callbacks = list(callbacks or [])

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        for callback in self.callbacks:
            method = getattr(callback, "on_event", None)
            if method:
                method(event, payload)
            elif callable(callback):
                callback(event, payload)


__all__ = ["CallbackList", "MetricLogger", "TrainingHook", "memory_usage_bytes"]
