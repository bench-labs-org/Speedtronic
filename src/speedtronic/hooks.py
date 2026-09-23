"""Optional hook and callback helpers."""

from __future__ import annotations

from typing import Any, Callable


def on_event(callback: Callable[[str, dict[str, Any]], None]):
    """Return a callback wrapper suitable for ``MetricLogger(hooks=...)``."""

    def wrapped(event: str, payload: dict[str, Any]) -> None:
        callback(event, payload)

    return wrapped


__all__ = ["on_event"]
