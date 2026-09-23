"""Small public helpers for user modules."""

from __future__ import annotations

from typing import Any


def set_gradient_checkpointing(module: Any, enabled: bool = True) -> Any:
    """Enable a module's explicit gradient-checkpointing convention.

    Speedtronic intentionally does not inspect transformer internals.  A custom
    module opts in by exposing ``set_gradient_checkpointing(enabled)``.
    """

    hook = getattr(module, "set_gradient_checkpointing", None)
    if not callable(hook):
        raise AttributeError(
            "module must expose set_gradient_checkpointing(enabled) to use this feature"
        )
    hook(bool(enabled))
    return module


__all__ = ["set_gradient_checkpointing"]
