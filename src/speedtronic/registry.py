"""Model registry used to keep the training engine architecture-neutral."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ModelFactory = Callable[..., Any]


class ModelRegistry:
    _builtin_names = {"reference_transformer", "gpt"}

    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {}

    def register(self, name: str, factory: ModelFactory | None = None):
        """Register a factory, usable as a decorator or a normal function."""

        def decorate(fn: ModelFactory) -> ModelFactory:
            if not name or not callable(fn):
                raise ValueError("a non-empty model name and callable factory are required")
            self._factories[name] = fn
            return fn

        return decorate(factory) if factory is not None else decorate

    def get(self, name: str) -> ModelFactory:
        if name not in self._factories and name in self._builtin_names:
            from . import model  # noqa: F401
        try:
            return self._factories[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "none"
            raise KeyError(f"unknown model '{name}'; registered models: {available}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._factories) | self._builtin_names))

    def build(self, name: str, **kwargs: Any) -> Any:
        return self.get(name)(**kwargs)


registry = ModelRegistry()


def register_model(name: str, factory: ModelFactory | None = None):
    return registry.register(name, factory)


def build_model(config: Any, **overrides: Any) -> Any:
    """Build a model from a :class:`~speedtronic.config.ModelConfig`.

    Extra keyword arguments are useful for custom models and are intentionally
    passed through without interpretation.
    """

    name = getattr(config, "name", "reference_transformer")
    kwargs = {
        "vocab_size": config.vocab_size,
        "block_size": config.max_seq_len,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_kv_head": config.n_kv_head,
        "d_model": config.d_model,
        "d_ff": config.d_ff,
        "dropout": config.dropout,
        "tie_weights": config.tie_weights,
        "rope_base": config.rope_base,
    }
    kwargs.update(overrides)
    return registry.build(name, **kwargs)


__all__ = ["ModelRegistry", "build_model", "register_model", "registry"]
