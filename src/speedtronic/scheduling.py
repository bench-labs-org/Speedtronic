"""Dependency-aware forward/backward stream scheduling for v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class StageInfo:
    name: str
    stream_index: int


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def _expand_stage_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find disjoint, meaningful stage roots without nesting hooks."""

    explicit = getattr(model, "ooo_stage_names", None)
    if explicit is not None:
        modules = dict(model.named_modules())
        selected: list[tuple[str, nn.Module]] = []
        seen_names: set[str] = set()
        for raw_name in explicit:
            name = str(raw_name)
            if name in seen_names:
                raise ValueError(f"duplicate ooo stage name: {name!r}")
            seen_names.add(name)
            if name in modules and modules[name] is not model:
                selected.append((name, modules[name]))
        names = [name for name, _ in selected]
        for index, name in enumerate(names):
            for other in names[index + 1 :]:
                if name.startswith(other + ".") or other.startswith(name + "."):
                    raise ValueError(
                        f"ooo stage names must be disjoint; {name!r} and {other!r} are nested"
                    )
        if selected:
            return selected

    selected = []
    seen: set[int] = set()

    def visit(name: str, module: nn.Module) -> None:
        children = list(module.named_children())
        if not children:
            if module is not model and id(module) not in seen:
                seen.add(id(module))
                selected.append((name, module))
            return
        for child_name, child in children:
            child_full = f"{name}.{child_name}" if name else child_name
            visit(child_full, child)

    for child_name, child in model.named_children():
        visit(child_name, child)
    return selected


class StageStreamScheduler:
    """Run disjoint module stages on CUDA streams for autograd's DAG engine.

    PyTorch's autograd engine already performs dependency-aware out-of-order
    node execution.  The engine still routes a node to the forward stream that
    created it, so this scheduler assigns disjoint forward stages to multiple
    streams.  Cross-stream inputs wait on their producer and ``record_stream``
    protects allocator reuse.  CPU and MPS intentionally fall back to the
    standard sequential path.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        *,
        num_streams: int = 4,
        logger: Any | None = None,
    ) -> None:
        self.logger = logger
        self.device = torch.device(device)
        self.requested_streams = max(1, min(8, int(num_streams)))
        self.enabled = False
        self.reason = ""
        self.stages: list[StageInfo] = []
        self._streams: list[torch.cuda.Stream] = []
        self._stage_streams: dict[int, torch.cuda.Stream] = {}
        self._producer: dict[int, torch.cuda.Stream] = {}
        self._handles: list[Any] = []
        self._contexts: list[tuple[Any, torch.cuda.Stream]] = []
        self._current_stream: torch.cuda.Stream | None = None
        self._installed = False

        if self.device.type != "cuda":
            self.reason = f"device {self.device.type} has no supported CUDA stream scheduler"
            return
        if not hasattr(torch, "cuda") or not torch.cuda.is_available():
            self.reason = "CUDA is unavailable"
            return
        try:
            modules = _expand_stage_modules(model)
            if len(modules) < 2:
                self.reason = "fewer than two independent module stages were found"
                return
            count = min(self.requested_streams, len(modules))
            self._streams = [torch.cuda.Stream(device=self.device) for _ in range(count)]
            self._install_hooks(modules)
            self.stages = [
                StageInfo(name, index % count) for index, (name, _) in enumerate(modules)
            ]
            self._stage_streams = {
                id(module): self._streams[index % count]
                for index, (_, module) in enumerate(modules)
            }
            self.enabled = True
            self._installed = True
            self.reason = "enabled"
        except Exception as exc:  # degrade, never fail training
            self.remove_hooks()
            self.enabled = False
            self.reason = f"stream scheduling setup failed: {exc}"
            if self.logger is not None:
                if hasattr(self.logger, "warning"):
                    self.logger.warning("ooo_backprop disabled: %s", self.reason)
                elif hasattr(self.logger, "emit"):
                    self.logger.emit("ooo_backprop", {"enabled": False, "reason": self.reason})

    def _install_hooks(self, modules: list[tuple[str, nn.Module]]) -> None:
        for _, module in modules:
            self._handles.append(module.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
            self._handles.append(module.register_forward_hook(self._post_hook))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()
        self._installed = False

    def _pre_hook(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if not self.enabled or self._current_stream is None:
            return None
        stream = self._stage_streams.get(id(module))
        if stream is None:
            return None
        for tensor in _iter_tensors((args, kwargs)):
            producer = self._producer.get(id(tensor))
            if producer is not None and producer is not stream:
                stream.wait_stream(producer)
            if tensor.device.type == "cuda":
                tensor.record_stream(stream)
        context = torch.cuda.stream(stream)
        context.__enter__()
        self._contexts.append((context, stream))
        return None

    def _post_hook(self, module: nn.Module, args: tuple[Any, ...], output: Any):
        if not self._contexts:
            return None
        context, stream = self._contexts.pop()
        context.__exit__(None, None, None)
        # Unhooked parent/container operations run on the ambient stream.
        # Synchronize here so a stage can never be read by an untracked op
        # before its producer finishes; this is a correctness-first fallback.
        if self._current_stream is not None and self._current_stream != stream:
            self._current_stream.wait_stream(stream)
        for tensor in _iter_tensors(output):
            self._producer[id(tensor)] = stream
        return None

    def begin(self) -> None:
        if not self.enabled:
            return
        self._current_stream = torch.cuda.current_stream(self.device)
        self._producer.clear()
        for stream in self._streams:
            stream.wait_stream(self._current_stream)
        self._contexts.clear()

    def finish(self) -> None:
        if not self.enabled or self._current_stream is None:
            return
        # Ensure the host-visible loss/counter read is safe after stage work.
        for stream in self._streams:
            self._current_stream.wait_stream(stream)
        while self._contexts:
            context, _ = self._contexts.pop()
            context.__exit__(None, None, None)
        self._producer.clear()
        self._current_stream = None

    def dispose(self) -> None:
        self.finish()
        self.remove_hooks()
        self.enabled = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "streams": len(self._streams),
            "stages": [stage.name for stage in self.stages],
        }


__all__ = ["StageInfo", "StageStreamScheduler"]
