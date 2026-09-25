"""Startup shape-efficiency warnings for configured batches and model GEMMs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
from torch import nn

from .precision import PrecisionPlan


@dataclass(frozen=True)
class ShapeProfile:
    device_type: str
    precision: str
    alignment: int | None
    source: str


@dataclass(frozen=True)
class ShapeWarning:
    code: str
    field: str
    value: int
    suggested: int | None
    alignment: int
    reason: str


@dataclass
class ShapeReport:
    profile: ShapeProfile
    warnings: list[ShapeWarning] = field(default_factory=list)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": asdict(self.profile),
            "warning_count": self.warning_count,
            "warnings": [asdict(item) for item in self.warnings],
        }


def _suggest(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def resolve_shape_profile(
    device: torch.device | str,
    precision: PrecisionPlan,
    *,
    requested_alignment: int | str | None = "auto",
) -> ShapeProfile:
    device = torch.device(device)
    requested = "auto" if requested_alignment is None else str(requested_alignment).lower()
    if requested not in {"auto", "none"}:
        alignment = int(requested)
        if alignment <= 0:
            raise ValueError("shape alignment must be positive")
        return ShapeProfile(device.type, precision.mode, alignment, "explicit")

    if device.type != "cuda" or precision.mode not in {"bf16", "fp16"}:
        return ShapeProfile(device.type, precision.mode, None, "device-default")
    return ShapeProfile(device.type, precision.mode, 8, "cuda-mixed-precision")


def _append_warning(
    warnings: list[ShapeWarning],
    seen: set[tuple[str, int, int]],
    *,
    code: str,
    name: str,
    value: int,
    alignment: int,
    reason: str,
    suppress_suggestion: bool = False,
) -> None:
    key = (name, value, alignment)
    if key in seen or value <= 0 or value % alignment == 0:
        return
    seen.add(key)
    warnings.append(
        ShapeWarning(
            code=code,
            field=name,
            value=value,
            suggested=None if suppress_suggestion else _suggest(value, alignment),
            alignment=alignment,
            reason=reason,
        )
    )


def _module_metadata(model: nn.Module) -> list[tuple[str, int]]:
    metadata: list[tuple[str, int]] = []
    hook = getattr(model, "speedtronic_shape_metadata", None)
    if callable(hook):
        try:
            value = hook()
            if isinstance(value, Mapping):
                metadata.extend((str(key), int(item)) for key, item in value.items())
                return metadata
        except Exception:
            pass

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            metadata.append((f"{name}.in_features", int(module.in_features)))
            metadata.append((f"{name}.out_features", int(module.out_features)))
        elif isinstance(module, nn.Embedding):
            metadata.append((f"{name}.embedding_dim", int(module.embedding_dim)))
        elif isinstance(module, nn.Conv2d):
            metadata.append((f"{name}.in_channels", int(module.in_channels)))
            metadata.append((f"{name}.out_channels", int(module.out_channels)))
    return metadata


def validate_startup_shapes(
    config: Any,
    model: nn.Module,
    device: torch.device | str,
    precision: PrecisionPlan,
    *,
    logger: Any | None = None,
) -> ShapeReport:
    """Inspect effective shapes and return non-fatal efficiency warnings."""

    shape_config = getattr(config, "shape_validation", None)
    enabled = bool(getattr(shape_config, "enabled", True))
    profile = resolve_shape_profile(
        device,
        precision,
        requested_alignment=getattr(shape_config, "alignment", "auto"),
    )
    report = ShapeReport(profile=profile)
    device_type = torch.device(device).type
    if not enabled or profile.alignment is None:
        return report
    if device_type in {"cpu", "mps"} and not bool(getattr(shape_config, "warn_on_cpu", False)):
        # An explicit alignment is a deliberate benchmarking request, so it
        # still needs an explicit opt-in on non-CUDA devices.
        return report

    data = getattr(config, "data", None)
    model_config = getattr(config, "model", None)
    alignment = profile.alignment
    seen: set[tuple[str, int, int]] = set()
    micro_batch = int(getattr(data, "micro_batch_size", 0) or 0)
    target_batch = int(getattr(data, "target_batch_size", 0) or 0)
    if getattr(shape_config, "check_batch", True):
        _append_warning(
            report.warnings,
            seen,
            code="batch_alignment",
            name="data.micro_batch_size",
            value=micro_batch,
            alignment=alignment,
            reason="kernel batches near this alignment are usually more efficient",
            # A suggested micro batch must remain divisible into the target
            # batch; otherwise the warning would recommend an invalid config.
            suppress_suggestion=target_batch > 0 and target_batch % alignment != 0,
        )
    if getattr(shape_config, "check_sequence", True):
        block_size = int(getattr(data, "block_size", 0) or 0)
        _append_warning(
            report.warnings,
            seen,
            code="sequence_alignment",
            name="data.block_size",
            value=block_size,
            alignment=alignment,
            reason="long unaligned sequence GEMMs can reduce accelerator utilization",
        )
        _append_warning(
            report.warnings,
            seen,
            code="token_alignment",
            name="data.micro_batch_size * data.block_size",
            value=micro_batch * block_size,
            alignment=alignment,
            reason="effective token count is an important GEMM dimension",
        )
    if getattr(shape_config, "check_model", True):
        for name, value in _module_metadata(model):
            _append_warning(
                report.warnings,
                seen,
                code="model_alignment",
                name=name,
                value=value,
                alignment=alignment,
                reason="unaligned projection dimensions can reduce tensor-core efficiency",
            )
    if getattr(shape_config, "check_vocab", False):
        _append_warning(
            report.warnings,
            seen,
            code="vocab_alignment",
            name="model.vocab_size",
            value=int(getattr(model_config, "vocab_size", 0) or 0),
            alignment=alignment,
            reason="vocabulary projections can benefit from aligned output widths",
        )

    if logger is not None:
        warning = getattr(logger, "warning", None)
        emit = getattr(logger, "emit", None)
        for item in report.warnings:
            if callable(warning):
                warning(
                    "shape warning: %s=%s is not a multiple of %s; consider %s (%s)",
                    item.field,
                    item.value,
                    item.alignment,
                    item.suggested,
                    item.reason,
                )
            if callable(emit):
                emit("shape_warning", asdict(item))
        if callable(emit):
            emit(
                "shape_profile",
                {
                    "device_type": profile.device_type,
                    "precision": profile.precision,
                    "alignment": profile.alignment,
                    "warning_count": report.warning_count,
                },
            )
    return report


__all__ = [
    "ShapeProfile",
    "ShapeReport",
    "ShapeWarning",
    "resolve_shape_profile",
    "validate_startup_shapes",
]
