"""Safetensors helpers with defensive tensor normalization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    # Clone after moving/contiguing: CPU parameters can share storage (for
    # example tied embeddings), which safetensors rejects.
    return value.detach().to(device="cpu").contiguous().clone()


def cpu_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {str(key): cpu_tensor(value) for key, value in state.items()}


def save_safetensors(
    state: dict[str, torch.Tensor],
    path: str | os.PathLike[str],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for Speedtronic Hub files") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Tied weights can share storage; safetensors requires independent buffers.
    normalized = {key: cpu_tensor(value) for key, value in state.items()}
    save_file(normalized, str(path), metadata=metadata or {})


def load_safetensors(path: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for Speedtronic Hub files") from exc
    return load_file(str(path), device="cpu")


def read_metadata(path: str | os.PathLike[str]) -> dict[str, str]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for Speedtronic Hub files") from exc
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def floating_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state.items()
        if value.is_floating_point() or value.is_complex()
    }


def compute_pseudo_gradient(
    baseline: dict[str, torch.Tensor], current: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Compute ``baseline - current`` for floating-point state tensors."""

    if set(baseline) != set(current):
        missing_current = sorted(set(baseline) - set(current))
        missing_baseline = sorted(set(current) - set(baseline))
        raise ValueError(
            "state key mismatch; "
            f"missing current={missing_current}, missing baseline={missing_baseline}"
        )
    delta: dict[str, torch.Tensor] = {}
    for key, base in baseline.items():
        value = current[key]
        if base.shape != value.shape:
            raise ValueError(
                f"shape mismatch for {key}: {tuple(base.shape)} vs {tuple(value.shape)}"
            )
        if base.is_floating_point() or base.is_complex():
            delta[key] = base.to(dtype=torch.float32) - value.to(dtype=torch.float32)
    return delta


def average_deltas(deltas: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not deltas:
        raise ValueError("cannot average an empty delta list")
    keys = set(deltas[0])
    if any(set(delta) != keys for delta in deltas[1:]):
        raise ValueError("all deltas must contain the same keys")
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        reference = deltas[0][key]
        total = torch.zeros_like(reference, dtype=torch.float32)
        for delta in deltas:
            if delta[key].shape != reference.shape:
                raise ValueError(f"shape mismatch for delta key {key}")
            total += delta[key].to(dtype=torch.float32)
        result[key] = total / len(deltas)
    return result


def parse_delta_path(path: str) -> tuple[str, int] | None:
    parts = path.split("/")
    if len(parts) != 3 or parts[0] != "nodes" or not parts[2].startswith("delta_"):
        return None
    try:
        step = int(parts[2][len("delta_") :].removesuffix(".safetensors"))
    except ValueError:
        return None
    return parts[1], step


def load_delta(path: str | os.PathLike[str]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    metadata = read_metadata(path)
    try:
        state = load_safetensors(path)
    except Exception:
        # Re-raise with a stable exception type while retaining the original as
        # the cause; the outer loop logs the file name and skips it.
        raise ValueError(f"unable to load safetensors file {path}") from None
    return state, metadata


def metadata_json(metadata: dict[str, str]) -> dict[str, Any]:
    try:
        return {key: json.loads(value) for key, value in metadata.items()}
    except (TypeError, ValueError):
        return dict(metadata)


__all__ = [
    "average_deltas",
    "compute_pseudo_gradient",
    "cpu_state_dict",
    "cpu_tensor",
    "floating_state",
    "load_delta",
    "load_safetensors",
    "metadata_json",
    "parse_delta_path",
    "read_metadata",
    "save_safetensors",
]
