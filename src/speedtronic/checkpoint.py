"""Atomic local checkpoint storage and discovery."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch


class CheckpointError(RuntimeError):
    pass


def atomic_torch_save(state: dict[str, Any], path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def atomic_json_dump(value: Any, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class CheckpointManager:
    """Manage local ``step_<n>.pt`` checkpoints and a latest pointer."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        every_steps: int = 500,
        keep_last: int | None = 3,
        enabled: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.every_steps = int(every_steps)
        self.keep_last = keep_last
        self.enabled = enabled
        if self.every_steps <= 0:
            raise ValueError("every_steps must be positive")

    def should_save(self, step: int) -> bool:
        return self.enabled and step > 0 and step % self.every_steps == 0

    def path_for(self, step: int) -> Path:
        return self.directory / f"step_{int(step):012d}.pt"

    def save(self, step: int, state: dict[str, Any]) -> Path:
        path = self.path_for(step)
        atomic_torch_save(state, path)
        atomic_json_dump({"step": int(step), "file": path.name}, self.directory / "latest.json")
        self.prune()
        return path

    def load_latest(self) -> dict[str, Any] | None:
        path = self.latest_path()
        if path is None:
            return None
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # older PyTorch
            state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            raise CheckpointError(f"checkpoint {path} does not contain a state dictionary")
        return state

    def latest_path(self) -> Path | None:
        pointer = self.directory / "latest.json"
        if pointer.exists():
            try:
                value = json.loads(pointer.read_text(encoding="utf-8"))
                candidate = self.directory / str(value["file"])
                if candidate.exists():
                    return candidate
            except (OSError, KeyError, ValueError, TypeError):
                pass
        candidates = self._checkpoint_paths()
        return candidates[-1] if candidates else None

    def _checkpoint_paths(self) -> list[Path]:
        if not self.directory.exists():
            return []
        paths = []
        for path in self.directory.glob("step_*.pt"):
            match = re.fullmatch(r"step_(\d+)\.pt", path.name)
            if match:
                paths.append((int(match.group(1)), path))
        return [path for _, path in sorted(paths)]

    def prune(self) -> None:
        if self.keep_last is None:
            return
        paths = self._checkpoint_paths()
        for path in paths[: -self.keep_last]:
            try:
                path.unlink()
            except OSError:
                pass

    def copy_to(self, destination: str | os.PathLike[str]) -> Path:
        source = self.latest_path()
        if source is None:
            raise CheckpointError("no checkpoint is available to copy")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": __import__("random").getstate(),
        "numpy": None,
        "torch": torch.get_rng_state(),
    }
    try:  # NumPy is optional and commonly present in training environments.
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    import random

    try:
        random.setstate(state["python"])
    except Exception:
        pass
    try:
        torch.set_rng_state(state["torch"])
    except Exception:
        pass
    try:
        import numpy as np

        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
    except Exception:
        pass
    try:
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception:
        pass


__all__ = [
    "CheckpointError",
    "CheckpointManager",
    "atomic_json_dump",
    "atomic_torch_save",
    "capture_rng_state",
    "restore_rng_state",
]
