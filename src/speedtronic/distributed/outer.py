"""Outer-loop optimizer and master polling loop for DumbDiLoCo."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from ..checkpoint import atomic_json_dump, atomic_torch_save
from .hub import HubClient
from .tensors import average_deltas, load_delta, parse_delta_path

LOGGER = logging.getLogger("speedtronic.diloco")


class NesterovOuterOptimizer:
    """Nesterov momentum SGD over a floating-point model state."""

    def __init__(self, state: dict[str, torch.Tensor], lr: float, momentum: float = 0.9) -> None:
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.momentum_buffers = {
            key: torch.zeros_like(value, device="cpu", dtype=torch.float32)
            for key, value in state.items()
            if value.is_floating_point() or value.is_complex()
        }

    @torch.no_grad()
    def step(self, state: dict[str, torch.Tensor], delta: dict[str, torch.Tensor]) -> None:
        """Apply one outer update in place on CPU state tensors."""

        for key, gradient in delta.items():
            if key not in state:
                raise KeyError(f"outer delta contains unknown state key: {key}")
            if state[key].shape != gradient.shape:
                raise ValueError(f"shape mismatch for outer state key {key}")
            if not (state[key].is_floating_point() or state[key].is_complex()):
                continue
            gradient = gradient.to(device=state[key].device, dtype=torch.float32)
            buffer = self.momentum_buffers.setdefault(
                key, torch.zeros_like(state[key], device=state[key].device, dtype=torch.float32)
            )
            buffer.mul_(self.momentum).add_(gradient)
            # PyTorch's Nesterov formulation uses grad + momentum * d_p.
            effective = gradient + self.momentum * buffer
            state[key].add_(effective.to(dtype=state[key].dtype), alpha=-self.lr)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in self.momentum_buffers.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.momentum_buffers = {
            str(key): value.detach().to(device="cpu", dtype=torch.float32).clone()
            for key, value in state.items()
        }


@dataclass
class OuterState:
    global_state: dict[str, torch.Tensor]
    outer_step: int = 0
    processed_deltas: set[tuple[str, int, int]] = field(default_factory=set)
    momentum: dict[str, torch.Tensor] = field(default_factory=dict)
    last_error: str | None = None


class MasterOuterLoop:
    """Poll and aggregate deltas without coupling polling to inner training.

    The loop owns a CPU copy of the global state and publishes a complete model
    plus metadata after every successful round.  A private thread is used for
    the master role, so the training thread can continue its local loop.
    """

    def __init__(
        self,
        hub: HubClient,
        state: dict[str, torch.Tensor],
        *,
        node_state_dir: str | Path,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        poll_interval: float = 60.0,
        on_global_update: Callable[[dict[str, torch.Tensor], int], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.hub = hub
        self.global_state = {key: value.detach().cpu().clone() for key, value in state.items()}
        self.state_dir = Path(node_state_dir)
        self.poll_interval = float(poll_interval)
        self.outer_lr = float(outer_lr)
        self.outer_momentum = float(outer_momentum)
        self.on_global_update = on_global_update
        self.logger = logger or LOGGER
        self.outer_step = 0
        self.processed_deltas: set[tuple[str, int, int]] = set()
        self.optimizer = NesterovOuterOptimizer(self.global_state, outer_lr, outer_momentum)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._load_local_state()

    @property
    def processed_filenames(self) -> set[str]:
        return {
            f"nodes/{node_id}/delta_{local_step}.safetensors"
            for node_id, local_step, _base_step in self.processed_deltas
        }

    @property
    def state_path(self) -> Path:
        return self.state_dir / "outer_state.pt"

    @property
    def metadata_path(self) -> Path:
        return self.state_dir / "outer_metadata.json"

    def _load_local_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = torch.load(self.state_path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover
            payload = torch.load(self.state_path, map_location="cpu")
        if not isinstance(payload, dict):
            return
        saved_state = payload.get("global_state")
        if saved_state:
            # Only replace matching tensors; this keeps forward-compatible
            # metadata additions from making an old checkpoint unusable.
            for key, value in saved_state.items():
                if key in self.global_state and tuple(self.global_state[key].shape) == tuple(
                    value.shape
                ):
                    self.global_state[key] = value.detach().cpu().clone()
        self.outer_step = int(payload.get("outer_step", 0))
        self.processed_deltas = {
            tuple(item) for item in payload.get("processed_deltas", []) if len(item) == 3
        }
        self.optimizer.load_state_dict(payload.get("momentum", {}))

    def _save_local_state(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            {
                "global_state": self.global_state,
                "outer_step": self.outer_step,
                "processed_deltas": [list(item) for item in sorted(self.processed_deltas)],
                "momentum": self.optimizer.state_dict(),
            },
            self.state_path,
        )
        atomic_json_dump(
            {
                "outer_step": self.outer_step,
                "processed_deltas": [list(item) for item in sorted(self.processed_deltas)],
                "updated_at": _utc_now(),
            },
            self.metadata_path,
        )

    def _delta_candidates(self) -> list[tuple[str, int, int, str]]:
        candidates: list[tuple[str, int, int, str]] = []
        for path in self.hub.delta_paths():
            parsed = parse_delta_path(path)
            if parsed is None:
                continue
            node_id, local_step = parsed
            # The base outer step is metadata-level information.  Use -1 as a
            # provisional key until the file is downloaded; duplicate names
            # from a restart are then conservatively considered once per name.
            candidates.append((node_id, local_step, -1, path))
        # Files with known metadata are filtered after loading.  Sorting makes
        # rounds deterministic and simplifies tests.
        return sorted(candidates, key=lambda item: (item[0], item[1], item[3]))

    def _download_delta(self, path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
        local = self.state_dir / "cache" / "deltas" / Path(path).name
        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = self.hub.download(path, local)
        return load_delta(downloaded)

    def sync_once(self) -> int:
        """Run one outer round and return the resulting outer step.

        Hub listing and downloads happen outside the state lock.  Only the
        short in-memory commit and local-state write are serialized, so a
        training-thread checkpoint never waits for a network round trip.
        """

        with self._lock:
            processed = set(self.processed_deltas)
            expected = {
                key
                for key, value in self.global_state.items()
                if value.is_floating_point() or value.is_complex()
            }
            shapes = {key: tuple(value.shape) for key, value in self.global_state.items()}

        candidates = self._delta_candidates()
        valid: list[tuple[tuple[str, int, int], dict[str, torch.Tensor], str]] = []
        for node_id, local_step, _provisional, path in candidates:
            try:
                delta, metadata = self._download_delta(path)
                base_step = int(metadata.get("base_outer_step", -1))
                identity = (node_id, local_step, base_step)
                if identity in processed:
                    continue
                # Validate shape/key compatibility before aggregation.
                if set(delta) != expected:
                    raise ValueError("delta keys do not match floating-point global state")
                for key, value in delta.items():
                    if tuple(value.shape) != shapes[key]:
                        raise ValueError(f"shape mismatch for {key}")
                valid.append((identity, delta, path))
            except Exception as exc:
                self.logger.warning("skipping unreadable delta %s: %s", path, exc)
        if not valid:
            self._emit_outer_event(len(candidates), 0)
            with self._lock:
                return self.outer_step

        # Do not include a provisional -1 identity in the final state.
        average = average_deltas([item[1] for item in valid])
        with self._lock:
            old_step = self.outer_step
            old_processed = set(self.processed_deltas)
            old_global = {key: value.clone() for key, value in self.global_state.items()}
            old_momentum = self.optimizer.state_dict()
            self.optimizer.step(self.global_state, average)
            self.outer_step += 1
            self.processed_deltas.update(item[0] for item in valid)
            candidate_global = {key: value.clone() for key, value in self.global_state.items()}
            candidate_step = self.outer_step

        try:
            # Network I/O is intentionally outside self._lock.
            self.hub.publish_global(
                candidate_global,
                outer_step=candidate_step,
                work_dir=self.state_dir / "outgoing",
                metadata_extra={"algorithm": "dumb_diloco", "optimizer": "nesterov_sgd"},
            )
        except Exception:
            # Do not mark a failed publication as processed.  Restore the
            # complete outer state so a retry applies the delta exactly once.
            with self._lock:
                self.global_state = old_global
                self.outer_step = old_step
                self.processed_deltas = old_processed
                self.optimizer.load_state_dict(old_momentum)
            raise
        try:
            with self._lock:
                self._save_local_state()
        except Exception as exc:
            # Publication already succeeded; rolling back here would reapply
            # the same deltas on the next round and diverge from the Hub.
            self.logger.warning("published global but could not persist local outer state: %s", exc)
        if self.on_global_update is not None:
            self.on_global_update(candidate_global, candidate_step)
        self._emit_outer_event(len(candidates), len(valid))
        return candidate_step

    def _emit_outer_event(self, found: int, valid: int) -> None:
        if hasattr(self.logger, "emit"):
            self.logger.emit(
                "outer_step",
                {
                    "outer_step": self.outer_step,
                    "deltas_found": found,
                    "deltas_included": valid,
                },
            )
        else:
            self.logger.info(
                "outer sync: step=%s found=%s valid=%s",
                self.outer_step,
                found,
                valid,
            )

    def adopt_global_state(
        self,
        state: dict[str, torch.Tensor],
        outer_step: int,
        *,
        reset_momentum: bool = False,
    ) -> None:
        """Recover a newer remotely published global state after a crash."""

        with self._lock:
            self.global_state = {key: value.detach().cpu().clone() for key, value in state.items()}
            self.outer_step = int(outer_step)
            if reset_momentum:
                self.optimizer = NesterovOuterOptimizer(
                    self.global_state, self.outer_lr, self.outer_momentum
                )
            self._save_local_state()

    def snapshot(self) -> tuple[int, dict[str, torch.Tensor]]:
        with self._lock:
            return self.outer_step, {key: value.clone() for key, value in self.global_state.items()}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="speedtronic-diloco-master", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sync_once()
            except Exception as exc:
                self.logger.warning("DumbDiLoCo outer poll failed: %s", exc)
            self._stop.wait(self.poll_interval)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, min(self.poll_interval, 10.0)))
        self._thread = None

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "outer_step": self.outer_step,
                "processed_deltas": [list(item) for item in sorted(self.processed_deltas)],
                "momentum": self.optimizer.state_dict(),
                "global_state": {key: value.clone() for key, value in self.global_state.items()},
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with self._lock:
            self.outer_step = int(state.get("outer_step", 0))
            self.processed_deltas = {
                tuple(item) for item in state.get("processed_deltas", []) if len(item) == 3
            }
            self.optimizer.load_state_dict(state.get("momentum", {}))
            saved = state.get("global_state", {})
            for key, value in saved.items():
                if key in self.global_state:
                    self.global_state[key] = value.detach().cpu().clone()
            self._save_local_state()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["MasterOuterLoop", "NesterovOuterOptimizer", "OuterState"]
