"""DumbDiLoCo coordinator integrated with the ordinary training loop."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .hub import GlobalMetadata, HubClient
from .outer import MasterOuterLoop
from .tensors import compute_pseudo_gradient, cpu_state_dict, load_safetensors

LOGGER = logging.getLogger("speedtronic.diloco")


@dataclass
class SyncResult:
    pushed: bool = False
    loaded_global: bool = False
    outer_step: int = 0


class DumbDiLoCoCoordinator:
    """Coordinate local inner loops and a Hub-backed outer loop.

    The trainer calls :meth:`after_optimizer_step` after each local optimizer
    step.  At ``inner_steps`` boundaries the coordinator computes and uploads a
    pseudo-gradient; at other steps it performs a cheap time-based poll for a
    newer global version.  The master additionally owns a background
    :class:`MasterOuterLoop` thread.
    """

    def __init__(
        self,
        config: Any,
        model: torch.nn.Module,
        *,
        hub: HubClient | None = None,
        state_dir: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if isinstance(config, dict):
            from ..config import SpeedtronicConfig

            config = SpeedtronicConfig.from_dict(config).distributed
        elif hasattr(config, "distributed"):
            config = config.distributed
        self.config = config
        self.model = model
        configured_node_id = getattr(config, "node_id", None)
        self.node_id = str(
            configured_node_id or os.getenv("HF_USERNAME") or os.getenv("USER") or "local"
        )
        self.role = str(config.role).lower()
        self.inner_steps = int(config.inner_steps)
        self.poll_interval = float(config.poll_interval)
        self.logger = logger or LOGGER
        self.hub = (
            hub
            if hub is not None
            else HubClient(
                config.repo_id,
                token=config.token,
                cache_dir=config.cache_dir,
                retry_initial=config.retry_initial,
                retry_max=config.retry_max,
                retry_attempts=config.retry_attempts,
            )
        )
        state_root = Path(state_dir or config.state_dir)
        # Keep per-node local metadata and caches separate when several nodes
        # share a filesystem (the Hub namespace is the only shared resource).
        self.state_dir = state_root / self.node_id
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._started = False
        self._stopped = False
        self._last_poll = 0.0
        self._last_global_step = -1
        self._last_uploaded_step: int | None = None
        self._local_step = 0
        self._baseline = cpu_state_dict(model.state_dict())
        self._outer_loop: MasterOuterLoop | None = None
        self._outer_snapshot: dict[str, torch.Tensor] | None = None
        self._outer_snapshot_step = -1
        self._pending_delta: dict[str, torch.Tensor] | None = None
        self._pending_delta_step: int | None = None

    @property
    def local_step(self) -> int:
        return self._local_step

    @property
    def last_global_step(self) -> int:
        return self._last_global_step

    @property
    def outer_step(self) -> int:
        if self._outer_loop is not None:
            return self._outer_loop.outer_step
        return max(self._last_global_step, 0)

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._stopped = False
            if self.role == "master":
                self._start_master()
            else:
                self._start_worker()
            self._started = True
            self._last_poll = time.monotonic()
            self.logger.info(
                "DumbDiLoCo started role=%s node_id=%s global_step=%s",
                self.role,
                self.node_id,
                self._last_global_step,
            )

    def _start_master(self) -> None:
        try:
            self.hub.create_repo(private=True)
        except Exception as exc:
            # Keep the local inner loop alive; the background outer poller will
            # retry repository access on its next interval.
            self.logger.warning("master Hub initialization failed; continuing locally: %s", exc)
        for collaborator in getattr(self.config, "collaborators", []) or []:
            try:
                self.hub.add_collaborator(collaborator, permission="write")
            except Exception as exc:
                self.logger.warning("could not grant write access to %s: %s", collaborator, exc)
        # Constructing the outer loop first lets a locally persisted master
        # state win over a stale remote model after a crash.
        initial_state = cpu_state_dict(self.model.state_dict())
        self._outer_loop = MasterOuterLoop(
            self.hub,
            initial_state,
            node_state_dir=self.state_dir,
            outer_lr=self.config.outer_lr,
            outer_momentum=self.config.outer_momentum,
            poll_interval=self.poll_interval,
            on_global_update=self._on_outer_update,
            logger=self.logger,
        )
        try:
            remote_metadata = self.hub.global_metadata()
        except Exception as exc:
            remote_metadata = None
            self.logger.warning("master could not read global metadata: %s", exc)
        if self._outer_loop.outer_step > 0 and (
            remote_metadata is None or remote_metadata.outer_step <= self._outer_loop.outer_step
        ):
            self._install_state(self._outer_loop.global_state)
            self._last_global_step = self._outer_loop.outer_step
        elif remote_metadata is not None:
            try:
                self._refresh_from_hub(remote_metadata)
                # If a publication completed remotely just before a master
                # crash, prefer that newer complete model over stale local
                # state.  The local processed set is retained; momentum is
                # reset because the corresponding remote momentum is private.
                if self._outer_loop.outer_step < remote_metadata.outer_step:
                    self._outer_loop.adopt_global_state(
                        cpu_state_dict(self.model.state_dict()),
                        remote_metadata.outer_step,
                        reset_momentum=True,
                    )
            except Exception as exc:
                self.logger.warning("master could not load remote global weights: %s", exc)
        else:
            # Bootstrap the shared global checkpoint.  If another master wins a
            # race, its metadata is used instead of overwriting it.  Failure is
            # non-fatal to local training; the poller can retry later.
            try:
                metadata = self.hub.publish_global(
                    self._outer_loop.global_state,
                    outer_step=0,
                    work_dir=self.state_dir / "outgoing",
                    metadata_extra={"algorithm": "dumb_diloco", "bootstrap": "true"},
                )
                self._last_global_step = metadata.outer_step
            except Exception as exc:
                self.logger.warning("global bootstrap deferred: %s", exc)
        self._outer_loop.start()

    def _start_worker(self) -> None:
        try:
            metadata = self.hub.global_metadata()
            if metadata is not None:
                self._refresh_from_hub(metadata)
                return
            self.logger.warning(
                "worker could not find global/step_count.json; continuing from local weights"
            )
        except Exception as exc:
            # Local inner-loop progress does not require a permanent Hub
            # connection.  A later boundary will retry the upload/poll path.
            self.logger.warning("initial worker Hub read failed; continuing locally: %s", exc)

    def _refresh_from_hub(self, metadata: GlobalMetadata) -> bool:
        if metadata.outer_step <= self._last_global_step:
            return False
        local_path = self.state_dir / "global" / f"latest-{metadata.outer_step}.safetensors"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            try:
                state = load_safetensors(local_path)
            except Exception:
                # A partial local cache entry is disposable; the Hub copy is
                # authoritative and will be retried below.
                local_path.unlink(missing_ok=True)
            else:
                self._install_state(state)
                self._last_global_step = metadata.outer_step
                return True
        downloaded = self.hub.download_global(local_path, metadata)
        state = load_safetensors(downloaded)
        self._install_state(state)
        self._last_global_step = metadata.outer_step
        return True

    def _install_state(self, state: dict[str, torch.Tensor]) -> None:
        # A global model may omit newly introduced non-parameter buffers.  Only
        # load compatible entries and retain the rest of the local module.
        current = self.model.state_dict()
        filtered = {
            key: value
            for key, value in state.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
        if filtered:
            self.model.load_state_dict(filtered, strict=False)
        self._baseline = cpu_state_dict(self.model.state_dict())

    def _on_outer_update(self, state: dict[str, torch.Tensor], outer_step: int) -> None:
        # Assignment is atomic and deliberately does not acquire the
        # coordinator lock: the outer thread may invoke this callback while a
        # checkpoint is taking the coordinator snapshot.
        self._outer_snapshot = {key: value.clone() for key, value in state.items()}
        self._outer_snapshot_step = outer_step

    def _refresh_master_snapshot(self) -> bool:
        loop = self._outer_loop
        if loop is None:
            return False
        step, state = loop.snapshot()
        if step <= self._last_global_step:
            return False
        self._install_state(state)
        self._last_global_step = step
        return True

    def _maybe_poll_global(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last_poll < self.poll_interval:
            return False
        self._last_poll = now
        try:
            if self.role == "master":
                loaded = self._refresh_master_snapshot()
                # A local snapshot is authoritative while the Hub is briefly
                # unavailable; polling it first avoids needless downloads.
                if not loaded and self._outer_snapshot_step > self._last_global_step:
                    self._install_state(self._outer_snapshot or {})
                    self._last_global_step = self._outer_snapshot_step
                    loaded = True
                return loaded
            metadata = self.hub.global_metadata()
            return self._refresh_from_hub(metadata) if metadata else False
        except Exception as exc:
            self.logger.warning("global poll failed: %s", exc)
            return False

    def _upload_delta(self, step: int) -> bool:
        current = cpu_state_dict(self.model.state_dict())
        try:
            delta = compute_pseudo_gradient(self._baseline, current)
        except Exception as exc:
            self.logger.error("cannot compute pseudo-gradient at step %s: %s", step, exc)
            return False
        try:
            self.hub.upload_delta(
                delta,
                node_id=self.node_id,
                local_step=step,
                base_outer_step=self._last_global_step,
                work_dir=self.state_dir / "outgoing",
                metadata={"algorithm": "dumb_diloco"},
            )
        except Exception as exc:
            # Keep training alive during transient Hub failures.  The next
            # boundary recomputes a cumulative delta from the same baseline.
            self.logger.warning("delta upload failed at step %s: %s", step, exc)
            return False
        self._last_uploaded_step = step
        self._baseline = current
        return True

    def after_optimizer_step(self, model: torch.nn.Module, step: int) -> bool:
        """Handle a local step; return whether the inner optimizer is reset."""

        if model is not self.model:
            raise ValueError("coordinator is attached to a different model")
        self._local_step = int(step)
        boundary = step % self.inner_steps == 0
        pushed = False
        if boundary:
            pushed = self._upload_delta(step)
        loaded = self._maybe_poll_global(force=boundary)
        # A new inner loop starts after a successful upload, even if the
        # global version has not changed yet.  This prevents stale Adam moments
        # from leaking across independent local objectives by default.
        return pushed or loaded

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "node_id": self.node_id,
                "role": self.role,
                "last_global_step": self._last_global_step,
                "last_uploaded_step": self._last_uploaded_step,
                "local_step": self._local_step,
                "baseline": self._baseline,
                "pending_delta": self._pending_delta,
                "pending_delta_step": self._pending_delta_step,
                "outer": self._outer_loop.state_dict() if self._outer_loop is not None else None,
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._last_global_step = int(state.get("last_global_step", self._last_global_step))
            self._last_uploaded_step = state.get("last_uploaded_step")
            self._local_step = int(state.get("local_step", self._local_step))
            baseline = state.get("baseline")
            if baseline:
                self._baseline = {
                    key: value.detach().cpu().clone() for key, value in baseline.items()
                }
            self._pending_delta = state.get("pending_delta")
            self._pending_delta_step = state.get("pending_delta_step")
            if self._outer_loop is not None and state.get("outer") is not None:
                self._outer_loop.load_state_dict(state["outer"])

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            if self._outer_loop is not None:
                self._outer_loop.stop()


DumbDiLoCo = DumbDiLoCoCoordinator


__all__ = ["DumbDiLoCo", "DumbDiLoCoCoordinator", "SyncResult"]
