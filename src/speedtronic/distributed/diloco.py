"""DumbDiLoCo coordinator integrated with the ordinary training loop."""

from __future__ import annotations

import logging
import os
import queue
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


@dataclass
class DeltaUploadJob:
    """Immutable CPU snapshot dispatched to the asynchronous uploader."""

    step: int
    base_outer_step: int
    generation: int
    delta: dict[str, torch.Tensor]
    target_state: dict[str, torch.Tensor]


@dataclass
class GlobalCandidate:
    outer_step: int
    state: dict[str, torch.Tensor]


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
        # A single tuple assignment is atomic in CPython, so the training
        # thread never observes a step number paired with a different snapshot.
        self._outer_snapshot_pair: tuple[int, dict[str, torch.Tensor]] | None = None
        self._accepting_uploads = True
        self._pending_delta: dict[str, torch.Tensor] | None = None
        self._pending_delta_step: int | None = None
        self.async_delta_upload = bool(getattr(config, "async_delta_upload", True))
        self.async_global_poll = bool(getattr(config, "async_global_poll", True))
        self.delta_upload_queue_size = int(getattr(config, "delta_upload_queue_size", 1))
        self.delta_upload_shutdown_timeout = float(
            getattr(config, "delta_upload_shutdown_timeout", 5.0)
        )
        self._upload_queue: queue.Queue[DeltaUploadJob] = queue.Queue(
            maxsize=self.delta_upload_queue_size
        )
        self._upload_stop = threading.Event()
        self._upload_thread: threading.Thread | None = None
        self._upload_busy = False
        self._pending_upload: DeltaUploadJob | None = None
        self._last_dispatched_step: int | None = None
        self._last_skipped_step: int | None = None
        self._baseline_generation = 0
        self._poll_queue: queue.Queue[bool] = queue.Queue(maxsize=1)
        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._poll_busy = False
        self._global_candidate: GlobalCandidate | None = None
        self._prepared_state: dict[str, Any] | None = None

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

    def _emit_event(self, event: str, payload: dict[str, Any]) -> None:
        if hasattr(self.logger, "emit"):
            self.logger.emit(event, payload)
        else:
            self.logger.info("%s %s", event, payload)

    def _start_io_workers(self) -> None:
        if self.async_delta_upload:
            self._upload_stop.clear()
            self._upload_thread = threading.Thread(
                target=self._upload_worker,
                name="speedtronic-diloco-upload",
                daemon=True,
            )
            self._upload_thread.start()
            if self._pending_upload is not None:
                try:
                    self._upload_queue.put_nowait(self._pending_upload)
                    self._upload_busy = True
                except queue.Full:
                    self.logger.warning("could not requeue pending delta upload")
                    self._pending_upload = None
                    self._pending_delta = None
                    self._pending_delta_step = None
                    self._upload_busy = False
        if self.async_global_poll and self.role == "worker":
            self._poll_stop.clear()
            self._poll_thread = threading.Thread(
                target=self._poll_worker,
                name="speedtronic-diloco-poll",
                daemon=True,
            )
            self._poll_thread.start()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._stopped = False
            self._accepting_uploads = True
            if self.role == "master":
                self._start_master()
            else:
                self._start_worker()
            self._start_io_workers()
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
        if self._prepared_state is not None:
            # A checkpointed model/baseline pair is authoritative at startup;
            # a newer remote global is discovered asynchronously later.
            self._apply_prepared_state()
            self._outer_loop.start()
            return
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
        if self._prepared_state is not None:
            # Preserve the checkpointed model/baseline pair; a newer global is
            # fetched asynchronously after the first optimizer boundary.
            self._apply_prepared_state()
            return
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

    def _load_global_state(self, metadata: GlobalMetadata) -> dict[str, torch.Tensor]:
        local_path = self.state_dir / "global" / f"latest-{metadata.outer_step}.safetensors"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            try:
                return load_safetensors(local_path)
            except Exception:
                local_path.unlink(missing_ok=True)
        downloaded = self.hub.download_global(local_path, metadata)
        return load_safetensors(downloaded)

    def _refresh_from_hub(self, metadata: GlobalMetadata) -> bool:
        if metadata.outer_step <= self._last_global_step:
            return False
        state = self._load_global_state(metadata)
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
        with self._lock:
            self._baseline = cpu_state_dict(self.model.state_dict())
            self._baseline_generation += 1

    def _on_outer_update(self, state: dict[str, torch.Tensor], outer_step: int) -> None:
        # Assignment is atomic and deliberately does not acquire the
        # coordinator lock: the outer thread may invoke this callback while a
        # checkpoint is taking the coordinator snapshot.  Keep the step and
        # tensors in one tuple so readers cannot pair mismatched values.
        snapshot = {key: value.clone() for key, value in state.items()}
        self._outer_snapshot_pair = (int(outer_step), snapshot)
        # Retain the separate fields for backwards-compatible inspection.
        self._outer_snapshot = snapshot
        self._outer_snapshot_step = int(outer_step)

    def _refresh_master_snapshot(self) -> bool:
        # The outer thread already produces a lock-free snapshot.  Reading the
        # tuple once keeps the training thread independent of network-bound
        # work and avoids a torn step/state pair.
        pair = self._outer_snapshot_pair
        if pair is None or pair[0] <= self._last_global_step:
            return False
        outer_step, snapshot = pair
        self._install_state(snapshot)
        self._last_global_step = outer_step
        return True

    def _install_async_candidate(self) -> bool:
        candidate = self._global_candidate
        if candidate is None:
            return False
        self._global_candidate = None
        if candidate.outer_step <= self._last_global_step:
            return False
        self._install_state(candidate.state)
        self._last_global_step = candidate.outer_step
        return True

    def _fetch_global_candidate(self) -> GlobalCandidate | None:
        metadata = self.hub.global_metadata()
        if metadata is None or metadata.outer_step <= self._last_global_step:
            return None
        return GlobalCandidate(metadata.outer_step, self._load_global_state(metadata))

    def _schedule_async_poll(self, *, force: bool) -> bool:
        now = time.monotonic()
        if not force and now - self._last_poll < self.poll_interval:
            return False
        if self._poll_thread is None or self._poll_busy:
            return False
        self._last_poll = now
        self._poll_busy = True
        try:
            self._poll_queue.put_nowait(True)
        except queue.Full:
            self._poll_busy = False
            return False
        return True

    def _poll_worker(self) -> None:
        while not self._poll_stop.is_set() or not self._poll_queue.empty():
            try:
                self._poll_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                candidate = self._fetch_global_candidate()
            except Exception as exc:
                self.logger.warning("async global poll failed: %s", exc)
                candidate = None
            with self._lock:
                self._poll_busy = False
                if candidate is not None:
                    self._global_candidate = candidate
            self._poll_queue.task_done()

    def _consume_async_poll(self) -> None:
        # Results are placed directly on the training thread's candidate slot
        # by the daemon poll worker.  This method remains a small compatibility
        # hook for callers that used the previous future-based lane.
        return None

    def _maybe_poll_global(self, *, force: bool = False) -> bool:
        self._consume_async_poll()
        loaded = self._install_async_candidate()
        if loaded:
            return True
        if self.async_global_poll:
            if self.role == "master":
                return self._refresh_master_snapshot()
            # A boundary request is dispatched to the I/O lane but never waits
            # for the network; the next step installs any completed candidate.
            self._schedule_async_poll(force=force)
            return False
        now = time.monotonic()
        if not force and now - self._last_poll < self.poll_interval:
            return False
        self._last_poll = now
        try:
            if self.role == "master":
                loaded = self._refresh_master_snapshot()
                return loaded
            metadata = self.hub.global_metadata()
            return self._refresh_from_hub(metadata) if metadata else False
        except Exception as exc:
            self.logger.warning("global poll failed: %s", exc)
            return False

    def _upload_delta_sync(self, step: int) -> bool:
        current = cpu_state_dict(self.model.state_dict())
        with self._lock:
            baseline = {key: value.clone() for key, value in self._baseline.items()}
            generation = self._baseline_generation
            base_outer_step = self._last_global_step
        try:
            delta = compute_pseudo_gradient(baseline, current)
        except Exception as exc:
            self.logger.error("cannot compute pseudo-gradient at step %s: %s", step, exc)
            return False
        try:
            self.hub.upload_delta(
                delta,
                node_id=self.node_id,
                local_step=step,
                base_outer_step=base_outer_step,
                work_dir=self.state_dir / "outgoing",
                metadata={"algorithm": "dumb_diloco"},
            )
        except Exception as exc:
            self.logger.warning("delta upload failed at step %s: %s", step, exc)
            return False
        with self._lock:
            self._last_uploaded_step = step
            if generation == self._baseline_generation:
                self._baseline = current
            else:
                self._emit_event(
                    "delta_upload_stale",
                    {"step": step, "generation": generation},
                )
        return True

    def _dispatch_delta(self, step: int) -> bool:
        with self._lock:
            if not self._accepting_uploads:
                return False
            if self._upload_busy or self._pending_upload is not None:
                self._last_skipped_step = step
                self._emit_event(
                    "delta_upload_skipped",
                    {"step": step, "reason": "upload_in_flight"},
                )
                return False
        current = cpu_state_dict(self.model.state_dict())
        with self._lock:
            if not self._accepting_uploads:
                return False
            baseline = {key: value.clone() for key, value in self._baseline.items()}
            generation = self._baseline_generation
            base_outer_step = self._last_global_step
        try:
            delta = compute_pseudo_gradient(baseline, current)
        except Exception as exc:
            self.logger.error("cannot compute pseudo-gradient at step %s: %s", step, exc)
            return False
        with self._lock:
            if self._upload_busy or self._pending_upload is not None:
                self._last_skipped_step = step
                self._emit_event(
                    "delta_upload_skipped",
                    {"step": step, "reason": "upload_in_flight"},
                )
                return False
            job = DeltaUploadJob(
                step=step,
                base_outer_step=base_outer_step,
                generation=generation,
                delta=delta,
                target_state=current,
            )
            self._upload_busy = True
            self._pending_upload = job
            self._pending_delta = job.delta
            self._pending_delta_step = step
        try:
            self._upload_queue.put_nowait(job)
        except queue.Full:
            with self._lock:
                self._upload_busy = False
                self._pending_upload = None
                self._pending_delta = None
                self._pending_delta_step = None
            self._last_skipped_step = step
            self._emit_event("delta_upload_skipped", {"step": step, "reason": "queue_full"})
            return False
        self._last_dispatched_step = step
        self._emit_event(
            "delta_upload_queued",
            {"step": step, "base_outer_step": job.base_outer_step},
        )
        return True

    def _upload_worker(self) -> None:
        while not self._upload_stop.is_set() or not self._upload_queue.empty():
            try:
                job = self._upload_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.hub.upload_delta(
                    job.delta,
                    node_id=self.node_id,
                    local_step=job.step,
                    base_outer_step=job.base_outer_step,
                    work_dir=self.state_dir / "outgoing",
                    metadata={"algorithm": "dumb_diloco"},
                )
            except Exception as exc:
                self.logger.warning("async delta upload failed at step %s: %s", job.step, exc)
                self._emit_event("delta_upload_failed", {"step": job.step, "error": str(exc)})
            else:
                with self._lock:
                    self._last_uploaded_step = job.step
                    if self._baseline_generation == job.generation:
                        self._baseline = job.target_state
                    else:
                        self._emit_event(
                            "delta_upload_stale",
                            {"step": job.step, "generation": job.generation},
                        )
                self._emit_event("delta_uploaded", {"step": job.step})
            finally:
                with self._lock:
                    self._upload_busy = False
                    self._pending_upload = None
                    self._pending_delta = None
                    self._pending_delta_step = None
                self._upload_queue.task_done()

    def _upload_delta(self, step: int) -> bool:
        if self.async_delta_upload:
            return self._dispatch_delta(step)
        return self._upload_delta_sync(step)

    def after_optimizer_step(self, model: torch.nn.Module, step: int) -> bool:
        """Handle a local step; return whether the inner optimizer is reset."""

        if model is not self.model:
            raise ValueError("coordinator is attached to a different model")
        self._local_step = int(step)
        boundary = step % self.inner_steps == 0
        loaded = self._install_async_candidate()
        pushed = self._upload_delta(step) if boundary else False
        if not loaded:
            loaded = self._maybe_poll_global(force=boundary)
        # A new inner loop starts after a queued upload or global installation.
        return pushed or loaded

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            pending = None
            if self._pending_upload is not None:
                job = self._pending_upload
                pending = {
                    "step": job.step,
                    "base_outer_step": job.base_outer_step,
                    "generation": job.generation,
                    "delta": {
                        key: value.detach().cpu().clone() for key, value in job.delta.items()
                    },
                    "target_state": {
                        key: value.detach().cpu().clone() for key, value in job.target_state.items()
                    },
                }
            return {
                "node_id": self.node_id,
                "role": self.role,
                "last_global_step": self._last_global_step,
                "last_uploaded_step": self._last_uploaded_step,
                "last_dispatched_step": self._last_dispatched_step,
                "last_skipped_step": self._last_skipped_step,
                "local_step": self._local_step,
                "baseline_generation": self._baseline_generation,
                "baseline": {
                    key: value.detach().cpu().clone() for key, value in self._baseline.items()
                },
                "pending_upload": pending,
                # Preserve v1-compatible fields for older inspection tooling.
                # A pending_upload already contains the delta, so do not clone
                # a second model-sized copy into legacy fields.
                "pending_delta": None if pending is not None else self._pending_delta,
                "pending_delta_step": None if pending is not None else self._pending_delta_step,
                "outer": self._outer_loop.state_dict() if self._outer_loop is not None else None,
            }

    def _restore_pending_upload(self, pending: dict[str, Any] | None) -> None:
        if pending is None:
            return
        self._pending_upload = DeltaUploadJob(
            step=int(pending["step"]),
            base_outer_step=int(pending["base_outer_step"]),
            generation=int(pending["generation"]),
            delta={key: value.detach().cpu().clone() for key, value in pending["delta"].items()},
            target_state={
                key: value.detach().cpu().clone() for key, value in pending["target_state"].items()
            },
        )
        self._pending_delta = self._pending_upload.delta
        self._pending_delta_step = self._pending_upload.step
        self._upload_busy = False

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._last_global_step = int(state.get("last_global_step", self._last_global_step))
            self._last_uploaded_step = state.get("last_uploaded_step")
            self._last_dispatched_step = state.get("last_dispatched_step")
            self._last_skipped_step = state.get("last_skipped_step")
            self._local_step = int(state.get("local_step", self._local_step))
            self._baseline_generation = int(
                state.get("baseline_generation", self._baseline_generation)
            )
            baseline = state.get("baseline")
            if baseline:
                self._baseline = {
                    key: value.detach().cpu().clone() for key, value in baseline.items()
                }
            self._pending_delta = state.get("pending_delta")
            self._pending_delta_step = state.get("pending_delta_step")
            self._restore_pending_upload(state.get("pending_upload"))
            if self._outer_loop is not None and state.get("outer") is not None:
                self._outer_loop.load_state_dict(state["outer"])

    def prepare_state(self, state: dict[str, Any]) -> None:
        """Stage resume state before startup performs any remote refresh."""

        with self._lock:
            self._prepared_state = dict(state)

    def _apply_prepared_state(self) -> None:
        with self._lock:
            state = getattr(self, "_prepared_state", None)
            if state is None:
                return
            self._last_global_step = int(state.get("last_global_step", self._last_global_step))
            self._last_uploaded_step = state.get("last_uploaded_step")
            self._local_step = int(state.get("local_step", self._local_step))
            self._baseline_generation = int(
                state.get("baseline_generation", self._baseline_generation)
            )
            baseline = state.get("baseline")
            if baseline:
                self._baseline = {
                    key: value.detach().cpu().clone() for key, value in baseline.items()
                }
            self._restore_pending_upload(state.get("pending_upload"))
            if self._outer_loop is not None and state.get("outer") is not None:
                self._outer_loop.load_state_dict(state["outer"])
            self._prepared_state = None

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._accepting_uploads = False
            self._upload_stop.set()
            thread = self._upload_thread
            poll_thread = self._poll_thread
            outer_loop = self._outer_loop
            self._poll_stop.set()
        # Never join the uploader while holding _lock: its completion path
        # needs the lock to clear the pending job.  The bounded join keeps
        # shutdown non-blocking when a Hub request is stuck.
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=self.delta_upload_shutdown_timeout)
            if thread.is_alive():
                self.logger.warning("async delta uploader did not stop within the timeout")
        if (
            poll_thread is not None
            and poll_thread.is_alive()
            and poll_thread is not threading.current_thread()
        ):
            poll_thread.join(timeout=self.delta_upload_shutdown_timeout)
            if poll_thread.is_alive():
                self.logger.warning("async global poller did not stop within the timeout")
        with self._lock:
            self._poll_thread = None
            self._started = False
        if outer_loop is not None:
            outer_loop.stop()


DumbDiLoCo = DumbDiLoCoCoordinator


__all__ = ["DumbDiLoCo", "DumbDiLoCoCoordinator", "SyncResult"]
