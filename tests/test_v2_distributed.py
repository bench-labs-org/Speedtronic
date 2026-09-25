import threading
import time

import torch
from torch import nn

from speedtronic.config import DistributedConfig
from speedtronic.distributed.diloco import DumbDiLoCoCoordinator
from speedtronic.distributed.hub import GlobalMetadata
from speedtronic.distributed.tensors import save_safetensors


class BlockingHub:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.uploaded = threading.Event()
        self.uploads = []
        self.fail = False

    def global_metadata(self):
        return None

    def upload_delta(self, state, *, node_id, local_step, base_outer_step, **kwargs):
        self.uploads.append((local_step, dict(state)))
        self.started.set()
        assert self.release.wait(timeout=5)
        if self.fail:
            raise RuntimeError("synthetic upload failure")
        self.uploaded.set()

    def download_global(self, local_path, metadata=None):
        raise FileNotFoundError("no global")


class PollHub:
    def __init__(self, state):
        self.state = state
        self.metadata_available = False
        self.uploaded = threading.Event()

    def global_metadata(self):
        if not self.metadata_available:
            return None
        return GlobalMetadata(1, "now", "global/latest.safetensors")

    def upload_delta(self, *args, **kwargs):
        self.uploaded.set()

    def download_global(self, local_path, metadata=None):
        save_safetensors(self.state, local_path)
        return local_path


def test_async_global_poll_installs_on_training_thread(tmp_path):
    model = nn.Linear(2, 2)
    replacement = {key: torch.full_like(value, 9.0) for key, value in model.state_dict().items()}
    hub = PollHub(replacement)
    coordinator = DumbDiLoCoCoordinator(
        DistributedConfig(
            enabled=True,
            role="worker",
            repo_id="org/run",
            inner_steps=1,
            async_delta_upload=True,
            async_global_poll=True,
            poll_interval=0.01,
        ),
        model,
        hub=hub,
        state_dir=tmp_path,
    )
    coordinator.start()
    try:
        assert coordinator.after_optimizer_step(model, 1) is True
        hub.metadata_available = True
        coordinator.after_optimizer_step(model, 2)
        deadline = time.time() + 2
        while coordinator._last_global_step < 1 and time.time() < deadline:
            coordinator.after_optimizer_step(model, 3)
            time.sleep(0.01)
        assert coordinator._last_global_step == 1
        assert torch.equal(model.weight, replacement["weight"])
    finally:
        coordinator.stop()


def test_coordinator_can_restart_after_stop(tmp_path):
    model = nn.Linear(2, 2)
    hub = PollHub({key: torch.full_like(value, 9.0) for key, value in model.state_dict().items()})
    coordinator = DumbDiLoCoCoordinator(
        DistributedConfig(
            enabled=True,
            role="worker",
            repo_id="org/run",
            inner_steps=1,
            async_delta_upload=True,
            async_global_poll=True,
        ),
        model,
        hub=hub,
        state_dir=tmp_path,
    )
    coordinator.start()
    coordinator.after_optimizer_step(model, 1)
    coordinator.stop()
    coordinator.start()
    try:
        assert coordinator._poll_thread is not None
        coordinator.after_optimizer_step(model, 2)
    finally:
        coordinator.stop()


def test_async_delta_dispatch_does_not_block_training(tmp_path):
    model = nn.Linear(2, 2)
    hub = BlockingHub()
    coordinator = DumbDiLoCoCoordinator(
        DistributedConfig(
            enabled=True,
            role="worker",
            repo_id="org/run",
            inner_steps=1,
            async_delta_upload=True,
            async_global_poll=False,
        ),
        model,
        hub=hub,
        state_dir=tmp_path,
    )
    coordinator.start()
    try:
        started = time.perf_counter()
        assert coordinator.after_optimizer_step(model, 1) is True
        elapsed = time.perf_counter() - started
        assert elapsed < 0.5
        assert hub.started.wait(timeout=1)

        # The next boundary must skip rather than wait for the first upload.
        started = time.perf_counter()
        assert coordinator.after_optimizer_step(model, 2) is False
        assert time.perf_counter() - started < 0.5
        state = coordinator.state_dict()
        assert state["pending_upload"] is not None
        assert state["pending_upload"]["step"] == 1

        hub.release.set()
        assert hub.uploaded.wait(timeout=2)
    finally:
        hub.release.set()
        coordinator.stop()


def test_prepared_resume_restores_pending_upload(tmp_path):
    model = nn.Linear(2, 2)
    hub = BlockingHub()
    config = DistributedConfig(
        enabled=True,
        role="worker",
        repo_id="org/run",
        inner_steps=1,
        async_delta_upload=True,
        async_global_poll=False,
    )
    first = DumbDiLoCoCoordinator(config, model, hub=hub, state_dir=tmp_path)
    first.start()
    assert first.after_optimizer_step(model, 1) is True
    assert hub.started.wait(timeout=1)
    state = first.state_dict()
    assert state["pending_upload"] is not None

    hub2 = BlockingHub()
    second = DumbDiLoCoCoordinator(config, model, hub=hub2, state_dir=tmp_path)
    second.prepare_state(state)
    second.start()
    try:
        assert hub2.started.wait(timeout=1)
        assert hub2.uploads[0][0] == 1
    finally:
        hub.release.set()
        hub2.release.set()
        first.stop()
        second.stop()


def test_synchronous_delta_mode_remains_available(tmp_path):
    model = nn.Linear(2, 2)
    hub = BlockingHub()
    hub.release.set()
    coordinator = DumbDiLoCoCoordinator(
        DistributedConfig(
            enabled=True,
            role="worker",
            repo_id="org/run",
            inner_steps=1,
            async_delta_upload=False,
            async_global_poll=False,
        ),
        model,
        hub=hub,
        state_dir=tmp_path,
    )
    coordinator.start()
    try:
        assert coordinator.after_optimizer_step(model, 1) is True
        assert hub.uploaded.is_set()
        assert coordinator._last_uploaded_step == 1
    finally:
        coordinator.stop()


def test_async_delta_failure_keeps_baseline_for_cumulative_retry(tmp_path):
    model = nn.Linear(2, 2)
    hub = BlockingHub()
    hub.release.set()
    hub.fail = True
    coordinator = DumbDiLoCoCoordinator(
        DistributedConfig(
            enabled=True,
            role="worker",
            repo_id="org/run",
            inner_steps=1,
            async_delta_upload=False,
            async_global_poll=False,
        ),
        model,
        hub=hub,
        state_dir=tmp_path,
    )
    coordinator.start()
    try:
        before = {key: value.clone() for key, value in coordinator._baseline.items()}
        assert coordinator.after_optimizer_step(model, 1) is False
        assert all(torch.equal(before[key], value) for key, value in coordinator._baseline.items())
    finally:
        coordinator.stop()
