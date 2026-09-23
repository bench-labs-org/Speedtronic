from pathlib import Path

import torch

from speedtronic.distributed.hub import HubClient
from speedtronic.distributed.outer import MasterOuterLoop
from speedtronic.distributed.tensors import load_safetensors


class FakeApi:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.collaborators: list[tuple[str, str]] = []

    def create_repo(self, *args, **kwargs):
        return None

    def add_collaborator(self, username, **kwargs):
        self.collaborators.append((username, kwargs["permission"]))

    def list_repo_files(self, *args, **kwargs):
        return list(self.files)

    def upload_file(self, path, path_in_repo, **kwargs):
        self.files[path_in_repo] = Path(path).read_bytes()

    def hf_hub_download(self, repo_id, filename, local_dir, **kwargs):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[filename])
        return path


def test_hub_transport_round_trip(tmp_path: Path):
    api = FakeApi()
    hub = HubClient("org/run", api=api, cache_dir=tmp_path / "cache", retry_attempts=1)
    hub.create_repo()
    hub.add_collaborator("alice")
    state = {"weight": torch.tensor([1.0, 2.0])}
    hub.publish_global(state, outer_step=0, work_dir=tmp_path / "outgoing")
    assert hub.global_metadata().outer_step == 0
    downloaded = hub.download_global(tmp_path / "global.safetensors")
    assert torch.equal(load_safetensors(downloaded)["weight"], state["weight"])
    hub.upload_delta(
        state,
        node_id="node-a",
        local_step=1,
        base_outer_step=0,
        work_dir=tmp_path / "outgoing",
    )
    assert hub.delta_paths() == ["nodes/node-a/delta_1.safetensors"]
    assert api.collaborators == [("alice", "write")]


def test_master_skips_corrupt_delta_and_persists_processed_set(tmp_path: Path):
    api = FakeApi()
    api.files["nodes/node-a/delta_1.safetensors"] = b"not safetensors"
    hub = HubClient("org/run", api=api, cache_dir=tmp_path / "cache", retry_attempts=1)
    loop = MasterOuterLoop(
        hub,
        {"weight": torch.zeros(1)},
        node_state_dir=tmp_path / "state",
        poll_interval=60,
    )
    assert loop.sync_once() == 0
    assert (tmp_path / "state" / "outer_state.pt").exists() is False
