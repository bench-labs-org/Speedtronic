"""Hugging Face Hub transport used as the DumbDiLoCo synchronization bus."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .tensors import save_safetensors


class HubError(RuntimeError):
    pass


class HubUnavailable(HubError):
    pass


@dataclass(frozen=True)
class GlobalMetadata:
    outer_step: int
    updated_at: str
    model_file: str = "global/latest.safetensors"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GlobalMetadata":
        return cls(
            outer_step=int(value.get("outer_step", value.get("step", 0))),
            updated_at=str(value.get("updated_at", "")),
            model_file=str(value.get("model_file", "global/latest.safetensors")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_step": self.outer_step,
            "updated_at": self.updated_at,
            "model_file": self.model_file,
        }


class HubClient:
    """A small retrying wrapper around ``huggingface_hub``.

    The wrapper intentionally exposes only repository-file operations.  It is
    easy to replace with a fake object in tests and keeps the distributed code
    independent of Hub SDK version details.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        api: Any | None = None,
        retry_initial: float = 1.0,
        retry_max: float = 60.0,
        retry_attempts: int = 6,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not repo_id:
            raise ValueError("repo_id is required")
        self.repo_id = repo_id
        self.token = token
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "speedtronic" / "hub")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retry_initial = retry_initial
        self.retry_max = retry_max
        self.retry_attempts = retry_attempts
        self.sleeper = sleeper
        self._api = api
        self._hf_hub_download = None

    @property
    def api(self) -> Any:
        if self._api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:  # pragma: no cover - installation issue
                raise HubUnavailable("huggingface_hub is required for DumbDiLoCo") from exc
            self._api = HfApi(token=self.token)
        return self._api

    def _retry(self, operation: str, callback: Callable[[], Any]) -> Any:
        delay = self.retry_initial
        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                return callback()
            except Exception as exc:  # Hub SDK uses several exception classes.
                last_error = exc
                if attempt + 1 >= self.retry_attempts:
                    break
                self.sleeper(delay)
                delay = min(self.retry_max, delay * 2)
        raise HubUnavailable(
            f"{operation} failed after {self.retry_attempts} attempts: {last_error}"
        ) from last_error

    def create_repo(self, *, private: bool = True) -> Any:
        return self._retry(
            "create repository",
            lambda: self.api.create_repo(
                self.repo_id, repo_type="model", private=private, exist_ok=True
            ),
        )

    def add_collaborator(self, username: str, permission: str = "write") -> Any:
        if permission not in {"read", "write", "admin"}:
            raise ValueError("permission must be read, write, or admin")
        if not hasattr(self.api, "add_collaborator"):
            raise HubError("the installed Hugging Face Hub SDK cannot add collaborators")

        def operation() -> Any:
            try:
                return self.api.add_collaborator(
                    username,
                    repo_id=self.repo_id,
                    repo_type="model",
                    permission=permission,
                )
            except TypeError as keyword_error:
                try:
                    return self.api.add_collaborator(username, self.repo_id, permission)
                except TypeError:
                    raise keyword_error

        return self._retry(f"grant {permission} access to {username}", operation)

    def list_files(self) -> list[str]:
        result = self._retry(
            "list repository files",
            lambda: self.api.list_repo_files(self.repo_id, repo_type="model"),
        )
        return sorted(str(getattr(item, "rfilename", item)) for item in (result or []))

    def file_exists(self, remote_path: str) -> bool:
        return remote_path in set(self.list_files())

    def upload(self, local_path: str | os.PathLike[str], remote_path: str) -> Any:
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(local_path)

        def operation() -> Any:
            kwargs = {
                "path_in_repo": remote_path,
                "repo_id": self.repo_id,
                "repo_type": "model",
            }
            try:
                return self.api.upload_file(path_or_fileobj=str(local_path), **kwargs)
            except TypeError as keyword_error:
                # Older HfApi releases used ``path_or_fileobj`` under the
                # positional name; keep compatibility with those versions and
                # lightweight test doubles.
                try:
                    return self.api.upload_file(str(local_path), **kwargs)
                except TypeError:
                    raise keyword_error

        return self._retry(f"upload {remote_path}", operation)

    def download(self, remote_path: str, local_path: str | os.PathLike[str] | None = None) -> Path:
        destination = (
            Path(local_path)
            if local_path is not None
            else self.cache_dir / "downloads" / remote_path
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="speedtronic-download-", dir=self.cache_dir))
        try:

            def operation() -> Path:
                # Newer SDKs expose this through HfApi; fall back to the
                # top-level helper for older supported versions.
                if hasattr(self.api, "hf_hub_download"):
                    result = self.api.hf_hub_download(
                        repo_id=self.repo_id,
                        filename=remote_path,
                        local_dir=temp_dir,
                        repo_type="model",
                        token=self.token,
                    )
                elif hasattr(self.api, "download_file"):
                    result = self.api.download_file(
                        repo_id=self.repo_id,
                        filename=remote_path,
                        local_dir=temp_dir,
                        repo_type="model",
                        token=self.token,
                    )
                else:
                    if self._hf_hub_download is None:
                        from huggingface_hub import hf_hub_download

                        self._hf_hub_download = hf_hub_download
                    result = self._hf_hub_download(
                        repo_id=self.repo_id,
                        filename=remote_path,
                        local_dir=temp_dir,
                        repo_type="model",
                        token=self.token,
                    )
                return Path(result)

            downloaded = self._retry(f"download {remote_path}", operation)
            shutil.copy2(downloaded, destination)
            return destination
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def read_json(self, remote_path: str, *, default: Any = None) -> Any:
        if not self.file_exists(remote_path):
            return default
        local = self.cache_dir / "json" / remote_path
        self.download(remote_path, local)
        return json.loads(local.read_text(encoding="utf-8"))

    def write_json(self, remote_path: str, value: Any) -> None:
        local = self.cache_dir / "outgoing" / remote_path
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_suffix(local.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, local)
        self.upload(local, remote_path)
        cached = self.cache_dir / "json" / remote_path
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, cached)

    def global_metadata(self) -> GlobalMetadata | None:
        try:
            value = self.read_json("global/step_count.json", default=None)
        except HubUnavailable:
            cached = self.cache_dir / "json" / "global" / "step_count.json"
            if not cached.exists():
                raise
            value = json.loads(cached.read_text(encoding="utf-8"))
        if value is None:
            cached = self.cache_dir / "json" / "global" / "step_count.json"
            if not cached.exists():
                return None
            value = json.loads(cached.read_text(encoding="utf-8"))
        return GlobalMetadata.from_dict(value)

    def download_global(
        self, local_path: str | os.PathLike[str], metadata: GlobalMetadata | None = None
    ) -> Path:
        metadata = metadata or self.global_metadata()
        remote = metadata.model_file if metadata else "global/latest.safetensors"
        return self.download(remote, local_path)

    def publish_global(
        self,
        state: dict[str, Any],
        *,
        outer_step: int,
        work_dir: str | os.PathLike[str] | None = None,
        updated_at: str | None = None,
        metadata_extra: dict[str, str] | None = None,
    ) -> GlobalMetadata:
        work_dir = Path(work_dir or self.cache_dir / "outgoing")
        work_dir.mkdir(parents=True, exist_ok=True)
        remote_model = "global/latest.safetensors"
        local_model = work_dir / "latest.safetensors"
        save_safetensors(state, local_model)
        # Weights must exist before the metadata that advertises them.  A failed
        # metadata upload is safe: the next publication retries and overwrites.
        self.upload(local_model, remote_model)
        metadata = GlobalMetadata(
            outer_step=int(outer_step),
            updated_at=updated_at or _utc_now(),
            model_file=remote_model,
        )
        value = metadata.to_dict()
        if metadata_extra:
            value.update({str(k): str(v) for k, v in metadata_extra.items()})
        self.write_json("global/step_count.json", value)
        return metadata

    def upload_delta(
        self,
        state: dict[str, Any],
        *,
        node_id: str,
        local_step: int,
        base_outer_step: int,
        work_dir: str | os.PathLike[str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        work_dir = Path(work_dir or self.cache_dir / "outgoing")
        work_dir.mkdir(parents=True, exist_ok=True)
        remote = f"nodes/{node_id}/delta_{int(local_step)}.safetensors"
        local = work_dir / f"{node_id}_delta_{int(local_step)}.safetensors"
        all_metadata = {
            "node_id": str(node_id),
            "local_step": str(int(local_step)),
            "base_outer_step": str(int(base_outer_step)),
        }
        if metadata:
            all_metadata.update({str(k): str(v) for k, v in metadata.items()})
        save_safetensors(state, local, metadata=all_metadata)
        self.upload(local, remote)
        return remote

    def delta_paths(self) -> list[str]:
        return [
            path
            for path in self.list_files()
            if path.startswith("nodes/") and path.endswith(".safetensors")
        ]


HubTransport = HubClient


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


__all__ = ["GlobalMetadata", "HubClient", "HubError", "HubTransport", "HubUnavailable"]
