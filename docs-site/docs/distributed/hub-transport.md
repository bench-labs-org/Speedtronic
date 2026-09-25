---
id: hub-transport
title: Hub Transport
sidebar_label: Hub Transport
description: HubClient API, retries, caching, repository operations, global publication, and delta upload format.
---

# Hub transport

## Public API

```python
from speedtronic import HubClient
# distributed alias: HubTransport
```

Errors:

```python
HubError(RuntimeError)
HubUnavailable(HubError)
```

## Constructor

```python
HubClient(
    repo_id,
    *,
    token=None,
    cache_dir=None,
    api=None,
    retry_initial=1.0,
    retry_max=60.0,
    retry_attempts=6,
    sleeper=time.sleep,
)
```

`repo_id` is required. The client always treats it as a Hugging Face **model** repository. `api` and `sleeper` are injection points for tests.

The `api` property lazily constructs `huggingface_hub.HfApi(token=token)`. If the SDK is missing, operations raise `HubUnavailable`.

## Retry behavior

Every network/repository operation is wrapped by `_retry`:

```text
attempt immediately
→ wait retry_initial
→ double delay after each failure
→ cap at retry_max
→ raise HubUnavailable after retry_attempts
```

With defaults, sleeps are 1, 2, 4, 8, and 16 seconds before the sixth final attempt.

The wrapper catches every `Exception`, including permanent authentication, malformed request, and programming errors. There is no jitter, retry classification, or explicit request timeout.

## Repository methods

### `create_repo(private=True)`

Calls:

```python
api.create_repo(
    repo_id,
    repo_type="model",
    private=private,
    exist_ok=True,
)
```

`exist_ok=True` does not verify that an existing repository is private.

### `add_collaborator(username, permission="write")`

Valid permissions:

```text
read
write
admin
```

Attempts the newer keyword API, then a positional compatibility form. Missing SDK support raises `HubError`.

### `list_files()`

Returns sorted repository filenames.

### `file_exists(remote_path)`

Builds a set from `list_files()` and checks membership. This is a full repository listing, not a direct existence API.

### `upload(local_path, remote_path)`

Verifies local existence and calls `api.upload_file` with compatibility handling.

### `download(remote_path, local_path=None)`

1. Creates a temporary directory under `cache_dir`.
2. Uses an API download method when available, then the top-level SDK helper.
3. Copies the downloaded file to the destination.
4. Removes the temporary directory in `finally`.

The final copy is not atomic.

## JSON methods

### `read_json(remote_path, default=None)`

Checks `file_exists`, downloads to `cache_dir/json/<remote_path>`, then parses JSON.

### `write_json(remote_path, value)`

1. Writes pretty, sorted JSON under `cache_dir/outgoing`.
2. Uses one deterministic `.tmp` path.
3. Replaces the local staging file.
4. Uploads it.
5. Copies the local result into the read cache.

The deterministic temporary filename is not process-safe.

## Global metadata

```python
@dataclass(frozen=True)
class GlobalMetadata:
    outer_step: int
    updated_at: str
    model_file: str = "global/latest.safetensors"
```

`from_dict()` accepts legacy `step`, missing metadata defaults, and coerces values to int/str. `updated_at` is informational; synchronization decisions use `outer_step`.

### `global_metadata()`

Attempts to read `global/step_count.json`. On `HubUnavailable`, it can fall back to cached JSON. It can also return cached metadata when the remote path is absent.

Malformed cached JSON is not converted into `HubUnavailable`.

## Global publication

```python
publish_global(
    state,
    *,
    outer_step,
    work_dir=None,
    updated_at=None,
    metadata_extra=None,
) -> GlobalMetadata
```

Publication sequence:

1. Save full state as local `latest.safetensors`.
2. Upload/overwrite `global/latest.safetensors`.
3. Construct metadata.
4. Write/upload `global/step_count.json`.
5. Return metadata.

Example:

```json
{
  "algorithm": "dumb_diloco",
  "model_file": "global/latest.safetensors",
  "optimizer": "nesterov_sgd",
  "outer_step": 7,
  "updated_at": "2026-01-01T12:00:00+00:00"
}
```

:::caution Non-atomic fixed paths

The model and metadata are separate overwrites of fixed paths. A reader can observe old metadata with a newly uploaded model. A metadata failure can leave new weights advertised under the old outer step.

:::

### `download_global(local_path, metadata=None)`

Uses `metadata.model_file` without an allow-list. Repository write access can redirect workers to another repo file.

## Delta publication

```python
upload_delta(
    state,
    *,
    node_id,
    local_step,
    base_outer_step,
    work_dir=None,
    metadata=None,
) -> str
```

Remote path:

```text
nodes/<node_id>/delta_<local_step>.safetensors
```

Safetensors metadata values are strings:

```json
{
  "algorithm": "dumb_diloco",
  "base_outer_step": "7",
  "local_step": "500",
  "node_id": "worker-1"
}
```

The outer loop derives node/local step from the path and reads `base_outer_step` from the header.

## `delta_paths()`

Returns all repository paths that start with `nodes/` and end in `.safetensors`. Parsing applies stricter path rules later.

## Local cache layout

```text
<cache_dir>/
├── downloads/<remote_path>
├── json/global/step_count.json
├── outgoing/global/step_count.json
└── speedtronic-download-<random>/   # temporary
```

The cache is not node-scoped and is not automatically rooted under run output.

## Failure behavior

All `_retry` failures become `HubUnavailable`. Higher coordinator layers usually catch and log those errors, allowing local training to continue. Because permanent errors are retried, an invalid token or repository name can consume the full backoff schedule on every operation.

## Security recommendations

- Use the SDK environment credential flow.
- Verify repository visibility after creation.
- Use one token with the minimum required scope.
- Restrict writer accounts.
- Do not treat `model_file` metadata as trusted input from an adversarial writer.
- Protect cache files from other local users.
- Monitor unexpected remote paths and updates.

Related: [Overview](./overview), [Outer Loop](./outer-loop), and [Security](../operations/security-and-limitations).
