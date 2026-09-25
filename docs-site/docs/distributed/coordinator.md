---
id: coordinator
title: DumbDiLoCo Coordinator
sidebar_label: Coordinator
description: Master/worker startup, baseline lifecycle, polling, uploads, state serialization, and reset signaling.
---

# DumbDiLoCo coordinator

## Public API

```python
from speedtronic import DumbDiLoCoCoordinator
# alias: DumbDiLoCo
```

Constructor:

```python
DumbDiLoCoCoordinator(
    config,
    model,
    *,
    hub=None,
    state_dir=None,
    logger=None,
)
```

`config` may be:

- `DistributedConfig`;
- a full `SpeedtronicConfig` exposing `.distributed`;
- a root mapping containing a `distributed` section.

The package-level `SyncResult` dataclass exists with `pushed`, `loaded_global`, and `outer_step`, but the current coordinator does not instantiate or return it.

## Node identity and paths

Identity resolution:

```text
config.node_id
→ HF_USERNAME
→ USER
→ "local"
```

Explicit IDs are config-validated. Environment-derived IDs are not revalidated.

State root:

```text
<state_root>/<node_id>/
```

`build_runtime()` resolves a relative `state_dir` under `run.output_dir`. The Hub `cache_dir` remains relative to the process working directory.

## Constructor state

The constructor captures an initial CPU baseline, initializes counters, and may create a `HubClient` or use an injected fake/transport.

Properties:

```python
coordinator.local_step
coordinator.last_global_step
coordinator.outer_step
```

## `start()`

The first call starts the role-specific lifecycle. `stop()` releases the
started flag after bounded shutdown, so a later `fit()` can start fresh
background lanes. If a Hub request outlives the timeout, its daemon thread
continues until the request returns and emits a shutdown warning.

### Master startup

1. Attempt private repository creation with `exist_ok=True`.
2. Attempt `write` collaborator grants.
3. Construct `MasterOuterLoop`; this loads local master state.
4. Read remote global metadata.
5. Select local state, remote state, or bootstrap.
6. Start the background outer thread.

Repository and collaborator failures do not stop local training.

### Worker startup

1. Read global metadata.
2. Download/install the referenced global if available.
3. Establish a new baseline.
4. Continue locally when metadata is absent or the read fails.

## Startup state selection

Let `L` be local outer step and `R` remote outer step.

| Condition | Selection |
|---|---|
| `L > 0` and remote unavailable or `R <= L` | Local global state wins |
| `R > L` | Remote model wins; outer state adopts it and resets momentum |
| `L = 0`, `R = 0` | Remote model is installed into the local model |
| No local state and no remote metadata | Bootstrap current model at outer step 0 |

Remote state does not contain processed deltas or Nesterov momentum.

## Global installation

`_install_state()` filters incoming tensors to keys present locally with identical shapes, then calls:

```python
model.load_state_dict(filtered, strict=False)
```

Missing keys, extra local keys, and shape mismatches are retained/ignored rather than raising. After installation, the complete local model becomes the new baseline.

## Polling

`after_optimizer_step(model, step)` verifies model identity, records local step, and computes:

```python
boundary = step % inner_steps == 0
```

At a boundary it uploads and force-polls. At other steps it polls only if `poll_interval` elapsed since the last poll.

Worker poll:

1. Read metadata.
2. Ignore versions `<= last_global_step`.
3. Use a cached per-version safetensors file when valid.
4. Delete and redownload a corrupt cache entry.
5. Install state and advance the baseline.

Master poll:

1. Read the outer loop's local snapshot under its lock.
2. Install it when newer.
3. Otherwise use the callback snapshot fallback.

## Delta upload

At an inner boundary:

```python
current = cpu_state_dict(model.state_dict())
delta = compute_pseudo_gradient(baseline, current)
hub.upload_delta(... base_outer_step=last_global_step ...)
```

Only after successful upload does the coordinator:

- set `_last_uploaded_step`;
- advance the baseline to `current`.

A failed upload leaves the baseline unchanged. The next boundary can therefore recompute a cumulative delta unless a global installation resets the baseline first.

## v2 asynchronous dispatch

With `async_delta_upload: true` (the v2 default), a boundary copies the model
state, computes the pseudo-gradient, and calls `put_nowait` on a one-slot
queue. The uploader thread performs serialization and Hub retries. The next
optimizer step is not held behind that network call. A second boundary while
the slot is occupied returns without waiting and emits
`delta_upload_skipped`.

A successful completion advances the baseline to the immutable target
snapshot only when the baseline generation is unchanged. A failed upload
leaves the baseline in place, so the next accepted boundary can send a
cumulative delta. A global installation increments the generation and prevents
an older in-flight completion from overwriting the new baseline.

`async_global_poll: true` submits metadata/state fetches to a separate worker.
The uploader/poll workers never mutate the live model; global state is
installed at the next training-thread callback.

## Optimizer-reset signal


The method returns:

```python
pushed or loaded
```

The trainer variable is named `weights_replaced`, but a true return can mean either:

- a delta upload succeeded; or
- newer global state was installed.

When `reset_inner_optimizer=true`, the trainer clears `optimizer.state` after any true result. This can occur mid-inner-loop after a global poll.

## State serialization

`state_dict()` returns:

```text
node_id
role
last_global_step
last_uploaded_step
local_step
baseline
pending_delta
pending_delta_step
outer (master only)
```

`pending_delta` and `pending_delta_step` are currently never populated by normal upload logic. `last_uploaded_step` is informational and does not suppress repeated uploads.

`load_state_dict()` restores counters, baseline, pending fields, and optional master outer state. It does not enforce that saved role/node ID matches the current process.

## Resume ordering caveat

The trainer:

1. calls `coordinator.start()`, which may install and baseline a fresh global;
2. then calls `coordinator.load_state_dict()` with deferred checkpoint state.

The checkpointed baseline can overwrite the baseline established by fresh global installation. On a worker resume, that can pair a newer model with an older baseline and produce a large unintended pseudo-gradient. This is a correctness limitation addressed by v2's prepared startup state.

## Stop behavior

`stop()` marks the coordinator stopped and asks the master outer thread to stop. It:

- does not force a partial delta;
- does not force a final outer sync;
- joins the thread for up to 10 seconds;
- does not clear `_started`.

A worker retains its latest local/global state. A master retains its local outer state after successful publication/save.

## Failure semantics

| Failure | Behavior |
|---|---|
| Initial repository/create/grant | Warning; local loop continues |
| Initial worker Hub read | Warning; local weights remain |
| Poll failure | Warning; model unchanged |
| Key/shape mismatch in delta | Error log; no upload |
| Upload failure | Warning; baseline not advanced |
| Corrupt local global cache | Delete and redownload |
| Wrong model passed to callback | `ValueError` stops training |

## Direct construction

```python
from speedtronic.config import DistributedConfig
from speedtronic.distributed import DumbDiLoCoCoordinator

config = DistributedConfig(
    enabled=True,
    role="worker",
    node_id="worker-1",
    repo_id="org/run",
)
coordinator = DumbDiLoCoCoordinator(config, model)
coordinator.start()
```

A real Hub token is resolved by the Hugging Face SDK; Speedtronic itself passes the configured token and does not define a separate token environment lookup.

Related: [Overview](./overview), [Outer Loop](./outer-loop), and [Security](../operations/security-and-limitations).
