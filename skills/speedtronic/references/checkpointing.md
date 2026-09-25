## Checkpoints and RNG

_Atomic checkpoint files, pointer discovery, retention, trainer state schema, RNG capture, and resume limitations._
### `CheckpointError`

A `RuntimeError` subclass used when a loaded artifact is not a state dictionary or no checkpoint exists for copying.

### Atomic helpers

#### `atomic_torch_save(state, path)`

1. Creates parent directories.
2. Writes to `.<name>.tmp-<pid>`.
3. Calls `torch.save`.
4. Replaces the destination with `os.replace`.
5. Removes a leftover temporary file in `finally`.

The rename is atomic on one filesystem, but files and parent directories are not explicitly `fsync`-ed for power-loss durability.

#### `atomic_json_dump(value, path)`

Writes indented JSON to a PID-specific temporary path and calls `os.replace`. It has no explicit temporary-file cleanup `finally` block.

### `CheckpointManager`

```python
CheckpointManager(
    directory,
    *,
    every_steps=500,
    keep_last=3,
    enabled=True,
)
```

Requires a positive interval even if disabled.

#### File layout

```text
<directory>/
├── latest.json
├── step_000000000002.pt
├── step_000000000004.pt
└── ...
```

`latest.json`:

```json
{"step": 4, "file": "step_000000000004.pt"}
```

#### `should_save(step)`

True when enabled, step is positive, and `step % every_steps == 0`.

#### `path_for(step)`

Returns a 12-digit filename: `step_<step:012d>.pt`.

#### `save(step, state)`

Directly writes the checkpoint, updates `latest.json`, and prunes. It does not itself check `enabled` or `should_save`; the trainer performs those checks.

#### `latest_path()`

If `latest.json` exists and points to an existing path, it trusts that pointer. Otherwise it scans and sorts `step_<digits>.pt` files and returns the highest step.

A process crash after writing a checkpoint but before updating the pointer can leave the pointer behind a higher valid file.

#### `load_latest()`

Loads the selected file to CPU with `weights_only=False`, then verifies that the result is a dictionary. It does not try an older checkpoint if the newest is corrupt.

#### `prune()`

Deletes old `step_*.pt` files to retain `keep_last`. `None` retains all files.

#### `copy_to(destination)`

Copies the file selected by `latest_path()`. It raises `CheckpointError` when none exists.

### Trainer checkpoint schema

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scaler": scaler_state_or_none,
    "scheduler": scheduler_state_or_none,
    "step": int,
    "samples": int,
    "tokens": int,
    "config": redacted_config_or_none,
    "rng": capture_rng_state(),
    "coordinator": coordinator_state_or_none,
    "precision": {
        "mode": "fp32",
        "autocast_enabled": False,
        "use_scaler": False,
    },
}
```

The stored config and precision plan are informational. Resume does not verify compatibility or reconstruct the current plan from them.

### RNG capture

`capture_rng_state()` records:

- Python `random` state;
- NumPy state when available;
- PyTorch CPU RNG state;
- all CUDA RNG states when CUDA is available.

MPS RNG is not captured. DataLoader generators and persistent worker RNG state are also not captured.

### Restore behavior

`restore_rng_state()` independently attempts each state restore and ignores individual exceptions.

`Trainer.resume()` loads model, optimizer, scheduler, counters, and RNG immediately. Scaler and coordinator states are deferred:

- scaler state is restored after `fit()` creates the scaler;
- coordinator state is applied after `start()` contacts or reconciles the Hub.

### Exact resume granularity

A checkpoint does not include:

- DataLoader iterator position;
- sampler state;
- epoch position;
- persistent worker state;
- arbitrary callback state;
- model train/eval mode.

Finite loaders restart iteration. `shuffle=False` can therefore repeat the beginning of a dataset after resume. RNG restoration does not recreate data-loader position.

### Final-step behavior

The trainer only checkpoints exact intervals. A run ending at step 3 with `every_steps=2` has no step-3 checkpoint. Resume returns to the most recent interval checkpoint and can repeat work after it.

### Coordinator state

When distributed mode is active, the checkpoint embeds coordinator state. For DumbDiLoCo this includes local/global counters, baseline, unused pending-delta fields, and the complete master outer state when present.

The checkpoint can therefore become large: it may contain the model, optimizer, baseline, global model, and momentum.

### v2 pending uploads

DumbDiLoCo checkpoints can include one immutable `pending_upload` job with its
step, base outer step, baseline generation, delta, and target snapshot. This
makes an in-flight boundary recoverable and bounded to one model-sized CPU
copy. Threads, queues, and locks are never serialized. See the
[DumbDiLoCo coordinator](references/distributed.md).

### Security


`.pt` files are loaded with pickle-capable `torch.load(..., weights_only=False)`. Treat checkpoint directories, distributed state directories, and serialized dataset files as trusted local files. A write-capable local attacker can execute code through a crafted payload.

### Operational recommendations

- Use short checkpoint intervals for expensive runs.
- Treat `max_steps` as absolute and retain a stable output directory.
- Copy important checkpoints outside the pruned directory.
- Protect `.pt` files with filesystem permissions.
- Do not use untrusted serialized datasets or state directories.
- Expect approximate rather than exact DataLoader replay.

See [Checkpoint and Resume](references/checkpointing.md) and [Security and Limitations](references/operations.md).


---

## Checkpoint and Resume

_Save, discover, retain, restore, and verify Speedtronic local training state._
### Configure checkpoints

```yaml
checkpoint:
  enabled: true
  directory: checkpoints
  every_steps: 100
  keep_last: 3
  resume: false
```

A relative directory is resolved under `run.output_dir`:

```text
<run.output_dir>/checkpoints
```

### Request resume

CLI:

```bash
speedtronic train \
  --config run.yaml \
  --output-dir runs/experiment \
  --resume
```

Python:

```python
from speedtronic.runtime import train_from_config

result = train_from_config(config, resume=True)
```

Configuration:

```yaml
checkpoint:
  resume: true
```

### What is restored

| State | Restored immediately | Notes |
|---|---:|---|
| Model parameters/buffers | Yes | Strict `load_state_dict` |
| AdamW state | Yes | Must match current optimizer structure |
| Scheduler state | Yes | `last_epoch` restored |
| Global step | Yes | Controls remaining work |
| Samples/tokens | Yes | Cumulative telemetry restored |
| CUDA GradScaler | Deferred | Loaded after `fit()` creates scaler |
| Coordinator | Deferred | Loaded after coordinator `start()` |
| Python/NumPy/Torch RNG | Yes | Best effort per subsystem |
| CUDA RNG | Yes when present | Best effort |
| DataLoader position | No | Iterator restarts |
| Epoch/sampler state | No | Not stored |
| Model train/eval mode | No | Module state remains as constructed |

### Absolute target behavior

```python
target_step = max(self.step, requested_target)
```

Examples:

| Restored step | Requested target | New updates |
|---:|---:|---:|
| 0 | 10 | 10 |
| 4 | 10 | 6 |
| 10 | 10 | 0 |
| 12 | 10 | 0 |

A resumed `TrainResult.steps` is the absolute final step.

### Two-step resume demonstration

```bash
speedtronic train \
  --config configs/smoke.yaml \
  --device cpu \
  --output-dir runs/resume-demo \
  --max-steps 2

speedtronic train \
  --config configs/smoke.yaml \
  --device cpu \
  --output-dir runs/resume-demo \
  --max-steps 4 \
  --resume
```

The CLI updates both `run.max_steps` and `scheduler.max_steps` for each invocation.

### Final-step gap

Saving occurs only at exact multiples of `every_steps`. A final off-interval step is not automatically saved.

| Final step | Interval | Latest saved |
|---:|---:|---:|
| 3 | 2 | 2 |
| 4 | 2 | 4 |
| 5 | 2 | 4 |

### Pointer behavior

`latest.json` points to a checkpoint file. If the pointer is missing or invalid, the manager scans filenames and chooses the highest step.

Crash windows and limitations:

- a higher checkpoint can exist while the pointer is stale;
- a valid pointer to a corrupt file does not fall back to an older one;
- atomic rename does not guarantee power-loss durability without `fsync`;
- temporary and data-file operations assume a local filesystem supporting atomic replacement.

### Retention

`keep_last: 3` prunes old `step_*.pt` files after a successful save. The pointer is not pruned.

Use `keep_last: null` for unlimited local retention, with disk-capacity planning.

### Verify a checkpoint

```python
import torch

state = torch.load(
    "runs/resume-demo/checkpoints/step_000000000002.pt",
    map_location="cpu",
    weights_only=False,
)

print(state.keys())
print(state["step"], state["samples"], state["tokens"])
```

Only inspect trusted files; `.pt` loading is pickle-capable.

### Distributed checkpoints

A distributed trainer checkpoint can include:

- local model and optimizer;
- coordinator counters/baseline;
- pending-delta fields;
- master global state;
- processed-delta set;
- outer momentum.

This duplicates large tensors and increases checkpoint size and memory pressure.

### Resume limitations

- DataLoader position is not exact.
- Optimizer updates skipped by a GradScaler overflow still advance the trainer's global step.
- The current coordinator resume ordering can install a fresh global model and then overwrite its baseline with checkpointed local state.
- A worker restart discards partial inner-loop progress unless captured by a later successful global installation path.
- Current config/model compatibility is not validated.

### Recovery checklist

1. Confirm the output directory is unchanged.
2. Inspect `latest.json`.
3. Load the selected file in a trusted environment.
4. Verify stored step and counters.
5. Use a target greater than the stored step.
6. Expect possible data repetition after restore.
7. Keep the previous checkpoint until the resumed run is validated.

Related: [Checkpoint API](references/checkpointing.md), [CLI](references/cli.md), and [DumbDiLoCo recovery](references/distributed.md).
