---
id: checkpoint-resume
title: Checkpoint and Resume
sidebar_label: Checkpoint & Resume
description: Save, discover, retain, restore, and verify Speedtronic local training state.
---

# Checkpoint and resume

## Configure checkpoints

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

## Request resume

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

## What is restored

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

## Absolute target behavior

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

## Two-step resume demonstration

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

## Final-step gap

Saving occurs only at exact multiples of `every_steps`. A final off-interval step is not automatically saved.

| Final step | Interval | Latest saved |
|---:|---:|---:|
| 3 | 2 | 2 |
| 4 | 2 | 4 |
| 5 | 2 | 4 |

## Pointer behavior

`latest.json` points to a checkpoint file. If the pointer is missing or invalid, the manager scans filenames and chooses the highest step.

Crash windows and limitations:

- a higher checkpoint can exist while the pointer is stale;
- a valid pointer to a corrupt file does not fall back to an older one;
- atomic rename does not guarantee power-loss durability without `fsync`;
- temporary and data-file operations assume a local filesystem supporting atomic replacement.

## Retention

`keep_last: 3` prunes old `step_*.pt` files after a successful save. The pointer is not pruned.

Use `keep_last: null` for unlimited local retention, with disk-capacity planning.

## Verify a checkpoint

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

## Distributed checkpoints

A distributed trainer checkpoint can include:

- local model and optimizer;
- coordinator counters/baseline;
- pending-delta fields;
- master global state;
- processed-delta set;
- outer momentum.

This duplicates large tensors and increases checkpoint size and memory pressure.

## Resume limitations

- DataLoader position is not exact.
- Optimizer updates skipped by a GradScaler overflow still advance the trainer's global step.
- The current coordinator resume ordering can install a fresh global model and then overwrite its baseline with checkpointed local state.
- A worker restart discards partial inner-loop progress unless captured by a later successful global installation path.
- Current config/model compatibility is not validated.

## Recovery checklist

1. Confirm the output directory is unchanged.
2. Inspect `latest.json`.
3. Load the selected file in a trusted environment.
4. Verify stored step and counters.
5. Use a target greater than the stored step.
6. Expect possible data repetition after restore.
7. Keep the previous checkpoint until the resumed run is validated.

Related: [Checkpoint API](../reference/checkpointing), [CLI](../reference/cli), and [DumbDiLoCo recovery](../distributed/coordinator).
