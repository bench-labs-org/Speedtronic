---
id: checkpointing
title: Checkpoints and RNG
sidebar_label: Checkpoints
description: Atomic checkpoint files, pointer discovery, retention, trainer state schema, RNG capture, and resume limitations.
---

# Checkpoints and RNG

## `CheckpointError`

A `RuntimeError` subclass used when a loaded artifact is not a state dictionary or no checkpoint exists for copying.

## Atomic helpers

### `atomic_torch_save(state, path)`

1. Creates parent directories.
2. Writes to `.<name>.tmp-<pid>`.
3. Calls `torch.save`.
4. Replaces the destination with `os.replace`.
5. Removes a leftover temporary file in `finally`.

The rename is atomic on one filesystem, but files and parent directories are not explicitly `fsync`-ed for power-loss durability.

### `atomic_json_dump(value, path)`

Writes indented JSON to a PID-specific temporary path and calls `os.replace`. It has no explicit temporary-file cleanup `finally` block.

## `CheckpointManager`

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

### File layout

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

### `should_save(step)`

True when enabled, step is positive, and `step % every_steps == 0`.

### `path_for(step)`

Returns a 12-digit filename: `step_<step:012d>.pt`.

### `save(step, state)`

Directly writes the checkpoint, updates `latest.json`, and prunes. It does not itself check `enabled` or `should_save`; the trainer performs those checks.

### `latest_path()`

If `latest.json` exists and points to an existing path, it trusts that pointer. Otherwise it scans and sorts `step_<digits>.pt` files and returns the highest step.

A process crash after writing a checkpoint but before updating the pointer can leave the pointer behind a higher valid file.

### `load_latest()`

Loads the selected file to CPU with `weights_only=False`, then verifies that the result is a dictionary. It does not try an older checkpoint if the newest is corrupt.

### `prune()`

Deletes old `step_*.pt` files to retain `keep_last`. `None` retains all files.

### `copy_to(destination)`

Copies the file selected by `latest_path()`. It raises `CheckpointError` when none exists.

## Trainer checkpoint schema

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

## RNG capture

`capture_rng_state()` records:

- Python `random` state;
- NumPy state when available;
- PyTorch CPU RNG state;
- all CUDA RNG states when CUDA is available.

MPS RNG is not captured. DataLoader generators and persistent worker RNG state are also not captured.

## Restore behavior

`restore_rng_state()` independently attempts each state restore and ignores individual exceptions.

`Trainer.resume()` loads model, optimizer, scheduler, counters, and RNG immediately. Scaler and coordinator states are deferred:

- scaler state is restored after `fit()` creates the scaler;
- coordinator state is applied after `start()` contacts or reconciles the Hub.

## Exact resume granularity

A checkpoint does not include:

- DataLoader iterator position;
- sampler state;
- epoch position;
- persistent worker state;
- arbitrary callback state;
- model train/eval mode.

Finite loaders restart iteration. `shuffle=False` can therefore repeat the beginning of a dataset after resume. RNG restoration does not recreate data-loader position.

## Final-step behavior

The trainer only checkpoints exact intervals. A run ending at step 3 with `every_steps=2` has no step-3 checkpoint. Resume returns to the most recent interval checkpoint and can repeat work after it.

## Coordinator state

When distributed mode is active, the checkpoint embeds coordinator state. For DumbDiLoCo this includes local/global counters, baseline, unused pending-delta fields, and the complete master outer state when present.

The checkpoint can therefore become large: it may contain the model, optimizer, baseline, global model, and momentum.

## v2 pending uploads

DumbDiLoCo checkpoints can include one immutable `pending_upload` job with its
step, base outer step, baseline generation, delta, and target snapshot. This
makes an in-flight boundary recoverable and bounded to one model-sized CPU
copy. Threads, queues, and locks are never serialized. See the
[DumbDiLoCo coordinator](../distributed/coordinator).

## Security


`.pt` files are loaded with pickle-capable `torch.load(..., weights_only=False)`. Treat checkpoint directories, distributed state directories, and serialized dataset files as trusted local files. A write-capable local attacker can execute code through a crafted payload.

## Operational recommendations

- Use short checkpoint intervals for expensive runs.
- Treat `max_steps` as absolute and retain a stable output directory.
- Copy important checkpoints outside the pruned directory.
- Protect `.pt` files with filesystem permissions.
- Do not use untrusted serialized datasets or state directories.
- Expect approximate rather than exact DataLoader replay.

See [Checkpoint and Resume](../tutorials/checkpoint-resume) and [Security and Limitations](../operations/security-and-limitations).
