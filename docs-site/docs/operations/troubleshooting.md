---
id: troubleshooting
title: Troubleshooting
sidebar_label: Troubleshooting
description: Diagnose configuration, model, data, precision, checkpoint, compilation, logging, CLI, and DumbDiLoCo failures.
---

# Troubleshooting

## Configuration errors

### `unknown configuration keys`

Speedtronic rejects unknown keys intentionally.

Check:

- section spelling;
- top-level aliases (`name`, `seed`, `device`, `max_steps`, `output_dir`, `log_every`);
- the current [configuration schema](../reference/configuration);
- whether `logging.hooks` was added—hooks are programmatic in 2.0.0.

### `batch sizes must be positive`

Set positive `micro_batch_size` and `target_batch_size`.

### `target_batch_size must be ... divisible`

Use:

```text
target_batch_size % micro_batch_size == 0
```

### Model dimensions invalid

For the bundled model:

- `n_kv_head` must divide `n_head`;
- `d_model` must divide by `n_head`;
- attention head dimension must be even for RoPE;
- all dimensions must be positive.

### Distributed repository missing

Enabled DumbDiLoCo requires:

```yaml
distributed:
  enabled: true
  repo_id: org/run
```

### Role changed unexpectedly

An enabled `single` role with a repository becomes `master`. Set `role: worker` explicitly on workers.

## Validation passes but training fails

`validate` checks schema and selected invariants only. It does not open files, build models, resolve hardware, or run data.

Common structurally valid but unrunnable configurations:

- missing `data.text_path`;
- unknown custom model name;
- model block size smaller than data block size;
- data vocabulary inconsistent with model vocabulary;
- requested unavailable CUDA/MPS device.

## Model errors

### `unknown model`

A custom model registered in one process is not automatically visible in a CLI process. Construct the runtime in the same process or add an application plugin import.

### `sequence length ... exceeds model block size`

The reference model rejects inputs longer than `model.max_seq_len`. Keep:

```text
data.block_size <= model.max_seq_len
```

for the bundled model.

### Unexpected `input_ids`, `labels`, or `attention_mask`

The trainer expands dictionary batches as keyword arguments. A custom model must accept the emitted keys or use a tuple batch/custom loader.

### Loss behaves incorrectly

A bare tensor is treated as a direct loss. Return `{"logits": logits}` plus compatible labels or return a loss mapping when using 3-D output tensors.

### Causal target appears shifted twice

Speedtronic 2.0 consumes the already-shifted labels emitted by its datasets. If a custom model or dataset still shifts the same labels, align the contract to one next-token label per input position.

## Data errors

### `data loader produced no batches`

The complete loader pass yielded no items. Check:

- empty iterable dataset;
- text file exists and is readable;
- final stream behavior;
- dataset filter removed all examples;
- `drop_last` with a dataset smaller than one batch.

### `data.dataset must ... exist`

The configured serialized path is absent or is not a PyTorch Dataset.

Never load an untrusted `.pt` dataset; `weights_only=False` permits pickle code execution.

### Duplicate text data with workers

The bundled text iterable dataset is not worker-sharded. Set `num_workers: 0` or implement worker-aware sharding.

### Broken custom tokenizer

The stream carries IDs, not raw text. Context-sensitive tokenizers can split merges at chunk boundaries. Use an incremental tokenizer or a whole-file preprocessing step.

### Out-of-range token IDs

The standard runtime passes `model.vocab_size` to the loader and ignores a separate `data.vocab_size`. Keep vocabularies aligned.

## Runtime and precision errors

### CUDA/MPS unavailable

`resolve_device()` raises when an explicit backend is unavailable. Use `--device cpu` or `auto`.

### Unexpected FP32 fallback

Expected cases include:

- auto on CPU/MPS;
- explicit CPU FP16;
- mixed precision on MPS;
- unsupported CUDA BF16.

The `train_start` event reports the resolved mode.

### CPU BF16 failure

Explicit CPU BF16 is not capability-checked. Use FP32 if the installed PyTorch/custom operators cannot run CPU autocast BF16.

### Fused AdamW not used

Fused construction is best-effort and may fail because of device placement or backend support. No warning is emitted by the fallback path.

## Compilation

### Compilation disabled automatically

A compile event reports construction/runtime fallback. The run continues eagerly.

### Model executes twice

The compiled forward raised, then the trainer reran the same batch eagerly. Look for model side effects or an actual model error masked by the broad fallback.

### `torch.compile` overhead on short runs

Compilation cost can dominate a four-step smoke test. Keep `compile: false` for smoke runs.

## Scheduler behavior

### Run ends during warmup

Set `scheduler.max_steps` and `warmup_steps` explicitly. Programmatic run-target overrides do not reliably update the default 1000-step scheduler.

### Logged LR is one step ahead

The optimizer steps before the scheduler advances, and metrics read the post-scheduler LR.

## GradScaler behavior

On CUDA FP16, an overflow can cause `GradScaler` to skip the optimizer update while Speedtronic still increments the global step and scheduler. A sudden high loss or unchanged parameters around an early step can indicate this.

## Checkpoint and resume

### No checkpoint found

Resume searches:

```text
<resolved output_dir>/<checkpoint.directory>/
```

Changing `--output-dir` changes the search location.

A warning is logged and a new run starts.

### Checkpoint is stale behind file scan

If `latest.json` is missing, the manager chooses the highest valid filename. If the pointer is valid, it trusts the pointer.

### Corrupt newest checkpoint

`load_latest()` does not fall back to an older file after a corrupt selected checkpoint. Restore a known-good copy or point the directory to one.

### Resume repeats data

DataLoader position and worker state are not checkpointed. Expect approximate replay.

### Final step not saved

Only exact checkpoint intervals save. A final off-interval step is lost from the latest checkpoint.

### Checkpoint shape mismatch

Current config/model compatibility is not checked before `load_state_dict`. Ensure identical architecture and state keys.

## Logging and integrations

### No W&B/TensorBoard configuration

`LoggingConfig` has no hooks field. Attach adapters programmatically.

### TensorBoard file remains open

Call `TensorboardHook.close()`.

### W&B run does not finish

The adapter has no public close method. Finish the run through the Wandb API or application lifecycle.

### JSONL has concurrent writes

The master outer thread and trainer can emit concurrently. The logger does not lock JSONL appends or hook invocation.

## DumbDiLoCo

### Worker cannot find global metadata

The master may not have bootstrapped yet, credentials may lack access, or the repository may be wrong. The worker continues locally and retries later.

### Repeated slow retries

Every exception, including 401/403 and 404, is retried with backoff. Verify repository/token configuration before waiting through repeated delays.

### Corrupt delta warnings

The master skips the file and retries it on later polls because invalid files are not marked processed. Remove or replace an unrecoverable remote file.

### Delta key/shape mismatch

All nodes must use the same floating-state key set and shapes. A custom architecture or version skew rejects the full delta.

### No global update

Check:

- delta was uploaded successfully;
- `outer_step` increased;
- `base_outer_step` and identities are valid;
- one active master is running;
- the worker is not stuck behind an invalid cache;
- model state shapes match.

### Repository grows continuously

Expected in 2.0.0. Deltas are not deleted, global caches are not pruned, and all listed candidates are downloaded.

### Unexpected optimizer reset

Optimizer state clears after a successful delta upload **or** global installation, not strictly at inner boundaries.

### Resume produces a large delta

Current startup/load ordering can pair a fresh global model with a checkpointed older baseline. Treat distributed resume as a known limitation and verify baseline/global step before training.

## v2 optimizer and systems issues

### Muon routes too few parameters

Check the parameter role heuristic and tied weights. Embeddings, heads, norms,
and biases intentionally remain on AdamW. A model can set
`_speedtronic_optimizer_role` to make an explicit choice.

### Muon fails on a sparse or complex gradient

Muon is defined for dense, real, floating-point 2-D matrices. Route sparse or
complex parameters to AdamW or provide a custom optimizer.

### OOO backprop does nothing

`ooo_backprop` is a no-op on CPU and MPS. It is also disabled when
`compile: true`, when fewer than two module stages are found, or when CUDA
streams cannot be installed. Inspect the `ooo_backprop` event.

### Shape warnings appear on CPU

Automatic CPU profiles are quiet. An explicit `shape_validation.alignment`
also needs `warn_on_cpu: true` to enable CPU benchmarking warnings. Set
`alignment: none` to disable them.

### Async delta is skipped

A `delta_upload_skipped` event with `reason=upload_in_flight` means the
one-slot queue is occupied. The next accepted boundary sends a cumulative
delta. Increase `delta_upload_shutdown_timeout` only for controlled shutdown
tests, not to make the inner loop block.

### Async poll does not install a new global immediately

Downloads are intentionally performed in the background and installed only at
a training-thread boundary. Inspect `delta_upload` and `ooo_backprop`/`shape`
events plus the outer-step metric.

## Diagnostics

For local issues, collect:

```text
resolved configuration
device and precision plan
start/target step
checkpoint pointer
first failing stack trace
metric records
exact model/data dimensions
node role/ID/repository when distributed
```

Never collect or print token values. Share redacted configuration and filenames instead.

Related: [Security and Limitations](./security-and-limitations), [Configuration](../reference/configuration), and [DumbDiLoCo](../distributed/overview).
