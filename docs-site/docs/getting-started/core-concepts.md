---
id: core-concepts
title: Core Concepts
sidebar_label: Core Concepts
description: Understand configuration, microbatch accumulation, global steps, checkpoints, and DumbDiLoCo terminology.
---

# Core concepts

## Configuration is the composition contract

`SpeedtronicConfig` is a tree of mutable dataclasses:

```text
SpeedtronicConfig
├── run
├── model
├── data
├── optimizer
├── scheduler
├── precision
├── checkpoint
├── logging
├── distributed
├── gradient_checkpointing
└── compile
```

The runtime composition root, `build_runtime()`, converts that declarative description into concrete objects. Configuration validation is deliberately structural and lightweight; it does not guarantee that a data path exists or that a custom model can train.

## Microbatch versus optimizer step

A loader emits `data.micro_batch_size` examples. The trainer consumes:

```text
accumulation_steps = target_batch_size / micro_batch_size
```

microbatches before one optimizer update. Every loss is divided by that count before backward propagation. The scheduler, global step counter, coordinator callback, and checkpoint cadence advance once per complete optimizer update.

If the final emitted microbatch is smaller than `micro_batch_size` and `drop_last=False`, the update can contain fewer than the nominal target batch.

## Absolute global steps

`run.max_steps`, `data.max_steps`, the `Trainer` constructor, and `Trainer.fit()` describe an absolute target:

```python
target_step = max(self.step, requested_target)
```

A trainer resumed at step 900 with a target of 1000 performs 100 steps. A trainer already at step 1000 performs none and returns `final_loss=None`.

## Model input contract

The built-in trainer recognizes two batch forms:

| Batch form | Forward call |
|---|---|
| `dict` | `model(**batch)` |
| `tuple` or `list` with at least two elements | `model(batch[0], batch[1])` |

The built-in causal loader emits:

```python
{
    "input_ids": LongTensor[batch, sequence],
    "labels": LongTensor[batch, sequence],
    "attention_mask": BoolTensor[batch, sequence],
}
```

A custom dataset that returns arbitrary dictionaries does not automatically receive a general-purpose collator; the bundled collator specifically understands causal dictionaries.

## Model output contract

A model can return:

- a scalar loss tensor;
- `{"loss": loss, ...}`;
- `{"logits": logits, ...}` plus compatible labels;
- a tuple/list whose first item is a loss;
- a tuple/list whose first item is 3-D causal logits.

A bare tensor is always interpreted as a loss, not logits. Extra metrics returned in an output mapping are not currently forwarded to `MetricLogger`.

## Precision resolution

`precision.mode: auto` means:

- CUDA with BF16 support → BF16;
- CUDA without BF16 support → FP16 plus `GradScaler`;
- CPU or MPS → FP32.

Explicit FP16 on CPU and mixed precision on MPS fall back to FP32. Unsupported BF16 on CUDA also falls back. Explicit CPU BF16 is not capability-checked by Speedtronic.

## Checkpoint versus distributed global state

A local trainer checkpoint contains model, optimizer, scheduler, scaler, counters, RNG, redacted config, and coordinator state. It is saved beneath the local filesystem.

DumbDiLoCo additionally maintains:

- a Hub-global model and outer step;
- a master's local processed-delta ledger and Nesterov momentum;
- per-node baselines and cached global files.

These are separate state domains. A worker restart begins from the latest global version it can read, not from an exact continuation of its DataLoader or partial inner loop.

## DumbDiLoCo vocabulary

| Term | Meaning in Speedtronic |
|---|---|
| Local step | One completed local AdamW optimizer update |
| Inner step | A local step; the name emphasizes that it belongs to the local objective |
| Inner boundary | `local_step % inner_steps == 0` |
| Baseline | Model state captured at the last successful upload or global installation |
| Pseudo-gradient | `baseline - current`; not an autograd gradient |
| Outer round | One successful master aggregation/publication |
| Outer step | Monotonic integer published in `global/step_count.json` |
| Node delta | Floating-state safetensors plus path/header metadata |
| Global model | Complete CPU-cloned `state_dict` published by the master |

## State ownership

| State | Primary owner | Local checkpoint | Hub |
|---|---|---:|---:|
| Parameters and buffers | `torch.nn.Module` | Yes | Global model |
| AdamW moments | `torch.optim.AdamW` | Yes | No |
| Scheduler | `LambdaLR` | Yes | No |
| CUDA scaler | `GradScaler` | Yes | No |
| Step/sample/token counters | `Trainer` | Yes | No |
| Global RNG | Python/NumPy/PyTorch | Yes | No |
| DataLoader position | PyTorch loader/workers | No | No |
| Global outer model | `MasterOuterLoop` | Master state | Yes |
| Processed deltas | `MasterOuterLoop` | Master state | No |
| Nesterov momentum | `MasterOuterLoop` | Master state | No |
| Worker baseline | Coordinator | Yes | No |

## Extension seams

- **Model factory:** `ModelRegistry` / `register_model`
- **Gradient checkpointing:** `model.set_gradient_checkpointing(enabled)`
- **Dataset and tokenizer:** `build_dataloader(dataset=..., tokenizer=...)`
- **Coordinator:** implement the `SyncCoordinator` protocol
- **Metrics:** callables or objects with `on_event(event, payload)`

See [architecture](../reference/architecture), [extensions](../reference/extensions), and [DumbDiLoCo](../distributed/overview).
