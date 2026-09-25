---
id: local-performance
title: Local Performance
sidebar_label: Local Performance
description: Tune accumulation, precision, workers, fused AdamW, compilation, gradient checkpointing, and reference-model attention.
---

# Local performance

This page describes mechanisms implemented in Speedtronic 2.0.0 and their trade-offs. Repository tests are CPU-focused and do not validate every hardware path.

## Gradient accumulation

```yaml
data:
  micro_batch_size: 1
  target_batch_size: 8
```

This produces eight microbatches per AdamW update. Each loss is divided by eight before backward propagation.

Accumulation:

- keeps per-microbatch parameter memory bounded;
- increases wall time for the same examples;
- allows the nominal target batch to exceed loader batch size;
- still counts one global optimizer step and one scheduler step.

The final microbatch may be smaller when `drop_last=False`.

## Precision

| Mode | Typical use |
|---|---|
| `fp32` | Maximum portability and numerical stability |
| `bf16` | CUDA devices with native BF16 support |
| `fp16` | CUDA fallback with dynamic loss scaling |
| `auto` | BF16, FP16, or FP32 based on device |

See [Precision](../reference/precision) for the exact matrix and caveats.

## DataLoader throughput

```yaml
data:
  num_workers: 2
  prefetch_factor: 2
  pin_memory: true
```

- Workers run in separate processes and can overlap data preparation.
- Persistent workers remain alive between loader epochs.
- Pinning accelerates CPU-to-CUDA transfers.
- Iterable datasets cannot be shuffled by a sampler.
- The bundled text stream duplicates full-file work across workers.

Benchmark worker changes rather than assuming monotonic speedup.

## Fused AdamW

```yaml
optimizer:
  fused: null
```

Speedtronic probes fused AdamW on CUDA and falls back if construction fails. The model is built on CPU before being moved, so the current probe and actual construction can disagree about device placement.

## `torch.compile`

```yaml
compile: true
```

or:

```yaml
compile:
  enabled: true
```

Compilation is best-effort. Construction failure continues eagerly. A runtime exception from a compiled forward disables compilation and retries the same batch eagerly.

The broad runtime catch can mask a model bug if eager execution succeeds, and rerunning can duplicate RNG or side effects. Enable it after establishing a correct eager baseline.

## Gradient checkpointing

```yaml
gradient_checkpointing: true
```

The model must expose:

```python
set_gradient_checkpointing(enabled: bool)
```

The reference transformer propagates the flag to every block.

Trade-off:

- lower activation memory;
- additional forward recomputation;
- potentially lower throughput;
- smaller feasible model/batch sizes.

## Reference attention

The reference model uses `torch.nn.functional.scaled_dot_product_attention`, allowing PyTorch to choose flash, memory-efficient, or math kernels without a separate attention package.

Supplying `attention_mask` creates a dense `(B, 1, T, T)` floating mask. This can reduce kernel efficiency. With no mask, the model uses `is_causal=True`.

## Scheduler and measured LR

The optimizer updates, then the scheduler advances, then metrics read the current LR. The logged LR is therefore the next step's LR, not the rate that produced the just-completed update.

## Throughput synchronization

Every microbatch executes:

```python
float(loss.detach().float().cpu())
```

On CUDA, this synchronizes the device on each microbatch. Token counting with an attention mask also performs `.sum().item()`. These operations can reduce throughput in a speed-focused engine.

Metric memory queries can synchronize as well.

## Fused outer operations

DumbDiLoCo outer aggregation occurs in FP32 on CPU. The Hub upload/download path is I/O-heavy and can dominate outer rounds for small deltas or large models.

## v2 additions

- [Hybrid Muon](../v2/optimizers) changes the optimizer algorithm, not just kernel scheduling.
- [OOO backprop](../v2/scheduling) is opt-in and CUDA-specific in its active path.
- [Shape validation](../v2/shape-validation) warns about unaligned effective dimensions.

Muon, Muon+, and cautious updates can compose with local performance settings,
but benchmark convergence as well as throughput.

## Recommended optimization order


1. Establish a correct eager FP32 baseline.
2. Tune microbatch and target batch.
3. Benchmark workers and prefetch.
4. Enable a supported precision mode.
5. Verify numerical behavior.
6. Benchmark fused AdamW on CUDA.
7. Add gradient checkpointing only when memory pressure requires it.
8. Add `torch.compile` last and verify eager fallback behavior.

## Measurement cautions

Speedtronic logs cumulative counters but computes rates from the start of the current `fit()` invocation. It measures wall time including synchronous Hub work and CPU loss transfer, but does not separate forward, backward, optimizer, checkpoint, or logging costs.

Related: [Precision](../reference/precision), [Data](../reference/data), and [Architecture](../reference/architecture).
