## Device and Precision

_Device selection, mixed-precision resolution, GradScaler, autocast, and fused AdamW behavior._
### `PrecisionPlan`

```python
@dataclass(frozen=True)
class PrecisionPlan:
    mode: str
    dtype: Any
    use_scaler: bool
    autocast_enabled: bool
    device_type: str
```

`is_mixed_precision` is true for `bf16` and `fp16`.

### `resolve_device(requested="auto")`

Auto order:

1. CUDA when `torch.cuda.is_available()`.
2. MPS when exposed and available.
3. CPU.

Explicit CUDA and MPS requests verify availability and raise `RuntimeError` when unavailable. Other device strings are passed to `torch.device` directly.

### `resolve_precision(config, device)`

The resolved plan is used by the trainer; parameters and buffers themselves are not globally cast to the mixed dtype.

#### Resolution matrix

| Device | Request | Resolved mode | Scaler | Autocast |
|---|---|---|---|---:|
| CUDA | `auto`, BF16 supported | `bf16` | No | Yes |
| CUDA | `auto`, BF16 unsupported | `fp16` | Yes | Yes |
| CUDA | explicit `bf16`, supported | `bf16` | No | Yes |
| CUDA | explicit `bf16`, unsupported/error | `fp32` | No | No |
| CUDA | explicit `fp16` | `fp16` | Yes | Yes |
| CPU | `auto` | `fp32` | No | No |
| CPU | explicit `fp32` | `fp32` | No | No |
| CPU | explicit `fp16` | `fp32` | No | No |
| CPU | explicit `bf16` | `bf16` | No | Yes |
| MPS | `auto` | `fp32` | No | No |
| MPS | explicit mixed mode | `fp32` | No | No |
| MPS | explicit `fp32` | `fp32` | No | No |

> **Caution** — Explicit CPU BF16
>
>
> The code does not check CPU BF16 capability. It preserves explicit BF16 on CPU and enables CPU autocast. Modern PyTorch may support it, but behavior depends on the installed PyTorch build and custom operators.
>

### `precision.dtype`

`PrecisionConfig.dtype` affects resolution only when `mode == "auto"`. For example:

```yaml
precision:
  mode: auto
  dtype: fp16
```

requests FP16. With `mode: fp32`, `dtype: bf16` is ignored.

### `autocast_context(plan, device)`

Returns `nullcontext()` for FP32. For mixed precision, returns `torch.autocast(device_type, dtype)`, with a compatibility fallback for older signatures. The `device` argument is accepted but not directly consulted; the plan's `device_type` controls the context.

### `make_grad_scaler(plan)`

Returns a scaler only for FP16 on CUDA. It prefers:

```python
torch.amp.GradScaler("cuda")
```

and falls back to `torch.cuda.amp.GradScaler()` for older PyTorch.

The trainer creates a new scaler on every `fit()` invocation, then restores deferred checkpoint state when present.

### `supports_fused_adamw(device)`

Returns false on non-CUDA devices. On CUDA it constructs a zero-step fused AdamW optimizer with a one-element parameter and catches `TypeError`, `RuntimeError`, and `ValueError`.

`build_optimizer()` currently builds the actual model on CPU and moves it later. A capability probe on a CUDA parameter can succeed while construction with CPU model parameters fails, causing an ordinary AdamW fallback.

### Fused AdamW configuration

```yaml
optimizer:
  fused: null   # probe
```

```yaml
optimizer:
  fused: true   # attempt
```

```yaml
optimizer:
  fused: false  # disable
```

A failed attempt is silent except for the absence of fused behavior.

### Performance implications

- BF16 generally avoids CUDA `GradScaler`.
- FP16 CUDA uses dynamic loss scaling.
- CPU auto precision stays FP32 for broad custom-module compatibility.
- MPS auto precision stays FP32.
- Microbatch loss-to-CPU conversion still synchronizes CUDA even when mixed precision reduces tensor compute cost.
- Metric memory queries can also synchronize the current accelerator.

### Test evidence

The repository directly tests CPU auto FP32 and explicit CPU FP16 fallback. CUDA, MPS, CPU BF16, fused AdamW, and scaler lifecycle are implementation-derived and not covered by repository tests.

See [Local Performance](references/precision-and-performance.md) and [Runtime and Trainer](references/architecture.md).


---

## Local Performance

_Tune accumulation, precision, workers, fused AdamW, compilation, gradient checkpointing, and reference-model attention._
This page describes mechanisms implemented in Speedtronic 2.0.0 and their trade-offs. Repository tests are CPU-focused and do not validate every hardware path.

### Gradient accumulation

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

### Precision

| Mode | Typical use |
|---|---|
| `fp32` | Maximum portability and numerical stability |
| `bf16` | CUDA devices with native BF16 support |
| `fp16` | CUDA fallback with dynamic loss scaling |
| `auto` | BF16, FP16, or FP32 based on device |

See [Precision](references/precision-and-performance.md) for the exact matrix and caveats.

### DataLoader throughput

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

### Fused AdamW

```yaml
optimizer:
  fused: null
```

Speedtronic probes fused AdamW on CUDA and falls back if construction fails. The model is built on CPU before being moved, so the current probe and actual construction can disagree about device placement.

### `torch.compile`

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

### Gradient checkpointing

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

### Reference attention

The reference model uses `torch.nn.functional.scaled_dot_product_attention`, allowing PyTorch to choose flash, memory-efficient, or math kernels without a separate attention package.

Supplying `attention_mask` creates a dense `(B, 1, T, T)` floating mask. This can reduce kernel efficiency. With no mask, the model uses `is_causal=True`.

### Scheduler and measured LR

The optimizer updates, then the scheduler advances, then metrics read the current LR. The logged LR is therefore the next step's LR, not the rate that produced the just-completed update.

### Throughput synchronization

Every microbatch executes:

```python
float(loss.detach().float().cpu())
```

On CUDA, this synchronizes the device on each microbatch. Token counting with an attention mask also performs `.sum().item()`. These operations can reduce throughput in a speed-focused engine.

Metric memory queries can synchronize as well.

### Fused outer operations

DumbDiLoCo outer aggregation occurs in FP32 on CPU. The Hub upload/download path is I/O-heavy and can dominate outer rounds for small deltas or large models.

### v2 additions

- [Hybrid Muon](references/optimizers.md) changes the optimizer algorithm, not just kernel scheduling.
- [OOO backprop](references/scheduling-and-shapes.md) is opt-in and CUDA-specific in its active path.
- [Shape validation](references/scheduling-and-shapes.md) warns about unaligned effective dimensions.

Muon, Muon+, and cautious updates can compose with local performance settings,
but benchmark convergence as well as throughput.

### Recommended optimization order


1. Establish a correct eager FP32 baseline.
2. Tune microbatch and target batch.
3. Benchmark workers and prefetch.
4. Enable a supported precision mode.
5. Verify numerical behavior.
6. Benchmark fused AdamW on CUDA.
7. Add gradient checkpointing only when memory pressure requires it.
8. Add `torch.compile` last and verify eager fallback behavior.

### Measurement cautions

Speedtronic logs cumulative counters but computes rates from the start of the current `fit()` invocation. It measures wall time including synchronous Hub work and CPU loss transfer, but does not separate forward, backward, optimizer, checkpoint, or logging costs.

Related: [Precision](references/precision-and-performance.md), [Data](references/data.md), and [Architecture](references/architecture.md).
