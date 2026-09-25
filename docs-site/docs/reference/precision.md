---
id: precision
title: Device and Precision
sidebar_label: Precision
description: Device selection, mixed-precision resolution, GradScaler, autocast, and fused AdamW behavior.
---

# Device and precision

## `PrecisionPlan`

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

## `resolve_device(requested="auto")`

Auto order:

1. CUDA when `torch.cuda.is_available()`.
2. MPS when exposed and available.
3. CPU.

Explicit CUDA and MPS requests verify availability and raise `RuntimeError` when unavailable. Other device strings are passed to `torch.device` directly.

## `resolve_precision(config, device)`

The resolved plan is used by the trainer; parameters and buffers themselves are not globally cast to the mixed dtype.

### Resolution matrix

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

:::caution Explicit CPU BF16

The code does not check CPU BF16 capability. It preserves explicit BF16 on CPU and enables CPU autocast. Modern PyTorch may support it, but behavior depends on the installed PyTorch build and custom operators.

:::

## `precision.dtype`

`PrecisionConfig.dtype` affects resolution only when `mode == "auto"`. For example:

```yaml
precision:
  mode: auto
  dtype: fp16
```

requests FP16. With `mode: fp32`, `dtype: bf16` is ignored.

## `autocast_context(plan, device)`

Returns `nullcontext()` for FP32. For mixed precision, returns `torch.autocast(device_type, dtype)`, with a compatibility fallback for older signatures. The `device` argument is accepted but not directly consulted; the plan's `device_type` controls the context.

## `make_grad_scaler(plan)`

Returns a scaler only for FP16 on CUDA. It prefers:

```python
torch.amp.GradScaler("cuda")
```

and falls back to `torch.cuda.amp.GradScaler()` for older PyTorch.

The trainer creates a new scaler on every `fit()` invocation, then restores deferred checkpoint state when present.

## `supports_fused_adamw(device)`

Returns false on non-CUDA devices. On CUDA it constructs a zero-step fused AdamW optimizer with a one-element parameter and catches `TypeError`, `RuntimeError`, and `ValueError`.

`build_optimizer()` currently builds the actual model on CPU and moves it later. A capability probe on a CUDA parameter can succeed while construction with CPU model parameters fails, causing an ordinary AdamW fallback.

## Fused AdamW configuration

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

## Performance implications

- BF16 generally avoids CUDA `GradScaler`.
- FP16 CUDA uses dynamic loss scaling.
- CPU auto precision stays FP32 for broad custom-module compatibility.
- MPS auto precision stays FP32.
- Microbatch loss-to-CPU conversion still synchronizes CUDA even when mixed precision reduces tensor compute cost.
- Metric memory queries can also synchronize the current accelerator.

## Test evidence

The repository directly tests CPU auto FP32 and explicit CPU FP16 fallback. CUDA, MPS, CPU BF16, fused AdamW, and scaler lifecycle are implementation-derived and not covered by repository tests.

See [Local Performance](../tutorials/local-performance) and [Runtime and Trainer](./runtime-and-trainer).
