---
title: Shape Validation
sidebar_label: Shape Validation
description: Startup warnings for batch, sequence, token, and model GEMM alignment.
---

# Shape validation

Shape validation catches awkward tensor-core/GEMM dimensions before they
silently reduce throughput.

```yaml
shape_validation:
  enabled: true
  alignment: auto
  check_batch: true
  check_sequence: true
  check_model: true
  check_vocab: false
  warn_on_cpu: false
```

## Automatic profiles

| Device and resolved precision | Automatic alignment |
|---|---:|
| CUDA FP16/BF16 | 8 |
| CUDA FP32/TF32 | None by default; set an explicit alignment for a benchmark profile |
| CPU | None by default |
| MPS | None by default |

An explicit positive `alignment` overrides the profile. On CPU or MPS, set
`warn_on_cpu: true` to enable that explicit benchmark profile. `"none"`
disables warnings.

## Inspected values

- `data.micro_batch_size`;
- `data.block_size`;
- `micro_batch_size * block_size`;
- `nn.Linear` input/output features;
- embedding dimensions;
- 2-D convolution channels;
- optionally vocabulary/output width.

Warnings are deduplicated and include a nearby upward-aligned suggestion when
that suggestion remains valid for the configured target batch. They never
change configuration, memory use, or model architecture. Set
`alignment: none` to disable them.

## Example

```text
shape warning: data.micro_batch_size=3 is not a multiple of 8; consider 8 (kernel batches near this alignment are usually more efficient)
```

If the nearby micro-batch suggestion would violate the target-batch
divisibility rule, `suggested` is emitted as `null` instead.

The warning is a hint, not an error. A user may intentionally choose an
unaligned shape for memory, debugging, or data semantics.

## Custom models

A model may expose:

```python
def speedtronic_shape_metadata(self) -> Mapping[str, int]:
    return {"block1.mlp_hidden": 300}
```

The validator falls back to standard PyTorch module introspection when the
hook is absent or fails. Shape metadata is observational and has no effect on
training.

## Runtime integration

`build_runtime()` resolves precision first, runs validation, then constructs
the optimizer. Direct `Trainer` users receive the same validation on the first
`fit()` call. The `train_start` event includes alignment and warning count.
