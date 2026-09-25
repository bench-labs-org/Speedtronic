---
title: v2 Optimizers
sidebar_label: Muon & Cautious
description: Hybrid Muon routing, Newton–Schulz orthogonalization, Muon+, cautious masking, and configuration.
---

# v2 optimizers

## Configuration

AdamW remains the default. Use the mapping form for full control:

```yaml
optimizer:
  name: muon
  lr: 0.0003
  weight_decay: 0.1
  muon_plus: true
  cautious: true
```

The addendum's scalar form is also accepted:

```yaml
optimizer: muon
muon_plus: true
cautious: true
```

`muon_plus` is valid only with `name: muon`. `cautious` composes with either
base optimizer.

## Hybrid routing

Muon is intended for hidden 2-D matrix weights:

- attention Q/K/V/output projections;
- MLP gate/up/down projections;
- neutral 2-D custom matrix parameters.

These stay on AdamW:

- embeddings and positional embeddings;
- LM heads, classifiers, and output projections;
- RMSNorm/LayerNorm/BatchNorm parameters;
- biases, scalars, and non-2D tensors;
- frozen parameters.

The bundled reference transformer routes seven hidden matrices to Muon and
four unique parameters to AdamW when weights are tied.

A module can explicitly override the heuristic:

```python
module._speedtronic_optimizer_role = "muon"  # or "adamw"
```

The role must be valid for a floating 2-D parameter.

## Newton–Schulz

`newton_schulz(matrix, steps=5, eps=1e-7)` uses the quintic iteration with
coefficients `(3.4445, -4.7750, 2.0315)` in pure PyTorch. Tall matrices are
transposed internally, normalized, iterated, and transposed back. Zero
matrices remain finite.

There is no SVD, Triton kernel, custom CUDA extension, or hardware-specific
dependency.

## Muon update

For each routed matrix:

```text
B ← momentum · B + (1 - momentum) · gradient
Q ← lerp(gradient, B, momentum)       # Nesterov form
U ← NewtonSchulz(Q)
W ← (1 - learning_rate · weight_decay) · W
    - learning_rate · max(1, rows / columns)^0.5 · U
```

Momentum state is stored as FP32 (or higher) independently of parameter
precision. Sparse and complex gradients are rejected with a clear error.

## Muon+

Muon+ adds post-orthogonalization normalization. The default `row_col` mode
first normalizes rows, then columns, with an epsilon inside each square root.
It adds no persistent optimizer state.

```yaml
optimizer:
  name: muon
  muon_plus: true
```

Muon may change which solution a model selects, not only how quickly it trains.
Its convergence/simplicity-bias trade-off is an active research question, so
it is not a guaranteed free lunch.

## Cautious updates

With `cautious: true`, Speedtronic snapshots parameters, lets the base
optimizer compute its update, and keeps entries whose update direction aligns
with the current gradient:

```text
mask = (observed_update · gradient > 0)
parameter = old - masked_update
```

The wrapper adds no mask history to the checkpoint. Because the wrapper is
composable with arbitrary optimizers, the observed update includes the base
optimizer's decoupled weight-decay contribution; native Muon and AdamW state
remain intact.

## Scheduler and checkpoint behavior

The hybrid optimizer exposes both groups through one outer optimizer, so the
existing `LambdaLR` updates both learning rates. Its state dictionary contains
ordinary Adam moments and Muon momentum buffers. Clearing inner optimizer state
after a distributed sync clears both branches.

Cautious is a facade over the base optimizer and delegates its state,
zero-grad, and checkpoint operations.

## CPU/CUDA/MPS behavior

Muon and Muon+ are pure PyTorch and work on all three devices. Fused AdamW is
only attempted for CUDA and can silently fall back to ordinary AdamW. Cautious
wrapping disables the fused path so the observed update is available for
masking.

## Evidence

The v2 test suite covers routing, tied weights, zero/rectangular Newton–Schulz
matrices, Muon+ normalization, hybrid stepping, cautious wrapping, sparse
rejection, scheduler updates, and CPU runtime smoke training. CUDA fused and
stream paths remain hardware-dependent and are documented separately.
