---
title: Migrating to 2.0.0
sidebar_label: Migration
description: Changes from Speedtronic 0.1.0 and compatibility guidance for v2 configurations.
---

# Migrating to 2.0.0

## Version and package identity

```text
speedtronic==2.0.0
```

The PyPI distribution and import package remain exactly `speedtronic`.

## Optimizer migration

AdamW remains the default. No v1 configuration needs to change.

```yaml
optimizer: muon
muon_plus: true
cautious: true
```

For explicit fields:

```yaml
optimizer:
  name: muon
  muon_plus: true
  cautious: true
  muon_momentum: 0.95
  muon_ns_steps: 5
  muon_norm_eps: 1.0e-8
```

Muon receives hidden 2-D matrices; embeddings, heads, normalization weights,
and biases remain on AdamW. Checkpoint state therefore contains both AdamW
moments and Muon momentum when Muon is enabled.

## Causal labels

Speedtronic 2.0 consumes the already-shifted labels emitted by its datasets.
The reference model no longer shifts same-length labels a second time. Custom
models should return one next-token label per input position, or use the
model's own loss mapping.

## Scheduler target

A run target shorter than the historical 1000-step scheduler default now
synchronizes the default schedule to `run.max_steps`. Set
`scheduler.max_steps` explicitly when a longer schedule is intentional.

## Distributed defaults

DumbDiLoCo now defaults to:

```yaml
distributed:
  async_delta_upload: true
  async_global_poll: true
  delta_upload_queue_size: 1
  delta_upload_overflow: skip
  delta_upload_shutdown_timeout: 5.0
```

Set both async flags to `false` for the v1 synchronous behavior during
debugging or deterministic transport tests.

## Shape and OOO settings

```yaml
shape_validation:
  enabled: true

ooo_backprop: false
ooo_streams: 4
```

OOO backprop is opt-in and unavailable as an active path on CPU/MPS; those
devices use standard sequential backward.

## Checkpoint compatibility

Old checkpoints remain readable where their model and optimizer state shapes
match. Muon/Muon+/cautious state is not interchangeable with an AdamW-only
checkpoint. Resume with a changed optimizer algorithm should start a new run or
use a deliberately migrated checkpoint.

The bundled trainer checkpoint now redacts configuration secrets and includes
DumbDiLoCo pending-upload state when a job is in flight.

## Deferred items

Sophia, mixture-of-experts layers, FP8 training, and multi-GPU pipeline or
tensor parallelism are not part of 2.0.0. AoT scheduling is documented as
subsumed/deferred by `torch.compile` rather than exposed as a duplicate flag.
