---
title: Out-of-Order Backprop
sidebar_label: OOO Backprop
description: Opt-in CUDA stream scheduling for autograd's dependency-aware backward engine, with portable fallback.
---

# Out-of-order backprop

```yaml
ooo_backprop: true
ooo_streams: 4
```

The feature is off by default and opt-in because its benefit is model and
hardware dependent.

## What PyTorch already does

PyTorch's autograd engine already computes ready nodes from the autograd DAG
rather than forcing a strict Python reverse-layer loop. Its remaining stream
behavior follows the stream that created each forward node. Speedtronic's v2
scheduler supplies streams to disjoint module stages, records producer
dependencies, and synchronizes each stage back to the ambient stream before an
untracked parent operation runs. This preserves correctness while keeping the
scheduling boundary explicit. It is a conservative first implementation and
does not promise kernel overlap on every model.

## Stage selection

1. A model may expose `ooo_stage_names`, a sequence of dotted module names.
2. Otherwise the scheduler expands container modules into disjoint leaf
   stages.
3. Each stage is assigned round-robin to at most `ooo_streams` streams.

For every stage:

- inputs wait on the recorded producer stream;
- tensors crossing streams use `record_stream` for allocator safety;
- stage outputs are associated with their producer stream;
- the ambient stream waits for each stage before untracked parent/container
  operations continue, avoiding a read of an unfinished stage;
- the current stream waits for all stage streams before host-visible loss use.

## Failure and fallback

| Condition | Behavior |
|---|---|
| CPU or MPS | Standard sequential backward; event reports disabled |
| CUDA unavailable | Standard sequential backward |
| Fewer than two stages | Standard sequential backward |
| Setup failure | Hooks removed, warning/event emitted, training continues |
| `compile: true` | OOO disabled with a warning; compiled path wins |
| CUDA graph capture | Not supported by this scheduler |

## Honest performance expectations

Published out-of-order backprop work reports roughly 1.03–1.58× single-GPU
throughput depending on the model, but those results use a lower-level
implementation. Speedtronic's first pure-PyTorch path prioritizes dependency
correctness and conservative synchronization; it is intended to provide a
portable scheduling seam, not a fixed multiplier. A compute-saturated model may
see no speedup, and a small model may see only kernel-overhead changes.

Speedtronic does not claim a fixed multiplier. Benchmark warm-up separately
from steady state and compare against the same model, shapes, precision, and
seed.

## Reference-model behavior

The bundled transformer is mostly a sequential chain at block granularity.
Its value is still explicit stream placement and reduced ambiguity about
producer dependencies, but it cannot expose a large reorderable DAG. Custom
models with independent towers or branches have more scheduling freedom.

## Correctness

The feature changes stream placement, not gradient mathematics. Autograd's
`.grad` accumulation, gradient scaling, clipping, accumulation, and coordinator
boundaries remain in the trainer. CPU parity is the default test path; CUDA
parity and race testing should be run on the target accelerator before enabling
the flag in production.

## Interaction with compilation

`ooo_backprop` and `torch.compile` are mutually exclusive in v2. Dynamo does
not expose a stable API for tracing arbitrary `torch.cuda.stream` contexts
inside a compiled region. Choose one scheduling mode and benchmark it.

## AoT scheduling decision

The addendum's ahead-of-time kernel scheduling item is **deferred as
subsumed** for this release. Inductor already performs graph capture, fusion,
topological scheduling, and autotuning inside `torch.compile`. Speedtronic does
not expose a duplicate `aot_scheduling` flag that would depend on private
PyTorch APIs or claim a hardware-specific speedup.
