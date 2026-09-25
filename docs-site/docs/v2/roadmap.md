---
title: Deferred Techniques
sidebar_label: Deferred & Roadmap
description: Explicit disposition of Sophia, MoE, FP8, AoT scheduling, and parallelism items from the v2 addendum.
---

# Deferred techniques and roadmap

The v2 addendum deliberately separates implemented work from ideas that
require more evidence. Speedtronic records those decisions so they do not
become undocumented dead flags.

## Implemented in 2.0.0

- Hybrid Muon and AdamW routing.
- Muon+ post-orthogonalization normalization.
- Cautious update masking.
- CUDA stream-backed dependency-aware backprop scheduling.
- Startup shape-alignment warnings.
- Non-blocking DumbDiLoCo delta dispatch and global polling.

## AoT kernel scheduling: deferred/subsumed

`torch.compile` and Inductor already perform graph capture, fusion,
topological scheduling, and autotuning. A second public AoT flag would
duplicate that work and would require private PyTorch APIs or CUDA-specific
dependencies. Speedtronic records the item as **deferred/subsumed** rather than
shipping a flag that could not be portable or honestly benchmarked.

## Sophia: deferred

Sophia's second-order Hessian estimate can help some model scales, but its
complexity and adoption consistency do not yet justify adding a second
optimizer family to the v2 release. It remains a candidate for a future
version after Muon integration is validated.

## MoE layers: deferred

MoE is an architecture decision for a reference-model variant, not a training
engine technique. It would affect parameter routing, memory, and checkpoint
compatibility.

## FP8: deferred and hardware-detected if revisited

FP8 requires newer accelerator classes and must not become a default or a
hardware-locked dependency. Any future support would be capability-detected
and opt-in, following the existing BF16 detection pattern.

## Multi-GPU parallelism: unchanged non-goal

Pipeline and tensor parallelism remain out of scope, as do FSDP-style
partitioning and a hosted dashboard.

## Muon convergence caveat

Muon's reported speed advantage may come with a different simplicity bias or
selected solution. Users should compare quality and not only steps per second.
