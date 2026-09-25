---
title: Speedtronic v2
sidebar_label: v2 Overview
description: Overview of the v2 optimizer, scheduling, shape-efficiency, and asynchronous distributed additions.
---

# Speedtronic v2

Version 2.0.0 adds optimization and systems techniques that can be enabled
independently while preserving CPU, CUDA, and MPS fallbacks.

## What is new

| Area | Feature | Default |
|---|---|---:|
| Optimizer | Hybrid Muon + AdamW routing | AdamW |
| Optimizer | Muon+ post-orthogonal normalization | Off |
| Optimizer | Cautious sign-aligned update wrapper | Off |
| Systems | CUDA stream-backed out-of-order backprop (conservative) | Off |
| Systems | Startup shape-alignment warnings | On when applicable |
| Distributed | Asynchronous delta dispatch and global polling | On |

## Design rules

- Muon's Newton–Schulz iteration uses only PyTorch operations.
- CPU and MPS never require CUDA streams; they use the standard path.
- Shape findings are warnings, not startup failures.
- Distributed Hub work is dispatched away from the training thread by default.
- Features compose, but `torch.compile` and stream scheduling are mutually
  exclusive in this release.
- Deferred techniques are documented explicitly rather than exposed as dead
  configuration flags.

## Pages

- [v2 optimizers](/docs/v2/optimizers)
- [Out-of-order backprop scheduling](/docs/v2/scheduling)
- [Shape validation](/docs/v2/shape-validation)
- [0.1.0 to 2.0.0 migration](/docs/v2/migration)
- [Deferred and roadmap decisions](/docs/v2/roadmap)
