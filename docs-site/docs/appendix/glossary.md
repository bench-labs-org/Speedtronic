---
id: glossary
title: Glossary
sidebar_label: Glossary
description: Definitions of Speedtronic training, model, checkpoint, and DumbDiLoCo terminology.
---

# Glossary

## Absolute step

A global optimizer-step target rather than a count of additional updates. A trainer at step 900 targeting 1000 performs 100 updates.

## Accumulation

Multiple microbatches whose gradients are summed and applied in one optimizer update. Speedtronic divides each microbatch loss by the accumulation count.

## AdamW

The decoupled weight-decay optimizer used by the framework. Optional fused mode is attempted only on supported CUDA setups.

## AMP

Automatic mixed precision. Speedtronic uses autocast for BF16/FP16 and a CUDA `GradScaler` for FP16.

## Baseline

In DumbDiLoCo, the model state captured at the last successful delta upload or global installation. The next boundary delta is baseline minus current.

## BF16

A floating-point format with a wider exponent range and lower precision than FP32. Commonly native on supported CUDA hardware.

## Block size

The number of input positions in a causal model sample. The reference model rejects sequences longer than its configured block size.

## Causal LM loss

Next-token cross-entropy computed from logits and shifted labels, usually ignoring padding index `-100`.

## Checkpoint

A local pickle-backed state file containing model and training state, optionally including coordinator state.

## Compile fallback

Speedtronic's behavior of disabling `torch.compile` and rerunning a failed compiled forward eagerly.

## DataLoader position

The current epoch, sampler, batch, and worker iterator state. Speedtronic 2.0.0 does not checkpoint this state.

## Distributed

The optional DumbDiLoCo mode. It does not use NCCL or `torch.distributed` collectives.

## Dilation/outer step

See **Outer step**.

## DumbDiLoCo

Speedtronic's Hub-transport, DiLoCo-style protocol: local steps, pseudo-gradient upload, mean aggregation, Nesterov outer update, and global-weight polling.

## FP16

A half-precision floating-point format. On CUDA it uses `GradScaler`; CPU explicit FP16 falls back to FP32.

## FP32

Standard 32-bit floating-point execution. It is the portable default on CPU/MPS and the fallback for several unsupported mixed modes.

## GQA

Grouped-query attention. The reference model projects fewer K/V heads than query heads and repeats K/V groups to match query-head count.

## Global model

The complete model `state_dict` published by the master at `global/latest.safetensors`.

## Global step

In normal local training, the cumulative optimizer update count. In DumbDiLoCo Hub metadata, `outer_step` is the global model version; avoid using the same word for both in operational logs.

## Gradient accumulation

See **Accumulation**.

## Gradient checkpointing

Activation-memory optimization that recomputes selected forward work during backward. The model opts in through `set_gradient_checkpointing(enabled)`.

## Hub

The Hugging Face Hub. Speedtronic uses a model repository as mutable file transport for global state and deltas.

## Inner boundary

A local step divisible by `distributed.inner_steps`.

## Inner loop/step

The local optimization loop. In Speedtronic's coordinator, an inner step is one local optimizer update after gradient accumulation.

## LambdaLR

A PyTorch scheduler driven by a function of the optimizer-step count. Speedtronic builds warmup and cosine/constant factors.

## Local step

One completed local AdamW optimizer update and scheduler step before the coordinator callback.

## LR

Learning rate.

## Microbatch

One loader batch consumed before an optimizer update. The Trainer forward/backward path runs once per microbatch.

## MPS

Apple Metal Performance Shaders backend exposed by PyTorch. Speedtronic auto-selects it after CUDA and before CPU, but uses FP32 by default.

## Nesterov momentum SGD

The outer optimizer update:

```text
buffer = momentum * buffer + gradient
effective = gradient + momentum * buffer
state -= outer_lr * effective
```

## Node ID

The distributed participant's unique path-safe identity, used for `nodes/<node_id>/...` and local state directories.

## Outer loop

Master-side aggregation and global optimization over uploaded pseudo-gradients.

## Outer optimizer

`NesterovOuterOptimizer`, which updates the CPU global state from the mean valid delta.

## Outer round

One successful `MasterOuterLoop.sync_once()` aggregation and publication.

## Outer step

The monotonic global model version stored in `global/step_count.json`. The name `outer_step` distinguishes it from local optimizer steps.

## Parameter server

A centralized service hosting model/optimizer state. DumbDiLoCo does not use one; the Hub is file transport.

## Persistent workers

DataLoader worker processes kept alive across epochs. Enabled automatically when `num_workers > 0`.

## Pin memory

Page-locking CPU tensor storage to accelerate CUDA transfers. Configured explicitly or inferred for resolved CUDA devices.

## Prefetch factor

Number of batches each DataLoader worker prepares in advance. Defaults to 2 when workers are enabled.

## Pseudo-gradient

```text
baseline weights - current local weights
```

It approximates a direction for the outer global update and is not an autograd gradient.

## Registry

The process-global mapping from model name to factory. Built-ins are `reference_transformer` and `gpt`.

## Resume

Loading model, optimizer, scheduler, scaler, counters, RNG, and optional coordinator state from a local checkpoint. It is not exact DataLoader replay.

## RMSNorm

Root-mean-square layer normalization with a learned scale, implemented in the reference model.

## RoPE

Rotary positional embedding. Query and key vectors are rotated by position-dependent angles before attention.

## Safetensors

A tensor serialization format used for global models and deltas. It avoids pickle execution for tensor files but does not authenticate writers.

## Scheduler horizon

`scheduler.max_steps`, the step count used to shape the cosine schedule. It can differ from the actual run target if not configured explicitly.

## SDPA

PyTorch's `scaled_dot_product_attention`, used by the reference model for backend-selected attention kernels.

## Staleness

How old a worker's baseline/global version is. DumbDiLoCo records `base_outer_step` but does not use it to reject or weight stale deltas.

## Step accumulation

See **Accumulation**.

## Straggler

A participant that completes local work later than others. DumbDiLoCo does not wait for stragglers or enforce fixed rounds.

## SwiGLU

A gated MLP using `SiLU(gate(x)) * up(x)` followed by a down projection.

## Target batch

The nominal number of examples represented by one optimizer update, calculated as microbatch size times accumulation steps.

## Token rate

Tokens per second computed by the trainer from input tensor shape or attention-mask sum during the current invocation.

## Trainer

The architecture-neutral loop in `trainer.py` that moves batches, executes forward/backward, updates optimizer/scheduler, emits metrics, checkpoints, and calls the coordinator.

## Trusted writers

The DumbDiLoCo security assumption: every account with Hub write permission is honest and authorized to influence global weights.

## Worker

A DumbDiLoCo participant that trains locally, uploads deltas, and installs newer global weights but does not aggregate outer updates.

## Warmup

Initial scheduler steps where the LR rises linearly to its configured starting value.

Related: [Core Concepts](../getting-started/core-concepts), [Architecture](../reference/architecture), and [DumbDiLoCo](../distributed/overview).
