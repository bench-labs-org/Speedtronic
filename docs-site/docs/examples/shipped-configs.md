---
id: shipped-configs
title: Shipped Configurations
sidebar_label: Shipped Configs
description: Line-by-line behavior of configs/smoke.yaml, configs/v2_smoke.yaml, and configs/diloco.yaml.
---

# Shipped configurations

## `configs/smoke.yaml`

A small, runnable, synthetic CPU-oriented run.

```yaml
name: smoke
seed: 1234
max_steps: 4
output_dir: runs/smoke
```

Top-level aliases populate `run`.

| Setting | Effect |
|---|---|
| `name: smoke` | Descriptive run name |
| `seed: 1234` | Seeds global RNGs and synthetic data |
| `max_steps: 4` | Four absolute optimizer steps |
| `output_dir: runs/smoke` | Checkpoints resolve to `runs/smoke/checkpoints` |

### Model

```yaml
model:
  name: reference_transformer
  vocab_size: 128
  max_seq_len: 32
  n_layer: 2
  n_head: 4
  n_kv_head: 2
  d_model: 64
  d_ff: 128
```

The model has:

- 2 decoder blocks;
- 4 query heads;
- 2 KV heads with a repeat factor of 2;
- hidden width 64 and head width 16;
- vocabulary 128;
- context length 32;
- tied embeddings by default;
- zero dropout by default;
- gradient checkpointing disabled.

The bundled model contains 66,880 parameters in this configuration.

### Data

```yaml
data:
  synthetic: true
  num_tokens: 256
  block_size: 32
  micro_batch_size: 1
  target_batch_size: 2
  num_workers: 0
```

This creates 256 synthetic samples, each with 32 input tokens and 32 labels. Each optimizer update consumes two microbatches.

### Optimizer and scheduler

```yaml
optimizer:
  lr: 0.0003
  weight_decay: 0.01
scheduler:
  warmup_steps: 1
  max_steps: 4
```

Other defaults remain:

```text
betas=(0.9, 0.95)
eps=1e-8
scheduler=cosine
min_lr_ratio=0.1
```

### Precision and compilation

```yaml
precision:
  mode: fp32
compile: false
```

This is the safest CPU configuration and disables compile overhead/fallback paths.

### Checkpointing

```yaml
checkpoint:
  enabled: true
  directory: checkpoints
  every_steps: 2
  keep_last: 2
```

With the output base, the effective directory is:

```text
runs/smoke/checkpoints
```

Saves occur at steps 2 and 4.

### Logging

```yaml
logging:
  level: INFO
  every_steps: 1
```

Events go to stdout every step.

### Run it

```bash
speedtronic train --config configs/smoke.yaml --device cpu
```

This configuration is the recommended local smoke test.

---

## `configs/v2_smoke.yaml`

A CPU-safe run that exercises the v2 optimizer, cautious wrapper, shape
profile, OOO fallback, and asynchronous distributed defaults without Hub
access.

```yaml
optimizer:
  name: muon
  muon_plus: true
  cautious: true
shape_validation:
  enabled: true
  alignment: auto
ooo_backprop: true
ooo_streams: 2
distributed:
  enabled: false
  async_delta_upload: true
  async_global_poll: true
```

On CPU, `ooo_backprop` intentionally follows the standard sequential path. The
Muon/AdamW hybrid, Muon+ normalization, cautious masking, and shape validation
still run.

## `configs/diloco.yaml`

A template for a larger local text run with distributed fields present but disabled.

:::caution Not runnable as shipped

It references `data/train.txt`, which is not included in the repository. Create or change that path before training. `speedtronic validate` performs schema validation and does not open the file.

:::

### Run

```yaml
name: local-demo
seed: 1234
max_steps: 1000
output_dir: runs/local-demo
```

This requests 1000 local optimizer steps.

### Model

```yaml
model:
  name: reference_transformer
  vocab_size: 512
  max_seq_len: 128
  n_layer: 4
  n_head: 8
  n_kv_head: 2
  d_model: 256
  d_ff: 768
```

Architecture:

- 4 layers;
- 8 query heads;
- 2 KV heads;
- hidden width 256;
- head width 32;
- context 128;
- FFN configured as 768;
- vocabulary 512.

### Text data

```yaml
data:
  text_path: data/train.txt
  block_size: 128
  micro_batch_size: 1
  target_batch_size: 8
  num_workers: 2
  prefetch_factor: 2
  pin_memory: true
```

This requests eight-way accumulation.

:::warning Worker I/O

`TextFileTokenDataset` assigns disjoint blocks to workers, but each worker still
reads the source file to discover them. Use a pre-sharded dataset for very
large files.

:::

### Optimization

```yaml
optimizer:
  lr: 0.0003
  weight_decay: 0.1
scheduler:
  warmup_steps: 100
  max_steps: 1000
```

This uses 100 warmup optimizer steps and a 1000-step cosine horizon.

### Performance features

```yaml
precision:
  mode: auto
compile: true
gradient_checkpointing: true
```

Auto precision chooses by device. Compilation and activation checkpointing are best-effort.

### Disabled distributed section

```yaml
distributed:
  enabled: false
  mode: dumb_diloco
  role: single
  node_id: local
  inner_steps: 500
  poll_interval: 60
  outer_lr: 0.7
  outer_momentum: 0.9
  repo_id: your-org/your-run
  state_dir: .speedtronic/diloco
```

No coordinator is created while `enabled` is false. The values are a template for later enablement and must be replaced with a real unique repository/node configuration.

### Checkpoints and logs

```yaml
checkpoint:
  enabled: true
  directory: checkpoints
  every_steps: 500
  keep_last: 3
logging:
  level: INFO
  file: runs/local-demo/train.log
  every_steps: 10
```

Effective checkpoint directory:

```text
runs/local-demo/checkpoints
```

The log path is not rebased under output; it is already explicitly rooted at `runs/local-demo/train.log` relative to the process working directory.

### Prepare and run

Create:

```text
data/train.txt
```

or change `text_path` to an absolute path, then run:

```bash
speedtronic train --config configs/diloco.yaml
```

## Configuration comparison

| Property | `smoke.yaml` | `v2_smoke.yaml` | `diloco.yaml` |
|---|---|---|---|
| Runnable as shipped | Yes | Yes | No, missing text path |
| Steps | 4 | 2 | 1000 |
| Data | Synthetic | Synthetic | Text stream |
| Layers | 2 | 1 | 4 |
| Hidden width | 64 | 32 | 256 |
| Accumulation | 2 | 2 | 8 |
| Workers | 0 | 0 | 2 |
| Precision | FP32 | FP32 | Auto |
| Optimizer | AdamW | Muon + cautious | AdamW template |
| Compile | No | No | Yes |
| OOO backprop | Off | Enabled, CPU fallback | Off |
| Gradient checkpointing | No | No | Yes |
| Distributed | Disabled | Disabled, async defaults | Disabled template |

Related: [Configuration](../reference/configuration), [Quickstart](../getting-started/quickstart), and [Shipped Examples](./shipped-examples).
