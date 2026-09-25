---
id: shipped-examples
title: Shipped Examples
sidebar_label: Shipped Examples
description: Walkthrough of examples/train_reference.py and examples/diloco_master.yaml.
---

# Shipped examples

## `examples/train_reference.py`

A programmatic CPU FP32 run on synthetic data.

```python
from speedtronic.runtime import train_from_config

if __name__ == "__main__":
    result = train_from_config(
        {
            "run": {
                "name": "reference-example",
                "max_steps": 10,
                "device": "cpu",
                "output_dir": "runs/reference-example",
            },
            "model": {
                "name": "reference_transformer",
                "vocab_size": 128,
                "max_seq_len": 32,
                "n_layer": 2,
                "n_head": 4,
                "n_kv_head": 2,
                "d_model": 64,
                "d_ff": 128,
            },
            "data": {
                "block_size": 32,
                "num_tokens": 256,
                "micro_batch_size": 1,
                "target_batch_size": 2,
            },
            "precision": {"mode": "fp32"},
            "checkpoint": {
                "enabled": True,
                "directory": "checkpoints",
                "every_steps": 5,
            },
        }
    )
    print(result)
```

### What it configures

| Area | Value |
|---|---|
| Target | 10 global steps |
| Device | CPU |
| Model | 2-layer, 64-wide reference transformer |
| Data | 256 synthetic samples of block size 32 |
| Accumulation | 2 microbatches per update |
| Precision | FP32 |
| Checkpoint | Every 5 steps |
| Scheduler | Implicit default 1000-step cosine horizon |

:::warning Scheduler horizon

The run stops at 10 steps while the scheduler still has a 1000-step horizon. Add an explicit scheduler section to align the schedule.

:::

### Run it

From the repository root:

```bash
python examples/train_reference.py
```

Artifacts:

```text
runs/reference-example/checkpoints/
├── latest.json
├── step_000000000005.pt
└── step_000000000010.pt
```

The final dataclass output includes:

```text
TrainResult(
  steps=10,
  samples=20,
  tokens=640,
  final_loss=<value>,
  elapsed_s=<value>,
  metrics=[...10 records...],
)
```

Exact loss and elapsed time vary.

## `examples/diloco_master.yaml`

A Hub-backed master example using synthetic data.

:::caution External dependency

`repo_id` is a placeholder. Running this example contacts Hugging Face Hub and requires credentials. It is not an offline or automatically runnable test.

:::

### Run and model

```yaml
name: diloco-example
max_steps: 10000
output_dir: runs/diloco-master
```

The local master trains for 10,000 optimizer steps.

Model:

```yaml
model:
  name: reference_transformer
  vocab_size: 512
  max_seq_len: 128
  n_layer: 2
  n_head: 8
  n_kv_head: 2
  d_model: 256
  d_ff: 768
```

### Synthetic data

```yaml
data:
  block_size: 128
  num_tokens: 100000
  micro_batch_size: 1
  target_batch_size: 8
```

The field creates 100,000 synthetic samples, not a literal 100,000-token budget.

### Optimizer and scheduler

```yaml
optimizer:
  lr: 0.0003
scheduler:
  warmup_steps: 100
  max_steps: 10000
```

### Distributed section

```yaml
distributed:
  enabled: true
  mode: dumb_diloco
  role: master
  node_id: master-1
  collaborators: []
  repo_id: your-org/your-diloco-run
  token: null
  inner_steps: 500
  poll_interval: 60
  outer_lr: 0.7
  outer_momentum: 0.9
  state_dir: .speedtronic/diloco
```

Before running:

1. Replace `repo_id`.
2. Set `HF_TOKEN` in the environment.
3. Verify repository privacy.
4. Add only trusted collaborator usernames.
5. Keep `node_id: master-1` unique to this master.
6. Ensure all workers use identical model dimensions and compatible state keys.

Run:

```bash
export HF_TOKEN='...'
speedtronic train --config examples/diloco_master.yaml
```

Do not put the token in the YAML file.

### Checkpoints

```yaml
checkpoint:
  enabled: true
  directory: checkpoints
  every_steps: 500
  keep_last: 3
```

The master checkpoint can include its local trainer state plus a second full global state and momentum, making files large.

### Logging

```yaml
logging:
  level: INFO
  file: runs/diloco-master/train.log
  every_steps: 10
```

The master outer thread can also emit `outer_step` events independently of the training-step cadence.

## No shipped worker configuration

The repository contains a master YAML but no dedicated worker YAML. Copy the configuration, change:

```yaml
role: worker
node_id: worker-1
```

and use the same model, data contract, and repository.

## Example limitations

- Neither example is a test of real Hub failure recovery.
- The Python example inherits the scheduler-horizon mismatch.
- The master example is a placeholder and can create a real repository.
- Examples do not demonstrate data sharding across nodes.
- Examples do not demonstrate custom metric hooks.

Related: [DumbDiLoCo](../distributed/overview), [Shipped Configs](./shipped-configs), and [Security](../operations/security-and-limitations).
