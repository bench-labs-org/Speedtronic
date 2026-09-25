---
id: quickstart
title: Local Quickstart
sidebar_label: Quickstart
description: Install Speedtronic and run the bundled four-step CPU smoke configuration.
---

# Local quickstart

This path runs entirely from a source checkout. It uses synthetic data, does not download a model or dataset, and does not require Hugging Face credentials.

## Prerequisites

- Python 3.10 or newer
- PyTorch 2.1 or newer
- A writable checkout
- For CPU-only installation, install an appropriate PyTorch CPU wheel first when needed

## Install from source

```bash
cd /path/to/speedtronic
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The package installs the `speedtronic` console command. The equivalent module entry point is:

```bash
python -m speedtronic --help
```

## Validate the configuration

```bash
speedtronic validate --config configs/smoke.yaml
```

Validation parses and normalizes configuration. It does **not** instantiate the model, open the selected data, resolve hardware, or run a forward pass.

The effective smoke configuration has these important properties:

| Property | Value |
|---|---|
| Target | 4 global optimizer steps |
| Model | 2-layer reference transformer, 66,880 parameters |
| Data | Deterministic synthetic token blocks |
| Microbatch | 1 sample |
| Target batch | 2 samples |
| Accumulation | 2 microbatches per optimizer update |
| Precision | FP32 |
| Device | `auto` in YAML; pass `--device cpu` to force it |
| Checkpoint interval | Every 2 steps |
| Retention | Last 2 checkpoints |

## Run four steps

```bash
speedtronic train --config configs/smoke.yaml --device cpu
```

A successful run ends with output similar to:

```text
completed steps=4 samples=8 tokens=256 loss=<value>
```

Loss is stochastic for synthetic input and initialization state, so the exact value is not contractual.

Artifacts are written beneath:

```text
runs/smoke/checkpoints/
├── latest.json
├── step_000000000002.pt
└── step_000000000004.pt
```

The pointer is:

```json
{"step": 4, "file": "step_000000000004.pt"}
```

## Resume semantics

`max_steps` is an absolute global target, not a count of additional work:

```bash
speedtronic train --config configs/smoke.yaml --device cpu --resume
```

If the newest checkpoint is at step 4, this invocation performs no optimizer updates. To demonstrate resume, use a temporary output directory, run two steps, then resume to four:

```bash
speedtronic train \
  --config configs/smoke.yaml \
  --device cpu \
  --output-dir runs/resume-demo \
  --max-steps 2

speedtronic train \
  --config configs/smoke.yaml \
  --device cpu \
  --output-dir runs/resume-demo \
  --max-steps 4 \
  --resume
```

The second command reports cumulative `steps=4`, even though it performed two updates.

## Python equivalent

```python
from speedtronic.runtime import train_from_config

result = train_from_config(
    {
        "run": {
            "name": "demo",
            "max_steps": 4,
            "device": "cpu",
            "output_dir": "runs/python-demo",
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
            "micro_batch_size": 1,
            "target_batch_size": 2,
            "num_tokens": 32,
        },
        "precision": {"mode": "fp32"},
        "scheduler": {"warmup_steps": 1, "max_steps": 4},
    }
)

print(result.steps, result.samples, result.tokens, result.final_loss)
```

:::tip Configure the scheduler horizon explicitly

`SchedulerConfig.max_steps` defaults to 1000. When a shorter `run.max_steps` is
set and the scheduler is left at that default, Speedtronic 2.0 automatically
adopts the run target so the schedule ends with training. Set
`scheduler.max_steps` explicitly when you want a horizon that differs from the
run target.

:::

## What the four steps execute

For each global step:

1. Fetch two synthetic microbatches.
2. Run each through the reference model in FP32.
3. Compute a causal-language-model loss.
4. Backpropagate each loss divided by two.
5. Clip only if `optimizer.grad_clip` is configured.
6. Perform one AdamW update.
7. Advance the cosine scheduler.
8. Emit `train_step` metrics.
9. Save a checkpoint at steps 2 and 4.

Continue with [core concepts](./core-concepts) or the [architecture reference](../reference/architecture).
