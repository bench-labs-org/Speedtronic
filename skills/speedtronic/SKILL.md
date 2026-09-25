---
name: speedtronic
description: >-
  Build, configure, run, debug, and optimize training jobs with Speedtronic, the
  GPU-agnostic PyTorch training framework. Use when working with Speedtronic
  configs or YAML, the `speedtronic train`/`validate` CLI, the Trainer and
  runtime helpers, hybrid Muon/Muon+/cautious optimizers, out-of-order
  backprop and shape validation, gradient accumulation and mixed precision,
  checkpoints and resume, DumbDiLoCo multi-node training over the Hugging Face
  Hub, custom models or datasets, or the reference transformer. Triggers on
  "speedtronic", trainer/runtime/config APIs, Muon optimizer routing, OOO
  backprop, DumbDiLoCo, or Speedtronic troubleshooting and packaging.
license: Apache-2.0
compatibility: Requires Python 3.10+ and PyTorch 2.1+. Runs on CPU, CUDA, or MPS.
metadata:
  author: BenchLabs
  version: "2.0.0"
  upstream: https://github.com/bench-labs-org/Speedtronic
  docs: https://speedtronic-docs.pages.dev/
---

# Speedtronic

Speedtronic is a configuration-driven, architecture-neutral PyTorch training
framework. You supply an `nn.Module` and a config; it supplies the loop,
precision handling, accumulation, checkpointing, metrics, and optional
Hub-backed DumbDiLoCo coordination.

Current version: **2.0.0** (`pip install speedtronic`).

## When to use this skill

Use it for anything involving the Speedtronic package: writing or debugging a
config, using the CLI, wiring the `Trainer`/`runtime`, choosing an optimizer,
enabling v2 systems features, checkpoint/resume, DumbDiLoCo, or extending the
model registry and data pipeline.

## Core mental model

A run is assembled from independent pieces, all driven by one validated config:

```text
SpeedtronicConfig
  ├── model      registered nn.Module factory
  ├── data       dataset + loader (micro_batch_size → accumulation)
  ├── optimizer  adamw (default) | muon (hybrid Muon + AdamW)
  ├── scheduler  cosine | constant
  ├── precision  auto | fp32 | bf16 | fp16
  ├── checkpoint  atomic save, retention, resume
  ├── logging     stdout/file/JSONL + programmatic hooks
  ├── shape_validation  startup efficiency warnings
  ├── ooo_backprop     opt-in CUDA stream scheduling
  └── distributed      DumbDiLoCo master/worker coordination
```

`build_runtime(config)` wires all of it and returns a `Trainer`; `train_from_config`
runs it. Unknown config keys are **rejected** rather than ignored.

## Quick start

```bash
pip install speedtronic

# validate config (parses only; does not build the model or read data)
speedtronic validate --config configs/smoke.yaml

# run the shipped CPU smoke config
speedtronic train --config configs/smoke.yaml --device cpu

# exercise the v2 optimizer/systems path
speedtronic train --config configs/v2_smoke.yaml --device cpu
```

Python equivalent:

```python
from speedtronic.runtime import train_from_config

result = train_from_config({
    "run": {"max_steps": 4, "device": "cpu"},
    "model": {"name": "reference_transformer", "vocab_size": 128,
              "max_seq_len": 32, "n_layer": 2, "n_head": 4,
              "n_kv_head": 2, "d_model": 64, "d_ff": 128},
    "data": {"micro_batch_size": 1, "target_batch_size": 2, "num_tokens": 64},
    "precision": {"mode": "fp32"},
})
print(result.steps, result.final_loss)
```

## Minimal working config

```yaml
run:      { max_steps: 100, device: auto, output_dir: runs/demo }
model:    { name: reference_transformer, vocab_size: 256, max_seq_len: 64 }
data:     { micro_batch_size: 2, target_batch_size: 8 }
optimizer: { name: adamw, lr: 0.0003 }
precision: { mode: auto }
```

## Key behaviors that are easy to get wrong

These are the highest-value facts. Verify against the reference files before
asserting anything else.

- **Accumulation.** `micro_batch_size` is what the loader emits;
  `target_batch_size` is the effective batch. `target_batch_size` must be
  divisible by `micro_batch_size`; the trainer derives
  `accumulation_steps = target // micro` and backpropagates `loss / K`.
- **`max_steps` is an absolute target, not a count of extra work.** Resuming a
  run already at the target performs no updates.
- **Scheduler horizon.** `scheduler.max_steps` defaults to 1000, but when a
  shorter `run.max_steps` is set and the scheduler is left at that default,
  Speedtronic adopts the run target. Set it explicitly to diverge.
- **Causal labels are already shifted.** Speedtronic datasets emit one
  next-token label per input position. Do **not** shift again in a custom model
  or trainer path, or the first target is skipped.
- **Muon is hybrid, not global.** Only hidden 2-D matrices go to Muon.
  Embeddings, LM heads/classifiers, norm weights, and biases stay on AdamW.
  Switching `optimizer.name: muon` does not move embeddings to Muon.
- **Muon+ requires Muon.** `muon_plus: true` with `name: adamw` is a config
  error.
- **OOO backprop is opt-in and CUDA-only in its active path.** It is a no-op on
  CPU/MPS, disabled when `compile: true`, and currently correctness-first
  (conservative stream synchronization), so expect no guaranteed speedup.
- **AoT scheduling is deliberately not implemented.** It is deferred as
  subsumed by `torch.compile`; there is no `aot_scheduling` flag.
- **DumbDiLoCo is asynchronous by default.** One upload may be in flight; a
  boundary that finds the slot busy skips and logs rather than blocking. Set
  `async_delta_upload: false` and `async_global_poll: false` for the
  synchronous path.
- **Shape validation warns, it never fails.** It never mutates config or model.
- **Secrets.** `speedtronic validate` redacts `distributed.token`; checkpoints
  and `config.save()` redact it too. Prefer `HF_TOKEN` over YAML tokens.
- **Resume is approximate.** DataLoader position, sampler state, and worker RNG
  are not checkpointed.

## Optimizer selection (v2)

```yaml
optimizer: muon        # scalar shorthand
muon_plus: true        # root-level alias also accepted
cautious: true
```

Full mapping form:

```yaml
optimizer:
  name: muon            # adamw | muon
  lr: 0.0003
  muon_plus: false
  cautious: false
  muon_momentum: 0.95
  muon_ns_steps: 5
  muon_norm_eps: 1.0e-8
```

Muon uses a pure-PyTorch Newton–Schulz iteration — no custom CUDA kernel, no
SVD, no hardware-locked dependency. A module can override routing with
`_speedtronic_optimizer_role = "muon" | "adamw"`.

Muon may change which solution a model selects, not only how fast it trains.
Benchmark convergence, not just throughput.

## Custom models and data

A model maps a batch to a loss: return a tensor, `{"loss": ...}`, or a tuple
whose first element is the loss. Expose `set_gradient_checkpointing(enabled)`
to support `gradient_checkpointing: true`.

```python
import torch
from torch import nn
from torch.nn import functional as F
from speedtronic import register_model

class SmallCausalLM(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, labels=None, attention_mask=None):
        logits = self.output(self.embedding(input_ids))
        if labels is None:
            return {"logits": logits}
        return {"logits": logits, "loss": F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
            ignore_index=-100)}

@register_model("my_model")
def make_model(**kwargs):
    return SmallCausalLM(kwargs["vocab_size"], kwargs["d_model"])
```

Registration is process-local; a separate `speedtronic train` process needs an
application import step to discover a custom factory.

For data, pass any `Dataset`/`IterableDataset` via `build_dataloader(...,
dataset=..., tokenizer=...)`.

## Reference map

Load only what the task needs — these are large files.

| Task | Read |
|---|---|
| Install, first run, core concepts | `references/getting-started.md` |
| Every config field and validation rule | `references/configuration.md` |
| Trainer/runtime internals, architecture | `references/architecture.md` |
| Muon, Muon+, cautious, deferred roadmap | `references/optimizers.md` |
| OOO backprop, shape validation | `references/scheduling-and-shapes.md` |
| v2 overview and 0.1.0 → 2.0.0 migration | `references/v2-overview-and-migration.md` |
| Datasets, collators, DataLoader | `references/data.md` |
| Reference transformer, registry, extensions | `references/models-and-extensions.md` |
| Precision, fused AdamW, perf tuning | `references/precision-and-performance.md` |
| Checkpoints, resume semantics | `references/checkpointing.md` |
| DumbDiLoCo, Hub transport, outer loop | `references/distributed.md` |
| Metrics, hooks, W&B/TensorBoard | `references/observability.md` |
| CLI flags and exit codes | `references/cli.md` |
| Packaging, testing, troubleshooting, security | `references/operations.md` |
| Shipped configs and examples | `references/examples-and-configs.md` |
| Repo/module map and evidence labels | `references/project-structure.md` |
| Full AST signature inventory | `references/api-inventory.md` |
| Glossary and test-to-source map | `references/glossary-and-tests.md` |

## Debugging order

1. `speedtronic validate --config <file>` — catches config errors first.
2. Confirm the run reaches `train_start` and read its `shape_warning_count`
   and `ooo_backprop` events.
3. Check `train_step` metrics: loss, `lr`, throughput, microbatch count.
4. For distributed, read the `delta_upload_*`, `outer_step`, and
   `shape_warning` events rather than assuming transport succeeded.
5. Consult `references/operations.md` for known limitations and
   `references/troubleshooting`-style symptom tables inside it.

## Non-goals

No FSDP, pipeline/tensor parallelism, model zoo, tokenizer training, hosted
dashboard, or public/adversarial distributed participation. FP8, MoE layers,
and Sophia are deferred. Treat 2.0.0 as alpha.
