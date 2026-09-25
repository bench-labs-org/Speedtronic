<div align="center">

<img width="256" height="256" alt="Speedtronic Logo" src="https://github.com/user-attachments/assets/3389569b-430f-49f3-98b8-7c2d73dca8dc" />

# Speedtronic

*GPU-agnostic AI training, built for speed.*

</div>

[![PyPI version](https://img.shields.io/pypi/v/speedtronic.svg)](https://pypi.org/project/speedtronic/)

**Be Fast, Be Efficient.**

Speedtronic is an open-source PyTorch training framework. You bring your own `nn.Module`, dataset, and a single YAML config; it runs an efficient training loop on CPU, CUDA, or MPS, with optional asynchronous multi-node training (DumbDiLoCo) synced through a Hugging Face Hub repo instead of a dedicated parameter server.

## What you get

- Architecture-neutral training engine: works with any `nn.Module`, not just the bundled model.
- Reference GPT-style decoder-only transformer (RoPE, GQA, SwiGLU, tied embeddings) so the repo runs out of the box.
- Single config file per run: local and distributed runs share the same code path.
- Efficiency defaults: auto mixed precision, gradient accumulation, streaming data loading with prefetch, fused AdamW when available, opt-in `torch.compile` and gradient checkpointing.
- v2 optimizer stack: hybrid Muon/AdamW routing, Muon+, and cautious sign-aligned updates.
- v2 systems stack: conservative CUDA stream-backed out-of-order backprop, startup shape-alignment warnings, and non-blocking DumbDiLoCo dispatch.
- Local resumable checkpoints (model, optimizer, scheduler, RNG, step counters).
- DumbDiLoCo: DiLoCo-style local-steps-then-sync, with the Hub as transport. One node is master, the rest are workers. No extra infrastructure to host.
- Pluggable metrics: stdout/file/JSONL built in, W&B and TensorBoard via optional hooks.

## v2 additions

```yaml
optimizer: muon
muon_plus: true
cautious: true

shape_validation:
  enabled: true
  alignment: auto

ooo_backprop: false
```

Muon routes hidden 2-D matrices through a pure-PyTorch Newton–Schulz
orthogonalizer and leaves embeddings, heads, normalization weights, biases, and
other tensors on AdamW. `cautious` masks updates that do not align with the
current gradient. `ooo_backprop` uses dependency-aware CUDA streams and falls
back to standard sequential backward on CPU/MPS. Shape validation emits
startup warnings, never hard failures.

Muon may change which solution a model selects as well as how quickly it
trains; compare convergence, not just throughput. Ahead-of-time scheduling is
deferred as subsumed by `torch.compile`. See
[CHANGELOG.md](CHANGELOG.md) and the [v2 documentation](https://speedtronic-docs.pages.dev/docs/v2).

## Requirements

- Python >= 3.10
- PyTorch >= 2.1 (install the CPU or CUDA wheel appropriate for your machine first)
- `safetensors`, `PyYAML`, `numpy`, `huggingface-hub` (installed automatically)

## Install

```bash
pip install speedtronic
# optional logging integrations
pip install 'speedtronic[logging]'
```

For a source checkout:

```bash
pip install -e .
# dev tools (pytest, ruff)
pip install -e '.[dev]'
```

## Quickstart (60 seconds)

Train the bundled reference model on synthetic data (CPU-safe, no downloads):

```bash
speedtronic train --config configs/smoke.yaml
```

Exercise the v2 Muon/cautious/systems path on CPU:

```bash
speedtronic train --config configs/v2_smoke.yaml
```

Validate a config without training:

```bash
speedtronic validate --config configs/smoke.yaml
```

## PDF documentation

The full documentation is also available as a single print-ready PDF
(Letter, ~257 pages, with a linked table of contents, running headers, page
numbers, and rendered Mermaid diagrams):

```text
docs-site/static/pdf/speedtronic-documentation.pdf
```

It is rebuilt from the Docusaurus site with a headless browser, so the PDF can
never drift from the site content:

```bash
cd docs-site
npm ci
python -m pip install playwright pymupdf
python -m playwright install chromium
npm run build && npm run build:pdf
```

The PDF is also served from the docs site at `/pdf/speedtronic-documentation.pdf`.

## Agent skill

The full documentation is also packaged as an
[Agent Skill](https://agentskills.io) so coding agents can load it on demand:

```text
skills/speedtronic/
├── SKILL.md          # entry point, config/CLI/optimizer essentials
└── references/       # 18 topic-focused reference files
```

`SKILL.md` stays small (progressive disclosure) and routes to `references/`
for detail. Copy or symlink `skills/speedtronic` into your agent's skills
directory, e.g. `.claude/skills/speedtronic/`, then validate it with:

```bash
pip install skills-ref
agentskills validate skills/speedtronic
```

Same run from Python:

```python
from speedtronic import SpeedtronicConfig
from speedtronic.runtime import train_from_config

config = SpeedtronicConfig.from_dict({
    "run": {"name": "demo", "max_steps": 100, "device": "cpu"},
    "model": {
        "name": "reference_transformer",
        "vocab_size": 256,
        "max_seq_len": 64,
        "n_layer": 2,
        "n_head": 4,
        "n_kv_head": 2,
        "d_model": 64,
    },
    "data": {"micro_batch_size": 2, "target_batch_size": 8},
    "precision": {"mode": "fp32"},
})
result = train_from_config(config)
print(result.steps, result.final_loss)
```

CLI flags override the file without editing it:

```bash
speedtronic train --config configs/smoke.yaml --max-steps 100 --output-dir runs/demo --resume
```

## Project layout

```text
src/speedtronic/
  config.py        # SpeedtronicConfig: validation, YAML/JSON/dict loading
  trainer.py       # Trainer: accumulation, precision, compile, checkpoint hooks
  runtime.py       # build_runtime / train_from_config: wiring from config to Trainer
  model.py         # reference transformer (example model, not a framework dependency)
  registry.py      # model registry for custom architectures
  data.py          # synthetic + streaming text datasets, dataloader builder
  precision.py     # device and dtype capability detection
  optimizers.py    # v2 Muon, Muon+, cautious wrapper, hybrid routing
  shapes.py        # startup shape-alignment warnings
  scheduling.py    # v2 CUDA stream-backed backprop scheduling
  checkpoint.py    # atomic local checkpoints
  profiling.py     # MetricLogger, stdout/file/JSONL output, hook fan-out
  integrations.py  # optional W&B / TensorBoard hooks
  cli.py           # speedtronic train | validate
  distributed/     # DumbDiLoCo: hub.py, diloco.py, outer.py, tensors.py
configs/
  smoke.yaml       # minimal CPU run
  diloco.yaml      # local template with distributed section
examples/
  train_reference.py
  diloco_master.yaml
tests/             # CPU-only, Hub fakes (no network credentials needed)
```

## Configuration

One `SpeedtronicConfig` drives everything. Load from YAML, JSON, a JSON/YAML string, or a plain dict. Unknown keys are rejected so typos fail fast.

| Section | Controls |
|---|---|
| `run` | name, seed, device (`auto`/`cpu`/`cuda`/`mps`), `max_steps`, `output_dir` |
| `model` | registered model name + architecture params |
| `data` | `micro_batch_size` / `target_batch_size` (accumulation derived), text path or synthetic source, workers, prefetch, pin_memory, shuffle |
| `optimizer` | AdamW or hybrid Muon, Muon+, cautious updates, LR, betas, decay, clipping |
| `scheduler` | `cosine` (default) or `constant`, warmup steps, min LR ratio |
| `precision` | `auto`/`bf16`/`fp16`/`fp32`; auto picks bf16 on capable CUDA, fp32 elsewhere |
| `compile` | opt-in `torch.compile`; logs and continues uncompiled if unsupported |
| `gradient_checkpointing` | calls the model's `set_gradient_checkpointing()` hook if present |
| `checkpoint` | directory, interval, retention (`keep_last`), resume flag |
| `logging` | level, file, JSONL file, step cadence (hooks are programmatic) |
| `shape_validation` | startup shape alignment profile and warning toggles |
| `distributed` | DumbDiLoCo role, node id, Hub repo, async dispatch, inner/outer loop tuning |
| `ooo_backprop` / `ooo_streams` | opt-in CUDA stage scheduling |

Top-level aliases (`name`, `max_steps`, `output_dir`, `seed`, `device`) are accepted as shorthand for `run.*`.

## Bring your own model

The engine only requires a module that maps a batch to a loss. Return a tensor, a `{"loss": ...}` dict, or a `(loss, ...)` tuple. Dict batches are passed as kwargs; tuple batches are passed as `(inputs, labels)`.

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
        return {
            "logits": logits,
            "loss": F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            ),
        }

@register_model("my_model")
def make_model(**kwargs):
    return SmallCausalLM(kwargs["vocab_size"], kwargs["d_model"])
```

```yaml
model:
  name: my_model
  vocab_size: 256
  d_model: 256
```

Registration is process-local. A separate `speedtronic train` process needs an
application import step or a plugin loader to discover a custom factory.

For activation checkpointing, expose `set_gradient_checkpointing(enabled)` on your module; the trainer calls it and warns if it is absent.

## Bring your own data

- Default: deterministic synthetic token stream (good for smoke tests and benchmarking the loop).
- `data.text_path`: streams a UTF-8 file in fixed-size token blocks without loading it into RAM.
- `build_dataloader(..., dataset=..., tokenizer=...)`: pass any PyTorch `Dataset`/`IterableDataset` directly.
- `micro_batch_size` is what the loader emits; `target_batch_size` is the effective batch after accumulation. The trainer handles the math and keeps mixed precision correct across accumulation steps.

## Efficiency behavior

- **Precision:** `auto` uses bf16 where `torch.cuda.is_bf16_supported()`, fp16 with `GradScaler` on CUDA otherwise, fp32 on CPU/MPS. Explicit modes are honored unless the hardware cannot run them, in which case the run falls back to fp32 instead of crashing.
- **`torch.compile`:** never a hard failure. Unsupported combinations log and continue eagerly, including a runtime fallback if a compiled step throws.
- **Fused AdamW:** used only when construction with `fused=True` succeeds on CUDA.
- **Attention:** the reference model uses `scaled_dot_product_attention` so PyTorch picks flash / memory-efficient / math per device. No flash-attn dependency.
- **Data:** `num_workers`, `prefetch_factor`, and `pin_memory` are configurable; iterable datasets skip sampler shuffling (implement shuffling in the stream if needed).
- **Muon:** hidden 2-D matrices use pure-PyTorch Newton–Schulz; AdamW handles embeddings, heads, normalization, and biases.
- **Muon+:** opt-in post-orthogonalization normalization after Muon.
- **Cautious:** opt-in sign-aligned update masking for either base optimizer.
- **OOO backprop:** opt-in CUDA stream scheduling; CPU/MPS use standard backward.
- **Shape validation:** warns on unaligned batch/sequence/model dimensions with a nearby suggestion.

## Checkpoints and resume

Checkpoints store model weights, optimizer state, scheduler state, step/sample/token counters, RNG state, redacted config copy, and (for DumbDiLoCo) coordinator state. Files are written to temp + atomic rename; a `latest.json` pointer tracks the newest checkpoint.

```bash
speedtronic train --config configs/smoke.yaml --resume
```

`max_steps` is a global step target, so resuming with the same config continues rather than restarting. Saved Hub tokens are redacted (`<redacted>`) in stored configs; provide tokens via environment at runtime.

## DumbDiLoCo (distributed without a parameter server)

Each node trains locally for `inner_steps` optimizer steps, dispatches a pseudo-gradient (`weights_at_last_sync - current_weights`) to its own `nodes/<node_id>/` folder in a Hub repo, and periodically pulls the newest global weights. v2 dispatches Hub I/O in a bounded background lane, so a boundary does not wait for the upload. The master aggregates whatever deltas arrived since the last round (simple mean), applies Nesterov momentum SGD, and publishes `global/latest.safetensors` + `global/step_count.json`. Nodes never share a clock and never block on stragglers.

```yaml
distributed:
  enabled: true
  mode: dumb_diloco
  role: master       # worker on the other machines
  node_id: node-a    # unique per participant, path-safe
  repo_id: my-org/my-run
  # token: prefer HF_TOKEN env var over YAML
  inner_steps: 500
  poll_interval: 60
  outer_lr: 0.7
  outer_momentum: 0.9
```

Repo layout on the Hub:

```text
<run>/
  global/latest.safetensors
  global/step_count.json
  nodes/<node_id>/delta_<local_step>.safetensors
```

Operational notes:

- The master creates the repo (private by default) and attempts `collaborators` grants where the installed Hub SDK supports it; otherwise grant write access in the repo settings UI.
- `node_id` defaults to `HF_USERNAME` / `USER` / `local` if unset; set it explicitly per machine.
- Transient Hub failures retry with exponential backoff and never kill local training; a failed delta upload is retried as a cumulative delta at the next boundary.
- A corrupt delta is logged with its filename and skipped for that round.
- A restarted worker resumes from the latest global weights; a pending asynchronous upload can be restored from the local checkpoint.
- `distributed.reset_inner_optimizer` (default `true`) clears Adam/Muon state at a successful sync event.
- `async_delta_upload` and `async_global_poll` default to `true`; set both to `false` for the v1 synchronous behavior.
- A one-slot upload queue skips and logs rather than blocking when an upload is still in flight.

Security model: trusted collaborators only. Anyone with a valid write token for the repo can influence the global model. There is no cryptographic node identity, Byzantine fault tolerance, or staleness weighting. Do not point this at a public-write repo.

## Metrics

Every run logs `loss`, `lr`, `steps_per_sec`, `samples_per_sec`, `tokens_per_sec`, memory when available, shape warnings, OOO scheduling state, and — for DumbDiLoCo — outer step plus delta dispatch/upload/skip events. Output goes to stdout by default, optionally to a file and/or JSONL (`logging.file`, `logging.json_file`). `MetricLogger(hooks=[...])` accepts `(event, payload)` callables or objects with `on_event`; W&B and TensorBoard adapters live in `integrations.py` and remain optional.

## Development

```bash
pip install -e '.[dev]'
pytest -q
ruff check src tests examples
ruff format --check src tests examples
python -m compileall src
```

Tests are CPU-only. Hub tests use fakes; no network or credentials required.

## Scope and non-goals

- One reference model, not a model zoo.
- No tokenizer training tooling.
- No FSDP or pipeline parallelism.
- No open/adversarial participation mode.
- No hosted dashboard; CLI + config files only.
- Mid-inner-loop worker resume remains approximate; v2 restores the latest global and a bounded pending upload.
- AoT kernel scheduling is deferred as subsumed by `torch.compile`.
- Sophia, MoE layers, FP8 training, and multi-GPU pipeline/tensor parallelism remain deferred or out of scope.

## License

Apache-2.0. See `LICENSE`.
