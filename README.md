# Speedtronic

<img width="512" height="512" alt="Speedtronic Logo" src="https://github.com/user-attachments/assets/3389569b-430f-49f3-98b8-7c2d73dca8dc" />

**Be Fast, Be Efficient.**

Speedtronic is an open-source PyTorch training framework. You bring your own `nn.Module`, dataset, and a single YAML config; it runs an efficient training loop on CPU, CUDA, or MPS, with optional asynchronous multi-node training (DumbDiLoCo) synced through a Hugging Face Hub repo instead of a dedicated parameter server.

## What you get

- Architecture-neutral training engine: works with any `nn.Module`, not just the bundled model.
- Reference GPT-style decoder-only transformer (RoPE, GQA, SwiGLU, tied embeddings) so the repo runs out of the box.
- Single config file per run: local and distributed runs share the same code path.
- Efficiency defaults: auto mixed precision, gradient accumulation, streaming data loading with prefetch, fused AdamW when available, opt-in `torch.compile` and gradient checkpointing.
- Local resumable checkpoints (model, optimizer, scheduler, RNG, step counters).
- DumbDiLoCo: DiLoCo-style local-steps-then-sync, with the Hub as transport. One node is master, the rest are workers. No extra infrastructure to host.
- Pluggable metrics: stdout/file/JSONL built in, W&B and TensorBoard via optional hooks.

## Requirements

- Python >= 3.10
- PyTorch >= 2.1 (install the CPU or CUDA wheel appropriate for your machine first)
- `safetensors`, `PyYAML`, `numpy`, `huggingface-hub` (installed automatically)

## Install

```bash
pip install -e .
# dev tools (pytest, ruff)
pip install -e '.[dev]'
# optional logging integrations
pip install -e '.[logging]'
```

## Quickstart (60 seconds)

Train the bundled reference model on synthetic data (CPU-safe, no downloads):

```bash
speedtronic train --config configs/smoke.yaml
```

Validate a config without training:

```bash
speedtronic validate --config configs/smoke.yaml
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
| `optimizer` | AdamW LR, betas, eps, weight decay, optional fused, grad clip |
| `scheduler` | `cosine` (default) or `constant`, warmup steps, min LR ratio |
| `precision` | `auto`/`bf16`/`fp16`/`fp32`; auto picks bf16 on capable CUDA, fp32 elsewhere |
| `compile` | opt-in `torch.compile`; logs and continues uncompiled if unsupported |
| `gradient_checkpointing` | calls the model's `set_gradient_checkpointing()` hook if present |
| `checkpoint` | directory, interval, retention (`keep_last`), resume flag |
| `logging` | level, file, JSONL file, step cadence, hook list |
| `distributed` | DumbDiLoCo role, node id, Hub repo, inner/outer loop tuning (see below) |

Top-level aliases (`name`, `max_steps`, `output_dir`, `seed`, `device`) are accepted as shorthand for `run.*`.

## Bring your own model

The engine only requires a module that maps a batch to a loss. Return a tensor, a `{"loss": ...}` dict, or a `(loss, ...)` tuple. Dict batches are passed as kwargs; tuple batches are passed as `(inputs, labels)`.

```python
import torch
from speedtronic import register_model

@register_model("my_model")
def make_model(**kwargs):
    return torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(d_model=kwargs["d_model"], nhead=kwargs.get("n_head", 4)),
        num_layers=kwargs.get("n_layer", 2),
    )
```

```yaml
model:
  name: my_model
  d_model: 256
```

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

## Checkpoints and resume

Checkpoints store model weights, optimizer state, scheduler state, step/sample/token counters, RNG state, redacted config copy, and (for DumbDiLoCo) coordinator state. Files are written to temp + atomic rename; a `latest.json` pointer tracks the newest checkpoint.

```bash
speedtronic train --config configs/smoke.yaml --resume
```

`max_steps` is a global step target, so resuming with the same config continues rather than restarting. Saved Hub tokens are redacted (`<redacted>`) in stored configs; provide tokens via environment at runtime.

## DumbDiLoCo (distributed without a parameter server)

Each node trains locally for `inner_steps` AdamW steps, uploads a pseudo-gradient (`weights_at_last_sync - current_weights`) to its own `nodes/<node_id>/` folder in a Hub repo, and periodically pulls the newest global weights. The master aggregates whatever deltas arrived since the last round (simple mean), applies Nesterov momentum SGD, and publishes `global/latest.safetensors` + `global/step_count.json`. Nodes never share a clock and never block on stragglers.

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
- A restarted worker resumes from the latest global weights (at most one partial inner loop is lost). A restarted master resumes global weights, outer step, processed-delta set, and Nesterov momentum from its local `state_dir`.
- `distributed.reset_inner_optimizer` (default `true`) clears Adam state at each inner-loop boundary so stale moments do not leak across sync rounds.

Security model: trusted collaborators only. Anyone with a valid write token for the repo can influence the global model. There is no cryptographic node identity, Byzantine fault tolerance, or staleness weighting. Do not point this at a public-write repo.

## Metrics

Every run logs `loss`, `lr`, `steps/sec`, `samples/sec`, `tokens/sec`, memory (when the backend exposes it), and — for DumbDiLoCo — outer step plus deltas found/included per round. Output goes to stdout by default, optionally to a file and/or JSONL (`logging.file`, `logging.json_file`). `MetricLogger(hooks=[...])` accepts `(event, payload)` callables or objects with `on_event`; W&B and TensorBoard adapters live in `integrations.py` and stay optional dependencies.

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
- Mid-inner-loop worker resume is out of scope (restart from latest global).

## License

Apache-2.0. See `LICENSE`.
