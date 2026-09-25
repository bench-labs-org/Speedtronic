---
id: configuration
title: Configuration Reference
sidebar_label: Configuration
description: Every Speedtronic configuration class and field, including defaults, validation, normalization, path semantics, aliases, and precedence.
---

# Configuration reference

Speedtronic configuration is implemented by mutable dataclasses in `speedtronic.config`. Each section validates selected invariants in `__post_init__`; the aggregate applies cross-section defaults.

## Accepted forms

```python
from speedtronic import SpeedtronicConfig, load_config

config = SpeedtronicConfig.from_dict({...})
config = SpeedtronicConfig.from_yaml("run.yaml")
config = SpeedtronicConfig.load("run.json")
config = SpeedtronicConfig.load("name: demo\nmax_steps: 10")
config = load_config({"max_steps": 10})
```

`load()` heuristically distinguishes inline YAML/JSON from paths. Unknown root and nested mapping keys raise `ConfigError`.

## Root sections

```yaml
run: {}
model: {}
data: {}
optimizer: {}
scheduler: {}
precision: {}
checkpoint: {}
logging: {}
distributed: {}
gradient_checkpointing: false
compile: false
```

## `RunConfig`

| Field | Type | Default | Meaning and validation |
|---|---|---:|---|
| `name` | `str` | `speedtronic-run` | Descriptive run name; not used as a filesystem path |
| `seed` | `int` | `1234` | Seeds Python, NumPy when available, PyTorch CPU, and CUDA |
| `device` | `str` | `auto` | Requested device; `auto` selects CUDA, MPS, then CPU |
| `max_steps` | `int` | `1000` | Positive absolute local optimizer-step target |
| `output_dir` | `str` | `runs` | Base for relative checkpoint and distributed state paths |
| `log_every` | `int \| None` | `null` | Positive; copied to logging cadence only when logging cadence remains `1` |

## `ModelConfig`

| Field | Type | Default | Meaning and validation |
|---|---|---:|---|
| `name` | `str` | `reference_transformer` | Registry key; membership is not checked during config parsing |
| `vocab_size` | `int` | `512` | Positive vocabulary size |
| `max_seq_len` | `int` | `128` | Positive reference-model context limit; mapped to `block_size` |
| `n_layer` | `int` | `4` | Positive layer count |
| `n_head` | `int` | `8` | Positive query-head count |
| `n_kv_head` | `int \| None` | `null` | Resolves to `n_head`; positive and divides `n_head` |
| `d_model` | `int` | `256` | Positive hidden width; divisible by `n_head` |
| `d_ff` | `int \| None` | `null` | Resolves to `4 * d_model`; positive |
| `dropout` | `float` | `0.0` | Must satisfy `0 <= dropout < 1` |
| `tie_weights` | `bool` | `true` | Reference-model embedding/output weight tying |
| `rope_base` | `float` | `10000.0` | Rotary base; no positivity check in `ModelConfig` |
| `gradient_checkpointing` | `bool` | `false` | Synchronized with root-level flag |

These fields are reference-transformer-oriented even though the trainer itself is architecture-neutral.

## `DataConfig`

| Field | Type | Default | Meaning and validation |
|---|---|---:|---|
| `text_path` | `str \| null` | `null` | UTF-8 text file; selected before serialized/synthetic data |
| `dataset` | `str \| null` | `null` | Serialized dataset path in YAML; may hold a Dataset object programmatically |
| `synthetic` | `bool` | `true` | Enables synthetic fallback |
| `num_tokens` | `int` | `100000` | Positive; synthetic implementation uses it as **sample count** |
| `vocab_size` | `int \| null` | `null` | Positive if explicit; aggregate normally fills from model |
| `block_size` | `int \| null` | `null` | Positive after aggregate initialization; normally model context length |
| `micro_batch_size` | `int` | `1` | Positive loader batch size |
| `target_batch_size` | `int` | `1` | Positive, at least microbatch, and divisible by microbatch |
| `num_workers` | `int` | `0` | Non-negative DataLoader worker count |
| `prefetch_factor` | `int \| null` | `null` | Positive if explicit; defaults to 2 when workers are enabled |
| `pin_memory` | `bool \| null` | `null` | Explicit override or runtime device-derived default |
| `shuffle` | `bool` | `false` | Applied to map-style datasets; forced off for iterable datasets |
| `drop_last` | `bool` | `true` | Drop incomplete loader batches |
| `seed` | `int \| null` | `null` | Aggregate fills from `run.seed` |
| `max_steps` | `int \| null` | `null` | Positive; used as local target when `run` target is not explicitly overridden |

Derived accumulation:

```python
config.data.accumulation_steps
# target_batch_size // micro_batch_size
```

:::caution `num_tokens` naming

For synthetic data, `num_tokens=256` creates 256 samples, each containing `block_size` input tokens and one next-token target. It is not a literal total-token budget.

:::

## `OptimizerConfig`

| Field | Type | Default | Meaning and validation |
|---|---|---:|---|
| `name` | `str` | `adamw` | Only `adamw` is supported |
| `lr` | `float` | `0.0003` | Positive learning rate |
| `betas` | `tuple[float, float]` | `(0.9, 0.95)` | Exactly two values in `[0, 1)` |
| `eps` | `float` | `1e-8` | Positive Adam epsilon |
| `weight_decay` | `float` | `0.1` | Non-negative |
| `fused` | `bool \| null` | `null` | Auto-probe, force, or disable fused AdamW |
| `grad_clip` | `float \| null` | `null` | Positive global gradient-norm limit |

Fused construction is best-effort. Failure falls back to regular AdamW.

## `SchedulerConfig`

| Field | Type | Default | Meaning and validation |
|---|---|---:|---|
| `name` | `str` | `cosine` | `cosine` or `constant` |
| `warmup_steps` | `int` | `100` | Non-negative optimizer-step warmup |
| `max_steps` | `int` | `1000` | Positive schedule horizon |
| `min_lr_ratio` | `float` | `0.1` | Cosine floor in `[0, 1]`; ignored by constant schedule |

The warmup factor is `(step + 1) / warmup_steps`, floored at `1e-8`. The cosine factor clamps progress to `[0, 1]`.

:::warning Schedule target mismatch

`SchedulerConfig` rejects `max_steps <= 0`, making the aggregate branch that attempts to infer it from `run.max_steps` unreachable. If a Python config omits `scheduler.max_steps`, it remains 1000 even when `run.max_steps` is 10. The CLI `--max-steps` override updates both fields; direct Python overrides do not.

:::

## `PrecisionConfig`

| Field | Type | Default | Meaning and validation |
|---|---|---:|---|
| `mode` | `str` | `auto` | `auto`, `bf16`, `fp16`, or `fp32` |
| `dtype` | `str \| null` | `null` | `bf16`, `fp16`, or `fp32`; affects resolution only when mode is `auto` |

A string section is accepted as shorthand:

```yaml
precision: bf16
```

Conflicting values such as `mode: fp32` and `dtype: bf16` are accepted; the explicit mode wins and `dtype` is ignored.

## `CheckpointConfig`

| Field | Type | Default | Meaning and validation |
|---|---|---:|---|
| `enabled` | `bool` | `true` | Enable interval saves from the trainer |
| `directory` | `str` | `checkpoints` | Relative paths are rooted under `run.output_dir` |
| `every_steps` | `int` | `500` | Positive global-step interval |
| `keep_last` | `int \| null` | `3` | Positive retained count or `null` for unlimited |
| `resume` | `bool` | `false` | Request local checkpoint loading during runtime construction |

Saving happens only at exact interval boundaries. The trainer does not force a final checkpoint when the final step is off interval.

## `LoggingConfig`

| Field | Type | Default | Meaning and validation |
|---|---|---:|---|
| `level` | `str` | `INFO` | Uppercased; unknown names map to INFO in the logger |
| `file` | `str \| null` | `null` | Optional text output; relative to process working directory |
| `every_steps` | `int` | `1` | Positive cadence for `train_step` text, JSONL, and hook delivery |
| `json_file` | `str \| null` | `null` | Optional JSONL output; relative to process working directory |

There is no YAML `hooks` field. W&B, TensorBoard, and custom hooks require programmatic logger construction or mutation.

## `DistributedConfig`

| Field | Type | Default | Meaning and validation |
|---|---|---:|---|
| `enabled` | `bool` | `false` | Construct `DumbDiLoCoCoordinator` |
| `mode` | `str` | `dumb_diloco` | `dumb_diloco` or disabled `single` |
| `role` | `str` | `single` | `single`, `master`, or `worker` |
| `node_id` | `str \| null` | `null` | Explicit path-safe unique ID |
| `collaborators` | `list[str]` | `[]` | Master attempts to grant each user `write` |
| `inner_steps` | `int` | `500` | Positive local steps between uploads |
| `poll_interval` | `float` | `60.0` | Positive worker/master polling interval |
| `outer_lr` | `float` | `0.7` | Positive Nesterov outer learning rate |
| `outer_momentum` | `float` | `0.9` | Value in `[0, 1)` |
| `repo_id` | `str \| null` | `null` | Required for enabled distributed mode |
| `token` | `str \| null` | `null` | Optional Hub token; prefer SDK environment authentication |
| `cache_dir` | `str` | `.speedtronic/hub` | Hub cache, relative to process working directory |
| `state_dir` | `str` | `.speedtronic/diloco` | Relative state root under `run.output_dir`, then `/<node_id>` |
| `retry_initial` | `float` | `1.0` | Positive first retry delay |
| `retry_max` | `float` | `60.0` | At least `retry_initial` |
| `retry_attempts` | `int` | `6` | Positive total attempts per Hub operation |
| `reset_inner_optimizer` | `bool` | `true` | Clear optimizer state after successful upload/load callback |

When enabled with `role: single`, the role becomes `master` because a repository is required. The code treats this as a master-by-default convenience.

## Root booleans

```yaml
gradient_checkpointing: true
compile: true
```

`compile` also accepts:

```yaml
compile:
  enabled: true
```

Top-level values pass through `bool(...)`; quoted strings such as `"false"` become truthy. YAML booleans should remain unquoted.

## Aggregate normalization

`SpeedtronicConfig.__post_init__` performs:

1. Positive run target check.
2. Distributed role fallback.
3. `data.block_size = model.max_seq_len` when null.
4. `data.vocab_size = model.vocab_size` when null.
5. `data.seed = run.seed` when null.
6. `logging.every_steps = run.log_every` when the former remains 1.
7. Scheduler fallback that is only reachable for a non-positive value.
8. `data.max_steps` scheduler adoption when the scheduler still has its default 1000 horizon.
9. If either gradient-checkpointing flag is true, set both root and model flags true.

## Top-level aliases

These keys are moved into `run` before schema validation:

```yaml
name: demo
seed: 42
device: cpu
max_steps: 100
output_dir: runs/demo
log_every: 10
```

If both a top-level alias and nested `run` field exist, the top-level alias overwrites the nested value for that field.

Compatibility section aliases:

```yaml
hub: {}       # accepted only when distributed is absent
diloco: {}    # accepted only when distributed is absent
```

## Serialization and redaction

```python
plain = config.to_dict()
safe = config.to_dict(redact_secrets=True)
yaml_text = config.to_yaml(redact_secrets=True)
path = config.save("run.yaml")
```

`save()` always redacts `distributed.token`. `to_dict()` and `to_yaml()` default to **unredacted** output. Checkpoints also request redaction.

## Path semantics

| Setting | Relative-path base |
|---|---|
| `checkpoint.directory` | `run.output_dir` |
| `distributed.state_dir` | `run.output_dir`, then `node_id` |
| `distributed.cache_dir` | Process working directory |
| `logging.file` | Process working directory |
| `logging.json_file` | Process working directory |
| `data.text_path` | Process working directory |
| serialized `data.dataset` | Process working directory |

## v2 optimizer fields

```yaml
optimizer: muon
muon_plus: true
cautious: true
```

The canonical mapping form is:

```yaml
optimizer:
  name: muon          # adamw or muon
  lr: 0.0003
  muon_plus: false
  cautious: false
  muon_momentum: 0.95
  muon_ns_steps: 5
  muon_norm_eps: 0.00000001
```

`optimizer: muon` and the root-level `muon_plus`/`cautious` forms are
normalized into `OptimizerConfig`. Conflicting duplicate values are rejected.
`muon_plus` requires `name: muon`; `cautious` works with either optimizer.

## v2 systems fields

```yaml
shape_validation:
  enabled: true
  alignment: auto
  check_batch: true
  check_sequence: true
  check_model: true
  check_vocab: false
  warn_on_cpu: false

ooo_backprop: false
ooo_streams: 4
```

`ooo_streams` is constrained to 1–8. Shape warnings are non-fatal. See the
[v2 optimizer](../v2/optimizers), [scheduling](../v2/scheduling), and
[shape validation](../v2/shape-validation) pages.

## v2 distributed fields

```yaml
distributed:
  async_delta_upload: true
  delta_upload_queue_size: 1
  delta_upload_overflow: skip
  delta_upload_shutdown_timeout: 5.0
  async_global_poll: true
```

Queue size is intentionally one in v2. An occupied upload slot causes the
next boundary to skip and log rather than block the inner loop. Set both async
flags to `false` for the legacy synchronous transport path.


```yaml
run:
  name: local
  seed: 1234
  device: cpu
  max_steps: 4
  output_dir: runs/local

model:
  name: reference_transformer
  vocab_size: 128
  max_seq_len: 32
  n_layer: 2
  n_head: 4
  n_kv_head: 2
  d_model: 64
  d_ff: 128

data:
  synthetic: true
  num_tokens: 32
  block_size: 32
  micro_batch_size: 1
  target_batch_size: 2
  num_workers: 0

optimizer:
  lr: 0.0003
  weight_decay: 0.01

scheduler:
  name: cosine
  warmup_steps: 1
  max_steps: 4

precision:
  mode: fp32

checkpoint:
  enabled: true
  directory: checkpoints
  every_steps: 2
  keep_last: 2

logging:
  level: INFO
  every_steps: 1
```

For loader and model implementation details, continue to [Data](./data) and [Reference model](./reference-model). For execution, see [Runtime and Trainer](./runtime-and-trainer).
