---
id: extensions
title: Models, Data, Modules, and Hooks
sidebar_label: Extensions
description: Public extension seams for model factories, gradient checkpointing, datasets, coordinators, metric callbacks, and optional integrations.
---

# Extension points

## Model registry

### `ModelRegistry`

```python
registry = ModelRegistry()
registry.register(name, factory=None)
registry.get(name)
registry.names()
registry.build(name, **kwargs)
```

`register()` works as a decorator or direct function. Registration silently overwrites an existing name.

### Built-in lazy loading

The package-level global registry knows that `reference_transformer` and `gpt` are built-ins. If a name is absent, `get()` imports `speedtronic.model`; the model decorators then populate the **global** registry.

A newly constructed `ModelRegistry()` advertises built-in names through `names()` but importing the model does not register factories into that separate instance. Use the exported global `registry` for normal custom registration.

### `register_model(name, factory=None)`

Public decorator/function backed by the global registry.

```python
import torch
from speedtronic import register_model

@register_model("my_model")
def make_model(**kwargs):
    return torch.nn.Linear(kwargs["d_model"], kwargs["vocab_size"])
```

### `build_model(config, **overrides)`

Always forwards these fields:

```text
vocab_size
block_size
max_seq_len-derived model context
n_layer
n_head
n_kv_head
d_model
d_ff
dropout
tie_weights
rope_base
```

Explicit `overrides` replace standard fields. Custom factories should accept `**kwargs` or deliberately ignore unrelated reference-model parameters.

## Configuration-driven custom models

Unknown custom model-specific configuration keys are rejected. A model factory must obtain architecture-specific values inside Python or through the standard fields.

Registration is process-local. A CLI process loading only YAML has no plugin import mechanism and cannot discover a factory registered only in another process.

To make a CLI-visible model:

1. Put registration in an importable Python module.
2. Import that module before `build_runtime()`.
3. Construct the runtime programmatically, or add a supported plugin loader in the application.

## Trainer model contract

Accept a dictionary batch as keyword arguments or a tuple batch as `(inputs, labels)`. Return a direct loss, mapping with loss, or compatible causal logits. See [Runtime and Trainer](./runtime-and-trainer).

## Gradient checkpointing convention

### Model hook

```python
def set_gradient_checkpointing(self, enabled: bool = True):
    ...
```

The trainer calls the hook once during construction. Missing or failing hooks warn and continue.

### Public helper

```python
from speedtronic import set_gradient_checkpointing

set_gradient_checkpointing(model, True)
```

`module_utils.set_gradient_checkpointing()` raises `AttributeError` when the model does not expose the hook, unlike the trainer's warning-only behavior.

## Dataset extension

Use:

```python
build_dataloader(config.data, dataset=my_dataset, tokenizer=my_tokenizer)
```

The built-in collator recognizes causal dictionaries, equal-width tuples/lists, and tensors. For other tasks, build a `DataLoader` directly and pass it to `Trainer`.

## Sync coordinator extension

Implement:

```python
class MyCoordinator:
    def start(self): ...
    def after_optimizer_step(self, model, step): ...
    def stop(self): ...
    def state_dict(self): ...
    def load_state_dict(self, state): ...
```

Contract details:

- `start()` is called only when work remains.
- `after_optimizer_step()` runs after the local scheduler step.
- A true result can clear optimizer state.
- `stop()` runs in `finally`.
- `state_dict()` is embedded in the next local checkpoint.
- `load_state_dict()` runs after coordinator startup.

## Metric hooks

A hook can be:

```python
def hook(event: str, payload: dict[str, Any]) -> None:
    ...
```

or an object with `on_event`. Hook failures do not stop training.

See [Observability](./observability) for built-in W&B and TensorBoard adapters.

## Optional integrations

| Adapter | Dependency | Construction | Close behavior |
|---|---|---|---|
| `WandbHook` | `wandb` | Calls `wandb.init` | No public close method |
| `TensorboardHook` | TensorBoard writer | Creates `SummaryWriter` | Has `close()` |

Install with:

```bash
pip install 'speedtronic[logging]'
```

## Stability categories

| Category | Examples |
|---|---|
| Public top-level API | `SpeedtronicConfig`, `Trainer`, `register_model`, `HubClient` |
| Public module API | `build_dataloader`, `CheckpointManager`, tensor helpers |
| Compatibility alias | `GPT`, `TrainingEngine`, `DumbDiLoCo`, `HubTransport`, `Config` |
| Declarative but currently unused | `SyncResult`, `OuterState`, pending coordinator delta fields |
| Internal extension seam | `SyncCoordinator` protocol and model checkpoint hook |
| Private implementation | Names beginning with `_`; not stable API |

The [generated inventory](./generated-source-inventory) lists all of these declarations and source lines.
