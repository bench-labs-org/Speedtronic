---
id: observability
title: Observability
sidebar_label: Observability
description: Send Speedtronic metrics to files, JSONL, W&B, TensorBoard, or custom Python hooks.
---

# Observability

## Built-in text output

Every event is written to stdout by default:

```text
train_start start_step=0 target_step=100 ...
train_step step=1 loss=2.31 ...
checkpoint step=50 path=...
train_end steps=100 elapsed_s=12.3 loss=1.84
```

The master distributed loop can also emit `outer_step` events.

## Text file

```yaml
logging:
  level: INFO
  file: runs/demo/train.log
  every_steps: 10
```

Relative logging paths are resolved from the process working directory, not `run.output_dir`.

## JSONL file

```yaml
logging:
  json_file: runs/demo/metrics.jsonl
  every_steps: 1
```

Each line is a self-contained JSON event.

## Cadence

`logging.every_steps` filters `train_step` events before text, JSONL, and hook dispatch. Lifecycle, checkpoint, compile, and outer events are not filtered by this field.

The trainer still retains every metric dictionary in `TrainResult.metrics`, so long runs can accumulate memory even when output cadence is sparse.

## Custom callable hook

```python
events = []


def collect(event: str, payload: dict) -> None:
    events.append((event, dict(payload)))


trainer, _ = build_runtime(config)
trainer.logger.hooks.append(collect)
result = trainer.fit()
```

A hook exception is caught and logged; it does not stop training.

## Object hook

```python
class JsonlHook:
    def on_event(self, event, payload):
        ...
```

`MetricLogger` prefers `hook.on_event` when present and otherwise calls the object directly.

## W&B

Install:

```bash
pip install 'speedtronic[logging]'
```

Attach:

```python
from speedtronic.integrations import WandbHook

trainer, _ = build_runtime(config)
wandb_hook = WandbHook(project="speedtronic")
trainer.logger.hooks.append(wandb_hook)
result = trainer.fit()
```

Every emitted event calls:

```python
wandb.log({**payload, "event": event})
```

The adapter currently has no public `close()`/finish method.

## TensorBoard

```python
from speedtronic.integrations import TensorboardHook

tb_hook = TensorboardHook(log_dir="runs/tensorboard")
trainer.logger.hooks.append(tb_hook)
result = trainer.fit()
tb_hook.close()
```

Numeric values are written to `<event>/<key>`.

## Direct logger construction

```python
from speedtronic.profiling import MetricLogger

logger = MetricLogger(
    level="INFO",
    file="train.log",
    json_file="metrics.jsonl",
    every_steps=5,
    hooks=[collect],
)
```

Attach the logger to a directly constructed `Trainer`.

## YAML limitation

`LoggingConfig` has no `hooks` list. Adding one to YAML raises `ConfigError`. Programmatic attachment is required in 2.0.0.

## Memory metrics

`MetricLogger` adds `memory_bytes` from the current CUDA or MPS device when available. It does not accept a trainer device parameter, so a non-current CUDA device can report the wrong allocation.

## Distributed events

The master emits:

```json
{
  "event": "outer_step",
  "outer_step": 4,
  "deltas_found": 3,
  "deltas_included": 2
}
```

An empty round still emits the event with zero included deltas. Because the outer loop runs on a background thread, hook implementations should be thread-safe.

## Operational recommendations

- Close file handlers explicitly for long-lived processes.
- Close TensorBoard writers explicitly.
- Finish W&B runs explicitly.
- Keep hooks cheap and nonblocking.
- Use JSONL for durable machine-readable telemetry.
- Monitor hook warnings; failures are intentionally isolated.
- Avoid placing secrets in metric payloads.

Related: [Observability API](../reference/observability) and [DumbDiLoCo](../distributed/overview).
