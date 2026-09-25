---
id: observability
title: Metrics, Events, and Hooks
sidebar_label: Observability
description: MetricLogger, memory reporting, event payloads, cadence, JSONL output, and hook error isolation.
---

# Metrics, events, and hooks

## `TrainingHook`

A structural protocol:

```python
class TrainingHook(Protocol):
    def on_event(self, event: str, payload: dict[str, Any]) -> None: ...
```

A plain callable with the same signature is also accepted.

## `memory_usage_bytes()`

Attempts to return:

- `torch.cuda.memory_allocated()` when CUDA is available;
- `torch.mps.current_allocated_memory()` when MPS is available;
- otherwise `None`.

The function queries the current/default device rather than accepting the trainer's device explicitly. Memory can therefore be reported for the wrong CUDA device in a multi-GPU setup.

## `MetricLogger`

```python
MetricLogger(
    level="INFO",
    file=None,
    json_file=None,
    every_steps=1,
    stream=None,
    hooks=None,
)
```

Creates one dedicated Python logger with propagation disabled. It writes to stdout by default or to a text file. JSONL output is optional.

### `emit(event, payload)`

1. Copies the payload.
2. Adds invocation-local `elapsed_s` when absent.
3. Adds accelerator `memory_bytes` when absent and available.
4. For `train_step`, returns early when the step is not divisible by `every_steps`.
5. Writes a text log line.
6. Appends one JSON object to `json_file` when configured.
7. Dispatches to every hook.
8. Catches and logs hook exceptions.

Cadence filtering occurs before hook delivery, so hooks do not receive skipped training steps.

Memory is queried before cadence filtering, so skipped steps can still incur a device query.

### Text output

```text
2026-01-01 12:00:00 | INFO | train_step step=1 loss=1.2 lr=0.0003 ...
```

### JSONL output

Each line is:

```json
{"event":"train_step","step":1,"loss":1.2,"elapsed_s":0.01}
```

Non-JSON-native values use `default=str`.

### `close()`

Flushes and removes the logger's Speedtronic handler and closes file handlers. `train_from_config()` does not call `close()` automatically, so repeated programmatic runs can retain open resources until garbage collection.

## `CallbackList`

```python
CallbackList(callbacks=None)
```

Fan-out adapter whose `on_event` invokes each callback's `on_event` method or callable form.

## Event catalog

| Event | Producer | Typical payload |
|---|---|---|
| `train_start` | Trainer | start/target step, device, precision, accumulation, parameter count |
| `train_step` | Trainer | loss, LR, rates, counters, microbatches |
| `checkpoint` | Trainer | step, checkpoint path |
| `train_end` | Trainer | absolute steps, elapsed time, final loss |
| `compile` | Trainer | enabled flag and failure reason |
| `outer_step` | Master outer loop | outer step, candidates found, deltas included |
| `delta_upload_queued` | Coordinator | local step, base outer step |
| `delta_uploaded` | Async uploader | local step |
| `delta_upload_skipped` | Coordinator | step and overflow/in-flight reason |
| `delta_upload_failed` | Async uploader | step and error |
| `delta_upload_stale` | Async uploader | step and baseline generation |
| `shape_profile` | Runtime | device, precision, alignment, warning count |
| `shape_warning` | Runtime | field, value, suggested alignment |
| `ooo_backprop` | Trainer | enabled, reason, streams, stage names |

## Hook adapter

`speedtronic.hooks.on_event(callback)` returns a two-argument wrapper. It is useful when adapting a function whose natural signature differs from `(event, payload)`.

## W&B adapter

```python
from speedtronic.integrations import WandbHook

hook = WandbHook(project="my-project")
```

Requires the `logging` extra or `wandb`. Construction calls `wandb.init`; every event calls `wandb.log({**payload, "event": event})`.

The adapter has no `close()` method and does not finish the run.

## TensorBoard adapter

```python
from speedtronic.integrations import TensorboardHook

hook = TensorboardHook(log_dir="runs/tensorboard")
```

Requires `tensorboard` or PyTorch's TensorBoard writer. It writes each numeric payload value to:

```text
<event>/<key>
```

Call `hook.close()` explicitly. `MetricLogger.close()` does not automatically call hook close methods.

## Attaching hooks through configuration

YAML does not have a logging hook field. The runtime's `build_logger()` does not attach W&B or TensorBoard.

Programmatic pattern:

```python
from speedtronic.integrations import WandbHook
from speedtronic.profiling import MetricLogger
from speedtronic.runtime import build_runtime

trainer, _ = build_runtime(config)
hook = WandbHook(project="speedtronic")
trainer.logger.hooks.append(hook)
result = trainer.fit()
hook._run.finish()  # adapter currently has no public close
```

Direct construction is also possible:

```python
logger = MetricLogger(hooks=[WandbHook(project="speedtronic")])
trainer = Trainer(..., logger=logger)
```

## Hook failure semantics

Hook exceptions are isolated from training. The logger writes a warning and continues. This protects the loop but can hide broken observability integrations unless warnings are monitored.

## Concurrency caveat

DumbDiLoCo's master outer thread can emit `outer_step` while the training thread emits other events. `MetricLogger` does not synchronize JSONL appends or hook calls across those threads.

See [Observability Tutorial](../tutorials/observability) and [DumbDiLoCo](../distributed/overview).
