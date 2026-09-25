## Metrics, Events, and Hooks

_MetricLogger, memory reporting, event payloads, cadence, JSONL output, and hook error isolation._
### `TrainingHook`

A structural protocol:

```python
class TrainingHook(Protocol):
    def on_event(self, event: str, payload: dict[str, Any]) -> None: ...
```

A plain callable with the same signature is also accepted.

### `memory_usage_bytes()`

Attempts to return:

- `torch.cuda.memory_allocated()` when CUDA is available;
- `torch.mps.current_allocated_memory()` when MPS is available;
- otherwise `None`.

The function queries the current/default device rather than accepting the trainer's device explicitly. Memory can therefore be reported for the wrong CUDA device in a multi-GPU setup.

### `MetricLogger`

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

#### `emit(event, payload)`

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

#### Text output

```text
2026-01-01 12:00:00 | INFO | train_step step=1 loss=1.2 lr=0.0003 ...
```

#### JSONL output

Each line is:

```json
{"event":"train_step","step":1,"loss":1.2,"elapsed_s":0.01}
```

Non-JSON-native values use `default=str`.

#### `close()`

Flushes and removes the logger's Speedtronic handler and closes file handlers. `train_from_config()` does not call `close()` automatically, so repeated programmatic runs can retain open resources until garbage collection.

### `CallbackList`

```python
CallbackList(callbacks=None)
```

Fan-out adapter whose `on_event` invokes each callback's `on_event` method or callable form.

### Event catalog

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

### Hook adapter

`speedtronic.hooks.on_event(callback)` returns a two-argument wrapper. It is useful when adapting a function whose natural signature differs from `(event, payload)`.

### W&B adapter

```python
from speedtronic.integrations import WandbHook

hook = WandbHook(project="my-project")
```

Requires the `logging` extra or `wandb`. Construction calls `wandb.init`; every event calls `wandb.log({**payload, "event": event})`.

The adapter has no `close()` method and does not finish the run.

### TensorBoard adapter

```python
from speedtronic.integrations import TensorboardHook

hook = TensorboardHook(log_dir="runs/tensorboard")
```

Requires `tensorboard` or PyTorch's TensorBoard writer. It writes each numeric payload value to:

```text
<event>/<key>
```

Call `hook.close()` explicitly. `MetricLogger.close()` does not automatically call hook close methods.

### Attaching hooks through configuration

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

### Hook failure semantics

Hook exceptions are isolated from training. The logger writes a warning and continues. This protects the loop but can hide broken observability integrations unless warnings are monitored.

### Concurrency caveat

DumbDiLoCo's master outer thread can emit `outer_step` while the training thread emits other events. `MetricLogger` does not synchronize JSONL appends or hook calls across those threads.

See [Observability Tutorial](references/observability.md) and [DumbDiLoCo](references/distributed.md).


---

## Observability

_Send Speedtronic metrics to files, JSONL, W&B, TensorBoard, or custom Python hooks._
### Built-in text output

Every event is written to stdout by default:

```text
train_start start_step=0 target_step=100 ...
train_step step=1 loss=2.31 ...
checkpoint step=50 path=...
train_end steps=100 elapsed_s=12.3 loss=1.84
```

The master distributed loop can also emit `outer_step` events.

### Text file

```yaml
logging:
  level: INFO
  file: runs/demo/train.log
  every_steps: 10
```

Relative logging paths are resolved from the process working directory, not `run.output_dir`.

### JSONL file

```yaml
logging:
  json_file: runs/demo/metrics.jsonl
  every_steps: 1
```

Each line is a self-contained JSON event.

### Cadence

`logging.every_steps` filters `train_step` events before text, JSONL, and hook dispatch. Lifecycle, checkpoint, compile, and outer events are not filtered by this field.

The trainer still retains every metric dictionary in `TrainResult.metrics`, so long runs can accumulate memory even when output cadence is sparse.

### Custom callable hook

```python
events = []


def collect(event: str, payload: dict) -> None:
    events.append((event, dict(payload)))


trainer, _ = build_runtime(config)
trainer.logger.hooks.append(collect)
result = trainer.fit()
```

A hook exception is caught and logged; it does not stop training.

### Object hook

```python
class JsonlHook:
    def on_event(self, event, payload):
        ...
```

`MetricLogger` prefers `hook.on_event` when present and otherwise calls the object directly.

### W&B

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

### TensorBoard

```python
from speedtronic.integrations import TensorboardHook

tb_hook = TensorboardHook(log_dir="runs/tensorboard")
trainer.logger.hooks.append(tb_hook)
result = trainer.fit()
tb_hook.close()
```

Numeric values are written to `<event>/<key>`.

### Direct logger construction

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

### YAML limitation

`LoggingConfig` has no `hooks` list. Adding one to YAML raises `ConfigError`. Programmatic attachment is required in 2.0.0.

### Memory metrics

`MetricLogger` adds `memory_bytes` from the current CUDA or MPS device when available. It does not accept a trainer device parameter, so a non-current CUDA device can report the wrong allocation.

### Distributed events

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

### Operational recommendations

- Close file handlers explicitly for long-lived processes.
- Close TensorBoard writers explicitly.
- Finish W&B runs explicitly.
- Keep hooks cheap and nonblocking.
- Use JSONL for durable machine-readable telemetry.
- Monitor hook warnings; failures are intentionally isolated.
- Avoid placing secrets in metric payloads.

Related: [Observability API](references/observability.md) and [DumbDiLoCo](references/distributed.md).
