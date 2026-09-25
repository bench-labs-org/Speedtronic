## Command-Line Interface

_Complete reference for speedtronic train, speedtronic validate, and python -m speedtronic._
The package installs one console entry point:

```text
speedtronic = speedtronic.cli:main
```

The module entry point delegates to the same function:

```bash
python -m speedtronic --help
```

### `speedtronic train`

```text
speedtronic train --config PATH
  [--resume]
  [--device DEVICE]
  [--max-steps N]
  [--output-dir PATH]
```

| Option | Required | Behavior |
|---|---:|---|
| `--config` | Yes | YAML or JSON configuration path |
| `--resume` | No | Load the latest local checkpoint if one exists |
| `--device` | No | Override `run.device`; commonly `auto`, `cpu`, `cuda`, or `mps` |
| `--max-steps` | No | Override the absolute local optimizer-step target and scheduler horizon |
| `--output-dir` | No | Override `run.output_dir` before runtime composition |

After completion, the CLI prints:

```text
completed steps=<global-step> samples=<cumulative-samples> tokens=<cumulative-tokens> loss=<value-or-None>
```

The counters are cumulative and include restored state.

### Override order

CLI processing occurs after configuration construction:

1. Load and validate the file.
2. Mutate `run.output_dir` when `--output-dir` is present.
3. Mutate both `run.max_steps` and `scheduler.max_steps` when `--max-steps` is present.
4. Pass `device` and `max_steps` again as runtime keyword overrides.
5. Load the checkpoint from the resolved output directory when resume is active.

Because checkpoint paths are constructed after output override, resume searches the overridden output directory.

### `speedtronic validate`

```text
speedtronic validate --config PATH [--format {yaml,json}]
```

Validation:

- loads YAML or JSON;
- rejects unknown mapping keys;
- applies defaults and section validation;
- prints the normalized configuration;
- does not import the runtime eagerly.

It does not:

- build or register a model;
- open a data file;
- load a serialized dataset;
- resolve the requested device;
- check hardware precision support;
- perform a model forward;
- connect to Hugging Face Hub.

> **Caution** — Secret exposure
>
>
> `validate` calls `to_dict()` and `to_yaml()` without secret redaction. A token placed under `distributed.token` can be printed. Prefer `HF_TOKEN`/SDK authentication and avoid validating a file that contains a secret.
>

### Exit behavior

| Outcome | Exit code |
|---|---:|
| Success | `0` |
| Handled configuration, file, key, runtime, type, or value error | `2` |

The explicit exception tuple does not cover every possible lower-level `OSError` or import failure.

### Examples

#### Four CPU steps

```bash
speedtronic train --config configs/smoke.yaml --device cpu
```

#### Resume an existing target

```bash
speedtronic train --config configs/smoke.yaml --device cpu --resume
```

#### Override output and target

```bash
speedtronic train \
  --config configs/smoke.yaml \
  --output-dir runs/experiment-42 \
  --max-steps 100
```

#### Validate as JSON

```bash
speedtronic validate --config configs/smoke.yaml --format json
```

### Programmatic equivalent

```python
from speedtronic import SpeedtronicConfig
from speedtronic.runtime import train_from_config

config = SpeedtronicConfig.load("configs/smoke.yaml")
result = train_from_config(
    config,
    resume=True,
    device="cpu",
    max_steps=100,
)
```

`train_from_config()` accepts a config object or dictionary and returns `TrainResult`; see [Runtime and Trainer](references/architecture.md).
