---
id: generated-source-inventory
title: Generated Source Inventory
sidebar_label: Full AST Inventory
description: Exhaustive generated inventory of every local Python module, class, function, method, test, configuration, and example.
---

# Generated source inventory

:::info
This page is generated from the local repository by `scripts/generate_source_inventory.py` during every documentation build. It inventories every implementation module and every top-level class/function/method without importing PyTorch or contacting the network.
:::

## Implementation modules

### `src/speedtronic/__init__.py`

**Import path:** `speedtronic`<br />
**Purpose:** Speedtronic: fast, efficient training with optional DumbDiLoCo..<br />
**Lines:** 108

**Imported modules:** `__future__.annotations`, `.config.CheckpointConfig`, `.config.Config`, `.config.ConfigError`, `.config.DataConfig`, `.config.DistributedConfig`, `.config.LoggingConfig`, `.config.ModelConfig`, `.config.OptimizerConfig`, `.config.PrecisionConfig`, `.config.RunConfig`, `.config.SchedulerConfig`, `.config.ShapeValidationConfig`, `.config.SpeedtronicConfig`, `.config.load_config`, `.config.load_yaml_config`, `.module_utils.set_gradient_checkpointing`, `.optimizers.CautiousOptimizer`, `.optimizers.HybridOptimizer`, `.optimizers.Muon`, `.optimizers.ParameterRouting`, `.optimizers.newton_schulz`, `.optimizers.post_polar_normalize`, `.optimizers.route_parameters`, `.registry.ModelRegistry`, `.registry.build_model`, `.registry.register_model`, `.registry.registry`, `.scheduling.StageInfo`, `.scheduling.StageStreamScheduler`, `.shapes.ShapeProfile`, `.shapes.ShapeReport`, `.shapes.ShapeWarning`, `.shapes.resolve_shape_profile`, `.shapes.validate_startup_shapes`

#### Function `__getattr__` {#api-speedtronic-__getattr__}

```python
def __getattr__(name: str)
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `__all__` | 42 |
| constant/alias | `__version__` | 87 |


### `src/speedtronic/__main__.py`

**Import path:** `speedtronic.__main__`<br />
**Purpose:** No module docstring.<br />
**Lines:** 4

**Imported modules:** `.cli.main`


### `src/speedtronic/checkpoint.py`

**Import path:** `speedtronic.checkpoint`<br />
**Purpose:** Atomic local checkpoint storage and discovery..<br />
**Lines:** 176

**Imported modules:** `__future__.annotations`, `json`, `os`, `re`, `shutil`, `pathlib.Path`, `typing.Any`, `torch`

#### Class `CheckpointError` {#api-speedtronic-checkpoint-CheckpointError}

```python
class CheckpointError(RuntimeError)
```

#### Function `atomic_torch_save` {#api-speedtronic-checkpoint-atomic_torch_save}

```python
def atomic_torch_save(state: dict[str, Any], path: str | os.PathLike[str]) -> None
```

#### Function `atomic_json_dump` {#api-speedtronic-checkpoint-atomic_json_dump}

```python
def atomic_json_dump(value: Any, path: str | os.PathLike[str]) -> None
```

#### Class `CheckpointManager` {#api-speedtronic-checkpoint-CheckpointManager}

```python
class CheckpointManager
```

Manage local ``step_<n>.pt`` checkpoints and a latest pointer.

##### Method `__init__` {#api-speedtronic-checkpoint-CheckpointManager-__init__}

```python
def __init__(self, directory: str | os.PathLike[str], *, every_steps: int=500, keep_last: int | None=3, enabled: bool=True) -> None
```

##### Method `should_save` {#api-speedtronic-checkpoint-CheckpointManager-should_save}

```python
def should_save(self, step: int) -> bool
```

##### Method `path_for` {#api-speedtronic-checkpoint-CheckpointManager-path_for}

```python
def path_for(self, step: int) -> Path
```

##### Method `save` {#api-speedtronic-checkpoint-CheckpointManager-save}

```python
def save(self, step: int, state: dict[str, Any]) -> Path
```

##### Method `load_latest` {#api-speedtronic-checkpoint-CheckpointManager-load_latest}

```python
def load_latest(self) -> dict[str, Any] \| None
```

##### Method `latest_path` {#api-speedtronic-checkpoint-CheckpointManager-latest_path}

```python
def latest_path(self) -> Path \| None
```

##### Method `_checkpoint_paths` {#api-speedtronic-checkpoint-CheckpointManager-_checkpoint_paths}

```python
def _checkpoint_paths(self) -> list[Path]
```

##### Method `prune` {#api-speedtronic-checkpoint-CheckpointManager-prune}

```python
def prune(self) -> None
```

##### Method `copy_to` {#api-speedtronic-checkpoint-CheckpointManager-copy_to}

```python
def copy_to(self, destination: str | os.PathLike[str]) -> Path
```

#### Function `capture_rng_state` {#api-speedtronic-checkpoint-capture_rng_state}

```python
def capture_rng_state() -> dict[str, Any]
```

#### Function `restore_rng_state` {#api-speedtronic-checkpoint-restore_rng_state}

```python
def restore_rng_state(state: dict[str, Any] | None) -> None
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `__all__` | 169 |


### `src/speedtronic/cli.py`

**Import path:** `speedtronic.cli`<br />
**Purpose:** Command-line interface for Speedtronic..<br />
**Lines:** 75

**Imported modules:** `__future__.annotations`, `argparse`, `json`, `sys`, `typing.Sequence`, `.config.ConfigError`, `.config.SpeedtronicConfig`

#### Function `_parser` {#api-speedtronic-cli-_parser}

```python
def _parser() -> argparse.ArgumentParser
```

#### Function `main` {#api-speedtronic-cli-main}

```python
def main(argv: Sequence[str] | None=None) -> int
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `__all__` | 75 |


### `src/speedtronic/config.py`

**Import path:** `speedtronic.config`<br />
**Purpose:** Configuration objects and loading helpers for Speedtronic. The configuration is intentionally data-first: a run can be represented by a YAML file or by constructing the same dataclasses in Python.<br />
**Lines:** 631

**Imported modules:** `__future__.annotations`, `json`, `os`, `dataclasses.asdict`, `dataclasses.dataclass`, `dataclasses.field`, `dataclasses.fields`, `dataclasses.is_dataclass`, `pathlib.Path`, `typing.Any`, `typing.Mapping`, `typing.TypeVar`

#### Class `ConfigError` {#api-speedtronic-config-ConfigError}

```python
class ConfigError(ValueError)
```

Raised when a run configuration is invalid.

#### Class `ModelConfig` {#api-speedtronic-config-ModelConfig}

```python
class ModelConfig
```

##### Method `__post_init__` {#api-speedtronic-config-ModelConfig-__post_init__}

```python
def __post_init__(self) -> None
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `name` | 28 |
| field | `vocab_size` | 29 |
| field | `max_seq_len` | 30 |
| field | `n_layer` | 31 |
| field | `n_head` | 32 |
| field | `n_kv_head` | 33 |
| field | `d_model` | 34 |
| field | `d_ff` | 35 |
| field | `dropout` | 36 |
| field | `tie_weights` | 37 |
| field | `rope_base` | 38 |
| field | `gradient_checkpointing` | 39 |

#### Class `DataConfig` {#api-speedtronic-config-DataConfig}

```python
class DataConfig
```

Data and batching configuration.

``micro_batch_size`` is the batch size emitted by the loader.  The trainer
accumulates that many batches to reach ``target_batch_size`` when the
latter is larger.

##### Method `__post_init__` {#api-speedtronic-config-DataConfig-__post_init__}

```python
def __post_init__(self) -> None
```

##### Method `accumulation_steps` {#api-speedtronic-config-DataConfig-accumulation_steps}

```python
def accumulation_steps(self) -> int
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `text_path` | 71 |
| field | `dataset` | 72 |
| field | `synthetic` | 73 |
| field | `num_tokens` | 74 |
| field | `vocab_size` | 75 |
| field | `block_size` | 76 |
| field | `micro_batch_size` | 77 |
| field | `target_batch_size` | 78 |
| field | `num_workers` | 79 |
| field | `prefetch_factor` | 80 |
| field | `pin_memory` | 81 |
| field | `shuffle` | 82 |
| field | `drop_last` | 83 |
| field | `seed` | 84 |
| field | `max_steps` | 85 |

#### Class `OptimizerConfig` {#api-speedtronic-config-OptimizerConfig}

```python
class OptimizerConfig
```

##### Method `__post_init__` {#api-speedtronic-config-OptimizerConfig-__post_init__}

```python
def __post_init__(self) -> None
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `name` | 112 |
| field | `lr` | 113 |
| field | `betas` | 114 |
| field | `eps` | 115 |
| field | `weight_decay` | 116 |
| field | `fused` | 117 |
| field | `grad_clip` | 118 |
| field | `muon_plus` | 119 |
| field | `cautious` | 120 |
| field | `muon_momentum` | 121 |
| field | `muon_ns_steps` | 122 |
| field | `muon_norm_eps` | 123 |

#### Class `SchedulerConfig` {#api-speedtronic-config-SchedulerConfig}

```python
class SchedulerConfig
```

##### Method `__post_init__` {#api-speedtronic-config-SchedulerConfig-__post_init__}

```python
def __post_init__(self) -> None
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `name` | 151 |
| field | `warmup_steps` | 152 |
| field | `max_steps` | 153 |
| field | `min_lr_ratio` | 154 |

#### Class `PrecisionConfig` {#api-speedtronic-config-PrecisionConfig}

```python
class PrecisionConfig
```

##### Method `__post_init__` {#api-speedtronic-config-PrecisionConfig-__post_init__}

```python
def __post_init__(self) -> None
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `mode` | 168 |
| field | `dtype` | 169 |

#### Class `CheckpointConfig` {#api-speedtronic-config-CheckpointConfig}

```python
class CheckpointConfig
```

##### Method `__post_init__` {#api-speedtronic-config-CheckpointConfig-__post_init__}

```python
def __post_init__(self) -> None
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `enabled` | 183 |
| field | `directory` | 184 |
| field | `every_steps` | 185 |
| field | `keep_last` | 186 |
| field | `resume` | 187 |

#### Class `LoggingConfig` {#api-speedtronic-config-LoggingConfig}

```python
class LoggingConfig
```

##### Method `__post_init__` {#api-speedtronic-config-LoggingConfig-__post_init__}

```python
def __post_init__(self) -> None
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `level` | 198 |
| field | `file` | 199 |
| field | `every_steps` | 200 |
| field | `json_file` | 201 |

#### Class `ShapeValidationConfig` {#api-speedtronic-config-ShapeValidationConfig}

```python
class ShapeValidationConfig
```

##### Method `__post_init__` {#api-speedtronic-config-ShapeValidationConfig-__post_init__}

```python
def __post_init__(self) -> None
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `enabled` | 211 |
| field | `alignment` | 212 |
| field | `check_batch` | 213 |
| field | `check_sequence` | 214 |
| field | `check_model` | 215 |
| field | `check_vocab` | 216 |
| field | `warn_on_cpu` | 217 |

#### Class `DistributedConfig` {#api-speedtronic-config-DistributedConfig}

```python
class DistributedConfig
```

##### Method `__post_init__` {#api-speedtronic-config-DistributedConfig-__post_init__}

```python
def __post_init__(self) -> None
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `enabled` | 247 |
| field | `mode` | 248 |
| field | `role` | 249 |
| field | `node_id` | 250 |
| field | `collaborators` | 251 |
| field | `inner_steps` | 252 |
| field | `poll_interval` | 253 |
| field | `outer_lr` | 254 |
| field | `outer_momentum` | 255 |
| field | `repo_id` | 256 |
| field | `token` | 257 |
| field | `cache_dir` | 258 |
| field | `state_dir` | 259 |
| field | `retry_initial` | 260 |
| field | `retry_max` | 261 |
| field | `retry_attempts` | 262 |
| field | `reset_inner_optimizer` | 263 |
| field | `async_delta_upload` | 264 |
| field | `delta_upload_queue_size` | 265 |
| field | `delta_upload_overflow` | 266 |
| field | `delta_upload_shutdown_timeout` | 267 |
| field | `async_global_poll` | 268 |

#### Class `RunConfig` {#api-speedtronic-config-RunConfig}

```python
class RunConfig
```

##### Method `__post_init__` {#api-speedtronic-config-RunConfig-__post_init__}

```python
def __post_init__(self) -> None
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `name` | 321 |
| field | `seed` | 322 |
| field | `device` | 323 |
| field | `max_steps` | 324 |
| field | `output_dir` | 325 |
| field | `log_every` | 326 |

#### Class `SpeedtronicConfig` {#api-speedtronic-config-SpeedtronicConfig}

```python
class SpeedtronicConfig
```

##### Method `__post_init__` {#api-speedtronic-config-SpeedtronicConfig-__post_init__}

```python
def __post_init__(self) -> None
```

##### Method `accumulation_steps` {#api-speedtronic-config-SpeedtronicConfig-accumulation_steps}

```python
def accumulation_steps(self) -> int
```

##### Method `validate` {#api-speedtronic-config-SpeedtronicConfig-validate}

```python
def validate(self) -> 'SpeedtronicConfig'
```

Validate cross-section constraints and return ``self``.

##### Method `to_dict` {#api-speedtronic-config-SpeedtronicConfig-to_dict}

```python
def to_dict(self, *, redact_secrets: bool=False) -> dict[str, Any]
```

##### Method `to_yaml` {#api-speedtronic-config-SpeedtronicConfig-to_yaml}

```python
def to_yaml(self, *, redact_secrets: bool=False) -> str
```

##### Method `save` {#api-speedtronic-config-SpeedtronicConfig-save}

```python
def save(self, path: str | os.PathLike[str]) -> Path
```

##### Method `from_dict` {#api-speedtronic-config-SpeedtronicConfig-from_dict}

```python
def from_dict(cls, values: Mapping[str, Any] | None=None) -> 'SpeedtronicConfig'
```

##### Method `from_yaml` {#api-speedtronic-config-SpeedtronicConfig-from_yaml}

```python
def from_yaml(cls, path: str | os.PathLike[str]) -> 'SpeedtronicConfig'
```

##### Method `load` {#api-speedtronic-config-SpeedtronicConfig-load}

```python
def load(cls, source: str | os.PathLike[str] | Mapping[str, Any]) -> 'SpeedtronicConfig'
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `run` | 337 |
| field | `model` | 338 |
| field | `data` | 339 |
| field | `optimizer` | 340 |
| field | `scheduler` | 341 |
| field | `precision` | 342 |
| field | `checkpoint` | 343 |
| field | `logging` | 344 |
| field | `distributed` | 345 |
| field | `shape_validation` | 346 |
| field | `gradient_checkpointing` | 347 |
| field | `compile` | 348 |
| field | `ooo_backprop` | 349 |
| field | `ooo_streams` | 350 |

#### Function `load_config` {#api-speedtronic-config-load_config}

```python
def load_config(source: str | os.PathLike[str] | Mapping[str, Any]) -> SpeedtronicConfig
```

#### Function `load_yaml_config` {#api-speedtronic-config-load_yaml_config}

```python
def load_yaml_config(path: str | os.PathLike[str]) -> SpeedtronicConfig
```

#### Class `_CompileSection` {#api-speedtronic-config-_CompileSection}

```python
class _CompileSection
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `enabled` | 572 |

#### Function `_build_dataclass` {#api-speedtronic-config-_build_dataclass}

```python
def _build_dataclass(cls: type[T], value: Any, path: str) -> T
```

#### Function `_reject_unknown` {#api-speedtronic-config-_reject_unknown}

```python
def _reject_unknown(value: Mapping[str, Any], cls: type[Any], path: str) -> None
```

#### Function `_jsonable` {#api-speedtronic-config-_jsonable}

```python
def _jsonable(value: Any) -> Any
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `Config` | 556 |
| constant/alias | `T` | 567 |
| constant/alias | `__all__` | 615 |


### `src/speedtronic/data.py`

**Import path:** `speedtronic.data`<br />
**Purpose:** Dataset and dataloader utilities. The default data path is a small synthetic token stream so the project is usable immediately.<br />
**Lines:** 269

**Imported modules:** `__future__.annotations`, `collections.abc.Callable`, `collections.abc.Iterable`, `collections.abc.Iterator`, `pathlib.Path`, `typing.Any`, `torch`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torch.utils.data.IterableDataset`, `torch.utils.data.get_worker_info`

#### Class `CharTokenizer` {#api-speedtronic-data-CharTokenizer}

```python
class CharTokenizer
```

A deterministic byte-level tokenizer useful for examples and tests.

##### Method `__init__` {#api-speedtronic-data-CharTokenizer-__init__}

```python
def __init__(self, vocab_size: int=256) -> None
```

##### Method `encode` {#api-speedtronic-data-CharTokenizer-encode}

```python
def encode(self, text: str) -> list[int]
```

##### Method `__call__` {#api-speedtronic-data-CharTokenizer-__call__}

```python
def __call__(self, text: str) -> list[int]
```

#### Class `SyntheticTokenDataset` {#api-speedtronic-data-SyntheticTokenDataset}

```python
class SyntheticTokenDataset(Dataset[dict[str, torch.Tensor]])
```

A reproducible random-token dataset for smoke tests and examples.

##### Method `__init__` {#api-speedtronic-data-SyntheticTokenDataset-__init__}

```python
def __init__(self, num_samples: int=10000, block_size: int=128, vocab_size: int=512, seed: int=1234) -> None
```

##### Method `__len__` {#api-speedtronic-data-SyntheticTokenDataset-__len__}

```python
def __len__(self) -> int
```

##### Method `__getitem__` {#api-speedtronic-data-SyntheticTokenDataset-__getitem__}

```python
def __getitem__(self, index: int) -> dict[str, torch.Tensor]
```

#### Class `TextFileTokenDataset` {#api-speedtronic-data-TextFileTokenDataset}

```python
class TextFileTokenDataset(IterableDataset[dict[str, torch.Tensor]])
```

Stream a text file in fixed-size token blocks.

``tokenizer`` may be a callable or an object with ``encode``.  The file is
read incrementally; only one block is materialized at a time.

##### Method `__init__` {#api-speedtronic-data-TextFileTokenDataset-__init__}

```python
def __init__(self, path: str | Path, block_size: int, tokenizer: Callable[[str], Any] | Any | None=None, vocab_size: int=256) -> None
```

##### Method `_encode` {#api-speedtronic-data-TextFileTokenDataset-_encode}

```python
def _encode(self, text: str) -> list[int]
```

##### Method `__iter__` {#api-speedtronic-data-TextFileTokenDataset-__iter__}

```python
def __iter__(self) -> Iterator[dict[str, torch.Tensor]]
```

#### Function `collate_causal` {#api-speedtronic-data-collate_causal}

```python
def collate_causal(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]
```

#### Function `collate_batch` {#api-speedtronic-data-collate_batch}

```python
def collate_batch(batch: list[Any]) -> Any
```

Collate causal dictionaries or homogeneous tuple/list datasets.

#### Function `_cycle` {#api-speedtronic-data-_cycle}

```python
def _cycle(loader: Iterable[Any]) -> Iterator[Any]
```

#### Function `infinite_batches` {#api-speedtronic-data-infinite_batches}

```python
def infinite_batches(loader: Iterable[Any]) -> Iterator[Any]
```

Cycle a finite loader, making max-step runs independent of data size.

#### Function `build_dataloader` {#api-speedtronic-data-build_dataloader}

```python
def build_dataloader(config: Any, *, dataset: Dataset | IterableDataset | None=None, tokenizer: Any | None=None, pin_memory_device: bool=False, vocab_size: int | None=None) -> DataLoader
```

Build a loader from a DataConfig.

``dataset`` is the primary extension point for user-provided datasets.  A
text path takes precedence over the synthetic fallback unless a dataset is
explicitly supplied.

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `__all__` | 261 |


### `src/speedtronic/distributed/__init__.py`

**Import path:** `speedtronic.distributed`<br />
**Purpose:** DumbDiLoCo public API..<br />
**Lines:** 31

**Imported modules:** `.diloco.DeltaUploadJob`, `.diloco.DumbDiLoCo`, `.diloco.DumbDiLoCoCoordinator`, `.diloco.SyncResult`, `.hub.GlobalMetadata`, `.hub.HubClient`, `.hub.HubError`, `.hub.HubTransport`, `.hub.HubUnavailable`, `.outer.MasterOuterLoop`, `.outer.NesterovOuterOptimizer`, `.tensors.average_deltas`, `.tensors.compute_pseudo_gradient`, `.tensors.load_delta`, `.tensors.load_safetensors`, `.tensors.save_safetensors`

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `__all__` | 14 |


### `src/speedtronic/distributed/diloco.py`

**Import path:** `speedtronic.distributed.diloco`<br />
**Purpose:** DumbDiLoCo coordinator integrated with the ordinary training loop..<br />
**Lines:** 723

**Imported modules:** `__future__.annotations`, `logging`, `os`, `queue`, `threading`, `time`, `dataclasses.dataclass`, `pathlib.Path`, `typing.Any`, `torch`, `.hub.GlobalMetadata`, `.hub.HubClient`, `.outer.MasterOuterLoop`, `.tensors.compute_pseudo_gradient`, `.tensors.cpu_state_dict`, `.tensors.load_safetensors`

#### Class `SyncResult` {#api-speedtronic-distributed-diloco-SyncResult}

```python
class SyncResult
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `pushed` | 25 |
| field | `loaded_global` | 26 |
| field | `outer_step` | 27 |

#### Class `DeltaUploadJob` {#api-speedtronic-distributed-diloco-DeltaUploadJob}

```python
class DeltaUploadJob
```

Immutable CPU snapshot dispatched to the asynchronous uploader.

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `step` | 34 |
| field | `base_outer_step` | 35 |
| field | `generation` | 36 |
| field | `delta` | 37 |
| field | `target_state` | 38 |

#### Class `GlobalCandidate` {#api-speedtronic-distributed-diloco-GlobalCandidate}

```python
class GlobalCandidate
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `outer_step` | 43 |
| field | `state` | 44 |

#### Class `DumbDiLoCoCoordinator` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator}

```python
class DumbDiLoCoCoordinator
```

Coordinate local inner loops and a Hub-backed outer loop.

The trainer calls :meth:`after_optimizer_step` after each local optimizer
step.  At ``inner_steps`` boundaries the coordinator computes and uploads a
pseudo-gradient; at other steps it performs a cheap time-based poll for a
newer global version.  The master additionally owns a background
:class:`MasterOuterLoop` thread.

##### Method `__init__` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-__init__}

```python
def __init__(self, config: Any, model: torch.nn.Module, *, hub: HubClient | None=None, state_dir: str | Path | None=None, logger: logging.Logger | None=None) -> None
```

##### Method `local_step` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-local_step}

```python
def local_step(self) -> int
```

##### Method `last_global_step` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-last_global_step}

```python
def last_global_step(self) -> int
```

##### Method `outer_step` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-outer_step}

```python
def outer_step(self) -> int
```

##### Method `_emit_event` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_emit_event}

```python
def _emit_event(self, event: str, payload: dict[str, Any]) -> None
```

##### Method `_start_io_workers` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_start_io_workers}

```python
def _start_io_workers(self) -> None
```

##### Method `start` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-start}

```python
def start(self) -> None
```

##### Method `_start_master` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_start_master}

```python
def _start_master(self) -> None
```

##### Method `_start_worker` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_start_worker}

```python
def _start_worker(self) -> None
```

##### Method `_load_global_state` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_load_global_state}

```python
def _load_global_state(self, metadata: GlobalMetadata) -> dict[str, torch.Tensor]
```

##### Method `_refresh_from_hub` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_refresh_from_hub}

```python
def _refresh_from_hub(self, metadata: GlobalMetadata) -> bool
```

##### Method `_install_state` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_install_state}

```python
def _install_state(self, state: dict[str, torch.Tensor]) -> None
```

##### Method `_on_outer_update` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_on_outer_update}

```python
def _on_outer_update(self, state: dict[str, torch.Tensor], outer_step: int) -> None
```

##### Method `_refresh_master_snapshot` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_refresh_master_snapshot}

```python
def _refresh_master_snapshot(self) -> bool
```

##### Method `_install_async_candidate` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_install_async_candidate}

```python
def _install_async_candidate(self) -> bool
```

##### Method `_fetch_global_candidate` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_fetch_global_candidate}

```python
def _fetch_global_candidate(self) -> GlobalCandidate \| None
```

##### Method `_schedule_async_poll` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_schedule_async_poll}

```python
def _schedule_async_poll(self, *, force: bool) -> bool
```

##### Method `_poll_worker` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_poll_worker}

```python
def _poll_worker(self) -> None
```

##### Method `_consume_async_poll` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_consume_async_poll}

```python
def _consume_async_poll(self) -> None
```

##### Method `_maybe_poll_global` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_maybe_poll_global}

```python
def _maybe_poll_global(self, *, force: bool=False) -> bool
```

##### Method `_upload_delta_sync` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_upload_delta_sync}

```python
def _upload_delta_sync(self, step: int) -> bool
```

##### Method `_dispatch_delta` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_dispatch_delta}

```python
def _dispatch_delta(self, step: int) -> bool
```

##### Method `_upload_worker` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_upload_worker}

```python
def _upload_worker(self) -> None
```

##### Method `_upload_delta` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_upload_delta}

```python
def _upload_delta(self, step: int) -> bool
```

##### Method `after_optimizer_step` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-after_optimizer_step}

```python
def after_optimizer_step(self, model: torch.nn.Module, step: int) -> bool
```

Handle a local step; return whether the inner optimizer is reset.

##### Method `state_dict` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-state_dict}

```python
def state_dict(self) -> dict[str, Any]
```

##### Method `_restore_pending_upload` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_restore_pending_upload}

```python
def _restore_pending_upload(self, pending: dict[str, Any] | None) -> None
```

##### Method `load_state_dict` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-load_state_dict}

```python
def load_state_dict(self, state: dict[str, Any]) -> None
```

##### Method `prepare_state` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-prepare_state}

```python
def prepare_state(self, state: dict[str, Any]) -> None
```

Stage resume state before startup performs any remote refresh.

##### Method `_apply_prepared_state` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-_apply_prepared_state}

```python
def _apply_prepared_state(self) -> None
```

##### Method `stop` {#api-speedtronic-distributed-diloco-DumbDiLoCoCoordinator-stop}

```python
def stop(self) -> None
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `LOGGER` | 20 |
| constant/alias | `DumbDiLoCo` | 720 |
| constant/alias | `__all__` | 723 |


### `src/speedtronic/distributed/hub.py`

**Import path:** `speedtronic.distributed.hub`<br />
**Purpose:** Hugging Face Hub transport used as the DumbDiLoCo synchronization bus..<br />
**Lines:** 329

**Imported modules:** `__future__.annotations`, `json`, `os`, `shutil`, `tempfile`, `time`, `dataclasses.dataclass`, `pathlib.Path`, `typing.Any`, `typing.Callable`, `.tensors.save_safetensors`

#### Class `HubError` {#api-speedtronic-distributed-hub-HubError}

```python
class HubError(RuntimeError)
```

#### Class `HubUnavailable` {#api-speedtronic-distributed-hub-HubUnavailable}

```python
class HubUnavailable(HubError)
```

#### Class `GlobalMetadata` {#api-speedtronic-distributed-hub-GlobalMetadata}

```python
class GlobalMetadata
```

##### Method `from_dict` {#api-speedtronic-distributed-hub-GlobalMetadata-from_dict}

```python
def from_dict(cls, value: dict[str, Any]) -> 'GlobalMetadata'
```

##### Method `to_dict` {#api-speedtronic-distributed-hub-GlobalMetadata-to_dict}

```python
def to_dict(self) -> dict[str, Any]
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `outer_step` | 27 |
| field | `updated_at` | 28 |
| field | `model_file` | 29 |

#### Class `HubClient` {#api-speedtronic-distributed-hub-HubClient}

```python
class HubClient
```

A small retrying wrapper around ``huggingface_hub``.

The wrapper intentionally exposes only repository-file operations.  It is
easy to replace with a fake object in tests and keeps the distributed code
independent of Hub SDK version details.

##### Method `__init__` {#api-speedtronic-distributed-hub-HubClient-__init__}

```python
def __init__(self, repo_id: str, *, token: str | None=None, cache_dir: str | os.PathLike[str] | None=None, api: Any | None=None, retry_initial: float=1.0, retry_max: float=60.0, retry_attempts: int=6, sleeper: Callable[[float], None]=time.sleep) -> None
```

##### Method `api` {#api-speedtronic-distributed-hub-HubClient-api}

```python
def api(self) -> Any
```

##### Method `_retry` {#api-speedtronic-distributed-hub-HubClient-_retry}

```python
def _retry(self, operation: str, callback: Callable[[], Any]) -> Any
```

##### Method `create_repo` {#api-speedtronic-distributed-hub-HubClient-create_repo}

```python
def create_repo(self, *, private: bool=True) -> Any
```

##### Method `add_collaborator` {#api-speedtronic-distributed-hub-HubClient-add_collaborator}

```python
def add_collaborator(self, username: str, permission: str='write') -> Any
```

##### Method `list_files` {#api-speedtronic-distributed-hub-HubClient-list_files}

```python
def list_files(self) -> list[str]
```

##### Method `file_exists` {#api-speedtronic-distributed-hub-HubClient-file_exists}

```python
def file_exists(self, remote_path: str) -> bool
```

##### Method `upload` {#api-speedtronic-distributed-hub-HubClient-upload}

```python
def upload(self, local_path: str | os.PathLike[str], remote_path: str) -> Any
```

##### Method `download` {#api-speedtronic-distributed-hub-HubClient-download}

```python
def download(self, remote_path: str, local_path: str | os.PathLike[str] | None=None) -> Path
```

##### Method `read_json` {#api-speedtronic-distributed-hub-HubClient-read_json}

```python
def read_json(self, remote_path: str, *, default: Any=None) -> Any
```

##### Method `write_json` {#api-speedtronic-distributed-hub-HubClient-write_json}

```python
def write_json(self, remote_path: str, value: Any) -> None
```

##### Method `global_metadata` {#api-speedtronic-distributed-hub-HubClient-global_metadata}

```python
def global_metadata(self) -> GlobalMetadata \| None
```

##### Method `download_global` {#api-speedtronic-distributed-hub-HubClient-download_global}

```python
def download_global(self, local_path: str | os.PathLike[str], metadata: GlobalMetadata | None=None) -> Path
```

##### Method `publish_global` {#api-speedtronic-distributed-hub-HubClient-publish_global}

```python
def publish_global(self, state: dict[str, Any], *, outer_step: int, work_dir: str | os.PathLike[str] | None=None, updated_at: str | None=None, metadata_extra: dict[str, str] | None=None) -> GlobalMetadata
```

##### Method `upload_delta` {#api-speedtronic-distributed-hub-HubClient-upload_delta}

```python
def upload_delta(self, state: dict[str, Any], *, node_id: str, local_step: int, base_outer_step: int, work_dir: str | os.PathLike[str] | None=None, metadata: dict[str, str] | None=None) -> str
```

##### Method `delta_paths` {#api-speedtronic-distributed-hub-HubClient-delta_paths}

```python
def delta_paths(self) -> list[str]
```

#### Function `_utc_now` {#api-speedtronic-distributed-hub-_utc_now}

```python
def _utc_now() -> str
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `HubTransport` | 320 |
| constant/alias | `__all__` | 329 |


### `src/speedtronic/distributed/outer.py`

**Import path:** `speedtronic.distributed.outer`<br />
**Purpose:** Outer-loop optimizer and master polling loop for DumbDiLoCo..<br />
**Lines:** 361

**Imported modules:** `__future__.annotations`, `logging`, `threading`, `dataclasses.dataclass`, `dataclasses.field`, `datetime.datetime`, `datetime.timezone`, `pathlib.Path`, `typing.Any`, `typing.Callable`, `torch`, `..checkpoint.atomic_json_dump`, `..checkpoint.atomic_torch_save`, `.hub.HubClient`, `.tensors.average_deltas`, `.tensors.load_delta`, `.tensors.parse_delta_path`

#### Class `NesterovOuterOptimizer` {#api-speedtronic-distributed-outer-NesterovOuterOptimizer}

```python
class NesterovOuterOptimizer
```

Nesterov momentum SGD over a floating-point model state.

##### Method `__init__` {#api-speedtronic-distributed-outer-NesterovOuterOptimizer-__init__}

```python
def __init__(self, state: dict[str, torch.Tensor], lr: float, momentum: float=0.9) -> None
```

##### Method `step` {#api-speedtronic-distributed-outer-NesterovOuterOptimizer-step}

```python
def step(self, state: dict[str, torch.Tensor], delta: dict[str, torch.Tensor]) -> None
```

Apply one outer update in place on CPU state tensors.

##### Method `state_dict` {#api-speedtronic-distributed-outer-NesterovOuterOptimizer-state_dict}

```python
def state_dict(self) -> dict[str, torch.Tensor]
```

##### Method `load_state_dict` {#api-speedtronic-distributed-outer-NesterovOuterOptimizer-load_state_dict}

```python
def load_state_dict(self, state: dict[str, torch.Tensor]) -> None
```

#### Class `OuterState` {#api-speedtronic-distributed-outer-OuterState}

```python
class OuterState
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `global_state` | 65 |
| field | `outer_step` | 66 |
| field | `processed_deltas` | 67 |
| field | `momentum` | 68 |
| field | `last_error` | 69 |

#### Class `MasterOuterLoop` {#api-speedtronic-distributed-outer-MasterOuterLoop}

```python
class MasterOuterLoop
```

Poll and aggregate deltas without coupling polling to inner training.

The loop owns a CPU copy of the global state and publishes a complete model
plus metadata after every successful round.  A private thread is used for
the master role, so the training thread can continue its local loop.

##### Method `__init__` {#api-speedtronic-distributed-outer-MasterOuterLoop-__init__}

```python
def __init__(self, hub: HubClient, state: dict[str, torch.Tensor], *, node_state_dir: str | Path, outer_lr: float=0.7, outer_momentum: float=0.9, poll_interval: float=60.0, on_global_update: Callable[[dict[str, torch.Tensor], int], None] | None=None, logger: logging.Logger | None=None) -> None
```

##### Method `processed_filenames` {#api-speedtronic-distributed-outer-MasterOuterLoop-processed_filenames}

```python
def processed_filenames(self) -> set[str]
```

##### Method `state_path` {#api-speedtronic-distributed-outer-MasterOuterLoop-state_path}

```python
def state_path(self) -> Path
```

##### Method `metadata_path` {#api-speedtronic-distributed-outer-MasterOuterLoop-metadata_path}

```python
def metadata_path(self) -> Path
```

##### Method `_load_local_state` {#api-speedtronic-distributed-outer-MasterOuterLoop-_load_local_state}

```python
def _load_local_state(self) -> None
```

##### Method `_save_local_state` {#api-speedtronic-distributed-outer-MasterOuterLoop-_save_local_state}

```python
def _save_local_state(self) -> None
```

##### Method `_delta_candidates` {#api-speedtronic-distributed-outer-MasterOuterLoop-_delta_candidates}

```python
def _delta_candidates(self) -> list[tuple[str, int, int, str]]
```

##### Method `_download_delta` {#api-speedtronic-distributed-outer-MasterOuterLoop-_download_delta}

```python
def _download_delta(self, path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]
```

##### Method `sync_once` {#api-speedtronic-distributed-outer-MasterOuterLoop-sync_once}

```python
def sync_once(self) -> int
```

Run one outer round and return the resulting outer step.

Hub listing and downloads happen outside the state lock.  Only the
short in-memory commit and local-state write are serialized, so a
training-thread checkpoint never waits for a network round trip.

##### Method `_emit_outer_event` {#api-speedtronic-distributed-outer-MasterOuterLoop-_emit_outer_event}

```python
def _emit_outer_event(self, found: int, valid: int) -> None
```

##### Method `adopt_global_state` {#api-speedtronic-distributed-outer-MasterOuterLoop-adopt_global_state}

```python
def adopt_global_state(self, state: dict[str, torch.Tensor], outer_step: int, *, reset_momentum: bool=False) -> None
```

Recover a newer remotely published global state after a crash.

##### Method `snapshot` {#api-speedtronic-distributed-outer-MasterOuterLoop-snapshot}

```python
def snapshot(self) -> tuple[int, dict[str, torch.Tensor]]
```

##### Method `start` {#api-speedtronic-distributed-outer-MasterOuterLoop-start}

```python
def start(self) -> None
```

##### Method `_run` {#api-speedtronic-distributed-outer-MasterOuterLoop-_run}

```python
def _run(self) -> None
```

##### Method `stop` {#api-speedtronic-distributed-outer-MasterOuterLoop-stop}

```python
def stop(self) -> None
```

##### Method `state_dict` {#api-speedtronic-distributed-outer-MasterOuterLoop-state_dict}

```python
def state_dict(self) -> dict[str, Any]
```

##### Method `load_state_dict` {#api-speedtronic-distributed-outer-MasterOuterLoop-load_state_dict}

```python
def load_state_dict(self, state: dict[str, Any]) -> None
```

#### Function `_utc_now` {#api-speedtronic-distributed-outer-_utc_now}

```python
def _utc_now() -> str
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `LOGGER` | 18 |
| constant/alias | `__all__` | 361 |


### `src/speedtronic/distributed/tensors.py`

**Import path:** `speedtronic.distributed.tensors`<br />
**Purpose:** Safetensors helpers with defensive tensor normalization..<br />
**Lines:** 148

**Imported modules:** `__future__.annotations`, `json`, `os`, `pathlib.Path`, `typing.Any`, `torch`

#### Function `cpu_tensor` {#api-speedtronic-distributed-tensors-cpu_tensor}

```python
def cpu_tensor(value: torch.Tensor) -> torch.Tensor
```

#### Function `cpu_state_dict` {#api-speedtronic-distributed-tensors-cpu_state_dict}

```python
def cpu_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]
```

#### Function `save_safetensors` {#api-speedtronic-distributed-tensors-save_safetensors}

```python
def save_safetensors(state: dict[str, torch.Tensor], path: str | os.PathLike[str], *, metadata: dict[str, str] | None=None) -> None
```

#### Function `load_safetensors` {#api-speedtronic-distributed-tensors-load_safetensors}

```python
def load_safetensors(path: str | os.PathLike[str]) -> dict[str, torch.Tensor]
```

#### Function `read_metadata` {#api-speedtronic-distributed-tensors-read_metadata}

```python
def read_metadata(path: str | os.PathLike[str]) -> dict[str, str]
```

#### Function `floating_state` {#api-speedtronic-distributed-tensors-floating_state}

```python
def floating_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]
```

#### Function `compute_pseudo_gradient` {#api-speedtronic-distributed-tensors-compute_pseudo_gradient}

```python
def compute_pseudo_gradient(baseline: dict[str, torch.Tensor], current: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]
```

Compute ``baseline - current`` for floating-point state tensors.

#### Function `average_deltas` {#api-speedtronic-distributed-tensors-average_deltas}

```python
def average_deltas(deltas: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]
```

#### Function `parse_delta_path` {#api-speedtronic-distributed-tensors-parse_delta_path}

```python
def parse_delta_path(path: str) -> tuple[str, int] \| None
```

#### Function `load_delta` {#api-speedtronic-distributed-tensors-load_delta}

```python
def load_delta(path: str | os.PathLike[str]) -> tuple[dict[str, torch.Tensor], dict[str, str]]
```

#### Function `metadata_json` {#api-speedtronic-distributed-tensors-metadata_json}

```python
def metadata_json(metadata: dict[str, str]) -> dict[str, Any]
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `__all__` | 136 |


### `src/speedtronic/hooks.py`

**Import path:** `speedtronic.hooks`<br />
**Purpose:** Optional hook and callback helpers..<br />
**Lines:** 17

**Imported modules:** `__future__.annotations`, `typing.Any`, `typing.Callable`

#### Function `on_event` {#api-speedtronic-hooks-on_event}

```python
def on_event(callback: Callable[[str, dict[str, Any]], None])
```

Return a callback wrapper suitable for ``MetricLogger(hooks=...)``.

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `__all__` | 17 |


### `src/speedtronic/integrations.py`

**Import path:** `speedtronic.integrations`<br />
**Purpose:** Optional logging integrations loaded only when explicitly requested..<br />
**Lines:** 38

**Imported modules:** `__future__.annotations`, `typing.Any`

#### Class `WandbHook` {#api-speedtronic-integrations-WandbHook}

```python
class WandbHook
```

##### Method `__init__` {#api-speedtronic-integrations-WandbHook-__init__}

```python
def __init__(self, project: str | None=None, **settings: Any) -> None
```

##### Method `on_event` {#api-speedtronic-integrations-WandbHook-on_event}

```python
def on_event(self, event: str, payload: dict[str, Any]) -> None
```

#### Class `TensorboardHook` {#api-speedtronic-integrations-TensorboardHook}

```python
class TensorboardHook
```

##### Method `__init__` {#api-speedtronic-integrations-TensorboardHook-__init__}

```python
def __init__(self, log_dir: str='runs') -> None
```

##### Method `on_event` {#api-speedtronic-integrations-TensorboardHook-on_event}

```python
def on_event(self, event: str, payload: dict[str, Any]) -> None
```

##### Method `close` {#api-speedtronic-integrations-TensorboardHook-close}

```python
def close(self) -> None
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `__all__` | 38 |


### `src/speedtronic/model.py`

**Import path:** `speedtronic.model`<br />
**Purpose:** Reference decoder-only transformer used by the examples and smoke tests..<br />
**Lines:** 345

**Imported modules:** `__future__.annotations`, `dataclasses.dataclass`, `typing.Any`, `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.checkpoint.checkpoint`, `.registry.register_model`

#### Class `GPTConfig` {#api-speedtronic-model-GPTConfig}

```python
class GPTConfig
```

##### Method `__post_init__` {#api-speedtronic-model-GPTConfig-__post_init__}

```python
def __post_init__(self) -> None
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `vocab_size` | 18 |
| field | `block_size` | 19 |
| field | `n_layer` | 20 |
| field | `n_head` | 21 |
| field | `n_kv_head` | 22 |
| field | `d_model` | 23 |
| field | `d_ff` | 24 |
| field | `dropout` | 25 |
| field | `tie_weights` | 26 |
| field | `rope_base` | 27 |

#### Class `RMSNorm` {#api-speedtronic-model-RMSNorm}

```python
class RMSNorm(nn.Module)
```

##### Method `__init__` {#api-speedtronic-model-RMSNorm-__init__}

```python
def __init__(self, dim: int, eps: float=1e-05) -> None
```

##### Method `forward` {#api-speedtronic-model-RMSNorm-forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

#### Function `_rotate_half` {#api-speedtronic-model-_rotate_half}

```python
def _rotate_half(x: torch.Tensor) -> torch.Tensor
```

#### Function `apply_rope` {#api-speedtronic-model-apply_rope}

```python
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor
```

Apply rotary embeddings to ``x`` with shape ``(B, H, T, D)``.

#### Class `RotaryEmbedding` {#api-speedtronic-model-RotaryEmbedding}

```python
class RotaryEmbedding(nn.Module)
```

##### Method `__init__` {#api-speedtronic-model-RotaryEmbedding-__init__}

```python
def __init__(self, head_dim: int, base: float=10000.0) -> None
```

##### Method `forward` {#api-speedtronic-model-RotaryEmbedding-forward}

```python
def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]
```

#### Class `CausalSelfAttention` {#api-speedtronic-model-CausalSelfAttention}

```python
class CausalSelfAttention(nn.Module)
```

##### Method `__init__` {#api-speedtronic-model-CausalSelfAttention-__init__}

```python
def __init__(self, config: GPTConfig) -> None
```

##### Method `forward` {#api-speedtronic-model-CausalSelfAttention-forward}

```python
def forward(self, x: torch.Tensor, rope: RotaryEmbedding, attention_mask: torch.Tensor | None=None) -> torch.Tensor
```

#### Class `SwiGLU` {#api-speedtronic-model-SwiGLU}

```python
class SwiGLU(nn.Module)
```

##### Method `__init__` {#api-speedtronic-model-SwiGLU-__init__}

```python
def __init__(self, config: GPTConfig) -> None
```

##### Method `forward` {#api-speedtronic-model-SwiGLU-forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

#### Class `TransformerBlock` {#api-speedtronic-model-TransformerBlock}

```python
class TransformerBlock(nn.Module)
```

##### Method `__init__` {#api-speedtronic-model-TransformerBlock-__init__}

```python
def __init__(self, config: GPTConfig) -> None
```

##### Method `forward` {#api-speedtronic-model-TransformerBlock-forward}

```python
def forward(self, x: torch.Tensor, rope: RotaryEmbedding, attention_mask: torch.Tensor | None=None) -> torch.Tensor
```

##### Method `_forward` {#api-speedtronic-model-TransformerBlock-_forward}

```python
def _forward(self, x: torch.Tensor, rope: RotaryEmbedding, attention_mask: torch.Tensor | None=None) -> torch.Tensor
```

#### Class `ReferenceTransformer` {#api-speedtronic-model-ReferenceTransformer}

```python
class ReferenceTransformer(nn.Module)
```

A compact GPT-style model with RoPE, GQA, SwiGLU, and weight tying.

##### Method `__init__` {#api-speedtronic-model-ReferenceTransformer-__init__}

```python
def __init__(self, **kwargs: Any) -> None
```

##### Method `_init_weights` {#api-speedtronic-model-ReferenceTransformer-_init_weights}

```python
def _init_weights(module: nn.Module) -> None
```

##### Method `set_gradient_checkpointing` {#api-speedtronic-model-ReferenceTransformer-set_gradient_checkpointing}

```python
def set_gradient_checkpointing(self, enabled: bool=True) -> None
```

##### Method `forward` {#api-speedtronic-model-ReferenceTransformer-forward}

```python
def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None=None, attention_mask: torch.Tensor | None=None, **_: Any) -> dict[str, torch.Tensor]
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `GPT` | 324 |
| constant/alias | `Transformer` | 325 |
| constant/alias | `ReferenceModel` | 326 |
| constant/alias | `GPTModel` | 327 |
| constant/alias | `TransformerConfig` | 328 |
| constant/alias | `__all__` | 331 |


### `src/speedtronic/module_utils.py`

**Import path:** `speedtronic.module_utils`<br />
**Purpose:** Small public helpers for user modules..<br />
**Lines:** 24

**Imported modules:** `__future__.annotations`, `typing.Any`

#### Function `set_gradient_checkpointing` {#api-speedtronic-module_utils-set_gradient_checkpointing}

```python
def set_gradient_checkpointing(module: Any, enabled: bool=True) -> Any
```

Enable a module's explicit gradient-checkpointing convention.

Speedtronic intentionally does not inspect transformer internals.  A custom
module opts in by exposing ``set_gradient_checkpointing(enabled)``.

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `__all__` | 24 |


### `src/speedtronic/optimizers.py`

**Import path:** `speedtronic.optimizers`<br />
**Purpose:** v2 optimizers: pure-PyTorch Muon, Muon+, cautious wrapping, and hybrid routing..<br />
**Lines:** 552

**Imported modules:** `__future__.annotations`, `dataclasses.dataclass`, `typing.Any`, `typing.Iterable`, `typing.Mapping`, `typing.Sequence`, `torch`, `torch.nn`, `torch.optim.Optimizer`

#### Class `ParameterRouting` {#api-speedtronic-optimizers-ParameterRouting}

```python
class ParameterRouting
```

Deterministic parameter ownership for the hybrid optimizer.

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `muon` | 17 |
| field | `adamw` | 18 |
| field | `skipped` | 19 |

#### Function `_work_dtype` {#api-speedtronic-optimizers-_work_dtype}

```python
def _work_dtype(value: torch.Tensor) -> torch.dtype
```

#### Function `newton_schulz` {#api-speedtronic-optimizers-newton_schulz}

```python
def newton_schulz(matrix: torch.Tensor, steps: int=5, eps: float=1e-07) -> torch.Tensor
```

Approximate the orthogonal polar factor with a quintic iteration.

The implementation intentionally uses only PyTorch matrix operations.  It
does not require a custom CUDA extension, SVD, or a device-specific package.

#### Function `post_polar_normalize` {#api-speedtronic-optimizers-post_polar_normalize}

```python
def post_polar_normalize(update: torch.Tensor, *, mode: str='row_col', eps: float=1e-08) -> torch.Tensor
```

Apply Muon+'s inexpensive post-orthogonalization normalization.

#### Function `_parameter_owners` {#api-speedtronic-optimizers-_parameter_owners}

```python
def _parameter_owners(model: nn.Module) -> Mapping[int, tuple[str, nn.Module, str]]
```

#### Function `_is_embedding_or_head` {#api-speedtronic-optimizers-_is_embedding_or_head}

```python
def _is_embedding_or_head(name: str, module: nn.Module) -> bool
```

#### Function `route_parameters` {#api-speedtronic-optimizers-route_parameters}

```python
def route_parameters(model: nn.Module) -> ParameterRouting
```

Route trainable parameters to Muon or AdamW by role and dimensionality.

Hidden ``nn.Linear`` matrices are eligible for Muon. Embeddings, heads,
normalization parameters, biases, non-2D tensors, and frozen parameters are
sent to AdamW or omitted.  A module can opt into a route with
``_speedtronic_optimizer_role = "muon"`` or ``"adamw"``.

#### Class `Muon` {#api-speedtronic-optimizers-Muon}

```python
class Muon(Optimizer)
```

A pure-PyTorch Muon optimizer for routed 2-D parameter groups.

##### Method `__init__` {#api-speedtronic-optimizers-Muon-__init__}

```python
def __init__(self, params: Iterable[torch.Tensor], lr: float=0.0003, momentum: float=0.95, nesterov: bool=True, ns_steps: int=5, weight_decay: float=0.0, eps: float=1e-07, norm_eps: float=1e-08, muon_plus: bool=False) -> None
```

##### Method `step` {#api-speedtronic-optimizers-Muon-step}

```python
def step(self, closure: Any | None=None) -> Any
```

#### Class `HybridOptimizer` {#api-speedtronic-optimizers-HybridOptimizer}

```python
class HybridOptimizer(Optimizer)
```

One optimizer facade combining Muon matrices and AdamW parameters.

##### Method `__init__` {#api-speedtronic-optimizers-HybridOptimizer-__init__}

```python
def __init__(self, muon_params: Sequence[torch.Tensor], adamw_params: Sequence[torch.Tensor], *, muon_lr: float, adamw_lr: float, betas: tuple[float, float], eps: float, weight_decay: float, fused: bool=False, muon_momentum: float=0.95, muon_ns_steps: int=5, muon_norm_eps: float=1e-08, muon_plus: bool=False) -> None
```

##### Method `_bind_children` {#api-speedtronic-optimizers-HybridOptimizer-_bind_children}

```python
def _bind_children(self) -> None
```

##### Method `_sync_children` {#api-speedtronic-optimizers-HybridOptimizer-_sync_children}

```python
def _sync_children(self) -> None
```

##### Method `step` {#api-speedtronic-optimizers-HybridOptimizer-step}

```python
def step(self, closure: Any | None=None) -> Any
```

##### Method `load_state_dict` {#api-speedtronic-optimizers-HybridOptimizer-load_state_dict}

```python
def load_state_dict(self, state_dict: dict[str, Any]) -> None
```

#### Class `CautiousOptimizer` {#api-speedtronic-optimizers-CautiousOptimizer}

```python
class CautiousOptimizer(Optimizer)
```

Composable cautious wrapper around a Speedtronic/native optimizer.

The wrapper snapshots parameters, lets the base optimizer calculate its
update, and masks entries whose observed update does not align with the
current gradient.  It intentionally supports arbitrary base optimizers;
native AdamW and Muon remain available for exact direction-specific
integrations.

##### Method `__init__` {#api-speedtronic-optimizers-CautiousOptimizer-__init__}

```python
def __init__(self, base: Optimizer) -> None
```

##### Method `step` {#api-speedtronic-optimizers-CautiousOptimizer-step}

```python
def step(self, closure: Any | None=None) -> Any
```

##### Method `zero_grad` {#api-speedtronic-optimizers-CautiousOptimizer-zero_grad}

```python
def zero_grad(self, set_to_none: bool=True) -> None
```

##### Method `state_dict` {#api-speedtronic-optimizers-CautiousOptimizer-state_dict}

```python
def state_dict(self) -> dict[str, Any]
```

##### Method `load_state_dict` {#api-speedtronic-optimizers-CautiousOptimizer-load_state_dict}

```python
def load_state_dict(self, state_dict: dict[str, Any]) -> None
```

##### Method `add_param_group` {#api-speedtronic-optimizers-CautiousOptimizer-add_param_group}

```python
def add_param_group(self, param_group: dict[str, Any]) -> None
```

#### Function `build_v2_optimizer` {#api-speedtronic-optimizers-build_v2_optimizer}

```python
def build_v2_optimizer(model: nn.Module, config: Any, device: torch.device) -> Optimizer
```

Build AdamW, hybrid Muon/AdamW, and cautious variants from config.

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `__all__` | 543 |


### `src/speedtronic/precision.py`

**Import path:** `speedtronic.precision`<br />
**Purpose:** Hardware capability detection and mixed-precision helpers..<br />
**Lines:** 147

**Imported modules:** `__future__.annotations`, `contextlib.nullcontext`, `dataclasses.dataclass`, `typing.Any`

#### Class `PrecisionPlan` {#api-speedtronic-precision-PrecisionPlan}

```python
class PrecisionPlan
```

##### Method `is_mixed_precision` {#api-speedtronic-precision-PrecisionPlan-is_mixed_precision}

```python
def is_mixed_precision(self) -> bool
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `mode` | 12 |
| field | `dtype` | 13 |
| field | `use_scaler` | 14 |
| field | `autocast_enabled` | 15 |
| field | `device_type` | 16 |

#### Function `resolve_device` {#api-speedtronic-precision-resolve_device}

```python
def resolve_device(requested: str | Any='auto') -> Any
```

#### Function `resolve_precision` {#api-speedtronic-precision-resolve_precision}

```python
def resolve_precision(config: Any, device: Any) -> PrecisionPlan
```

Resolve a precision config against the actual device.

The explicit mode wins, except that an explicit unsupported mixed mode
falls back to a safe mode rather than making a run unusable.

#### Function `autocast_context` {#api-speedtronic-precision-autocast_context}

```python
def autocast_context(plan: PrecisionPlan, device: Any)
```

#### Function `make_grad_scaler` {#api-speedtronic-precision-make_grad_scaler}

```python
def make_grad_scaler(plan: PrecisionPlan)
```

#### Function `supports_fused_adamw` {#api-speedtronic-precision-supports_fused_adamw}

```python
def supports_fused_adamw(device: Any) -> bool
```

#### Function `parameter_count` {#api-speedtronic-precision-parameter_count}

```python
def parameter_count(model: Any) -> int
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `__all__` | 140 |


### `src/speedtronic/profiling.py`

**Import path:** `speedtronic.profiling`<br />
**Purpose:** Lightweight stdout/file metrics and pluggable training hooks..<br />
**Lines:** 147

**Imported modules:** `__future__.annotations`, `json`, `logging`, `sys`, `threading`, `time`, `dataclasses.dataclass`, `pathlib.Path`, `typing.Any`, `typing.Callable`, `typing.Protocol`, `typing.TextIO`

#### Class `TrainingHook` {#api-speedtronic-profiling-TrainingHook}

```python
class TrainingHook(Protocol)
```

##### Method `on_event` {#api-speedtronic-profiling-TrainingHook-on_event}

```python
def on_event(self, event: str, payload: dict[str, Any]) -> None
```

#### Function `memory_usage_bytes` {#api-speedtronic-profiling-memory_usage_bytes}

```python
def memory_usage_bytes() -> int \| None
```

Return allocated accelerator memory when the backend exposes it.

#### Class `MetricLogger` {#api-speedtronic-profiling-MetricLogger}

```python
class MetricLogger
```

##### Method `__post_init__` {#api-speedtronic-profiling-MetricLogger-__post_init__}

```python
def __post_init__(self) -> None
```

##### Method `emit` {#api-speedtronic-profiling-MetricLogger-emit}

```python
def emit(self, event: str, payload: Metric) -> None
```

##### Method `_emit_locked` {#api-speedtronic-profiling-MetricLogger-_emit_locked}

```python
def _emit_locked(self, event: str, payload: Metric) -> None
```

##### Method `log` {#api-speedtronic-profiling-MetricLogger-log}

```python
def log(self, level: int, message: str) -> None
```

##### Method `debug` {#api-speedtronic-profiling-MetricLogger-debug}

```python
def debug(self, message: str, *args: Any) -> None
```

##### Method `info` {#api-speedtronic-profiling-MetricLogger-info}

```python
def info(self, message: str, *args: Any) -> None
```

##### Method `warning` {#api-speedtronic-profiling-MetricLogger-warning}

```python
def warning(self, message: str, *args: Any) -> None
```

##### Method `error` {#api-speedtronic-profiling-MetricLogger-error}

```python
def error(self, message: str, *args: Any) -> None
```

##### Method `close` {#api-speedtronic-profiling-MetricLogger-close}

```python
def close(self) -> None
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `level` | 40 |
| field | `file` | 41 |
| field | `json_file` | 42 |
| field | `every_steps` | 43 |
| field | `stream` | 44 |
| field | `hooks` | 45 |

#### Function `_format_metrics` {#api-speedtronic-profiling-_format_metrics}

```python
def _format_metrics(metrics: Metric) -> str
```

#### Class `CallbackList` {#api-speedtronic-profiling-CallbackList}

```python
class CallbackList
```

Fan-out adapter for optional W&B/TensorBoard-style callbacks.

##### Method `__init__` {#api-speedtronic-profiling-CallbackList-__init__}

```python
def __init__(self, callbacks: list[Any] | None=None) -> None
```

##### Method `on_event` {#api-speedtronic-profiling-CallbackList-on_event}

```python
def on_event(self, event: str, payload: dict[str, Any]) -> None
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `Metric` | 14 |
| constant/alias | `Hook` | 15 |
| constant/alias | `__all__` | 147 |


### `src/speedtronic/registry.py`

**Import path:** `speedtronic.registry`<br />
**Purpose:** Model registry used to keep the training engine architecture-neutral..<br />
**Lines:** 75

**Imported modules:** `__future__.annotations`, `collections.abc.Callable`, `typing.Any`

#### Class `ModelRegistry` {#api-speedtronic-registry-ModelRegistry}

```python
class ModelRegistry
```

##### Method `__init__` {#api-speedtronic-registry-ModelRegistry-__init__}

```python
def __init__(self) -> None
```

##### Method `register` {#api-speedtronic-registry-ModelRegistry-register}

```python
def register(self, name: str, factory: ModelFactory | None=None)
```

Register a factory, usable as a decorator or a normal function.

##### Method `get` {#api-speedtronic-registry-ModelRegistry-get}

```python
def get(self, name: str) -> ModelFactory
```

##### Method `names` {#api-speedtronic-registry-ModelRegistry-names}

```python
def names(self) -> tuple[str, ...]
```

##### Method `build` {#api-speedtronic-registry-ModelRegistry-build}

```python
def build(self, name: str, **kwargs: Any) -> Any
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| class attribute | `_builtin_names` | 12 |

#### Function `register_model` {#api-speedtronic-registry-register_model}

```python
def register_model(name: str, factory: ModelFactory | None=None)
```

#### Function `build_model` {#api-speedtronic-registry-build_model}

```python
def build_model(config: Any, **overrides: Any) -> Any
```

Build a model from a :class:`~speedtronic.config.ModelConfig`.

Extra keyword arguments are useful for custom models and are intentionally
passed through without interpretation.

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `ModelFactory` | 8 |
| constant/alias | `registry` | 44 |
| constant/alias | `__all__` | 75 |


### `src/speedtronic/runtime.py`

**Import path:** `speedtronic.runtime`<br />
**Purpose:** Composition helpers that turn one config into a runnable trainer..<br />
**Lines:** 173

**Imported modules:** `__future__.annotations`, `math`, `os`, `random`, `pathlib.Path`, `typing.Any`, `torch`, `.checkpoint.CheckpointManager`, `.config.SpeedtronicConfig`, `.data.build_dataloader`, `.distributed.DumbDiLoCoCoordinator`, `.optimizers.build_v2_optimizer`, `.precision.resolve_device`, `.precision.resolve_precision`, `.profiling.MetricLogger`, `.registry.build_model`, `.shapes.ShapeReport`, `.shapes.validate_startup_shapes`, `.trainer.Trainer`, `.trainer.TrainResult`

#### Function `seed_everything` {#api-speedtronic-runtime-seed_everything}

```python
def seed_everything(seed: int) -> None
```

#### Function `build_optimizer` {#api-speedtronic-runtime-build_optimizer}

```python
def build_optimizer(model: torch.nn.Module, config: Any, device: Any) -> torch.optim.Optimizer
```

Build the configured AdamW, hybrid Muon, or cautious optimizer.

#### Function `build_scheduler` {#api-speedtronic-runtime-build_scheduler}

```python
def build_scheduler(optimizer: torch.optim.Optimizer, config: SpeedtronicConfig) -> torch.optim.lr_scheduler.LambdaLR
```

#### Function `build_logger` {#api-speedtronic-runtime-build_logger}

```python
def build_logger(config: SpeedtronicConfig) -> MetricLogger
```

#### Function `build_runtime` {#api-speedtronic-runtime-build_runtime}

```python
def build_runtime(config: SpeedtronicConfig | dict[str, Any], *, resume: bool=False, device: str | torch.device | None=None, max_steps: int | None=None) -> tuple[Trainer, CheckpointManager]
```

Build all v1 components from one configuration object.

#### Function `train_from_config` {#api-speedtronic-runtime-train_from_config}

```python
def train_from_config(config: SpeedtronicConfig | dict[str, Any], *, resume: bool=False, device: str | torch.device | None=None, max_steps: int | None=None) -> TrainResult
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `run` | 160 |
| constant/alias | `train` | 161 |
| constant/alias | `__all__` | 164 |


### `src/speedtronic/scheduling.py`

**Import path:** `speedtronic.scheduling`<br />
**Purpose:** Dependency-aware forward/backward stream scheduling for v2..<br />
**Lines:** 220

**Imported modules:** `__future__.annotations`, `dataclasses.dataclass`, `typing.Any`, `typing.Iterable`, `torch`, `torch.nn`

#### Class `StageInfo` {#api-speedtronic-scheduling-StageInfo}

```python
class StageInfo
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `name` | 14 |
| field | `stream_index` | 15 |

#### Function `_iter_tensors` {#api-speedtronic-scheduling-_iter_tensors}

```python
def _iter_tensors(value: Any) -> Iterable[torch.Tensor]
```

#### Function `_expand_stage_modules` {#api-speedtronic-scheduling-_expand_stage_modules}

```python
def _expand_stage_modules(model: nn.Module) -> list[tuple[str, nn.Module]]
```

Find disjoint, meaningful stage roots without nesting hooks.

#### Class `StageStreamScheduler` {#api-speedtronic-scheduling-StageStreamScheduler}

```python
class StageStreamScheduler
```

Run disjoint module stages on CUDA streams for autograd's DAG engine.

PyTorch's autograd engine already performs dependency-aware out-of-order
node execution.  The engine still routes a node to the forward stream that
created it, so this scheduler assigns disjoint forward stages to multiple
streams.  Cross-stream inputs wait on their producer and ``record_stream``
protects allocator reuse.  CPU and MPS intentionally fall back to the
standard sequential path.

##### Method `__init__` {#api-speedtronic-scheduling-StageStreamScheduler-__init__}

```python
def __init__(self, model: nn.Module, device: torch.device | str, *, num_streams: int=4, logger: Any | None=None) -> None
```

##### Method `_install_hooks` {#api-speedtronic-scheduling-StageStreamScheduler-_install_hooks}

```python
def _install_hooks(self, modules: list[tuple[str, nn.Module]]) -> None
```

##### Method `remove_hooks` {#api-speedtronic-scheduling-StageStreamScheduler-remove_hooks}

```python
def remove_hooks(self) -> None
```

##### Method `_pre_hook` {#api-speedtronic-scheduling-StageStreamScheduler-_pre_hook}

```python
def _pre_hook(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any])
```

##### Method `_post_hook` {#api-speedtronic-scheduling-StageStreamScheduler-_post_hook}

```python
def _post_hook(self, module: nn.Module, args: tuple[Any, ...], output: Any)
```

##### Method `begin` {#api-speedtronic-scheduling-StageStreamScheduler-begin}

```python
def begin(self) -> None
```

##### Method `finish` {#api-speedtronic-scheduling-StageStreamScheduler-finish}

```python
def finish(self) -> None
```

##### Method `dispose` {#api-speedtronic-scheduling-StageStreamScheduler-dispose}

```python
def dispose(self) -> None
```

##### Method `as_dict` {#api-speedtronic-scheduling-StageStreamScheduler-as_dict}

```python
def as_dict(self) -> dict[str, Any]
```

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `__all__` | 220 |


### `src/speedtronic/shapes.py`

**Import path:** `speedtronic.shapes`<br />
**Purpose:** Startup shape-efficiency warnings for configured batches and model GEMMs..<br />
**Lines:** 244

**Imported modules:** `__future__.annotations`, `dataclasses.asdict`, `dataclasses.dataclass`, `dataclasses.field`, `typing.Any`, `typing.Mapping`, `torch`, `torch.nn`, `.precision.PrecisionPlan`

#### Class `ShapeProfile` {#api-speedtronic-shapes-ShapeProfile}

```python
class ShapeProfile
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `device_type` | 16 |
| field | `precision` | 17 |
| field | `alignment` | 18 |
| field | `source` | 19 |

#### Class `ShapeWarning` {#api-speedtronic-shapes-ShapeWarning}

```python
class ShapeWarning
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `code` | 24 |
| field | `field` | 25 |
| field | `value` | 26 |
| field | `suggested` | 27 |
| field | `alignment` | 28 |
| field | `reason` | 29 |

#### Class `ShapeReport` {#api-speedtronic-shapes-ShapeReport}

```python
class ShapeReport
```

##### Method `warning_count` {#api-speedtronic-shapes-ShapeReport-warning_count}

```python
def warning_count(self) -> int
```

##### Method `as_dict` {#api-speedtronic-shapes-ShapeReport-as_dict}

```python
def as_dict(self) -> dict[str, Any]
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `profile` | 34 |
| field | `warnings` | 35 |

#### Function `_suggest` {#api-speedtronic-shapes-_suggest}

```python
def _suggest(value: int, alignment: int) -> int
```

#### Function `resolve_shape_profile` {#api-speedtronic-shapes-resolve_shape_profile}

```python
def resolve_shape_profile(device: torch.device | str, precision: PrecisionPlan, *, requested_alignment: int | str | None='auto') -> ShapeProfile
```

#### Function `_append_warning` {#api-speedtronic-shapes-_append_warning}

```python
def _append_warning(warnings: list[ShapeWarning], seen: set[tuple[str, int, int]], *, code: str, name: str, value: int, alignment: int, reason: str, suppress_suggestion: bool=False) -> None
```

#### Function `_module_metadata` {#api-speedtronic-shapes-_module_metadata}

```python
def _module_metadata(model: nn.Module) -> list[tuple[str, int]]
```

#### Function `validate_startup_shapes` {#api-speedtronic-shapes-validate_startup_shapes}

```python
def validate_startup_shapes(config: Any, model: nn.Module, device: torch.device | str, precision: PrecisionPlan, *, logger: Any | None=None) -> ShapeReport
```

Inspect effective shapes and return non-fatal efficiency warnings.

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `__all__` | 238 |


### `src/speedtronic/trainer.py`

**Import path:** `speedtronic.trainer`<br />
**Purpose:** Architecture-neutral training loop..<br />
**Lines:** 528

**Imported modules:** `__future__.annotations`, `time`, `dataclasses.dataclass`, `typing.Any`, `typing.Iterable`, `typing.Protocol`, `torch`, `torch.nn.functional`, `.checkpoint.CheckpointManager`, `.checkpoint.capture_rng_state`, `.checkpoint.restore_rng_state`, `.data.infinite_batches`, `.precision.PrecisionPlan`, `.precision.autocast_context`, `.precision.make_grad_scaler`, `.precision.resolve_precision`, `.profiling.MetricLogger`, `.scheduling.StageStreamScheduler`, `.shapes.ShapeReport`, `.shapes.validate_startup_shapes`

#### Class `SyncCoordinator` {#api-speedtronic-trainer-SyncCoordinator}

```python
class SyncCoordinator(Protocol)
```

##### Method `start` {#api-speedtronic-trainer-SyncCoordinator-start}

```python
def start(self) -> None
```

##### Method `after_optimizer_step` {#api-speedtronic-trainer-SyncCoordinator-after_optimizer_step}

```python
def after_optimizer_step(self, model: Any, step: int) -> bool \| None
```

##### Method `stop` {#api-speedtronic-trainer-SyncCoordinator-stop}

```python
def stop(self) -> None
```

##### Method `state_dict` {#api-speedtronic-trainer-SyncCoordinator-state_dict}

```python
def state_dict(self) -> dict[str, Any] \| None
```

##### Method `load_state_dict` {#api-speedtronic-trainer-SyncCoordinator-load_state_dict}

```python
def load_state_dict(self, state: dict[str, Any]) -> None
```

#### Class `TrainResult` {#api-speedtronic-trainer-TrainResult}

```python
class TrainResult
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| field | `steps` | 34 |
| field | `samples` | 35 |
| field | `tokens` | 36 |
| field | `final_loss` | 37 |
| field | `elapsed_s` | 38 |
| field | `metrics` | 39 |

#### Class `Trainer` {#api-speedtronic-trainer-Trainer}

```python
class Trainer
```

Train a user-provided module with optional local or DumbDiLoCo sync.

The model receives a batch dictionary when the loader emits dictionaries;
otherwise the first two tuple elements are passed as ``inputs, labels``.
A model may return a loss directly, a mapping with ``loss``, or a tuple
whose first element is the loss.

##### Method `__init__` {#api-speedtronic-trainer-Trainer-__init__}

```python
def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, dataloader: Iterable[dict[str, torch.Tensor]], *, device: torch.device | str, config: Any | None=None, scheduler: Any | None=None, precision: PrecisionPlan | None=None, logger: MetricLogger | None=None, checkpoint_manager: CheckpointManager | None=None, coordinator: SyncCoordinator | None=None, max_steps: int | None=None, start_step: int=0, shape_report: ShapeReport | None=None) -> None
```

##### Method `accumulation_steps` {#api-speedtronic-trainer-Trainer-accumulation_steps}

```python
def accumulation_steps(self) -> int
```

##### Method `_configure_features` {#api-speedtronic-trainer-Trainer-_configure_features}

```python
def _configure_features(self) -> None
```

##### Method `_setup_ooo_backprop` {#api-speedtronic-trainer-Trainer-_setup_ooo_backprop}

```python
def _setup_ooo_backprop(self) -> None
```

##### Method `_enable_compile` {#api-speedtronic-trainer-Trainer-_enable_compile}

```python
def _enable_compile(self) -> None
```

##### Method `_move_batch` {#api-speedtronic-trainer-Trainer-_move_batch}

```python
def _move_batch(self, batch: Any) -> Any
```

##### Method `_batch_labels` {#api-speedtronic-trainer-Trainer-_batch_labels}

```python
def _batch_labels(batch: Any) -> torch.Tensor \| None
```

##### Method `_loss_from_logits` {#api-speedtronic-trainer-Trainer-_loss_from_logits}

```python
def _loss_from_logits(cls, output: Any, batch: Any) -> torch.Tensor \| None
```

##### Method `_forward` {#api-speedtronic-trainer-Trainer-_forward}

```python
def _forward(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]
```

##### Method `_forward_with_compile_fallback` {#api-speedtronic-trainer-Trainer-_forward_with_compile_fallback}

```python
def _forward_with_compile_fallback(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]
```

##### Method `_batch_size_and_tokens` {#api-speedtronic-trainer-Trainer-_batch_size_and_tokens}

```python
def _batch_size_and_tokens(batch: Any) -> tuple[int, int]
```

##### Method `_optimizer_step` {#api-speedtronic-trainer-Trainer-_optimizer_step}

```python
def _optimizer_step(self, loss: torch.Tensor, final_microbatch: bool, loss_scale: float=1.0) -> float
```

##### Method `_reset_inner_optimizer_if_needed` {#api-speedtronic-trainer-Trainer-_reset_inner_optimizer_if_needed}

```python
def _reset_inner_optimizer_if_needed(self) -> None
```

##### Method `_checkpoint` {#api-speedtronic-trainer-Trainer-_checkpoint}

```python
def _checkpoint(self, *, force: bool=False) -> None
```

##### Method `resume` {#api-speedtronic-trainer-Trainer-resume}

```python
def resume(self, state: dict[str, Any]) -> None
```

##### Method `fit` {#api-speedtronic-trainer-Trainer-fit}

```python
def fit(self, max_steps: int | None=None) -> TrainResult
```

##### Method `_compiled_model` {#api-speedtronic-trainer-Trainer-_compiled_model}

```python
def _compiled_model(self) -> torch.nn.Module
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| class attribute | `train` | 504 |

#### Function `_checkpoint_config` {#api-speedtronic-trainer-_checkpoint_config}

```python
def _checkpoint_config(config: Any) -> Any
```

#### Class `_AutoPrecision` {#api-speedtronic-trainer-_AutoPrecision}

```python
class _AutoPrecision
```

**Class attributes and fields**

| Kind | Symbol | Line |
|---|---:|---:|
| class attribute | `mode` | 520 |
| class attribute | `dtype` | 521 |

**Module constants and aliases**

| Kind | Symbol | Line |
|---|---:|---:|
| constant/alias | `TrainingEngine` | 524 |
| constant/alias | `SpeedtronicTrainer` | 525 |
| constant/alias | `__all__` | 528 |


## Test modules

Every test function defined by the repository is listed here.

### `tests/test_config.py`

- `test_config::test_config_yaml_aliases_and_accumulation`
- `test_config::test_config_round_trip`
- `test_config::test_unknown_keys_and_invalid_batch`
- `test_config::test_distributed_role_inference`
- `test_config::test_secret_can_be_redacted_for_serialization`

### `tests/test_distributed.py`

- `test_distributed::test_pseudo_gradient_direction_and_average`
- `test_distributed::test_nesterov_outer_optimizer`
- `test_distributed::test_delta_path_parser`
- `test_distributed::test_tied_model_state_can_be_safetensors_serialized`

### `tests/test_hub.py`

- `test_hub::test_hub_transport_round_trip`
- `test_hub::test_master_skips_corrupt_delta_and_persists_processed_set`

### `tests/test_model.py`

- `test_model::test_reference_transformer_forward_and_loss`
- `test_model::test_reference_transformer_weight_tying_and_checkpoint_hook`
- `test_model::test_config_dimensions`

### `tests/test_precision_data.py`

- `test_precision_data::test_cpu_precision_defaults_to_fp32`
- `test_precision_data::test_streaming_text_dataset_and_tuple_collate`

### `tests/test_training.py`

- `test_training::test_trainer_runs_with_accumulation_and_checkpoint`
- `test_training::test_accumulation_scales_gradients_and_reports_mean_loss`
- `test_training::test_repeated_fit_restarts_coordinator_lifecycle`
- `test_training::test_runtime_builds_reference_model`

### `tests/test_v2_distributed.py`

- `test_v2_distributed::test_async_global_poll_installs_on_training_thread`
- `test_v2_distributed::test_coordinator_can_restart_after_stop`
- `test_v2_distributed::test_async_delta_dispatch_does_not_block_training`
- `test_v2_distributed::test_prepared_resume_restores_pending_upload`
- `test_v2_distributed::test_synchronous_delta_mode_remains_available`
- `test_v2_distributed::test_async_delta_failure_keeps_baseline_for_cumulative_retry`

### `tests/test_v2_optimizers.py`

- `test_v2_optimizers::test_v2_config_accepts_scalar_optimizer_and_root_flags`
- `test_v2_optimizers::test_muon_plus_requires_muon`
- `test_v2_optimizers::test_v2_config_rejects_invalid_systems_values`
- `test_v2_optimizers::test_newton_schulz_is_pure_and_finite`
- `test_v2_optimizers::test_newton_schulz_handles_rectangular_matrices`
- `test_v2_optimizers::test_post_polar_normalization_produces_unit_rows_and_columns`
- `test_v2_optimizers::test_reference_parameter_routing_is_role_aware_and_tied_safe`
- `test_v2_optimizers::test_hybrid_optimizer_updates_both_branches_and_scheduler`
- `test_v2_optimizers::test_adamw_never_builds_an_empty_optimizer_for_all_linear_model`
- `test_v2_optimizers::test_non_linear_custom_matrix_defaults_to_adamw`
- `test_v2_optimizers::test_hybrid_optimizer_forwards_closure_once`
- `test_v2_optimizers::test_cautious_wrapper_is_composable_and_finite`
- `test_v2_optimizers::test_muon_rejects_non_matrix_parameters`
- `test_v2_optimizers::test_muon_sparse_gradient_is_rejected`
- `test_v2_optimizers::test_v2_muon_runtime_smoke`

### `tests/test_v2_release.py`

- `test_v2_release::test_version_is_synchronized`
- `test_v2_release::test_cli_validate_redacts_configured_token`
- `test_v2_release::test_v2_config_serializes_all_new_sections`

### `tests/test_v2_systems.py`

- `test_v2_systems::test_cpu_shape_auto_profile_is_quiet_but_explicit_alignment_warns`
- `test_v2_systems::test_shape_report_is_non_fatal_and_deduplicated`
- `test_v2_systems::test_explicit_cpu_profile_requires_opt_in_and_avoids_invalid_batch_suggestion`
- `test_v2_systems::test_shape_validation_accepts_standard_library_logger`
- `test_v2_systems::test_scheduler_default_follows_run_target`
- `test_v2_systems::test_causal_inferred_loss_uses_already_shifted_labels`
- `test_v2_systems::test_out_of_order_scheduler_degrades_to_noop_on_cpu`
- `test_v2_systems::test_out_of_order_config_runs_without_changing_cpu_path`

## Shipped configurations and examples

- `configs/diloco.yaml`
- `configs/smoke.yaml`
- `configs/v2_smoke.yaml`
- `examples/diloco_master.yaml`
- `examples/train_reference.py`
