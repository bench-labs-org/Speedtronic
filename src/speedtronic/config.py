"""Configuration objects and loading helpers for Speedtronic.

The configuration is intentionally data-first: a run can be represented by a
YAML file or by constructing the same dataclasses in Python.  Unknown keys are
rejected early so a typo cannot silently disable an efficiency feature.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

try:  # PyYAML is a runtime dependency, but keep import errors useful.
    import yaml
except ImportError:  # pragma: no cover - exercised only in broken installs
    yaml = None  # type: ignore[assignment]


class ConfigError(ValueError):
    """Raised when a run configuration is invalid."""


@dataclass
class ModelConfig:
    name: str = "reference_transformer"
    vocab_size: int = 512
    max_seq_len: int = 128
    n_layer: int = 4
    n_head: int = 8
    n_kv_head: int | None = None
    d_model: int = 256
    d_ff: int | None = None
    dropout: float = 0.0
    tie_weights: bool = True
    rope_base: float = 10_000.0
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.max_seq_len <= 0:
            raise ConfigError("model.max_seq_len must be positive")
        if self.vocab_size <= 0:
            raise ConfigError("model.vocab_size must be positive")
        if self.n_layer <= 0 or self.n_head <= 0 or self.n_kv_head <= 0:
            raise ConfigError("model layer/head counts must be positive")
        if self.d_model <= 0 or self.d_ff <= 0:
            raise ConfigError("model dimensions must be positive")
        if self.n_head % self.n_kv_head != 0:
            raise ConfigError("model.n_head must be divisible by model.n_kv_head")
        if self.d_model % self.n_head != 0:
            raise ConfigError("model.d_model must be divisible by model.n_head")
        if not 0 <= self.dropout < 1:
            raise ConfigError("model.dropout must be in [0, 1)")


@dataclass
class DataConfig:
    """Data and batching configuration.

    ``micro_batch_size`` is the batch size emitted by the loader.  The trainer
    accumulates that many batches to reach ``target_batch_size`` when the
    latter is larger.
    """

    text_path: str | None = None
    dataset: str | None = None
    synthetic: bool = True
    num_tokens: int = 100_000
    vocab_size: int | None = None
    block_size: int | None = None
    micro_batch_size: int = 1
    target_batch_size: int = 1
    num_workers: int = 0
    prefetch_factor: int | None = None
    pin_memory: bool | None = None
    shuffle: bool = False
    drop_last: bool = True
    seed: int | None = None
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.micro_batch_size <= 0 or self.target_batch_size <= 0:
            raise ConfigError("batch sizes must be positive")
        if self.target_batch_size < self.micro_batch_size:
            raise ConfigError("target_batch_size must be >= micro_batch_size")
        if self.target_batch_size % self.micro_batch_size != 0:
            raise ConfigError("target_batch_size must be divisible by micro_batch_size")
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ConfigError("data.vocab_size must be positive")
        if self.num_tokens <= 0:
            raise ConfigError("data.num_tokens must be positive")
        if self.num_workers < 0:
            raise ConfigError("data.num_workers cannot be negative")
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ConfigError("data.prefetch_factor must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ConfigError("data.max_steps must be positive")

    @property
    def accumulation_steps(self) -> int:
        return self.target_batch_size // self.micro_batch_size


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    fused: bool | None = None
    grad_clip: float | None = None
    muon_plus: bool = False
    cautious: bool = False
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_norm_eps: float = 1e-8

    def __post_init__(self) -> None:
        self.name = str(self.name).strip().lower()
        if self.name not in {"adamw", "muon"}:
            raise ConfigError("optimizer.name must be 'adamw' or 'muon'")
        if self.lr <= 0:
            raise ConfigError("optimizer.lr must be positive")
        if len(self.betas) != 2 or not all(0 <= b < 1 for b in self.betas):
            raise ConfigError("optimizer.betas must contain two values in [0, 1)")
        if self.eps <= 0 or self.weight_decay < 0:
            raise ConfigError("optimizer eps must be positive and weight_decay non-negative")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ConfigError("optimizer.grad_clip must be positive")
        if not isinstance(self.muon_plus, bool) or not isinstance(self.cautious, bool):
            raise ConfigError("optimizer Muon/cautious flags must be booleans")
        if self.muon_plus and self.name != "muon":
            raise ConfigError("optimizer.muon_plus is only valid with optimizer.name=muon")
        if not 0 <= self.muon_momentum < 1:
            raise ConfigError("optimizer.muon_momentum must be in [0, 1)")
        if self.muon_ns_steps <= 0:
            raise ConfigError("optimizer.muon_ns_steps must be positive")
        if self.muon_norm_eps <= 0:
            raise ConfigError("optimizer.muon_norm_eps must be positive")


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 100
    max_steps: int = 1000
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        self.name = str(self.name).lower()
        if self.name not in {"cosine", "constant"}:
            raise ConfigError("scheduler.name must be 'cosine' or 'constant'")
        if self.warmup_steps < 0 or self.max_steps <= 0:
            raise ConfigError("scheduler steps must be non-negative/max_steps positive")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ConfigError("scheduler.min_lr_ratio must be in [0, 1]")


@dataclass
class PrecisionConfig:
    mode: str = "auto"  # auto, bf16, fp16, fp32
    dtype: str | None = None

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower()
        if self.dtype is not None:
            self.dtype = str(self.dtype).lower()
        if self.mode not in {"auto", "bf16", "fp16", "fp32"}:
            raise ConfigError("precision.mode must be auto, bf16, fp16, or fp32")
        if self.dtype is not None and self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ConfigError("precision.dtype must be bf16, fp16, or fp32")


@dataclass
class CheckpointConfig:
    enabled: bool = True
    directory: str = "checkpoints"
    every_steps: int = 500
    keep_last: int | None = 3
    resume: bool = False

    def __post_init__(self) -> None:
        if self.every_steps <= 0:
            raise ConfigError("checkpoint.every_steps must be positive")
        if self.keep_last is not None and self.keep_last <= 0:
            raise ConfigError("checkpoint.keep_last must be positive or null")


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str | None = None
    every_steps: int = 1
    json_file: str | None = None

    def __post_init__(self) -> None:
        if self.every_steps <= 0:
            raise ConfigError("logging.every_steps must be positive")
        self.level = self.level.upper()


@dataclass
class ShapeValidationConfig:
    enabled: bool = True
    alignment: int | str = "auto"
    check_batch: bool = True
    check_sequence: bool = True
    check_model: bool = True
    check_vocab: bool = False
    warn_on_cpu: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigError("shape_validation.enabled must be a boolean")
        if isinstance(self.alignment, str):
            self.alignment = self.alignment.strip().lower()
            if self.alignment not in {"auto", "none"}:
                raise ConfigError(
                    "shape_validation.alignment must be 'auto', 'none', or a positive integer"
                )
        elif self.alignment is None:
            self.alignment = "auto"
        elif not isinstance(self.alignment, int) or isinstance(self.alignment, bool):
            raise ConfigError("shape_validation.alignment must be 'auto' or a positive integer")
        if isinstance(self.alignment, int) and self.alignment <= 0:
            raise ConfigError("shape_validation.alignment must be positive")
        for field_name in (
            "check_batch",
            "check_sequence",
            "check_model",
            "check_vocab",
            "warn_on_cpu",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ConfigError(f"shape_validation.{field_name} must be a boolean")


@dataclass
class DistributedConfig:
    enabled: bool = False
    mode: str = "dumb_diloco"
    role: str = "single"
    node_id: str | None = None
    collaborators: list[str] = field(default_factory=list)
    inner_steps: int = 500
    poll_interval: float = 60.0
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    repo_id: str | None = None
    token: str | None = None
    cache_dir: str = ".speedtronic/hub"
    state_dir: str = ".speedtronic/diloco"
    retry_initial: float = 1.0
    retry_max: float = 60.0
    retry_attempts: int = 6
    reset_inner_optimizer: bool = True
    async_delta_upload: bool = True
    delta_upload_queue_size: int = 1
    delta_upload_overflow: str = "skip"
    delta_upload_shutdown_timeout: float = 5.0
    async_global_poll: bool = True

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower()
        self.role = str(self.role).lower()
        if self.mode not in {"dumb_diloco", "single"}:
            raise ConfigError("distributed.mode must be 'dumb_diloco' or 'single'")
        if self.enabled and self.mode != "dumb_diloco":
            raise ConfigError("distributed.mode must be 'dumb_diloco' when enabled")
        if self.role not in {"single", "master", "worker"}:
            raise ConfigError("distributed.role must be single, master, or worker")
        if self.enabled and self.role == "single":
            # A repo-bearing run with no explicit role is a convenient master
            # default; callers that want a worker must say so explicitly.
            self.role = "master" if self.repo_id else "worker"
        if self.node_id is not None:
            node_id = str(self.node_id).strip()
            if not node_id or node_id in {".", ".."} or "/" in node_id or "\\" in node_id:
                raise ConfigError("distributed.node_id must be a non-empty path-safe identifier")
        if not isinstance(self.collaborators, list) or any(
            not isinstance(item, str) or not item.strip() for item in self.collaborators
        ):
            raise ConfigError("distributed.collaborators must be a list of usernames")
        if self.inner_steps <= 0:
            raise ConfigError("distributed.inner_steps must be positive")
        if self.poll_interval <= 0:
            raise ConfigError("distributed.poll_interval must be positive")
        if self.outer_lr <= 0 or not 0 <= self.outer_momentum < 1:
            raise ConfigError("outer learning rate/momentum values are invalid")
        if self.retry_initial <= 0 or self.retry_max < self.retry_initial:
            raise ConfigError("distributed retry bounds are invalid")
        if self.retry_attempts < 1:
            raise ConfigError("distributed.retry_attempts must be positive")
        for field_name in (
            "reset_inner_optimizer",
            "async_delta_upload",
            "async_global_poll",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ConfigError(f"distributed.{field_name} must be a boolean")
        if self.delta_upload_queue_size != 1:
            raise ConfigError("distributed.delta_upload_queue_size must be 1 in v2")
        self.delta_upload_overflow = str(self.delta_upload_overflow).lower()
        if self.delta_upload_overflow != "skip":
            raise ConfigError("distributed.delta_upload_overflow must be 'skip'")
        if self.delta_upload_shutdown_timeout <= 0:
            raise ConfigError("distributed.delta_upload_shutdown_timeout must be positive")
        if self.enabled and not self.repo_id:
            raise ConfigError("distributed.repo_id is required for DumbDiLoCo")


@dataclass
class RunConfig:
    name: str = "speedtronic-run"
    seed: int = 1234
    device: str = "auto"
    max_steps: int = 1000
    output_dir: str = "runs"
    log_every: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ConfigError("run.max_steps must be positive")
        if self.log_every is not None and self.log_every <= 0:
            raise ConfigError("run.log_every must be positive")


@dataclass
class SpeedtronicConfig:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    shape_validation: ShapeValidationConfig = field(default_factory=ShapeValidationConfig)
    gradient_checkpointing: bool = False
    compile: bool = False
    ooo_backprop: bool = False
    ooo_streams: int = 4

    def __post_init__(self) -> None:
        if self.run.max_steps <= 0:
            raise ConfigError("run.max_steps must be positive")
        if not isinstance(self.ooo_backprop, bool):
            raise ConfigError("ooo_backprop must be a boolean")
        if not 1 <= int(self.ooo_streams) <= 8:
            raise ConfigError("ooo_streams must be between 1 and 8")
        if self.distributed.enabled and self.distributed.role == "single":
            self.distributed.role = "worker" if not self.distributed.repo_id else "master"
        if self.data.block_size is None:
            self.data.block_size = self.model.max_seq_len
        if self.data.vocab_size is None:
            self.data.vocab_size = self.model.vocab_size
        if self.data.seed is None:
            self.data.seed = self.run.seed
        if self.run.log_every is not None and self.logging.every_steps == 1:
            self.logging.every_steps = self.run.log_every
        if self.scheduler.max_steps <= 0:
            self.scheduler.max_steps = self.run.max_steps
        elif (
            self.scheduler.max_steps == SchedulerConfig.max_steps
            and self.run.max_steps != SchedulerConfig.max_steps
        ):
            # The historical default was 1000.  Keep the default schedule in
            # sync with a shorter explicit run target unless the user supplied
            # a different horizon after construction.
            self.scheduler.max_steps = self.run.max_steps
        if (
            self.data.max_steps is not None
            and self.scheduler.max_steps == SchedulerConfig.max_steps
        ):
            self.scheduler.max_steps = self.data.max_steps
        # A top-level value is authoritative when explicitly supplied.  The
        # nested model value remains useful for model construction.
        if self.gradient_checkpointing or self.model.gradient_checkpointing:
            self.gradient_checkpointing = True
            self.model.gradient_checkpointing = True

    @property
    def accumulation_steps(self) -> int:
        return self.data.accumulation_steps

    def validate(self) -> "SpeedtronicConfig":
        """Validate cross-section constraints and return ``self``."""
        if self.data.block_size is None or self.data.block_size <= 0:
            raise ConfigError("data.block_size must be positive")
        if self.data.block_size > self.model.max_seq_len:
            # The reference model can handle shorter sequences, so this is a
            # warning-worthy configuration rather than a hard failure.  Keep
            # it valid for custom models.
            pass
        if self.scheduler.max_steps > 0 and self.run.max_steps > 0:
            # A scheduler longer than the run is valid; it simply keeps the
            # final schedule point beyond this invocation.
            pass
        if self.distributed.enabled and not self.distributed.repo_id:
            raise ConfigError("distributed.repo_id is required")
        return self

    def to_dict(self, *, redact_secrets: bool = False) -> dict[str, Any]:
        result = _jsonable(asdict(self))
        if redact_secrets and result.get("distributed", {}).get("token"):
            result["distributed"]["token"] = "<redacted>"
        return result

    def to_yaml(self, *, redact_secrets: bool = False) -> str:
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to serialize YAML")
        return yaml.safe_dump(self.to_dict(redact_secrets=redact_secrets), sort_keys=False)

    def save(self, path: str | os.PathLike[str]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Config files are intended to be version-controlled; never persist a
        # Hub token accidentally.  Use HF_TOKEN or inject the token at runtime.
        if path.suffix.lower() in {".yaml", ".yml"}:
            path.write_text(self.to_yaml(redact_secrets=True), encoding="utf-8")
        else:
            path.write_text(
                json.dumps(self.to_dict(redact_secrets=True), indent=2) + "\n",
                encoding="utf-8",
            )
        return path

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None = None) -> "SpeedtronicConfig":
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise ConfigError("configuration root must be a mapping")
        values = dict(values)
        # Accept convenient run-level aliases used in the example files.  Do
        # this before unknown-key validation so ``name: ...`` is not confused
        # with a model field.
        run_aliases = {
            "name": "name",
            "seed": "seed",
            "device": "device",
            "output_dir": "output_dir",
            "max_steps": "max_steps",
            "log_every": "log_every",
        }
        run_raw = values.get("run", {})
        if run_raw is None:
            run_raw = {}
        if is_dataclass(run_raw) and not isinstance(run_raw, Mapping):
            run_raw = asdict(run_raw)
        if not isinstance(run_raw, Mapping):
            raise ConfigError("run must be a mapping")
        run_values = dict(run_raw)
        for alias, target in run_aliases.items():
            if alias in values:
                run_values[target] = values.pop(alias)
        if run_values:
            values["run"] = run_values
        if "hub" in values and "distributed" not in values:
            values["distributed"] = values.pop("hub")
        if "diloco" in values and "distributed" not in values:
            values["distributed"] = values.pop("diloco")
        optimizer_value = values.get("optimizer")
        if isinstance(optimizer_value, str):
            optimizer_value = {"name": optimizer_value}
            values["optimizer"] = optimizer_value
        optimizer_aliases = ("muon_plus", "cautious")
        for alias in optimizer_aliases:
            if alias not in values:
                continue
            if optimizer_value is None:
                optimizer_value = {}
            elif is_dataclass(optimizer_value) and not isinstance(optimizer_value, Mapping):
                optimizer_value = asdict(optimizer_value)
            if not isinstance(optimizer_value, Mapping):
                raise ConfigError("optimizer must be a mapping or optimizer name string")
            optimizer_value = dict(optimizer_value)
            if alias in optimizer_value and optimizer_value[alias] != values[alias]:
                raise ConfigError(f"conflicting optimizer.{alias} values")
            optimizer_value[alias] = values.pop(alias)
            values["optimizer"] = optimizer_value
        _reject_unknown(values, cls, "")
        gradient_value = bool(values.pop("gradient_checkpointing", False))
        ooo_value = bool(values.pop("ooo_backprop", False))
        ooo_streams = int(values.pop("ooo_streams", 4))
        compile_value = values.pop("compile", False)
        if isinstance(compile_value, Mapping):
            _reject_unknown(compile_value, _CompileSection, "compile")
            compile_value = bool(compile_value.get("enabled", False))

        config_types = {
            "run": RunConfig,
            "model": ModelConfig,
            "data": DataConfig,
            "optimizer": OptimizerConfig,
            "scheduler": SchedulerConfig,
            "precision": PrecisionConfig,
            "checkpoint": CheckpointConfig,
            "logging": LoggingConfig,
            "distributed": DistributedConfig,
            "shape_validation": ShapeValidationConfig,
        }
        kwargs: dict[str, Any] = {
            "gradient_checkpointing": gradient_value,
            "compile": bool(compile_value),
            "ooo_backprop": ooo_value,
            "ooo_streams": ooo_streams,
        }
        for name, config_type in config_types.items():
            if name in values:
                kwargs[name] = _build_dataclass(config_type, values.pop(name), name)
        if values:
            unknown = ", ".join(sorted(values))
            raise ConfigError(f"unknown configuration keys: {unknown}")
        result = cls(**kwargs)
        result.validate()
        return result

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "SpeedtronicConfig":
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load YAML")
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ConfigError("configuration root must be a mapping")
        return cls.from_dict(raw)

    @classmethod
    def load(cls, source: str | os.PathLike[str] | Mapping[str, Any]) -> "SpeedtronicConfig":
        if isinstance(source, Mapping):
            return cls.from_dict(source)
        if isinstance(source, str) and (
            "\n" in source
            or source.lstrip().startswith(("{", "["))
            or (":" in source and not Path(source).exists())
        ):
            if yaml is not None:
                raw = yaml.safe_load(source) or {}
                if not isinstance(raw, Mapping):
                    raise ConfigError("configuration root must be a mapping")
                return cls.from_dict(raw)
        path = Path(source)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return cls.from_yaml(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


Config = SpeedtronicConfig


def load_config(source: str | os.PathLike[str] | Mapping[str, Any]) -> SpeedtronicConfig:
    return SpeedtronicConfig.load(source)


def load_yaml_config(path: str | os.PathLike[str]) -> SpeedtronicConfig:
    return SpeedtronicConfig.from_yaml(path)


T = TypeVar("T")


@dataclass
class _CompileSection:
    enabled: bool = False


def _build_dataclass(cls: type[T], value: Any, path: str) -> T:
    if is_dataclass(value) and isinstance(value, cls):
        return value
    if value is None:
        return cls()
    if cls is PrecisionConfig and isinstance(value, str):
        value = {"mode": value}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    _reject_unknown(value, cls, path)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in value:
            continue
        item = value[f.name]
        if is_dataclass(f.type) and isinstance(item, Mapping):
            kwargs[f.name] = _build_dataclass(f.type, item, f"{path}.{f.name}")
        else:
            kwargs[f.name] = item
    return cls(**kwargs)


def _reject_unknown(value: Mapping[str, Any], cls: type[Any], path: str) -> None:
    allowed = {f.name for f in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        prefix = f"{path}: " if path else ""
        raise ConfigError(f"{prefix}unknown configuration keys: {', '.join(sorted(unknown))}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


__all__ = [
    "CheckpointConfig",
    "Config",
    "ConfigError",
    "DataConfig",
    "DistributedConfig",
    "LoggingConfig",
    "ModelConfig",
    "OptimizerConfig",
    "PrecisionConfig",
    "RunConfig",
    "SchedulerConfig",
    "ShapeValidationConfig",
    "SpeedtronicConfig",
    "load_config",
    "load_yaml_config",
]
