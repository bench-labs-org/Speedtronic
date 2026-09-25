"""Speedtronic: fast, efficient training with optional DumbDiLoCo."""

from __future__ import annotations

from .config import (
    CheckpointConfig,
    Config,
    ConfigError,
    DataConfig,
    DistributedConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    PrecisionConfig,
    RunConfig,
    SchedulerConfig,
    ShapeValidationConfig,
    SpeedtronicConfig,
    load_config,
    load_yaml_config,
)
from .module_utils import set_gradient_checkpointing
from .optimizers import (
    CautiousOptimizer,
    HybridOptimizer,
    Muon,
    ParameterRouting,
    newton_schulz,
    post_polar_normalize,
    route_parameters,
)
from .registry import ModelRegistry, build_model, register_model, registry
from .scheduling import StageInfo, StageStreamScheduler
from .shapes import (
    ShapeProfile,
    ShapeReport,
    ShapeWarning,
    resolve_shape_profile,
    validate_startup_shapes,
)

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
    "Trainer",
    "TrainResult",
    "CautiousOptimizer",
    "HybridOptimizer",
    "Muon",
    "ParameterRouting",
    "ShapeProfile",
    "ShapeReport",
    "ShapeWarning",
    "StageInfo",
    "StageStreamScheduler",
    "newton_schulz",
    "post_polar_normalize",
    "resolve_shape_profile",
    "route_parameters",
    "validate_startup_shapes",
    "ModelRegistry",
    "build_model",
    "load_config",
    "load_yaml_config",
    "register_model",
    "registry",
    "set_gradient_checkpointing",
    "GPT",
    "GPTConfig",
    "ReferenceTransformer",
    "DumbDiLoCo",
    "DumbDiLoCoCoordinator",
    "HubClient",
]

__version__ = "2.0.0"


def __getattr__(name: str):
    if name in {"Trainer", "TrainResult"}:
        from .trainer import Trainer, TrainResult

        return {"Trainer": Trainer, "TrainResult": TrainResult}[name]
    if name in {"GPT", "GPTConfig", "ReferenceTransformer"}:
        from . import model

        return getattr(model, name)
    if name in {"DumbDiLoCo", "DumbDiLoCoCoordinator", "HubClient"}:
        from .distributed.diloco import DumbDiLoCo, DumbDiLoCoCoordinator
        from .distributed.hub import HubClient

        return {
            "DumbDiLoCo": DumbDiLoCo,
            "DumbDiLoCoCoordinator": DumbDiLoCoCoordinator,
            "HubClient": HubClient,
        }[name]
    raise AttributeError(name)
