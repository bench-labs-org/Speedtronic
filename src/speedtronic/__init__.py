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
    SpeedtronicConfig,
    load_config,
    load_yaml_config,
)
from .module_utils import set_gradient_checkpointing
from .registry import ModelRegistry, build_model, register_model, registry

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
    "SpeedtronicConfig",
    "Trainer",
    "TrainResult",
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

__version__ = "0.1.0"


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
