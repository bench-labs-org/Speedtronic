"""Composition helpers that turn one config into a runnable trainer."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import torch

from .checkpoint import CheckpointManager
from .config import SpeedtronicConfig
from .data import build_dataloader
from .distributed import DumbDiLoCoCoordinator
from .precision import resolve_device, resolve_precision, supports_fused_adamw
from .profiling import MetricLogger
from .registry import build_model
from .trainer import Trainer, TrainResult


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: torch.nn.Module, config: Any, device: Any) -> torch.optim.Optimizer:
    optimizer_config = config.optimizer
    fused = (
        supports_fused_adamw(device)
        if optimizer_config.fused is None
        else bool(optimizer_config.fused)
    )
    kwargs: dict[str, Any] = {
        "lr": optimizer_config.lr,
        "betas": tuple(optimizer_config.betas),
        "eps": optimizer_config.eps,
        "weight_decay": optimizer_config.weight_decay,
    }
    if fused:
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError, ValueError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: SpeedtronicConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    scheduler = config.scheduler
    warmup = max(0, int(scheduler.warmup_steps))
    total = max(1, int(scheduler.max_steps))
    min_ratio = float(scheduler.min_lr_ratio)

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return max(1e-8, float(step + 1) / float(warmup))
        if scheduler.name == "constant":
            return 1.0
        progress = min(1.0, max(0.0, float(step - warmup) / max(1, total - warmup)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_logger(config: SpeedtronicConfig) -> MetricLogger:
    return MetricLogger(
        level=config.logging.level,
        file=config.logging.file,
        json_file=config.logging.json_file,
        every_steps=config.logging.every_steps,
    )


def build_runtime(
    config: SpeedtronicConfig | dict[str, Any],
    *,
    resume: bool = False,
    device: str | torch.device | None = None,
    max_steps: int | None = None,
) -> tuple[Trainer, CheckpointManager]:
    """Build all v1 components from one configuration object."""

    if isinstance(config, dict):
        config = SpeedtronicConfig.from_dict(config).validate()
    else:
        config.validate()
    seed_everything(config.run.seed)
    resolved_device = resolve_device(device or config.run.device)
    model = build_model(config.model)
    dataloader = build_dataloader(
        config.data,
        pin_memory_device=resolved_device.type == "cuda",
        vocab_size=config.model.vocab_size,
    )
    optimizer = build_optimizer(model, config, resolved_device)
    scheduler = build_scheduler(optimizer, config)
    checkpoint_directory = Path(config.checkpoint.directory)
    if not checkpoint_directory.is_absolute():
        checkpoint_directory = Path(config.run.output_dir) / checkpoint_directory
    checkpoint_manager = CheckpointManager(
        checkpoint_directory,
        every_steps=config.checkpoint.every_steps,
        keep_last=config.checkpoint.keep_last,
        enabled=config.checkpoint.enabled,
    )
    logger = build_logger(config)
    coordinator = None
    if config.distributed.enabled:
        state_directory = Path(config.distributed.state_dir)
        if not state_directory.is_absolute():
            state_directory = Path(config.run.output_dir) / state_directory
        coordinator = DumbDiLoCoCoordinator(
            config.distributed,
            model,
            state_dir=state_directory,
            logger=logger,
        )
    effective_max_steps = (
        max_steps if max_steps is not None else (config.data.max_steps or config.run.max_steps)
    )
    trainer = Trainer(
        model,
        optimizer,
        dataloader,
        device=resolved_device,
        config=config,
        scheduler=scheduler,
        precision=resolve_precision(config.precision, resolved_device),
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        coordinator=coordinator,
        max_steps=effective_max_steps,
    )
    if resume or config.checkpoint.resume:
        state = checkpoint_manager.load_latest()
        if state is not None:
            trainer.resume(state)
        else:
            logger.warning("resume requested but no local checkpoint was found")
    return trainer, checkpoint_manager


def train_from_config(
    config: SpeedtronicConfig | dict[str, Any],
    *,
    resume: bool = False,
    device: str | torch.device | None = None,
    max_steps: int | None = None,
) -> TrainResult:
    trainer, _ = build_runtime(config, resume=resume, device=device, max_steps=max_steps)
    return trainer.fit()


run = train_from_config
train = train_from_config


__all__ = [
    "build_logger",
    "build_optimizer",
    "build_runtime",
    "build_scheduler",
    "run",
    "seed_everything",
    "train",
    "train_from_config",
]
