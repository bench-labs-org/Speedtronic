import torch
from torch import nn

from speedtronic.checkpoint import CheckpointManager
from speedtronic.config import SpeedtronicConfig
from speedtronic.data import build_dataloader
from speedtronic.runtime import build_optimizer, build_runtime, build_scheduler
from speedtronic.trainer import Trainer


class TinyLM(nn.Module):
    def __init__(self, vocab=32, dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.output = nn.Linear(dim, vocab)

    def forward(self, input_ids, labels=None, attention_mask=None):
        logits = self.output(self.embedding(input_ids))
        if labels is None:
            return logits
        return {
            "logits": logits,
            "loss": nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)
            ),
        }


def _loader(config):
    data = config.data
    data.num_tokens = 64
    return build_dataloader(data, pin_memory_device=False)


def test_trainer_runs_with_accumulation_and_checkpoint(tmp_path):
    config = SpeedtronicConfig.from_dict(
        {
            "run": {"max_steps": 3, "device": "cpu", "output_dir": str(tmp_path)},
            "model": {
                "name": "unused",
                "vocab_size": 32,
                "d_model": 16,
                "n_head": 4,
                "max_seq_len": 8,
            },
            "data": {
                "block_size": 8,
                "micro_batch_size": 1,
                "target_batch_size": 2,
                "num_tokens": 64,
            },
            "precision": {"mode": "fp32"},
            "checkpoint": {"enabled": True, "directory": "checkpoints", "every_steps": 2},
        }
    )
    # A custom model is supplied directly; the config name is not used here.
    model = TinyLM()
    optimizer = build_optimizer(model, config, torch.device("cpu"))
    trainer = Trainer(
        model,
        optimizer,
        _loader(config),
        device="cpu",
        config=config,
        scheduler=build_scheduler(optimizer, config),
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints", every_steps=2),
        max_steps=3,
    )
    result = trainer.fit()
    assert result.steps == 3
    assert len(result.metrics) == 3
    assert (tmp_path / "checkpoints" / "latest.json").exists()


def test_accumulation_scales_gradients_and_reports_mean_loss():
    class ScalarModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, labels=None, attention_mask=None):
            target = labels.float().mean()
            return (self.value - target) ** 2

    config = SpeedtronicConfig.from_dict(
        {
            "data": {"micro_batch_size": 1, "target_batch_size": 2},
            "precision": {"mode": "fp32"},
        }
    )
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [
        {"input_ids": torch.tensor([0]), "labels": torch.tensor([0])},
        {"input_ids": torch.tensor([1]), "labels": torch.tensor([2])},
    ]
    result = Trainer(
        model,
        optimizer,
        batches,
        device="cpu",
        config=config,
        max_steps=1,
    ).fit()
    assert torch.allclose(model.value, torch.tensor(0.2))
    assert result.final_loss == 2.0


class CountingCoordinator:
    def __init__(self):
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    def after_optimizer_step(self, model, step):
        return False

    def stop(self):
        self.stops += 1

    def state_dict(self):
        return None

    def load_state_dict(self, state):
        return None


def test_repeated_fit_restarts_coordinator_lifecycle(tmp_path):
    config = SpeedtronicConfig.from_dict(
        {"run": {"max_steps": 1, "device": "cpu", "output_dir": str(tmp_path)}}
    )
    model = TinyLM()
    optimizer = build_optimizer(model, config, torch.device("cpu"))
    batches = [
        {"input_ids": torch.randint(0, 32, (1, 4)), "labels": torch.randint(0, 32, (1, 4))}
        for _ in range(4)
    ]
    coordinator = CountingCoordinator()
    trainer = Trainer(
        model,
        optimizer,
        batches,
        device="cpu",
        config=config,
        coordinator=coordinator,
        max_steps=1,
    )
    trainer.fit()
    trainer.fit(max_steps=2)
    assert coordinator.starts == 2
    assert coordinator.stops == 2


def test_runtime_builds_reference_model(tmp_path):
    config = SpeedtronicConfig.from_dict(
        {
            "run": {"max_steps": 1, "device": "cpu", "output_dir": str(tmp_path)},
            "model": {"vocab_size": 32, "max_seq_len": 8, "n_layer": 1, "n_head": 4, "d_model": 32},
            "data": {"block_size": 8, "num_tokens": 32},
        }
    )
    trainer, _ = build_runtime(config)
    assert isinstance(trainer.model, nn.Module)
    result = trainer.fit()
    assert result.steps == 1
