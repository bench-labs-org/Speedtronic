from pathlib import Path

import pytest

from speedtronic.config import ConfigError, SpeedtronicConfig


def test_config_yaml_aliases_and_accumulation():
    config = SpeedtronicConfig.from_dict(
        {
            "name": "unit",
            "max_steps": 7,
            "model": {"vocab_size": 128, "max_seq_len": 16, "d_model": 32, "n_head": 4},
            "data": {"micro_batch_size": 2, "target_batch_size": 6},
            "compile": {"enabled": True},
        }
    )
    assert config.run.name == "unit"
    assert config.run.max_steps == 7
    assert config.accumulation_steps == 3
    assert config.compile is True
    assert config.model.n_kv_head == 4


def test_config_round_trip(tmp_path: Path):
    config = SpeedtronicConfig.from_dict({"max_steps": 2, "precision": {"mode": "fp32"}})
    path = tmp_path / "config.yaml"
    config.save(path)
    loaded = SpeedtronicConfig.load(path)
    assert loaded.to_dict() == config.to_dict()


def test_unknown_keys_and_invalid_batch():
    with pytest.raises(ConfigError):
        SpeedtronicConfig.from_dict({"not_a_key": 1})
    with pytest.raises(ConfigError):
        SpeedtronicConfig.from_dict({"data": {"micro_batch_size": 3, "target_batch_size": 4}})


def test_distributed_role_inference():
    config = SpeedtronicConfig.from_dict({"distributed": {"enabled": True, "repo_id": "org/run"}})
    assert config.distributed.role == "master"


def test_secret_can_be_redacted_for_serialization(tmp_path: Path):
    config = SpeedtronicConfig.from_dict({"distributed": {"token": "secret", "repo_id": "org/run"}})
    assert config.to_dict()["distributed"]["token"] == "secret"
    assert config.to_dict(redact_secrets=True)["distributed"]["token"] == "<redacted>"
    path = tmp_path / "redacted.yaml"
    config.save(path)
    assert "secret" not in path.read_text(encoding="utf-8")
