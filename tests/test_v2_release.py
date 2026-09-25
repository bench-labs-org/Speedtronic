from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test environment
    import tomli as tomllib

from speedtronic import SpeedtronicConfig, __version__
from speedtronic.cli import main


def test_version_is_synchronized():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == "2.0.0"
    assert metadata["project"]["version"] == __version__


def test_cli_validate_redacts_configured_token(capsys, tmp_path):
    config = tmp_path / "secret.yaml"
    config.write_text(
        "distributed:\n  enabled: false\n  repo_id: org/run\n  token: super-secret\n",
        encoding="utf-8",
    )
    assert main(["validate", "--config", str(config)]) == 0
    output = capsys.readouterr().out
    assert "super-secret" not in output
    assert "<redacted>" in output


def test_v2_config_serializes_all_new_sections():
    config = SpeedtronicConfig.from_dict(
        {
            "optimizer": {"name": "muon", "muon_plus": True, "cautious": True},
            "shape_validation": {"alignment": 8},
            "ooo_backprop": True,
            "distributed": {
                "enabled": True,
                "repo_id": "org/run",
                "role": "master",
                "async_delta_upload": True,
            },
        }
    )
    value = config.to_dict()
    assert value["optimizer"]["muon_plus"] is True
    assert value["shape_validation"]["alignment"] == 8
    assert value["ooo_backprop"] is True
    assert value["distributed"]["async_delta_upload"] is True
