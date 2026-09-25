import logging

import torch

from speedtronic.config import SpeedtronicConfig
from speedtronic.model import ReferenceTransformer
from speedtronic.precision import PrecisionPlan
from speedtronic.runtime import build_runtime
from speedtronic.scheduling import StageStreamScheduler
from speedtronic.shapes import resolve_shape_profile, validate_startup_shapes
from speedtronic.trainer import Trainer


def test_cpu_shape_auto_profile_is_quiet_but_explicit_alignment_warns():
    plan = PrecisionPlan(
        mode="bf16",
        dtype=torch.bfloat16,
        use_scaler=False,
        autocast_enabled=True,
        device_type="cpu",
    )
    assert resolve_shape_profile("cpu", plan).alignment is None
    config = SpeedtronicConfig.from_dict(
        {
            "model": {"d_model": 12, "n_head": 3, "max_seq_len": 8, "n_layer": 1},
            "data": {"micro_batch_size": 1, "target_batch_size": 8, "block_size": 5},
            "shape_validation": {"alignment": 8, "warn_on_cpu": True},
        }
    )
    model = ReferenceTransformer(
        vocab_size=16,
        block_size=8,
        n_layer=1,
        n_head=3,
        n_kv_head=1,
        d_model=12,
        d_ff=24,
    )
    report = validate_startup_shapes(config, model, "cpu", plan)
    assert report.warning_count > 0
    assert any(item.code == "batch_alignment" for item in report.warnings)
    assert any(item.code == "model_alignment" for item in report.warnings)
    assert all(item.suggested is not None for item in report.warnings)


def test_shape_report_is_non_fatal_and_deduplicated():
    plan = PrecisionPlan("fp32", torch.float32, False, False, "cpu")
    config = SpeedtronicConfig.from_dict(
        {
            "model": {"d_model": 12, "n_head": 3, "max_seq_len": 8, "n_layer": 1},
            "data": {"micro_batch_size": 1, "target_batch_size": 8, "block_size": 5},
            "shape_validation": {"alignment": 8, "warn_on_cpu": True},
        }
    )
    model = ReferenceTransformer(
        vocab_size=16, block_size=8, n_layer=1, n_head=3, n_kv_head=1, d_model=12, d_ff=24
    )
    first = validate_startup_shapes(config, model, "cpu", plan)
    second = validate_startup_shapes(config, model, "cpu", plan)
    assert first.as_dict() == second.as_dict()
    assert first.profile.alignment == 8


def test_explicit_cpu_profile_requires_opt_in_and_avoids_invalid_batch_suggestion():
    plan = PrecisionPlan("fp32", torch.float32, False, False, "cpu")
    config = SpeedtronicConfig.from_dict(
        {
            "model": {"d_model": 16, "n_head": 4, "max_seq_len": 8, "n_layer": 1},
            "data": {"micro_batch_size": 3, "target_batch_size": 9, "block_size": 8},
            "shape_validation": {"alignment": 8, "warn_on_cpu": True},
        }
    )
    model = ReferenceTransformer(
        vocab_size=16, block_size=8, n_layer=1, n_head=4, n_kv_head=1, d_model=16, d_ff=32
    )
    report = validate_startup_shapes(config, model, "cpu", plan)
    batch = next(item for item in report.warnings if item.code == "batch_alignment")
    assert batch.suggested is None
    quiet = SpeedtronicConfig.from_dict(
        {
            "model": {"d_model": 16, "n_head": 4, "max_seq_len": 8, "n_layer": 1},
            "data": {"micro_batch_size": 3, "target_batch_size": 9, "block_size": 8},
            "shape_validation": {"alignment": 8},
        }
    )
    assert validate_startup_shapes(quiet, model, "cpu", plan).warning_count == 0


def test_shape_validation_accepts_standard_library_logger():
    plan = PrecisionPlan("fp32", torch.float32, False, False, "cpu")
    config = SpeedtronicConfig.from_dict(
        {
            "model": {"d_model": 12, "n_head": 3, "max_seq_len": 8, "n_layer": 1},
            "data": {"micro_batch_size": 1, "target_batch_size": 8, "block_size": 5},
            "shape_validation": {"alignment": 8, "warn_on_cpu": True},
        }
    )
    model = ReferenceTransformer(
        vocab_size=16, block_size=8, n_layer=1, n_head=3, n_kv_head=1, d_model=12, d_ff=24
    )
    report = validate_startup_shapes(config, model, "cpu", plan, logger=logging.getLogger(__name__))
    assert report.warning_count > 0


def test_scheduler_default_follows_run_target():
    config = SpeedtronicConfig.from_dict({"run": {"max_steps": 7}})
    assert config.scheduler.max_steps == 7


def test_causal_inferred_loss_uses_already_shifted_labels():
    logits = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    labels = torch.tensor([[0, 1, 2], [3, 0, 1]])
    loss = Trainer._loss_from_logits(logits, {"labels": labels})
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1))
    assert torch.allclose(loss, expected)


def test_out_of_order_scheduler_degrades_to_noop_on_cpu():
    model = ReferenceTransformer(
        vocab_size=16, block_size=8, n_layer=1, n_head=4, n_kv_head=2, d_model=32
    )
    scheduler = StageStreamScheduler(model, "cpu", num_streams=4)
    assert scheduler.enabled is False
    assert "cpu" in scheduler.reason
    scheduler.begin()
    scheduler.finish()
    scheduler.dispose()


def test_out_of_order_config_runs_without_changing_cpu_path(tmp_path):
    config = SpeedtronicConfig.from_dict(
        {
            "run": {"max_steps": 1, "device": "cpu", "output_dir": str(tmp_path)},
            "model": {
                "vocab_size": 16,
                "max_seq_len": 8,
                "n_layer": 1,
                "n_head": 4,
                "n_kv_head": 2,
                "d_model": 32,
            },
            "data": {"block_size": 8, "micro_batch_size": 1, "target_batch_size": 1},
            "precision": {"mode": "fp32"},
            "ooo_backprop": True,
            "ooo_streams": 2,
        }
    )
    trainer, _ = build_runtime(config)
    assert trainer._ooo_scheduler is not None
    assert trainer._ooo_scheduler.enabled is False
    result = trainer.fit()
    assert result.steps == 1
