import math

import pytest
import torch
from torch import nn

from speedtronic.config import ConfigError, SpeedtronicConfig
from speedtronic.model import ReferenceTransformer
from speedtronic.optimizers import (
    CautiousOptimizer,
    HybridOptimizer,
    Muon,
    newton_schulz,
    post_polar_normalize,
    route_parameters,
)
from speedtronic.runtime import build_optimizer, build_scheduler


def test_v2_config_accepts_scalar_optimizer_and_root_flags():
    config = SpeedtronicConfig.from_dict(
        {
            "optimizer": "muon",
            "muon_plus": True,
            "cautious": True,
            "ooo_backprop": True,
            "ooo_streams": 2,
        }
    )
    assert config.optimizer.name == "muon"
    assert config.optimizer.muon_plus is True
    assert config.optimizer.cautious is True
    assert config.ooo_backprop is True
    assert config.ooo_streams == 2


def test_muon_plus_requires_muon():
    with pytest.raises(ConfigError):
        SpeedtronicConfig.from_dict({"optimizer": {"name": "adamw", "muon_plus": True}})


def test_v2_config_rejects_invalid_systems_values():
    with pytest.raises(ConfigError):
        SpeedtronicConfig.from_dict({"ooo_streams": 0})
    with pytest.raises(ConfigError):
        SpeedtronicConfig.from_dict({"shape_validation": {"alignment": 0}})


def test_newton_schulz_is_pure_and_finite():
    original = torch.randn(5, 3)
    snapshot = original.clone()
    result = newton_schulz(original, steps=5)
    assert result.shape == original.shape
    assert torch.isfinite(result).all()
    assert torch.equal(original, snapshot)
    assert torch.isfinite(newton_schulz(torch.zeros(3, 4))).all()


def test_newton_schulz_handles_rectangular_matrices():
    for shape in ((3, 8), (8, 3), (1, 5), (5, 1)):
        result = newton_schulz(torch.randn(shape), steps=4)
        assert result.shape == shape
        assert torch.isfinite(result).all()


def test_post_polar_normalization_produces_unit_rows_and_columns():
    update = torch.randn(4, 3) * 10
    result = post_polar_normalize(update)
    assert torch.allclose(result.square().sum(dim=0).sqrt(), torch.ones(3), atol=1e-4)
    assert torch.isfinite(result).all()


def test_reference_parameter_routing_is_role_aware_and_tied_safe():
    model = ReferenceTransformer(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_kv_head=2,
        d_model=32,
        d_ff=64,
    )
    routing = route_parameters(model)
    assert len(routing.muon) == 7
    assert len(routing.adamw) == 4
    assert all(parameter.ndim == 2 for _, parameter in routing.muon)
    assert model.lm_head.weight is model.transformer["wte"].weight
    assert sum(parameter is model.transformer["wte"].weight for _, parameter in routing.adamw) == 1


def test_hybrid_optimizer_updates_both_branches_and_scheduler():
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    config = SpeedtronicConfig.from_dict(
        {"optimizer": {"name": "muon", "muon_plus": True}, "model": {"d_model": 4, "n_head": 1}}
    )
    optimizer = build_optimizer(model, config, torch.device("cpu"))
    assert isinstance(optimizer, HybridOptimizer)
    scheduler = build_scheduler(optimizer, config)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    scheduler.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert len(optimizer.state) >= 2


def test_adamw_never_builds_an_empty_optimizer_for_all_linear_model():
    model = nn.Linear(4, 3, bias=False)
    config = SpeedtronicConfig.from_dict({"optimizer": {"name": "adamw"}})
    optimizer = build_optimizer(model, config, torch.device("cpu"))
    assert sum(len(group["params"]) for group in optimizer.param_groups) == 1


def test_non_linear_custom_matrix_defaults_to_adamw():
    class Custom(nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = nn.Parameter(torch.zeros(3, 2))

    routing = route_parameters(Custom())
    assert len(routing.muon) == 0
    assert len(routing.adamw) == 1


def test_hybrid_optimizer_forwards_closure_once():
    model = nn.Linear(2, 2)
    config = SpeedtronicConfig.from_dict({"optimizer": {"name": "muon"}})
    optimizer = build_optimizer(model, config, torch.device("cpu"))
    calls = []

    def closure():
        calls.append(True)
        return torch.tensor(0.0)

    optimizer.step(closure)
    assert calls == [True]


def test_cautious_wrapper_is_composable_and_finite():
    model = nn.Linear(4, 3)
    base = torch.optim.AdamW(model.parameters(), lr=0.1)
    optimizer = CautiousOptimizer(base)
    model(torch.randn(5, 4)).sum().backward()
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert optimizer.state_dict().keys() == {"state", "param_groups"}


def test_muon_rejects_non_matrix_parameters():
    parameter = nn.Parameter(torch.zeros(3))
    optimizer = Muon([parameter], lr=0.1)
    parameter.grad = torch.ones(3)
    with pytest.raises(ValueError):
        optimizer.step()


def test_muon_sparse_gradient_is_rejected():
    parameter = nn.Parameter(torch.zeros(3, 3))
    optimizer = Muon([parameter], lr=0.1)
    parameter.grad = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]),
        torch.tensor([1.0]),
        (3, 3),
        check_invariants=False,
    )
    with pytest.raises(RuntimeError):
        optimizer.step()


def test_v2_muon_runtime_smoke(tmp_path):
    config = SpeedtronicConfig.from_dict(
        {
            "run": {"max_steps": 1, "device": "cpu", "output_dir": str(tmp_path)},
            "model": {
                "name": "reference_transformer",
                "vocab_size": 32,
                "max_seq_len": 8,
                "n_layer": 1,
                "n_head": 4,
                "n_kv_head": 2,
                "d_model": 32,
                "d_ff": 64,
            },
            "optimizer": {"name": "muon", "muon_plus": True},
            "data": {"block_size": 8, "micro_batch_size": 1, "target_batch_size": 2},
            "precision": {"mode": "fp32"},
            "scheduler": {"warmup_steps": 1, "max_steps": 1},
        }
    )
    optimizer = build_optimizer(
        ReferenceTransformer(
            vocab_size=32, block_size=8, n_layer=1, n_head=4, n_kv_head=2, d_model=32, d_ff=64
        ),
        config,
        torch.device("cpu"),
    )
    assert isinstance(optimizer, HybridOptimizer)
    assert math.isfinite(float(next(iter(optimizer.param_groups))["lr"]))
