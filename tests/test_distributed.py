import torch

from speedtronic.distributed.outer import NesterovOuterOptimizer
from speedtronic.distributed.tensors import (
    average_deltas,
    compute_pseudo_gradient,
    parse_delta_path,
)
from speedtronic.model import ReferenceTransformer


def test_pseudo_gradient_direction_and_average():
    baseline = {"w": torch.tensor([3.0, 5.0])}
    current = {"w": torch.tensor([1.0, 4.0])}
    delta = compute_pseudo_gradient(baseline, current)
    assert torch.equal(delta["w"], torch.tensor([2.0, 1.0]))
    mean = average_deltas([delta, {"w": torch.tensor([0.0, 3.0])}])
    assert torch.equal(mean["w"], torch.tensor([1.0, 2.0]))


def test_nesterov_outer_optimizer():
    state = {"w": torch.tensor([1.0, 1.0])}
    optimizer = NesterovOuterOptimizer(state, lr=0.1, momentum=0.9)
    optimizer.step(state, {"w": torch.tensor([1.0, 2.0])})
    # First effective gradient is g + mu * g = 1.9*g.
    assert torch.allclose(state["w"], torch.tensor([0.81, 0.62]))
    optimizer.step(state, {"w": torch.tensor([1.0, 2.0])})
    assert state["w"].shape == (2,)


def test_delta_path_parser():
    assert parse_delta_path("nodes/a/delta_12.safetensors") == ("a", 12)
    assert parse_delta_path("global/latest.safetensors") is None


def test_tied_model_state_can_be_safetensors_serialized(tmp_path):
    from speedtronic.distributed.tensors import cpu_state_dict, load_safetensors, save_safetensors

    model = ReferenceTransformer(vocab_size=32, block_size=8, n_layer=1, n_head=4, d_model=32)
    path = tmp_path / "state.safetensors"
    save_safetensors(cpu_state_dict(model.state_dict()), path)
    assert len(load_safetensors(path)) == len(model.state_dict())
