import torch

from speedtronic.model import GPTConfig, ReferenceTransformer


def test_reference_transformer_forward_and_loss():
    model = ReferenceTransformer(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_kv_head=2,
        d_model=32,
        d_ff=64,
    )
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    output = model(x, labels=y)
    assert output["logits"].shape == (2, 8, 32)
    assert output["loss"].ndim == 0
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_reference_transformer_weight_tying_and_checkpoint_hook():
    model = ReferenceTransformer(vocab_size=32, block_size=8, n_layer=1, n_head=4, d_model=32)
    assert model.lm_head.weight is model.transformer["wte"].weight
    model.set_gradient_checkpointing(True)
    assert all(block.gradient_checkpointing for block in model.transformer["h"])
    model.set_gradient_checkpointing(False)
    assert not any(block.gradient_checkpointing for block in model.transformer["h"])


def test_config_dimensions():
    config = GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=4, d_model=32)
    assert config.d_ff == 128
