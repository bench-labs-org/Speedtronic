from pathlib import Path

import torch

from speedtronic.data import TextFileTokenDataset, collate_batch
from speedtronic.precision import resolve_precision


def test_cpu_precision_defaults_to_fp32():
    plan = resolve_precision(
        type("Precision", (), {"mode": "auto", "dtype": None})(), torch.device("cpu")
    )
    assert plan.mode == "fp32"
    assert not plan.use_scaler
    explicit_fp16 = resolve_precision(
        type("Precision", (), {"mode": "fp16", "dtype": None})(), torch.device("cpu")
    )
    assert explicit_fp16.mode == "fp32"


def test_streaming_text_dataset_and_tuple_collate(tmp_path: Path):
    path = tmp_path / "train.txt"
    path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    dataset = TextFileTokenDataset(path, block_size=4, vocab_size=32)
    blocks = list(dataset)
    assert blocks and blocks[0]["input_ids"].shape == (4,)
    batch = collate_batch([(torch.ones(2), torch.zeros(2)), (torch.ones(2), torch.zeros(2))])
    assert isinstance(batch, tuple)
    assert batch[0].shape == (2, 2)
