"""Dataset and dataloader utilities.

The default data path is a small synthetic token stream so the project is
usable immediately.  Real text files are streamed rather than loaded wholly
into memory, and callers may provide any PyTorch ``Dataset``/tokenizer through
``build_dataloader``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info


class CharTokenizer:
    """A deterministic byte-level tokenizer useful for examples and tests."""

    def __init__(self, vocab_size: int = 256) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        # Byte values are folded into small demo vocabularies; real users
        # should provide their own tokenizer for a language-model vocabulary.
        return [value % self.vocab_size for value in text.encode("utf-8")]

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)


class SyntheticTokenDataset(Dataset[dict[str, torch.Tensor]]):
    """A reproducible random-token dataset for smoke tests and examples."""

    def __init__(
        self,
        num_samples: int = 10_000,
        block_size: int = 128,
        vocab_size: int = 512,
        seed: int = 1234,
    ) -> None:
        if min(num_samples, block_size, vocab_size) <= 0:
            raise ValueError("dataset sizes must be positive")
        self.num_samples = num_samples
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.num_samples
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        # Generator state is deterministic per item and avoids storing a large
        # token tensor in memory.
        generator = torch.Generator().manual_seed(self.generator.initial_seed() + index)
        tokens = torch.randint(
            self.vocab_size,
            (self.block_size + 1,),
            generator=generator,
            dtype=torch.long,
        )
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


class TextFileTokenDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Stream a text file in fixed-size token blocks.

    ``tokenizer`` may be a callable or an object with ``encode``.  The file is
    read incrementally; only one block is materialized at a time.
    """

    def __init__(
        self,
        path: str | Path,
        block_size: int,
        tokenizer: Callable[[str], Any] | Any | None = None,
        vocab_size: int = 256,
    ) -> None:
        self.path = Path(path)
        self.block_size = int(block_size)
        self.tokenizer = tokenizer or CharTokenizer(vocab_size)
        self.vocab_size = vocab_size
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

    def _encode(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            result = self.tokenizer.encode(text)
        else:
            result = self.tokenizer(text)
        if hasattr(result, "tolist"):
            result = result.tolist()
        return [int(token) for token in result]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        carry: list[int] = []
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        block_index = 0
        with self.path.open("r", encoding="utf-8") as handle:
            while True:
                chunk = handle.read(self.block_size * 4)
                if not chunk:
                    break
                tokens = self._encode(chunk)
                if not tokens:
                    continue
                combined = carry + tokens
                # Keep one token for the input/target shift.
                complete = (len(combined) // (self.block_size + 1)) * (self.block_size + 1)
                for start in range(0, complete, self.block_size + 1):
                    if block_index % worker_count == worker_id:
                        block = combined[start : start + self.block_size + 1]
                        tensor = torch.tensor(block, dtype=torch.long)
                        yield {"input_ids": tensor[:-1], "labels": tensor[1:]}
                    block_index += 1
                carry = combined[complete:]
        if len(carry) >= 2 and block_index % worker_count == worker_id:
            # Keep a short final block; collate_causal pads it and marks the
            # padded positions as ignored rather than training on fake tokens.
            block = torch.tensor(carry, dtype=torch.long)
            yield {"input_ids": block[:-1], "labels": block[1:]}


def collate_causal(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for row, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[row, :length] = item["input_ids"]
        labels[row, :length] = item["labels"]
        attention_mask[row, :length] = True
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def collate_batch(batch: list[Any]) -> Any:
    """Collate causal dictionaries or homogeneous tuple/list datasets."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    if isinstance(batch[0], dict):
        return collate_causal(batch)
    if isinstance(batch[0], (tuple, list)):
        width = min(len(item) for item in batch)
        if any(len(item) != width for item in batch):
            raise ValueError("tuple dataset items must have the same number of fields")
        columns = []
        for index in range(width):
            values = [item[index] for item in batch]
            if isinstance(values[0], torch.Tensor):
                shapes = {tuple(value.shape) for value in values}
                if len(shapes) != 1:
                    raise ValueError("tensor dataset fields must have equal shapes")
                columns.append(torch.stack(values))
            else:
                columns.append(values)
        return tuple(columns)
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch)
    return batch


def _cycle(loader: Iterable[Any]) -> Iterator[Any]:
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise RuntimeError("data loader produced no batches")


def infinite_batches(loader: Iterable[Any]) -> Iterator[Any]:
    """Cycle a finite loader, making max-step runs independent of data size."""

    return _cycle(loader)


def build_dataloader(
    config: Any,
    *,
    dataset: Dataset | IterableDataset | None = None,
    tokenizer: Any | None = None,
    pin_memory_device: bool = False,
    vocab_size: int | None = None,
) -> DataLoader:
    """Build a loader from a DataConfig.

    ``dataset`` is the primary extension point for user-provided datasets.  A
    text path takes precedence over the synthetic fallback unless a dataset is
    explicitly supplied.
    """

    block_size = int(getattr(config, "block_size", None) or 128)
    vocab_size = int(vocab_size or getattr(config, "vocab_size", None) or 512)
    if dataset is None:
        text_path = getattr(config, "text_path", None)
        dataset_spec = getattr(config, "dataset", None)
        if text_path:
            dataset = TextFileTokenDataset(text_path, block_size, tokenizer, vocab_size)
        elif isinstance(dataset_spec, (Dataset, IterableDataset)):
            dataset = dataset_spec
        elif dataset_spec is not None:
            dataset_path = Path(dataset_spec)
            if not dataset_path.exists():
                raise ValueError(
                    "data.dataset must be a Dataset object or an existing local "
                    f"path: {dataset_spec}"
                )
            try:
                dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
            except TypeError:  # older torch
                dataset = torch.load(dataset_path, map_location="cpu")
            if not isinstance(dataset, (Dataset, IterableDataset)):
                raise ValueError("serialized data.dataset must contain a PyTorch Dataset")
        elif not bool(getattr(config, "synthetic", True)):
            raise ValueError("data.synthetic=false requires data.text_path or data.dataset")
        else:
            dataset = SyntheticTokenDataset(
                num_samples=int(getattr(config, "num_tokens", 10_000)),
                block_size=block_size,
                vocab_size=vocab_size,
                seed=int(getattr(config, "seed", 0) or 0),
            )
    workers = int(getattr(config, "num_workers", 0) or 0)
    requested_pin = getattr(config, "pin_memory", None)
    pin = bool(pin_memory_device if requested_pin is None else requested_pin)
    shuffle = bool(getattr(config, "shuffle", False))
    if isinstance(dataset, IterableDataset):
        # PyTorch does not support shuffling an IterableDataset through the
        # DataLoader sampler; callers can implement shuffling in their stream.
        shuffle = False
    kwargs: dict[str, Any] = {
        "batch_size": int(config.micro_batch_size),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin,
        "drop_last": bool(getattr(config, "drop_last", True)),
        "collate_fn": collate_batch,
    }
    if workers > 0:
        prefetch = getattr(config, "prefetch_factor", None) or 2
        kwargs["prefetch_factor"] = int(prefetch)
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


__all__ = [
    "CharTokenizer",
    "SyntheticTokenDataset",
    "TextFileTokenDataset",
    "build_dataloader",
    "collate_batch",
    "collate_causal",
    "infinite_batches",
]
