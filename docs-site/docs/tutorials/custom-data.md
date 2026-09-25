---
id: custom-data
title: Bring Your Own Data
sidebar_label: Custom Data
description: Configure synthetic, text, serialized, and custom datasets while respecting the built-in collator contract.
---

# Bring your own data

Speedtronic supports several data paths, but the built-in collator is intentionally small. Choose the path that matches the model's batch contract.

## Synthetic data

Synthetic data is the default and requires no download:

```yaml
data:
  synthetic: true
  num_tokens: 128
  block_size: 32
  micro_batch_size: 2
  target_batch_size: 8
```

Each sample contains `block_size` input IDs and `block_size` next-token labels. The field `num_tokens` is passed as the sample count in the current implementation.

## Text file

```yaml
data:
  text_path: /absolute/path/to/train.txt
  block_size: 128
  micro_batch_size: 1
  target_batch_size: 8
  num_workers: 0
```

The bundled reader:

- reads UTF-8 incrementally;
- uses a deterministic byte tokenizer by default;
- emits full blocks and one optional short final block;
- pads shorter final samples to the longest sample in a batch.

Use zero workers unless the dataset itself shards by worker. The bundled stream does not.

## Custom tokenizer

Programmatic only:

```python
class Tokenizer:
    def encode(self, text: str):
        return my_integer_ids(text)


loader = build_dataloader(
    config.data,
    tokenizer=Tokenizer(),
    vocab_size=config.model.vocab_size,
)
```

A tokenizer should support incremental or boundary-safe behavior. The bundled stream carries token IDs across reads rather than raw text, which can split context-sensitive tokenizers.

## Injected Dataset

```python
from torch.utils.data import Dataset

from speedtronic.data import build_dataloader


class MyDataset(Dataset):
    def __len__(self):
        return 1000

    def __getitem__(self, index):
        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
        }


loader = build_dataloader(
    config.data,
    dataset=MyDataset(),
    vocab_size=config.model.vocab_size,
)
```

An explicit Dataset takes precedence over `text_path` and synthetic fallback.

## Built-in collator support

`collate_batch()` currently handles:

| Dataset item | Collation |
|---|---|
| Causal dictionary | Pads `input_ids`, `labels`, and `attention_mask` |
| Equal-width tuple/list | Stacks equal-shaped tensor fields |
| Tensor | Stacks into a batch |
| Other | Returns the list unchanged |

For arbitrary dictionaries, build the DataLoader with a custom `collate_fn` and inject the resulting loader into `Trainer`.

## Custom task example

```python
from torch.utils.data import DataLoader, Dataset


class PixelDataset(Dataset):
    def __getitem__(self, index):
        return pixels[index], targets[index]


def pixel_collate(items):
    inputs = torch.stack([item[0] for item in items])
    targets = torch.stack([item[1] for item in items])
    return inputs, targets


loader = DataLoader(
    PixelDataset(),
    batch_size=8,
    collate_fn=pixel_collate,
)
```

The model and loss can then be task-specific.

## Serialized Dataset

YAML can name an existing file:

```yaml
data:
  synthetic: false
  dataset: /trusted/path/dataset.pt
```

The file must deserialize to a `Dataset` or `IterableDataset`. It is loaded with `weights_only=False`, so only load trusted files.

## Source-selection order

```text
explicit dataset argument
→ text_path
→ Dataset object in config.dataset
→ serialized dataset path
→ synthetic fallback
→ error
```

## Workers, prefetch, and pinning

```yaml
data:
  num_workers: 2
  prefetch_factor: 4
  pin_memory: true
```

When workers are enabled, the loader uses persistent workers. `shuffle` is disabled automatically for iterable datasets; implement stream-local shuffling if needed.

## Infinite batching

A finite loader is cycled forever. An entirely empty pass raises:

```text
RuntimeError: data loader produced no batches
```

## Runtime vocabulary behavior

`build_runtime()` passes `model.vocab_size` explicitly to `build_dataloader()`. Therefore, a non-default `data.vocab_size` is ignored through the standard config path. Keep model and data vocabularies aligned or use direct loader construction.

## Checkpoint position

DataLoader position is not checkpointed. Resume restarts iteration and may repeat or skip examples relative to an uninterrupted process, particularly with `shuffle=False`.

## Checklist

- [ ] Dataset returns batches the model accepts.
- [ ] Labels are differentiable-loss compatible.
- [ ] Custom collator handles variable shapes.
- [ ] Iterable datasets shard workers if enabled.
- [ ] Tokenizer is safe at stream boundaries.
- [ ] Dataset files and checkpoints come from trusted sources.
- [ ] Model/data vocabulary is consistent.

Related: [Data API](../reference/data), [Custom Model](./custom-model), and [Checkpoint Resume](./checkpoint-resume).
