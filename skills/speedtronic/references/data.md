## Data Pipeline API

_Speedtronic tokenizers, datasets, collators, infinite batching, and DataLoader construction._
### Source-selection order

`build_dataloader()` selects the first available source:

1. Explicit `dataset=` argument.
2. `data.text_path`.
3. A live Dataset object stored in `data.dataset` when called programmatically.
4. A serialized dataset path.
5. Synthetic data when `synthetic=true`.
6. Error when synthetic data is disabled and no source exists.

### `CharTokenizer`

```python
CharTokenizer(vocab_size: int = 256)
```

UTF-8 byte tokenizer:

```python
[byte % vocab_size for byte in text.encode("utf-8")]
```

It is deterministic and stream-friendly but not reversible when `vocab_size < 256`.

Methods:

- `encode(text) -> list[int]`
- `__call__(text) -> list[int]`

### `SyntheticTokenDataset`

```python
SyntheticTokenDataset(
    num_samples=10_000,
    block_size=128,
    vocab_size=512,
    seed=1234,
)
```

A map-style dataset that returns:

```python
{
    "input_ids": tokens[:-1],
    "labels": tokens[1:],
}
```

Each index uses a generator seeded from the dataset seed plus the index, making item contents independent of access order. Negative indexes are normalized from the end.

`DataConfig.num_tokens` is passed as `num_samples` by the runtime, despite its name.

### `TextFileTokenDataset`

```python
TextFileTokenDataset(
    path,
    block_size,
    tokenizer=None,
    vocab_size=256,
)
```

An `IterableDataset` that reads UTF-8 text in chunks of `block_size * 4` characters. It combines tokenized chunks with a carry, emits groups of `block_size + 1` tokens, and retains the remainder. A final carry of at least two tokens is emitted as a short block.

The default tokenizer is `CharTokenizer`.

#### Tokenizer contract

A tokenizer may expose:

- `encode(text)`, or
- `__call__(text)`.

Tensor-like results are converted through `.tolist()`; every token is cast to `int`.

> **Caution** — Boundary-sensitive tokenizers
>
>
> The stream carries token IDs, not raw text. This is safe for the bundled byte tokenizer but can split merges or normalization-sensitive BPE/SentencePiece tokens at read boundaries.
>

> **Caution** — Iterable workers
>
>
> `TextFileTokenDataset` does not shard by DataLoader worker. With `num_workers > 1`, every worker can read the same file. Implement a worker-aware iterable dataset for parallel production text ingestion.
>

### `collate_causal(batch)`

Pads a list of causal dictionaries to the longest input:

| Field | Padding |
|---|---|
| `input_ids` | Integer zero |
| `labels` | Integer `-100` |
| `attention_mask` | Boolean false |

Every item must contain `input_ids` and `labels` with compatible shapes.

### `collate_batch(batch)`

| First item | Result |
|---|---|
| `dict` | Delegate to `collate_causal` |
| `tuple` or `list` | Transpose equal-width items; stack equal-shaped tensor columns |
| `torch.Tensor` | Stack tensors |
| Other | Return the original list |

Empty batches raise `ValueError`. Tuple/list item widths must match.

The function is not a universal PyTorch collator: arbitrary dictionaries are interpreted as causal language-model records.

### `infinite_batches(loader)`

Returns an iterator that repeatedly traverses the loader. If one complete pass yields no batches, it raises `RuntimeError("data loader produced no batches")`.

This makes global-step training independent of finite dataset length.

### `build_dataloader(...)`

```python
build_dataloader(
    config,
    *,
    dataset=None,
    tokenizer=None,
    pin_memory_device=False,
    vocab_size=None,
) -> DataLoader
```

#### Dataset construction

- Serialized datasets load through `torch.load(..., weights_only=False)` and must contain a PyTorch Dataset.
- Synthetic data receives `block_size`, explicit/runtime `vocab_size`, and `data.seed`.

#### Loader arguments

| Argument | Behavior |
|---|---|
| `batch_size` | `data.micro_batch_size` |
| `shuffle` | `data.shuffle`, forced false for iterable datasets |
| `num_workers` | `data.num_workers` |
| `pin_memory` | Explicit `data.pin_memory`, otherwise `pin_memory_device` |
| `drop_last` | `data.drop_last` |
| `collate_fn` | `collate_batch` |
| `prefetch_factor` | `data.prefetch_factor` or 2 when workers enabled |
| `persistent_workers` | True when workers enabled |

#### Precedence and vocab caveats

The explicit `vocab_size` argument wins. `build_runtime()` always passes `model.vocab_size`, so a separately configured `data.vocab_size` has no effect through the standard YAML path.

### Programmatic custom dataset

```python
from speedtronic.data import build_dataloader

loader = build_dataloader(
    config.data,
    dataset=my_dataset,
    tokenizer=my_tokenizer,
    pin_memory_device=False,
    vocab_size=config.model.vocab_size,
)
```

To use the custom loader through the config-only runtime, inject it or construct `Trainer` directly; `build_runtime()` does not expose dataset/tokenizer override parameters.

### Token and sample accounting

Trainer token counts are based on input tensor shape, not tokenizer semantics. A dictionary input with an attention mask uses the mask sum. This can differ from a tokenizer's reported token count after normalization or special tokens.

### Text configuration

```yaml
data:
  text_path: /absolute/path/to/train.txt
  block_size: 128
  micro_batch_size: 2
  target_batch_size: 8
  num_workers: 0
  prefetch_factor: null
  pin_memory: null
  shuffle: false
  drop_last: true
```

Use `num_workers: 0` with the bundled text dataset unless you provide a sharding implementation.

For the trainer-facing contracts, continue with [Runtime and Trainer](references/architecture.md) and [Custom data](references/data.md).


---

## Bring Your Own Data

_Configure synthetic, text, serialized, and custom datasets while respecting the built-in collator contract._
Speedtronic supports several data paths, but the built-in collator is intentionally small. Choose the path that matches the model's batch contract.

### Synthetic data

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

### Text file

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

### Custom tokenizer

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

### Injected Dataset

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

### Built-in collator support

`collate_batch()` currently handles:

| Dataset item | Collation |
|---|---|
| Causal dictionary | Pads `input_ids`, `labels`, and `attention_mask` |
| Equal-width tuple/list | Stacks equal-shaped tensor fields |
| Tensor | Stacks into a batch |
| Other | Returns the list unchanged |

For arbitrary dictionaries, build the DataLoader with a custom `collate_fn` and inject the resulting loader into `Trainer`.

### Custom task example

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

### Serialized Dataset

YAML can name an existing file:

```yaml
data:
  synthetic: false
  dataset: /trusted/path/dataset.pt
```

The file must deserialize to a `Dataset` or `IterableDataset`. It is loaded with `weights_only=False`, so only load trusted files.

### Source-selection order

```text
explicit dataset argument
→ text_path
→ Dataset object in config.dataset
→ serialized dataset path
→ synthetic fallback
→ error
```

### Workers, prefetch, and pinning

```yaml
data:
  num_workers: 2
  prefetch_factor: 4
  pin_memory: true
```

When workers are enabled, the loader uses persistent workers. `shuffle` is disabled automatically for iterable datasets; implement stream-local shuffling if needed.

### Infinite batching

A finite loader is cycled forever. An entirely empty pass raises:

```text
RuntimeError: data loader produced no batches
```

### Runtime vocabulary behavior

`build_runtime()` passes `model.vocab_size` explicitly to `build_dataloader()`. Therefore, a non-default `data.vocab_size` is ignored through the standard config path. Keep model and data vocabularies aligned or use direct loader construction.

### Checkpoint position

DataLoader position is not checkpointed. Resume restarts iteration and may repeat or skip examples relative to an uninterrupted process, particularly with `shuffle=False`.

### Checklist

- [ ] Dataset returns batches the model accepts.
- [ ] Labels are differentiable-loss compatible.
- [ ] Custom collator handles variable shapes.
- [ ] Iterable datasets shard workers if enabled.
- [ ] Tokenizer is safe at stream boundaries.
- [ ] Dataset files and checkpoints come from trusted sources.
- [ ] Model/data vocabulary is consistent.

Related: [Data API](references/data.md), [Custom Model](references/models-and-extensions.md), and [Checkpoint Resume](references/checkpointing.md).
