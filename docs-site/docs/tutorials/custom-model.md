---
id: custom-model
title: Bring Your Own Model
sidebar_label: Custom Model
description: Register and train a model that follows Speedtronic's batch, loss, and gradient-checkpointing contracts.
---

# Bring your own model

The training loop is architecture-neutral. A custom model must follow the batch and loss protocols and be constructible through a registry factory.

## Step 1: Follow the batch contract

Dictionary batches are passed as keyword arguments. The bundled loader emits:

```text
input_ids
labels
attention_mask
```

A tuple/list batch is passed as:

```python
model(batch[0], batch[1])
```

## Step 2: Return a usable loss

The model can return:

```python
loss_tensor
```

```python
{"loss": loss_tensor, "auxiliary": ...}
```

```python
{"logits": logits, ...}
```

or a tuple/list whose first element is a loss or compatible 3-D logits.

A bare tensor is always interpreted as a loss, not logits.

## Step 3: Register a factory

```python
import torch
from torch import nn
from torch.nn import functional as F

from speedtronic import register_model


class SmallCausalLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids,
        labels=None,
        attention_mask=None,
    ):
        logits = self.output(self.embedding(input_ids))
        if labels is None:
            return {"logits": logits}

        if labels.shape[1] == input_ids.shape[1]:
            labels = labels[:, 1:]

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return {"logits": logits, "loss": loss}


@register_model("small_causal_lm")
def make_small_causal_lm(**kwargs):
    return SmallCausalLM(
        vocab_size=kwargs["vocab_size"],
        d_model=kwargs["d_model"],
    )
```

Factories receive the standard model fields even when they are unrelated to the custom architecture. Accepting `**kwargs` avoids accidental construction errors.

## Step 4: Train programmatically

```python
from speedtronic.runtime import train_from_config

result = train_from_config(
    {
        "run": {
            "max_steps": 10,
            "device": "cpu",
            "output_dir": "runs/small-lm",
        },
        "model": {
            "name": "small_causal_lm",
            "vocab_size": 128,
            "max_seq_len": 32,
            "n_layer": 2,
            "n_head": 4,
            "n_kv_head": 2,
            "d_model": 64,
        },
        "data": {
            "block_size": 32,
            "micro_batch_size": 2,
            "target_batch_size": 4,
        },
        "scheduler": {"warmup_steps": 1, "max_steps": 10},
        "precision": {"mode": "fp32"},
    }
)
```

## CLI limitation

Registry entries are in-memory. A separate `speedtronic train --config ...` process does not import arbitrary application registration code. YAML can select a registered name, but it cannot discover a Python factory by itself.

For a CLI-visible custom model, add an application-owned import/plugin mechanism and construct the runtime after registration.

## Non-language-model model

A model can ignore causal keys by accepting them explicitly and returning any differentiable scalar loss:

```python
class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, labels=None, attention_mask=None):
        if labels is None:
            raise ValueError("labels are required")
        return (self.value - labels.float().mean()) ** 2
```

The built-in collator still produces language-model-shaped batches, so use a custom DataLoader for other batch structures.

## Direct Trainer injection

For complete control:

```python
from speedtronic.trainer import Trainer

trainer = Trainer(
    model,
    optimizer,
    dataloader,
    device="cpu",
    max_steps=100,
)
result = trainer.fit()
```

## Gradient checkpointing hook

```python
class MyModule(nn.Module):
    def set_gradient_checkpointing(self, enabled: bool = True):
        self.use_checkpointing = bool(enabled)
```

The trainer warns rather than fails if the hook is absent or raises.

## Model-returned metrics

The trainer currently discards additional output-dictionary metrics. If metrics are required, send them through a hook, a global collector, or a custom Trainer implementation.

## Checklist

- [ ] Model accepts the actual batch keys.
- [ ] Output begins with a differentiable loss under the trainer's interpretation.
- [ ] Factory accepts standard registry keywords.
- [ ] Registration occurs in the same process as runtime construction.
- [ ] Model context and data block sizes are compatible.
- [ ] Custom batch shapes have a matching collator.
- [ ] Gradient checkpointing exposes the optional hook.

Related: [Extensions](../reference/extensions), [Data](./custom-data), and [Runtime and Trainer](../reference/runtime-and-trainer).
