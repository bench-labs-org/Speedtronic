## Out-of-Order Backprop

_Opt-in CUDA stream scheduling for autograd's dependency-aware backward engine, with portable fallback._
```yaml
ooo_backprop: true
ooo_streams: 4
```

The feature is off by default and opt-in because its benefit is model and
hardware dependent.

### What PyTorch already does

PyTorch's autograd engine already computes ready nodes from the autograd DAG
rather than forcing a strict Python reverse-layer loop. Its remaining stream
behavior follows the stream that created each forward node. Speedtronic's v2
scheduler supplies streams to disjoint module stages, records producer
dependencies, and synchronizes each stage back to the ambient stream before an
untracked parent operation runs. This preserves correctness while keeping the
scheduling boundary explicit. It is a conservative first implementation and
does not promise kernel overlap on every model.

### Stage selection

1. A model may expose `ooo_stage_names`, a sequence of dotted module names.
2. Otherwise the scheduler expands container modules into disjoint leaf
   stages.
3. Each stage is assigned round-robin to at most `ooo_streams` streams.

For every stage:

- inputs wait on the recorded producer stream;
- tensors crossing streams use `record_stream` for allocator safety;
- stage outputs are associated with their producer stream;
- the ambient stream waits for each stage before untracked parent/container
  operations continue, avoiding a read of an unfinished stage;
- the current stream waits for all stage streams before host-visible loss use.

### Failure and fallback

| Condition | Behavior |
|---|---|
| CPU or MPS | Standard sequential backward; event reports disabled |
| CUDA unavailable | Standard sequential backward |
| Fewer than two stages | Standard sequential backward |
| Setup failure | Hooks removed, warning/event emitted, training continues |
| `compile: true` | OOO disabled with a warning; compiled path wins |
| CUDA graph capture | Not supported by this scheduler |

### Honest performance expectations

Published out-of-order backprop work reports roughly 1.03–1.58× single-GPU
throughput depending on the model, but those results use a lower-level
implementation. Speedtronic's first pure-PyTorch path prioritizes dependency
correctness and conservative synchronization; it is intended to provide a
portable scheduling seam, not a fixed multiplier. A compute-saturated model may
see no speedup, and a small model may see only kernel-overhead changes.

Speedtronic does not claim a fixed multiplier. Benchmark warm-up separately
from steady state and compare against the same model, shapes, precision, and
seed.

### Reference-model behavior

The bundled transformer is mostly a sequential chain at block granularity.
Its value is still explicit stream placement and reduced ambiguity about
producer dependencies, but it cannot expose a large reorderable DAG. Custom
models with independent towers or branches have more scheduling freedom.

### Correctness

The feature changes stream placement, not gradient mathematics. Autograd's
`.grad` accumulation, gradient scaling, clipping, accumulation, and coordinator
boundaries remain in the trainer. CPU parity is the default test path; CUDA
parity and race testing should be run on the target accelerator before enabling
the flag in production.

### Interaction with compilation

`ooo_backprop` and `torch.compile` are mutually exclusive in v2. Dynamo does
not expose a stable API for tracing arbitrary `torch.cuda.stream` contexts
inside a compiled region. Choose one scheduling mode and benchmark it.

### AoT scheduling decision

The addendum's ahead-of-time kernel scheduling item is **deferred as
subsumed** for this release. Inductor already performs graph capture, fusion,
topological scheduling, and autotuning inside `torch.compile`. Speedtronic does
not expose a duplicate `aot_scheduling` flag that would depend on private
PyTorch APIs or claim a hardware-specific speedup.


---

## Shape Validation

_Startup warnings for batch, sequence, token, and model GEMM alignment._
Shape validation catches awkward tensor-core/GEMM dimensions before they
silently reduce throughput.

```yaml
shape_validation:
  enabled: true
  alignment: auto
  check_batch: true
  check_sequence: true
  check_model: true
  check_vocab: false
  warn_on_cpu: false
```

### Automatic profiles

| Device and resolved precision | Automatic alignment |
|---|---:|
| CUDA FP16/BF16 | 8 |
| CUDA FP32/TF32 | None by default; set an explicit alignment for a benchmark profile |
| CPU | None by default |
| MPS | None by default |

An explicit positive `alignment` overrides the profile. On CPU or MPS, set
`warn_on_cpu: true` to enable that explicit benchmark profile. `"none"`
disables warnings.

### Inspected values

- `data.micro_batch_size`;
- `data.block_size`;
- `micro_batch_size * block_size`;
- `nn.Linear` input/output features;
- embedding dimensions;
- 2-D convolution channels;
- optionally vocabulary/output width.

Warnings are deduplicated and include a nearby upward-aligned suggestion when
that suggestion remains valid for the configured target batch. They never
change configuration, memory use, or model architecture. Set
`alignment: none` to disable them.

### Example

```text
shape warning: data.micro_batch_size=3 is not a multiple of 8; consider 8 (kernel batches near this alignment are usually more efficient)
```

If the nearby micro-batch suggestion would violate the target-batch
divisibility rule, `suggested` is emitted as `null` instead.

The warning is a hint, not an error. A user may intentionally choose an
unaligned shape for memory, debugging, or data semantics.

### Custom models

A model may expose:

```python
def speedtronic_shape_metadata(self) -> Mapping[str, int]:
    return {"block1.mlp_hidden": 300}
```

The validator falls back to standard PyTorch module introspection when the
hook is absent or fails. Shape metadata is observational and has no effect on
training.

### Runtime integration

`build_runtime()` resolves precision first, runs validation, then constructs
the optimizer. Direct `Trainer` users receive the same validation on the first
`fit()` call. The `train_start` event includes alignment and warning count.
