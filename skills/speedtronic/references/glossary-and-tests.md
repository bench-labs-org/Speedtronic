## Glossary

_Definitions of Speedtronic training, model, checkpoint, and DumbDiLoCo terminology._
### Absolute step

A global optimizer-step target rather than a count of additional updates. A trainer at step 900 targeting 1000 performs 100 updates.

### Accumulation

Multiple microbatches whose gradients are summed and applied in one optimizer update. Speedtronic divides each microbatch loss by the accumulation count.

### AdamW

The decoupled weight-decay optimizer used by the framework. Optional fused mode is attempted only on supported CUDA setups.

### AMP

Automatic mixed precision. Speedtronic uses autocast for BF16/FP16 and a CUDA `GradScaler` for FP16.

### Baseline

In DumbDiLoCo, the model state captured at the last successful delta upload or global installation. The next boundary delta is baseline minus current.

### BF16

A floating-point format with a wider exponent range and lower precision than FP32. Commonly native on supported CUDA hardware.

### Block size

The number of input positions in a causal model sample. The reference model rejects sequences longer than its configured block size.

### Causal LM loss

Next-token cross-entropy computed from logits and shifted labels, usually ignoring padding index `-100`.

### Checkpoint

A local pickle-backed state file containing model and training state, optionally including coordinator state.

### Compile fallback

Speedtronic's behavior of disabling `torch.compile` and rerunning a failed compiled forward eagerly.

### DataLoader position

The current epoch, sampler, batch, and worker iterator state. Speedtronic 2.0.0 does not checkpoint this state.

### Distributed

The optional DumbDiLoCo mode. It does not use NCCL or `torch.distributed` collectives.

### Dilation/outer step

See **Outer step**.

### DumbDiLoCo

Speedtronic's Hub-transport, DiLoCo-style protocol: local steps, pseudo-gradient upload, mean aggregation, Nesterov outer update, and global-weight polling.

### FP16

A half-precision floating-point format. On CUDA it uses `GradScaler`; CPU explicit FP16 falls back to FP32.

### FP32

Standard 32-bit floating-point execution. It is the portable default on CPU/MPS and the fallback for several unsupported mixed modes.

### GQA

Grouped-query attention. The reference model projects fewer K/V heads than query heads and repeats K/V groups to match query-head count.

### Global model

The complete model `state_dict` published by the master at `global/latest.safetensors`.

### Global step

In normal local training, the cumulative optimizer update count. In DumbDiLoCo Hub metadata, `outer_step` is the global model version; avoid using the same word for both in operational logs.

### Gradient accumulation

See **Accumulation**.

### Gradient checkpointing

Activation-memory optimization that recomputes selected forward work during backward. The model opts in through `set_gradient_checkpointing(enabled)`.

### Hub

The Hugging Face Hub. Speedtronic uses a model repository as mutable file transport for global state and deltas.

### Inner boundary

A local step divisible by `distributed.inner_steps`.

### Inner loop/step

The local optimization loop. In Speedtronic's coordinator, an inner step is one local optimizer update after gradient accumulation.

### LambdaLR

A PyTorch scheduler driven by a function of the optimizer-step count. Speedtronic builds warmup and cosine/constant factors.

### Local step

One completed local AdamW optimizer update and scheduler step before the coordinator callback.

### LR

Learning rate.

### Microbatch

One loader batch consumed before an optimizer update. The Trainer forward/backward path runs once per microbatch.

### MPS

Apple Metal Performance Shaders backend exposed by PyTorch. Speedtronic auto-selects it after CUDA and before CPU, but uses FP32 by default.

### Nesterov momentum SGD

The outer optimizer update:

```text
buffer = momentum * buffer + gradient
effective = gradient + momentum * buffer
state -= outer_lr * effective
```

### Node ID

The distributed participant's unique path-safe identity, used for `nodes/<node_id>/...` and local state directories.

### Outer loop

Master-side aggregation and global optimization over uploaded pseudo-gradients.

### Outer optimizer

`NesterovOuterOptimizer`, which updates the CPU global state from the mean valid delta.

### Outer round

One successful `MasterOuterLoop.sync_once()` aggregation and publication.

### Outer step

The monotonic global model version stored in `global/step_count.json`. The name `outer_step` distinguishes it from local optimizer steps.

### Parameter server

A centralized service hosting model/optimizer state. DumbDiLoCo does not use one; the Hub is file transport.

### Persistent workers

DataLoader worker processes kept alive across epochs. Enabled automatically when `num_workers > 0`.

### Pin memory

Page-locking CPU tensor storage to accelerate CUDA transfers. Configured explicitly or inferred for resolved CUDA devices.

### Prefetch factor

Number of batches each DataLoader worker prepares in advance. Defaults to 2 when workers are enabled.

### Pseudo-gradient

```text
baseline weights - current local weights
```

It approximates a direction for the outer global update and is not an autograd gradient.

### Registry

The process-global mapping from model name to factory. Built-ins are `reference_transformer` and `gpt`.

### Resume

Loading model, optimizer, scheduler, scaler, counters, RNG, and optional coordinator state from a local checkpoint. It is not exact DataLoader replay.

### RMSNorm

Root-mean-square layer normalization with a learned scale, implemented in the reference model.

### RoPE

Rotary positional embedding. Query and key vectors are rotated by position-dependent angles before attention.

### Safetensors

A tensor serialization format used for global models and deltas. It avoids pickle execution for tensor files but does not authenticate writers.

### Scheduler horizon

`scheduler.max_steps`, the step count used to shape the cosine schedule. It can differ from the actual run target if not configured explicitly.

### SDPA

PyTorch's `scaled_dot_product_attention`, used by the reference model for backend-selected attention kernels.

### Staleness

How old a worker's baseline/global version is. DumbDiLoCo records `base_outer_step` but does not use it to reject or weight stale deltas.

### Step accumulation

See **Accumulation**.

### Straggler

A participant that completes local work later than others. DumbDiLoCo does not wait for stragglers or enforce fixed rounds.

### SwiGLU

A gated MLP using `SiLU(gate(x)) * up(x)` followed by a down projection.

### Target batch

The nominal number of examples represented by one optimizer update, calculated as microbatch size times accumulation steps.

### Token rate

Tokens per second computed by the trainer from input tensor shape or attention-mask sum during the current invocation.

### Trainer

The architecture-neutral loop in `trainer.py` that moves batches, executes forward/backward, updates optimizer/scheduler, emits metrics, checkpoints, and calls the coordinator.

### Trusted writers

The DumbDiLoCo security assumption: every account with Hub write permission is honest and authorized to influence global weights.

### Worker

A DumbDiLoCo participant that trains locally, uploads deltas, and installs newer global weights but does not aggregate outer updates.

### Warmup

Initial scheduler steps where the LR rises linearly to its configured starting value.

Related: [Core Concepts](references/getting-started.md), [Architecture](references/architecture.md), and [DumbDiLoCo](references/distributed.md).


---

## Test Map

_Every Speedtronic repository test, its direct assertion, exercised modules, and important untested areas._
The repository defines the v1 suite plus v2 optimizer, systems, and asynchronous distributed tests. The generated source inventory is authoritative for the current count. The tables below map the v1 and v2 tests to their direct evidence.

### Configuration tests

#### `tests/test_config.py`

| Test | Direct assertion | Primary source |
|---|---|---|
| `test_config_yaml_aliases_and_accumulation` | Top-level run aliases, derived accumulation, compile mapping | `config.py` |
| `test_config_round_trip` | Redacted YAML save/load equality | `config.py` |
| `test_unknown_keys_and_invalid_batch` | Unknown root and invalid batch divisibility fail | `config.py` |
| `test_distributed_role_inference` | Enabled repo-bearing role becomes master | `config.py` |
| `test_secret_can_be_redacted_for_serialization` | Explicit redaction and redacted `save()` | `config.py` |

Not directly tested:

- rare section value constraints;
- JSON and inline source loading;
- `hub`/`diloco` aliases;
- top-level versus nested run precedence;
- malformed YAML/JSON normalization.

### Model tests

#### `tests/test_model.py`

| Test | Direct assertion | Primary source |
|---|---|---|
| `test_reference_transformer_forward_and_loss` | Logits/loss shapes and backward gradients | `model.py` |
| `test_reference_transformer_weight_tying_and_checkpoint_hook` | Tied storage and block flag propagation | `model.py` |
| `test_config_dimensions` | `GPTConfig.d_ff` defaults to `4 * d_model` | `model.py` |

Not directly tested:

- RoPE correctness;
- GQA equivalence/performance;
- SDPA mask behavior;
- causality and label alignment;
- SwiGLU shape;
- dropout/training mode;
- untied weights;
- invalid context lengths.

### Precision and data tests

#### `tests/test_precision_data.py`

| Test | Direct assertion | Primary source |
|---|---|---|
| `test_cpu_precision_defaults_to_fp32` | CPU auto FP32 and no scaler | `precision.py` |
| `test_cpu_precision_defaults_to_fp32` second assertion | Explicit CPU FP16 falls back to FP32 | `precision.py` |
| `test_streaming_text_dataset_and_tuple_collate` | Text block shape and tuple stacking | `data.py` |

Not directly tested:

- CUDA/MPS;
- CPU BF16;
- fused AdamW;
- scaler creation/restoration;
- DataLoader worker/prefetch/pin behavior;
- custom tokenizer;
- serialized datasets;
- empty streams.

### Training tests

#### `tests/test_training.py`

| Test | Direct assertion | Primary source |
|---|---|---|
| `test_trainer_runs_with_accumulation_and_checkpoint` | Three CPU steps, three metrics, checkpoint pointer | `trainer.py`, `checkpoint.py` |
| `test_accumulation_scales_gradients_and_reports_mean_loss` | Gradient scaling and mean loss value | `trainer.py` |
| `test_runtime_builds_reference_model` | Reference composition and one step | `runtime.py`, `model.py` |

Not directly tested:

- checkpoint contents and full resume equivalence;
- DataLoader position;
- scheduler trajectory;
- clipping;
- compile fallback;
- every model output form;
- real CLI train error mapping;
- real coordinator lifecycle with Hub credentials.

### Tensor and outer optimizer tests

#### `tests/test_distributed.py`

| Test | Direct assertion | Primary source |
|---|---|---|
| `test_pseudo_gradient_direction_and_average` | Baseline-current direction and equal mean | `distributed/tensors.py` |
| `test_nesterov_outer_optimizer` | First update equals `1.9 × gradient` at momentum 0.9 | `distributed/outer.py` |
| `test_delta_path_parser` | Valid node delta and rejected global path | `distributed/tensors.py` |
| `test_tied_model_state_can_be_safetensors_serialized` | Tied state serializes after clone | `distributed/tensors.py` |

Not directly tested:

- key/shape mismatch errors;
- non-floating/complex behavior;
- delta metadata parsing;
- outer momentum persistence;
- large/NaN values.

### Hub tests

#### `tests/test_hub.py`

| Test | Direct assertion | Primary source |
|---|---|---|
| `test_hub_transport_round_trip` | Fake global and delta round trip plus write grant | `distributed/hub.py` |
| `test_master_skips_corrupt_delta_and_persists_processed_set` | Corrupt delta is skipped and no state file is created | `distributed/outer.py` |

The second test does not actually verify successful processed-set persistence despite its name.

Not directly tested:

- retry/backoff;
- cached metadata fallback;
- real Hugging Face SDK versions;
- collaborator API compatibility;
- successful aggregation/publication;
- duplicate suppression;
- rollback/recovery;
- background thread lifecycle.

### v2 test modules

| File | Coverage |
|---|---|
| `tests/test_v2_optimizers.py` | Config, Newton–Schulz, Muon+, routing, hybrid/cautious stepping, sparse rejection |
| `tests/test_v2_systems.py` | Shape profiles, causal alignment, scheduler default, CPU OOO fallback |
| `tests/test_v2_distributed.py` | Non-blocking dispatch, async poll, prepared resume, queue skip, synchronous compatibility, failure baseline retention |
| `tests/test_v2_release.py` | Version sync, CLI redaction, v2 config serialization |

CUDA-only stream parity and hardware benchmarks are not represented as CPU
unit-test evidence.

### Module evidence matrix

| Module | Direct tests | Evidence level |
|---|---|---|
| `config.py` | `test_config.py` | Partial |
| `model.py` | `test_model.py` | Partial |
| `precision.py` | `test_precision_data.py` | CPU partial |
| `data.py` | `test_precision_data.py`, `test_training.py` | Partial |
| `trainer.py` | `test_training.py` | CPU partial |
| `runtime.py` | `test_training.py` | Minimal |
| `checkpoint.py` | `test_training.py` | Basic file/pointer only |
| `distributed/tensors.py` | `test_distributed.py` | Basic |
| `distributed/outer.py` | `test_distributed.py`, `test_hub.py`, `test_v2_distributed.py` | Basic plus lock-scope tests |
| `distributed/hub.py` | `test_hub.py` | Fake happy path |
| `distributed/diloco.py` | `test_v2_distributed.py` | Async dispatch/poll/resume coverage |
| `distributed/__init__.py` | Indirect imports | No direct test |
| `cli.py` | `test_v2_release.py` | Validation/redaction coverage |
| `registry.py` | None | No direct test |
| `profiling.py` | None | No direct test |
| `hooks.py` | None | No direct test |
| `integrations.py` | None | No direct test |
| `module_utils.py` | Indirect model hook only | No helper test |
| `__init__.py` | None | No export/lazy-load test |
| `__main__.py` | None | No module-entry test |

### Test philosophy

Tests are intended to be CPU-only and use fake Hub APIs. They make no real network calls and require no credentials. Hardware-specific claims remain implementation-derived unless directly exercised.

The complete source-level inventory also includes every test function generated from the AST. See [Generated Source Inventory](references/api-inventory.md).

Related: [Testing](references/operations.md), [Source Coverage](references/project-structure.md), and [Security and Limitations](references/operations.md).
