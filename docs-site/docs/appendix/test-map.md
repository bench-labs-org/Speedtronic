---
id: test-map
title: Test Map
sidebar_label: Test Map
description: Every Speedtronic repository test, its direct assertion, exercised modules, and important untested areas.
---

# Test map

The repository defines the v1 suite plus v2 optimizer, systems, and asynchronous distributed tests. The generated source inventory is authoritative for the current count. The tables below map the v1 and v2 tests to their direct evidence.

## Configuration tests

### `tests/test_config.py`

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

## Model tests

### `tests/test_model.py`

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

## Precision and data tests

### `tests/test_precision_data.py`

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

## Training tests

### `tests/test_training.py`

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

## Tensor and outer optimizer tests

### `tests/test_distributed.py`

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

## Hub tests

### `tests/test_hub.py`

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

## v2 test modules

| File | Coverage |
|---|---|
| `tests/test_v2_optimizers.py` | Config, Newton–Schulz, Muon+, routing, hybrid/cautious stepping, sparse rejection |
| `tests/test_v2_systems.py` | Shape profiles, causal alignment, scheduler default, CPU OOO fallback |
| `tests/test_v2_distributed.py` | Non-blocking dispatch, async poll, prepared resume, queue skip, synchronous compatibility, failure baseline retention |
| `tests/test_v2_release.py` | Version sync, CLI redaction, v2 config serialization |

CUDA-only stream parity and hardware benchmarks are not represented as CPU
unit-test evidence.

## Module evidence matrix

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

## Test philosophy

Tests are intended to be CPU-only and use fake Hub APIs. They make no real network calls and require no credentials. Hardware-specific claims remain implementation-derived unless directly exercised.

The complete source-level inventory also includes every test function generated from the AST. See [Generated Source Inventory](../reference/generated-source-inventory).

Related: [Testing](../operations/testing), [Source Coverage](../reference/source-inventory), and [Security and Limitations](../operations/security-and-limitations).
