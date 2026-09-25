---
id: testing
title: Testing and Development
sidebar_label: Testing
description: Development setup, prescribed quality commands, test inventory, coverage evidence, and major untested paths.
---

# Testing and development

## Development install

```bash
cd /path/to/speedtronic
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

The development extra installs:

```text
pytest>=7.4
ruff>=0.5
tomli>=2.0 on Python 3.10
```

Logging extras are separate.

## Prescribed checks

```bash
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m compileall src
```

Pytest uses the `tests/` directory and `-ra`. Ruff targets Python 3.10 and a 100-character line length.

## Test philosophy

The suite is CPU-only and uses fake Hub behavior where possible. It avoids network credentials and real accelerator requirements.

The current repository defines the v1 suite plus the v2 optimizer, systems, and asynchronous distributed tests. The generated inventory is the authoritative count.

## Test inventory

### `tests/test_config.py`

| Test | Verifies |
|---|---|
| `test_config_yaml_aliases_and_accumulation` | Top-level run aliases, accumulation derivation, compile mapping |
| `test_config_round_trip` | Redacted save/load and equality |
| `test_unknown_keys_and_invalid_batch` | Unknown root rejection and invalid batch divisibility |
| `test_distributed_role_inference` | Enabled repo-bearing config defaults to master |
| `test_secret_can_be_redacted_for_serialization` | Explicit serialization redaction and redacted file save |

### `tests/test_model.py`

| Test | Verifies |
|---|---|
| `test_reference_transformer_forward_and_loss` | Logit/loss shape and backward gradients |
| `test_reference_transformer_weight_tying_and_checkpoint_hook` | Shared weights and checkpoint flag propagation |
| `test_config_dimensions` | Default FFN width |

### `tests/test_precision_data.py`

| Test | Verifies |
|---|---|
| `test_cpu_precision_defaults_to_fp32` | CPU auto FP32 and explicit CPU FP16 fallback |
| `test_streaming_text_dataset_and_tuple_collate` | Text block emission and tuple collation |

### `tests/test_training.py`

| Test | Verifies |
|---|---|
| `test_trainer_runs_with_accumulation_and_checkpoint` | Three CPU steps, metric count, and checkpoint pointer |
| `test_accumulation_scales_gradients_and_reports_mean_loss` | Mathematical gradient accumulation and mean loss |
| `test_runtime_builds_reference_model` | Runtime composition and one reference-model step |

### `tests/test_distributed.py`

| Test | Verifies |
|---|---|
| `test_pseudo_gradient_direction_and_average` | Baseline-current sign and equal averaging |
| `test_nesterov_outer_optimizer` | First Nesterov update formula and state shape |
| `test_delta_path_parser` | Valid and invalid delta paths |
| `test_tied_model_state_can_be_safetensors_serialized` | Clone behavior for tied storage |

### `tests/test_hub.py`

| Test | Verifies |
|---|---|
| `test_hub_transport_round_trip` | Fake global/delta upload-download and collaborator grant |
| `test_master_skips_corrupt_delta_and_persists_processed_set` | Corrupt input is skipped and no state file is written |

### `tests/test_v2_optimizers.py`

Covers v2 config aliases, Newton–Schulz edge cases, Muon+ normalization,
role-aware routing, hybrid/AdamW construction, closure forwarding, cautious
masking, sparse rejection, and a CPU runtime smoke build.

### `tests/test_v2_systems.py`

Covers shape profiles and warning suggestions, standard-library logger
compatibility, causal label alignment, scheduler defaults, and CPU OOO
fallback.

### `tests/test_v2_distributed.py`

Covers asynchronous global polling, non-blocking delta dispatch, queue
overflow, prepared resume, synchronous transport, and baseline retention on
failure.

### `tests/test_v2_release.py`

Covers version synchronization, CLI redaction, and serialization of all v2
configuration sections.

## Coverage matrix

| Area | Direct coverage | Important gaps |
|---|---|---|
| Config basics | Good scalar/alias/redaction coverage | Rare nested types and hardware/model compatibility |
| Trainer accumulation | Good CPU scalar/list coverage | Resume equivalence, output-contract breadth |
| Reference model | Basic forward/loss | RoPE, GQA, masks, causality, eval mode |
| CPU precision | Partial | CUDA, MPS, fused, scaler resume |
| Data | Basic plus worker sharding | Prefetch, custom tokenizer, loader builder |
| Checkpoints | Basic existence and final-step save | Contents, retention, corruption, RNG |
| Hub transport | Fake happy path | Retries, real SDK variants, cache fallback |
| Outer optimizer | Basic formula and publish-path tests | Complex values and full recovery matrix |
| Outer loop | Corrupt skip and lock-scope tests | Real Hub aggregation and crash recovery |
| Coordinator | Async dispatch, poll, resume, stop coverage | Master/worker real Hub lifecycle |
| CLI | Validation redaction and exit code | Parsing, overrides, train error mapping |
| Logging/hooks | None | Cadence, JSONL, W&B, TensorBoard |
| Compile/checkpoint fallback | None | Trainer fallback behavior |
| Packaging/docs | Version, wheel, Twine, docs build gates | Platform matrix and CUDA runners |

## v2 verification

The v2 tests cover:

- scalar and mapping optimizer configuration;
- Muon routing, tied weights, Newton–Schulz, Muon+, and cautious wrapping;
- sparse/non-matrix rejection;
- CPU shape profiles and warning suggestions;
- causal label alignment and scheduler synchronization;
- CPU no-op OOO behavior;
- asynchronous delta dispatch, queue overflow, failure, and baseline retention;
- synchronous transport compatibility.

CUDA stream parity, fused AdamW, and hardware-specific speedups require a CUDA
runner and are not implied by the CPU suite.

## Documentation coverage build

The Docusaurus project has a separate coverage gate:

```bash
cd docs-site
npm ci
npm run build
```

Before compilation it:

1. parses every local source module with Python's AST;
2. generates one inventory page for all modules, classes, functions, methods, tests, configs, and examples;
3. validates inventory paths and symbol anchors;
4. validates sidebar coverage.

Docusaurus then fails on broken internal links and MDX issues.

## Adding a source module

When adding a Python module under `src/speedtronic`:

1. Curate its conceptual responsibility in an appropriate page.
2. Ensure the generated inventory includes all declarations.
3. Add or update tests.
4. Add a link from the module coverage page if needed.
5. Run all Python and docs checks.

The generator requires no manual inventory edits; it rewrites the generated page.

## Code style

The repository uses:

- Ruff `E`, `F`, and `I` rules;
- 100-character lines;
- `from __future__ import annotations`;
- dataclasses for configuration and results;
- explicit fallback logging for optional capabilities.

## Test-data safety

Do not add real credentials to configs or tests. The fake Hub API stores bytes in memory and does not use network access.

Related: [Test Map](../appendix/test-map), [Generated Source Inventory](../reference/generated-source-inventory), and [Troubleshooting](./troubleshooting).
