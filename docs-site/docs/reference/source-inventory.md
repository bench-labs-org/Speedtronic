---
id: source-inventory
title: Source Coverage
sidebar_label: Source Coverage
description: How this site covers every Speedtronic source file, symbol, test, configuration, and example.
---

# Source coverage

## Coverage claim

This documentation project inventories:

- all implementation modules under `src/speedtronic` (v1 plus v2);
- every top-level class and function;
- every direct class method;
- module constants and compatibility aliases;
- all v1 and v2 test functions in the repository;
- both shipped YAML configurations;
- both shipped examples;
- package metadata, source manifest, license, and ignore rules through the operational pages.

The exhaustive AST listing is generated during `npm run build` and appears as [Generated Source Inventory](./generated-source-inventory).

## Build-time coverage gate

`scripts/validate_coverage.py` fails the build when:

- a Python implementation file is absent from the generated inventory;
- a top-level class/function or class method lacks a generated anchor;
- a test file or test function is absent;
- a shipped config or example is absent;
- a documentation page is not represented in the sidebar.

The source inventory generator uses `ast.parse`; it does not import Speedtronic, load PyTorch, read credentials, or access the network.

## Implementation module map

| Module | Responsibility | Curated page |
|---|---|---|
| `__init__.py` | Eager/lazy public exports and version | [Generated inventory](./generated-source-inventory) |
| `__main__.py` | `python -m speedtronic` | [CLI](./cli) |
| `config.py` | Dataclasses, parsing, defaults, redaction | [Configuration](./configuration) |
| `runtime.py` | Composition helpers | [Runtime and Trainer](./runtime-and-trainer) |
| `trainer.py` | Optimization loop and state | [Runtime and Trainer](./runtime-and-trainer) |
| `data.py` | Tokenizers, datasets, collators, loader | [Data](./data) |
| `model.py` | Reference transformer | [Reference Model](./reference-model) |
| `registry.py` | Model factory registry | [Extensions](./extensions) |
| `precision.py` | Device and precision capability | [Precision](./precision) |
| `checkpoint.py` | Atomic persistence and RNG | [Checkpoints](./checkpointing) |
| `profiling.py` | Metrics and callbacks | [Observability](./observability) |
| `hooks.py` | Event wrapper | [Observability](./observability) |
| `integrations.py` | W&B and TensorBoard adapters | [Observability](./observability) |
| `module_utils.py` | Gradient-checkpointing helper | [Extensions](./extensions) |
| `cli.py` | Argument parser and command dispatch | [CLI](./cli) |
| `distributed/__init__.py` | Distributed re-exports | [DumbDiLoCo overview](../distributed/overview) |
| `distributed/diloco.py` | Master/worker coordinator | [Coordinator](../distributed/coordinator) |
| `distributed/hub.py` | Hub file transport and retries | [Hub Transport](../distributed/hub-transport) |
| `distributed/outer.py` | Outer optimizer and master loop | [Outer Loop](../distributed/outer-loop) |
| `distributed/tensors.py` | Safetensors and delta wire format | [Tensor Format](../distributed/tensor-format) |
| `optimizers.py` | Hybrid Muon, Muon+, cautious wrapper, routing | [v2 Optimizers](../v2/optimizers) |
| `shapes.py` | Startup shape profiles and warnings | [Shape Validation](../v2/shape-validation) |
| `scheduling.py` | CUDA stage streams and OOO fallback | [OOO Backprop](../v2/scheduling) |

## Supporting file map

| File | Coverage |
|---|---|
| `README.md` | User overview reconciled against implementation in all pages |
| `pyproject.toml` | [Packaging](../operations/packaging) |
| `MANIFEST.in` | [Packaging](../operations/packaging) |
| `LICENSE` | Apache-2.0 summary and [glossary](../appendix/glossary) |
| `.gitignore` | Generated/artifact guidance in operations pages |
| `configs/smoke.yaml` | [Shipped Configs](../examples/shipped-configs) |
| `configs/v2_smoke.yaml` | [Shipped Configs](../examples/shipped-configs) |
| `configs/diloco.yaml` | [Shipped Configs](../examples/shipped-configs) |
| `CHANGELOG.md` | Release and deferred-decision record |
| `v2_additions.md` | v2 implementation specification |
| `.github/workflows/ci.yml` | Python, documentation, and distribution gates |
| `examples/train_reference.py` | [Shipped Examples](../examples/shipped-examples) |
| `examples/diloco_master.yaml` | [Shipped Examples](../examples/shipped-examples) |

## Evidence labels

| Label | Meaning |
|---|---|
| Verified by tests | Directly asserted by a repository test |
| Example exercised | Present in a shipped runnable configuration or Python example |
| Implementation-derived | Read directly from source but not directly tested |
| External dependent | Requires real CUDA, MPS, compilation, filesystem failure, or Hub networking |

Documentation completeness does not imply every documented path is tested. The [test map](../appendix/test-map) makes that distinction explicit.
