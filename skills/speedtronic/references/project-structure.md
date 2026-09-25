## Source Coverage

_How this site covers every Speedtronic source file, symbol, test, configuration, and example._
### Coverage claim

This documentation project inventories:

- all implementation modules under `src/speedtronic` (v1 plus v2);
- every top-level class and function;
- every direct class method;
- module constants and compatibility aliases;
- all v1 and v2 test functions in the repository;
- both shipped YAML configurations;
- both shipped examples;
- package metadata, source manifest, license, and ignore rules through the operational pages.

The exhaustive AST listing is generated during `npm run build` and appears as [Generated Source Inventory](references/api-inventory.md).

### Build-time coverage gate

`scripts/validate_coverage.py` fails the build when:

- a Python implementation file is absent from the generated inventory;
- a top-level class/function or class method lacks a generated anchor;
- a test file or test function is absent;
- a shipped config or example is absent;
- a documentation page is not represented in the sidebar.

The source inventory generator uses `ast.parse`; it does not import Speedtronic, load PyTorch, read credentials, or access the network.

### Implementation module map

| Module | Responsibility | Curated page |
|---|---|---|
| `__init__.py` | Eager/lazy public exports and version | [Generated inventory](references/api-inventory.md) |
| `__main__.py` | `python -m speedtronic` | [CLI](references/cli.md) |
| `config.py` | Dataclasses, parsing, defaults, redaction | [Configuration](references/configuration.md) |
| `runtime.py` | Composition helpers | [Runtime and Trainer](references/architecture.md) |
| `trainer.py` | Optimization loop and state | [Runtime and Trainer](references/architecture.md) |
| `data.py` | Tokenizers, datasets, collators, loader | [Data](references/data.md) |
| `model.py` | Reference transformer | [Reference Model](references/models-and-extensions.md) |
| `registry.py` | Model factory registry | [Extensions](references/models-and-extensions.md) |
| `precision.py` | Device and precision capability | [Precision](references/precision-and-performance.md) |
| `checkpoint.py` | Atomic persistence and RNG | [Checkpoints](references/checkpointing.md) |
| `profiling.py` | Metrics and callbacks | [Observability](references/observability.md) |
| `hooks.py` | Event wrapper | [Observability](references/observability.md) |
| `integrations.py` | W&B and TensorBoard adapters | [Observability](references/observability.md) |
| `module_utils.py` | Gradient-checkpointing helper | [Extensions](references/models-and-extensions.md) |
| `cli.py` | Argument parser and command dispatch | [CLI](references/cli.md) |
| `distributed/__init__.py` | Distributed re-exports | [DumbDiLoCo overview](references/distributed.md) |
| `distributed/diloco.py` | Master/worker coordinator | [Coordinator](references/distributed.md) |
| `distributed/hub.py` | Hub file transport and retries | [Hub Transport](references/distributed.md) |
| `distributed/outer.py` | Outer optimizer and master loop | [Outer Loop](references/distributed.md) |
| `distributed/tensors.py` | Safetensors and delta wire format | [Tensor Format](references/distributed.md) |
| `optimizers.py` | Hybrid Muon, Muon+, cautious wrapper, routing | [v2 Optimizers](references/optimizers.md) |
| `shapes.py` | Startup shape profiles and warnings | [Shape Validation](references/scheduling-and-shapes.md) |
| `scheduling.py` | CUDA stage streams and OOO fallback | [OOO Backprop](references/scheduling-and-shapes.md) |

### Supporting file map

| File | Coverage |
|---|---|
| `README.md` | User overview reconciled against implementation in all pages |
| `pyproject.toml` | [Packaging](references/operations.md) |
| `MANIFEST.in` | [Packaging](references/operations.md) |
| `LICENSE` | Apache-2.0 summary and [glossary](references/glossary-and-tests.md) |
| `.gitignore` | Generated/artifact guidance in operations pages |
| `configs/smoke.yaml` | [Shipped Configs](references/examples-and-configs.md) |
| `configs/v2_smoke.yaml` | [Shipped Configs](references/examples-and-configs.md) |
| `configs/diloco.yaml` | [Shipped Configs](references/examples-and-configs.md) |
| `CHANGELOG.md` | Release and deferred-decision record |
| `.github/workflows/ci.yml` | Python, documentation, and distribution gates |
| `examples/train_reference.py` | [Shipped Examples](references/examples-and-configs.md) |
| `examples/diloco_master.yaml` | [Shipped Examples](references/examples-and-configs.md) |

### Evidence labels

| Label | Meaning |
|---|---|
| Verified by tests | Directly asserted by a repository test |
| Example exercised | Present in a shipped runnable configuration or Python example |
| Implementation-derived | Read directly from source but not directly tested |
| External dependent | Requires real CUDA, MPS, compilation, filesystem failure, or Hub networking |

Documentation completeness does not imply every documented path is tested. The [test map](references/glossary-and-tests.md) makes that distinction explicit.
