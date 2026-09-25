## Packaging and Release

_PyPI metadata, setuptools src layout, wheel/sdist contents, dependencies, version synchronization, and local release gates._
### Distribution identity

```text
name: speedtronic
version: 2.0.0
requires-python: >=3.10
license: Apache-2.0
```

The distribution and import package are both named `speedtronic`. The repository URL may use title case, but PyPI normalization and metadata are exactly `speedtronic`.

### Build system

```toml
[build-system]
requires = ["setuptools>=77.0.3", "wheel"]
build-backend = "setuptools.build_meta"
```

The modern SPDX license expression and `license-files = ["LICENSE"]` require a sufficiently recent setuptools backend.

### Runtime dependencies

```text
torch>=2.1
safetensors>=0.4
PyYAML>=6.0
numpy>=1.24
huggingface-hub>=0.23
```

Install the appropriate CPU or CUDA PyTorch wheel for the target environment when the default PyPI wheel is not appropriate.

### Optional dependencies

#### Development

```text
pytest>=7.4
ruff>=0.5
```

#### Logging

```text
wandb>=0.16
tensorboard>=2.14
```

#### Release

```text
build>=1.2
check-wheel-contents>=0.6
twine>=6.0
```

The release extra is for maintainers and is not required at runtime.

### Console entry point

```toml
[project.scripts]
speedtronic = "speedtronic.cli:main"
```

This creates:

```bash
speedtronic ...
```

The package also supports:

```bash
python -m speedtronic ...
```

### Source layout

```toml
[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["speedtronic*"]
namespaces = false
```

The wheel contains only regular packages matching `speedtronic*`:

```text
speedtronic/
speedtronic/distributed/
```

Tests, configs, and examples are not installed as wheel package data.

### Source distribution manifest

`MANIFEST.in` includes:

```text
LICENSE
README.md
pyproject.toml
configs/*.yaml
examples/*.py
examples/*.yaml
CHANGELOG.md
tests/*.py
```

It excludes bytecode and generated run/state directories.

### Wheel versus sdist

| Artifact | Intended use | Includes |
|---|---|---|
| Wheel | Installation | Importable package, metadata, license, entry point |
| Source archive | Source review/build | Wheel source plus tests, configs, examples, manifest, README, license |

Commands such as `speedtronic train --config configs/smoke.yaml` are source-checkout/source-archive examples. An arbitrary installed wheel does not guarantee that top-level `configs/` exists beside the environment.

### Version synchronization

The version appears in:

```text
pyproject.toml
src/speedtronic/__init__.py
```

They currently both contain `2.0.0`. A release should update both and rebuild.

### Local release sequence

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m compileall src

python -m pip install build twine check-wheel-contents
rm -rf dist build
python -m build
python -m twine check dist/*
check-wheel-contents dist/speedtronic-2.0.0-py3-none-any.whl
```

Install into a clean environment and test the console command before upload.

### Artifact inspection

For a wheel:

- verify `Name: speedtronic`;
- verify version and Python requirement;
- verify Apache license file;
- verify console entry point;
- verify `speedtronic` and `speedtronic.distributed` modules;
- reject unexpected top-level packages.

For an sdist:

- inspect `SOURCES.txt`/archive contents;
- ensure tests/configs/examples are included;
- ensure virtual environments, runs, checkpoints, and bytecode are absent.

### PyPI upload

The project is published at:

```text
https://pypi.org/project/speedtronic/
```

Do not print or inspect credential files. Let Twine resolve its existing non-interactive configuration or SDK credential environment. A PyPI release filename cannot be overwritten once uploaded; fix metadata and increment the version for a new release.

### Documentation build

The Docusaurus project is isolated under `docs-site` and does not affect the Python wheel.

```bash
cd docs-site
npm ci
npm run build
npm run serve
```

Generated directories:

```text
docs-site/node_modules/
docs-site/.docusaurus/
docs-site/build/
```

are ignored locally.

### Release checklist

- [ ] Version values agree.
- [ ] Changelog/release intent is defined outside the source if needed.
- [ ] Tests, lint, formatting, and compilation pass.
- [ ] Wheel and sdist build.
- [ ] Twine metadata check passes.
- [ ] Wheel contents are clean.
- [ ] Clean-environment installation works.
- [ ] CLI and smoke run work from the wheel/source archive.
- [ ] No secret is stored in source or distribution metadata.
- [ ] Documentation build passes without deployment.

Related: [Testing](references/operations.md), [Shipped Examples](references/examples-and-configs.md), and [Troubleshooting](references/operations.md).


---

## Testing and Development

_Development setup, prescribed quality commands, test inventory, coverage evidence, and major untested paths._
### Development install

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

### Prescribed checks

```bash
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m compileall src
```

Pytest uses the `tests/` directory and `-ra`. Ruff targets Python 3.10 and a 100-character line length.

### Test philosophy

The suite is CPU-only and uses fake Hub behavior where possible. It avoids network credentials and real accelerator requirements.

The current repository defines the v1 suite plus the v2 optimizer, systems, and asynchronous distributed tests. The generated inventory is the authoritative count.

### Test inventory

#### `tests/test_config.py`

| Test | Verifies |
|---|---|
| `test_config_yaml_aliases_and_accumulation` | Top-level run aliases, accumulation derivation, compile mapping |
| `test_config_round_trip` | Redacted save/load and equality |
| `test_unknown_keys_and_invalid_batch` | Unknown root rejection and invalid batch divisibility |
| `test_distributed_role_inference` | Enabled repo-bearing config defaults to master |
| `test_secret_can_be_redacted_for_serialization` | Explicit serialization redaction and redacted file save |

#### `tests/test_model.py`

| Test | Verifies |
|---|---|
| `test_reference_transformer_forward_and_loss` | Logit/loss shape and backward gradients |
| `test_reference_transformer_weight_tying_and_checkpoint_hook` | Shared weights and checkpoint flag propagation |
| `test_config_dimensions` | Default FFN width |

#### `tests/test_precision_data.py`

| Test | Verifies |
|---|---|
| `test_cpu_precision_defaults_to_fp32` | CPU auto FP32 and explicit CPU FP16 fallback |
| `test_streaming_text_dataset_and_tuple_collate` | Text block emission and tuple collation |

#### `tests/test_training.py`

| Test | Verifies |
|---|---|
| `test_trainer_runs_with_accumulation_and_checkpoint` | Three CPU steps, metric count, and checkpoint pointer |
| `test_accumulation_scales_gradients_and_reports_mean_loss` | Mathematical gradient accumulation and mean loss |
| `test_runtime_builds_reference_model` | Runtime composition and one reference-model step |

#### `tests/test_distributed.py`

| Test | Verifies |
|---|---|
| `test_pseudo_gradient_direction_and_average` | Baseline-current sign and equal averaging |
| `test_nesterov_outer_optimizer` | First Nesterov update formula and state shape |
| `test_delta_path_parser` | Valid and invalid delta paths |
| `test_tied_model_state_can_be_safetensors_serialized` | Clone behavior for tied storage |

#### `tests/test_hub.py`

| Test | Verifies |
|---|---|
| `test_hub_transport_round_trip` | Fake global/delta upload-download and collaborator grant |
| `test_master_skips_corrupt_delta_and_persists_processed_set` | Corrupt input is skipped and no state file is written |

#### `tests/test_v2_optimizers.py`

Covers v2 config aliases, Newton–Schulz edge cases, Muon+ normalization,
role-aware routing, hybrid/AdamW construction, closure forwarding, cautious
masking, sparse rejection, and a CPU runtime smoke build.

#### `tests/test_v2_systems.py`

Covers shape profiles and warning suggestions, standard-library logger
compatibility, causal label alignment, scheduler defaults, and CPU OOO
fallback.

#### `tests/test_v2_distributed.py`

Covers asynchronous global polling, non-blocking delta dispatch, queue
overflow, prepared resume, synchronous transport, and baseline retention on
failure.

#### `tests/test_v2_release.py`

Covers version synchronization, CLI redaction, and serialization of all v2
configuration sections.

### Coverage matrix

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

### v2 verification

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

### Documentation coverage build

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

### Adding a source module

When adding a Python module under `src/speedtronic`:

1. Curate its conceptual responsibility in an appropriate page.
2. Ensure the generated inventory includes all declarations.
3. Add or update tests.
4. Add a link from the module coverage page if needed.
5. Run all Python and docs checks.

The generator requires no manual inventory edits; it rewrites the generated page.

### Code style

The repository uses:

- Ruff `E`, `F`, and `I` rules;
- 100-character lines;
- `from __future__ import annotations`;
- dataclasses for configuration and results;
- explicit fallback logging for optional capabilities.

### Test-data safety

Do not add real credentials to configs or tests. The fake Hub API stores bytes in memory and does not use network access.

Related: [Test Map](references/glossary-and-tests.md), [Generated Source Inventory](references/api-inventory.md), and [Troubleshooting](references/operations.md).


---

## Troubleshooting

_Diagnose configuration, model, data, precision, checkpoint, compilation, logging, CLI, and DumbDiLoCo failures._
### Configuration errors

#### `unknown configuration keys`

Speedtronic rejects unknown keys intentionally.

Check:

- section spelling;
- top-level aliases (`name`, `seed`, `device`, `max_steps`, `output_dir`, `log_every`);
- the current [configuration schema](references/configuration.md);
- whether `logging.hooks` was added—hooks are programmatic in 2.0.0.

#### `batch sizes must be positive`

Set positive `micro_batch_size` and `target_batch_size`.

#### `target_batch_size must be ... divisible`

Use:

```text
target_batch_size % micro_batch_size == 0
```

#### Model dimensions invalid

For the bundled model:

- `n_kv_head` must divide `n_head`;
- `d_model` must divide by `n_head`;
- attention head dimension must be even for RoPE;
- all dimensions must be positive.

#### Distributed repository missing

Enabled DumbDiLoCo requires:

```yaml
distributed:
  enabled: true
  repo_id: org/run
```

#### Role changed unexpectedly

An enabled `single` role with a repository becomes `master`. Set `role: worker` explicitly on workers.

### Validation passes but training fails

`validate` checks schema and selected invariants only. It does not open files, build models, resolve hardware, or run data.

Common structurally valid but unrunnable configurations:

- missing `data.text_path`;
- unknown custom model name;
- model block size smaller than data block size;
- data vocabulary inconsistent with model vocabulary;
- requested unavailable CUDA/MPS device.

### Model errors

#### `unknown model`

A custom model registered in one process is not automatically visible in a CLI process. Construct the runtime in the same process or add an application plugin import.

#### `sequence length ... exceeds model block size`

The reference model rejects inputs longer than `model.max_seq_len`. Keep:

```text
data.block_size <= model.max_seq_len
```

for the bundled model.

#### Unexpected `input_ids`, `labels`, or `attention_mask`

The trainer expands dictionary batches as keyword arguments. A custom model must accept the emitted keys or use a tuple batch/custom loader.

#### Loss behaves incorrectly

A bare tensor is treated as a direct loss. Return `{"logits": logits}` plus compatible labels or return a loss mapping when using 3-D output tensors.

#### Causal target appears shifted twice

Speedtronic 2.0 consumes the already-shifted labels emitted by its datasets. If a custom model or dataset still shifts the same labels, align the contract to one next-token label per input position.

### Data errors

#### `data loader produced no batches`

The complete loader pass yielded no items. Check:

- empty iterable dataset;
- text file exists and is readable;
- final stream behavior;
- dataset filter removed all examples;
- `drop_last` with a dataset smaller than one batch.

#### `data.dataset must ... exist`

The configured serialized path is absent or is not a PyTorch Dataset.

Never load an untrusted `.pt` dataset; `weights_only=False` permits pickle code execution.

#### Duplicate text data with workers

The bundled text iterable dataset is not worker-sharded. Set `num_workers: 0` or implement worker-aware sharding.

#### Broken custom tokenizer

The stream carries IDs, not raw text. Context-sensitive tokenizers can split merges at chunk boundaries. Use an incremental tokenizer or a whole-file preprocessing step.

#### Out-of-range token IDs

The standard runtime passes `model.vocab_size` to the loader and ignores a separate `data.vocab_size`. Keep vocabularies aligned.

### Runtime and precision errors

#### CUDA/MPS unavailable

`resolve_device()` raises when an explicit backend is unavailable. Use `--device cpu` or `auto`.

#### Unexpected FP32 fallback

Expected cases include:

- auto on CPU/MPS;
- explicit CPU FP16;
- mixed precision on MPS;
- unsupported CUDA BF16.

The `train_start` event reports the resolved mode.

#### CPU BF16 failure

Explicit CPU BF16 is not capability-checked. Use FP32 if the installed PyTorch/custom operators cannot run CPU autocast BF16.

#### Fused AdamW not used

Fused construction is best-effort and may fail because of device placement or backend support. No warning is emitted by the fallback path.

### Compilation

#### Compilation disabled automatically

A compile event reports construction/runtime fallback. The run continues eagerly.

#### Model executes twice

The compiled forward raised, then the trainer reran the same batch eagerly. Look for model side effects or an actual model error masked by the broad fallback.

#### `torch.compile` overhead on short runs

Compilation cost can dominate a four-step smoke test. Keep `compile: false` for smoke runs.

### Scheduler behavior

#### Run ends during warmup

Set `scheduler.max_steps` and `warmup_steps` explicitly. Programmatic run-target overrides do not reliably update the default 1000-step scheduler.

#### Logged LR is one step ahead

The optimizer steps before the scheduler advances, and metrics read the post-scheduler LR.

### GradScaler behavior

On CUDA FP16, an overflow can cause `GradScaler` to skip the optimizer update while Speedtronic still increments the global step and scheduler. A sudden high loss or unchanged parameters around an early step can indicate this.

### Checkpoint and resume

#### No checkpoint found

Resume searches:

```text
<resolved output_dir>/<checkpoint.directory>/
```

Changing `--output-dir` changes the search location.

A warning is logged and a new run starts.

#### Checkpoint is stale behind file scan

If `latest.json` is missing, the manager chooses the highest valid filename. If the pointer is valid, it trusts the pointer.

#### Corrupt newest checkpoint

`load_latest()` does not fall back to an older file after a corrupt selected checkpoint. Restore a known-good copy or point the directory to one.

#### Resume repeats data

DataLoader position and worker state are not checkpointed. Expect approximate replay.

#### Final step not saved

Only exact checkpoint intervals save. A final off-interval step is lost from the latest checkpoint.

#### Checkpoint shape mismatch

Current config/model compatibility is not checked before `load_state_dict`. Ensure identical architecture and state keys.

### Logging and integrations

#### No W&B/TensorBoard configuration

`LoggingConfig` has no hooks field. Attach adapters programmatically.

#### TensorBoard file remains open

Call `TensorboardHook.close()`.

#### W&B run does not finish

The adapter has no public close method. Finish the run through the Wandb API or application lifecycle.

#### JSONL has concurrent writes

The master outer thread and trainer can emit concurrently. The logger does not lock JSONL appends or hook invocation.

### DumbDiLoCo

#### Worker cannot find global metadata

The master may not have bootstrapped yet, credentials may lack access, or the repository may be wrong. The worker continues locally and retries later.

#### Repeated slow retries

Every exception, including 401/403 and 404, is retried with backoff. Verify repository/token configuration before waiting through repeated delays.

#### Corrupt delta warnings

The master skips the file and retries it on later polls because invalid files are not marked processed. Remove or replace an unrecoverable remote file.

#### Delta key/shape mismatch

All nodes must use the same floating-state key set and shapes. A custom architecture or version skew rejects the full delta.

#### No global update

Check:

- delta was uploaded successfully;
- `outer_step` increased;
- `base_outer_step` and identities are valid;
- one active master is running;
- the worker is not stuck behind an invalid cache;
- model state shapes match.

#### Repository grows continuously

Expected in 2.0.0. Deltas are not deleted, global caches are not pruned, and all listed candidates are downloaded.

#### Unexpected optimizer reset

Optimizer state clears after a successful delta upload **or** global installation, not strictly at inner boundaries.

#### Resume produces a large delta

Current startup/load ordering can pair a fresh global model with a checkpointed older baseline. Treat distributed resume as a known limitation and verify baseline/global step before training.

### v2 optimizer and systems issues

#### Muon routes too few parameters

Check the parameter role heuristic and tied weights. Embeddings, heads, norms,
and biases intentionally remain on AdamW. A model can set
`_speedtronic_optimizer_role` to make an explicit choice.

#### Muon fails on a sparse or complex gradient

Muon is defined for dense, real, floating-point 2-D matrices. Route sparse or
complex parameters to AdamW or provide a custom optimizer.

#### OOO backprop does nothing

`ooo_backprop` is a no-op on CPU and MPS. It is also disabled when
`compile: true`, when fewer than two module stages are found, or when CUDA
streams cannot be installed. Inspect the `ooo_backprop` event.

#### Shape warnings appear on CPU

Automatic CPU profiles are quiet. An explicit `shape_validation.alignment`
also needs `warn_on_cpu: true` to enable CPU benchmarking warnings. Set
`alignment: none` to disable them.

#### Async delta is skipped

A `delta_upload_skipped` event with `reason=upload_in_flight` means the
one-slot queue is occupied. The next accepted boundary sends a cumulative
delta. Increase `delta_upload_shutdown_timeout` only for controlled shutdown
tests, not to make the inner loop block.

#### Async poll does not install a new global immediately

Downloads are intentionally performed in the background and installed only at
a training-thread boundary. Inspect `delta_upload` and `ooo_backprop`/`shape`
events plus the outer-step metric.

### Diagnostics

For local issues, collect:

```text
resolved configuration
device and precision plan
start/target step
checkpoint pointer
first failing stack trace
metric records
exact model/data dimensions
node role/ID/repository when distributed
```

Never collect or print token values. Share redacted configuration and filenames instead.

Related: [Security and Limitations](references/operations.md), [Configuration](references/configuration.md), and [DumbDiLoCo](references/distributed.md).


---

## Security and Limitations

_Trust boundaries, unsafe deserialization, Hub writer risks, known correctness issues, and non-goals in Speedtronic 2.0.0._
### Security boundaries

#### Pickle-capable local files

The following paths use `torch.load(..., weights_only=False)` or equivalent pickle-capable loading:

- local trainer checkpoints;
- serialized datasets configured through `data.dataset`;
- master `outer_state.pt`.

Only load files from trusted local sources. A write-capable local attacker can craft a checkpoint that executes code during deserialization.

#### Hub writer trust

DumbDiLoCo assumes a closed set of trusted repository writers. A collaborator with write access can:

- replace global model weights;
- replace node deltas;
- alter step metadata;
- set `model_file` to another repository path;
- overwrite the fixed global paths.

There is no:

- node signature;
- content hash in metadata;
- pinned Hub revision;
- compare-and-swap;
- Byzantine quorum;
- outlier rejection;
- contribution-size limit;
- identity verification;
- public/open participation mode.

#### Secret handling

- `config.save()` and trainer checkpoints redact `distributed.token`.
- `to_dict()` and `to_yaml()` are unredacted by default when called
  programmatically.
- `speedtronic validate` redacts the distributed token in both JSON and YAML
  output.
- Prefer Hugging Face SDK environment authentication.

Never inspect or print credential files during diagnostics.

#### Repository privacy

The master requests `private=True` when creating a repository, but `exist_ok=True` does not verify that an existing same-name repository is private. Verify visibility explicitly.

#### Path trust

- `node_id` is used in local and remote paths.
- Explicit IDs are checked for `/`, `\\`, `.`, and `..` forms.
- Environment-derived IDs are not revalidated.
- `model_file` metadata is followed without an allow-list.
- Checkpoint `latest.json` file names are joined to the checkpoint directory without strict filename-pattern validation.

#### Denial of service

A malicious delta can use valid keys/shapes but extreme values. The master will average and apply it. Hub operations also have broad retries with no request timeout.

### Known correctness and design issues

#### Causal label contract

Speedtronic 2.0 consumes the already-shifted labels emitted by its synthetic
and text datasets. Custom models must follow the same one-next-token-label per
input-position contract; shifting both dataset and model will skip a target.

#### Text dataset worker sharding

`TextFileTokenDataset` uses the PyTorch worker identity to yield disjoint
blocks to each worker. Each worker still reads the source file to discover
those blocks, so I/O scales with worker count; use a pre-sharded dataset for
large files.

#### Stream-unsafe arbitrary tokenizers

The text stream carries token IDs rather than raw text across chunk boundaries. BPE/SentencePiece-style merges can be split.

#### Scheduler inference

The historical 1000-step scheduler default is synchronized to a shorter
explicit `run.max_steps` target unless the scheduler horizon is set explicitly.

#### Incomplete resume

The checkpoint does not preserve DataLoader iterator, sampler, epoch, worker RNG, or persistent worker state. Resume can repeat or skip examples.

#### GradScaler step accounting

CUDA FP16 overflow can skip the optimizer update. The trainer emits
`optimizer_step_skipped` and does not increment the step, scheduler, or
coordinator boundary for that attempt; retries continue at the same step.

#### Fused AdamW device probe

The model is moved to the resolved device before optimizer construction, so
fused-AdamW capability probing and the actual parameters share a device. A
failed fused construction still falls back to ordinary AdamW.

#### Compile fallback breadth

Any compiled-forward exception disables compilation and reruns eagerly. A genuine model error can be hidden if eager execution succeeds; side effects/RNG can be duplicated.

#### Model training mode

The trainer explicitly calls `model.train()` before fitting. Custom `fit()`
callers that bypass this path remain responsible for their mode.

#### Checkpoint pointer consistency

A checkpoint file can be written before `latest.json` is updated. A valid pointer to a corrupt file does not fall back to older checkpoints. The trainer forces one final checkpoint at the end of a successful fit when the configured interval did not land on the final step.

#### Data vocabulary override

The runtime always passes `model.vocab_size` to the loader, so a separately configured `data.vocab_size` is ignored.

#### Configuration semantics

Validation is intentionally shallow. It does not check model registration, data paths, hardware, or model/data compatibility. Boolean strings are coerced with `bool(...)`.

### v2 limitations

- Muon is a research-backed optimizer whose convergence and simplicity-bias
  trade-offs are not fully characterized.
- Cautious masking observes the base optimizer's complete update, including
  decoupled weight decay, when wrapped around a third-party optimizer.
- OOO backprop is opt-in, CUDA-only in its active path, and mutually exclusive
  with `torch.compile`.
- Shape validation is advisory and does not guarantee a speedup.
- AoT scheduling is deferred as subsumed by `torch.compile`; no private
  scheduler API is shipped.
- Async Hub uploads are bounded to one in-flight job and may be skipped.
- Pending upload state increases checkpoint size by up to one model snapshot.

### Distributed correctness limitations

#### Fixed-path publication is not atomic

Weights and metadata are uploaded separately. Readers can observe mismatched versions, and a failure between uploads can leave new weights under old metadata.

#### Local processed-delta ledger is not remote

A fresh master or a crash after remote publication but before local persistence can aggregate historical deltas again.

#### Multiple masters are unsupported

There is no leader election or distributed lock. Multiple masters overwrite the same global paths and maintain conflicting state.

#### Unbounded storage and bandwidth

Remote delta files, local processed sets, per-version global caches, and downloaded-delta caches grow indefinitely. Every poll downloads all listed candidates to inspect headers before dedupe.

#### Delta cache collisions

The master cache uses only `delta_<step>.safetensors` basenames, so nodes with the same local step share a local path.

#### Broad retry policy

Authentication, 404, malformed request, and programming errors are retried like transient outages. Hub work is dispatched to bounded background lanes by default, so a stuck request can delay a future boundary but does not block the ordinary local inner step.

#### Coordinator restart lifecycle

A stopped v2 coordinator resets its started flag so a later `fit()` can start
fresh background lanes. A request that outlives the bounded shutdown timeout
remains a daemon-side resource warning.

#### Resume baseline ordering

Coordinator startup stages checkpoint state before remote refresh when the v2
`prepare_state` path is available. A checkpointed model/baseline pair therefore
wins at startup, and newer remote state is discovered by the background poll
lane.

#### Partial inner-loop recovery

A worker restart begins from the latest global and can discard local progress
since that publication. v2 restores one bounded pending upload job from a
coordinator checkpoint, but it does not reconstruct every uncommitted inner
optimizer step.

#### Reset semantics are broader than the name

Optimizer state clears after a successful upload or global load, including non-boundary global polls.

#### No distributed data sharding

Node ID, role, and rank do not affect data seed/order. Nodes using identical configurations can see identical synthetic samples.

#### Compatibility is shape-based

No model fingerprint, config hash, schema version, dtype migration, or cryptographic lineage is checked.

#### Non-floating buffers

Integer/boolean buffers are transported globally but excluded from deltas and outer optimization. Model-specific counter or running-state semantics are not handled.

#### Complex tensors

Complex state passes floating checks but arithmetic converts to float32, so imaginary values are not optimized as true complex tensors.

### API and maintenance limitations

- YAML cannot register arbitrary model factories.
- YAML cannot configure metric hooks.
- Standard config fields are transformer-oriented.
- Model-returned auxiliary metrics are discarded by the trainer.
- The default causal dictionary collator is not general-purpose.
- `MetricLogger.close()` is not called by `train_from_config()`.
- `TrainResult.metrics` grows on every step.
- Compatibility aliases and declarative-but-unused types add surface area.
- Repository tests do not cover CLI, logging, coordinator lifecycle, successful outer aggregation, resume, or real Hub networking.

### Performance limitations

- Per-microbatch loss-to-CPU transfer synchronizes CUDA.
- Attention-mask token counting synchronizes through `.item()`.
- Dense explicit attention masks can reduce SDPA efficiency.
- Fused AdamW may silently not engage.
- Compile and checkpoint costs are included in wall-clock step rates.
- Hub I/O is dispatched to bounded background lanes by default; synchronous
  mode remains available for deterministic debugging.

### Explicit non-goals

Speedtronic 2.0.0 does not provide:

- FSDP;
- pipeline parallelism;
- tensor parallelism;
- a model zoo;
- tokenizer training;
- a hosted dashboard;
- open/adversarial distributed participation;
- a dedicated parameter server;
- exact mid-inner-loop worker resume;
- broad model/data schema plugin loading.

### Recommended production posture

1. Treat version 2.0.0 as alpha.
2. Use trusted local checkpoint/dataset files only.
3. Add regression tests for any distributed deployment scenario.
4. Use a private Hub repository and one active master.
5. Grant write access only to trusted service accounts.
6. Explicitly set unique node IDs.
7. Protect state/cache directories with filesystem permissions.
8. Avoid YAML tokens and unredacted validation output.
9. Version Hub repositories per experiment.
10. Validate numerical behavior on target hardware before long runs.

Related: [DumbDiLoCo](references/distributed.md), [Checkpointing](references/checkpointing.md), and [Troubleshooting](references/operations.md).
