---
id: security-and-limitations
title: Security and Limitations
sidebar_label: Security & Limits
description: Trust boundaries, unsafe deserialization, Hub writer risks, known correctness issues, and non-goals in Speedtronic 2.0.0.
---

# Security and limitations

## Security boundaries

### Pickle-capable local files

The following paths use `torch.load(..., weights_only=False)` or equivalent pickle-capable loading:

- local trainer checkpoints;
- serialized datasets configured through `data.dataset`;
- master `outer_state.pt`.

Only load files from trusted local sources. A write-capable local attacker can craft a checkpoint that executes code during deserialization.

### Hub writer trust

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

### Secret handling

- `config.save()` and trainer checkpoints redact `distributed.token`.
- `to_dict()` and `to_yaml()` are unredacted by default when called
  programmatically.
- `speedtronic validate` redacts the distributed token in both JSON and YAML
  output.
- Prefer Hugging Face SDK environment authentication.

Never inspect or print credential files during diagnostics.

### Repository privacy

The master requests `private=True` when creating a repository, but `exist_ok=True` does not verify that an existing same-name repository is private. Verify visibility explicitly.

### Path trust

- `node_id` is used in local and remote paths.
- Explicit IDs are checked for `/`, `\\`, `.`, and `..` forms.
- Environment-derived IDs are not revalidated.
- `model_file` metadata is followed without an allow-list.
- Checkpoint `latest.json` file names are joined to the checkpoint directory without strict filename-pattern validation.

### Denial of service

A malicious delta can use valid keys/shapes but extreme values. The master will average and apply it. Hub operations also have broad retries with no request timeout.

## Known correctness and design issues

### Causal label contract

Speedtronic 2.0 consumes the already-shifted labels emitted by its synthetic
and text datasets. Custom models must follow the same one-next-token-label per
input-position contract; shifting both dataset and model will skip a target.

### Text dataset worker sharding

`TextFileTokenDataset` uses the PyTorch worker identity to yield disjoint
blocks to each worker. Each worker still reads the source file to discover
those blocks, so I/O scales with worker count; use a pre-sharded dataset for
large files.

### Stream-unsafe arbitrary tokenizers

The text stream carries token IDs rather than raw text across chunk boundaries. BPE/SentencePiece-style merges can be split.

### Scheduler inference

The historical 1000-step scheduler default is synchronized to a shorter
explicit `run.max_steps` target unless the scheduler horizon is set explicitly.

### Incomplete resume

The checkpoint does not preserve DataLoader iterator, sampler, epoch, worker RNG, or persistent worker state. Resume can repeat or skip examples.

### GradScaler step accounting

CUDA FP16 overflow can skip the optimizer update. The trainer emits
`optimizer_step_skipped` and does not increment the step, scheduler, or
coordinator boundary for that attempt; retries continue at the same step.

### Fused AdamW device probe

The model is moved to the resolved device before optimizer construction, so
fused-AdamW capability probing and the actual parameters share a device. A
failed fused construction still falls back to ordinary AdamW.

### Compile fallback breadth

Any compiled-forward exception disables compilation and reruns eagerly. A genuine model error can be hidden if eager execution succeeds; side effects/RNG can be duplicated.

### Model training mode

The trainer explicitly calls `model.train()` before fitting. Custom `fit()`
callers that bypass this path remain responsible for their mode.

### Checkpoint pointer consistency

A checkpoint file can be written before `latest.json` is updated. A valid pointer to a corrupt file does not fall back to older checkpoints. The trainer forces one final checkpoint at the end of a successful fit when the configured interval did not land on the final step.

### Data vocabulary override

The runtime always passes `model.vocab_size` to the loader, so a separately configured `data.vocab_size` is ignored.

### Configuration semantics

Validation is intentionally shallow. It does not check model registration, data paths, hardware, or model/data compatibility. Boolean strings are coerced with `bool(...)`.

## v2 limitations

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

## Distributed correctness limitations

### Fixed-path publication is not atomic

Weights and metadata are uploaded separately. Readers can observe mismatched versions, and a failure between uploads can leave new weights under old metadata.

### Local processed-delta ledger is not remote

A fresh master or a crash after remote publication but before local persistence can aggregate historical deltas again.

### Multiple masters are unsupported

There is no leader election or distributed lock. Multiple masters overwrite the same global paths and maintain conflicting state.

### Unbounded storage and bandwidth

Remote delta files, local processed sets, per-version global caches, and downloaded-delta caches grow indefinitely. Every poll downloads all listed candidates to inspect headers before dedupe.

### Delta cache collisions

The master cache uses only `delta_<step>.safetensors` basenames, so nodes with the same local step share a local path.

### Broad retry policy

Authentication, 404, malformed request, and programming errors are retried like transient outages. Hub work is dispatched to bounded background lanes by default, so a stuck request can delay a future boundary but does not block the ordinary local inner step.

### Coordinator restart lifecycle

A stopped v2 coordinator resets its started flag so a later `fit()` can start
fresh background lanes. A request that outlives the bounded shutdown timeout
remains a daemon-side resource warning.

### Resume baseline ordering

Coordinator startup stages checkpoint state before remote refresh when the v2
`prepare_state` path is available. A checkpointed model/baseline pair therefore
wins at startup, and newer remote state is discovered by the background poll
lane.

### Partial inner-loop recovery

A worker restart begins from the latest global and can discard local progress
since that publication. v2 restores one bounded pending upload job from a
coordinator checkpoint, but it does not reconstruct every uncommitted inner
optimizer step.

### Reset semantics are broader than the name

Optimizer state clears after a successful upload or global load, including non-boundary global polls.

### No distributed data sharding

Node ID, role, and rank do not affect data seed/order. Nodes using identical configurations can see identical synthetic samples.

### Compatibility is shape-based

No model fingerprint, config hash, schema version, dtype migration, or cryptographic lineage is checked.

### Non-floating buffers

Integer/boolean buffers are transported globally but excluded from deltas and outer optimization. Model-specific counter or running-state semantics are not handled.

### Complex tensors

Complex state passes floating checks but arithmetic converts to float32, so imaginary values are not optimized as true complex tensors.

## API and maintenance limitations

- YAML cannot register arbitrary model factories.
- YAML cannot configure metric hooks.
- Standard config fields are transformer-oriented.
- Model-returned auxiliary metrics are discarded by the trainer.
- The default causal dictionary collator is not general-purpose.
- `MetricLogger.close()` is not called by `train_from_config()`.
- `TrainResult.metrics` grows on every step.
- Compatibility aliases and declarative-but-unused types add surface area.
- Repository tests do not cover CLI, logging, coordinator lifecycle, successful outer aggregation, resume, or real Hub networking.

## Performance limitations

- Per-microbatch loss-to-CPU transfer synchronizes CUDA.
- Attention-mask token counting synchronizes through `.item()`.
- Dense explicit attention masks can reduce SDPA efficiency.
- Fused AdamW may silently not engage.
- Compile and checkpoint costs are included in wall-clock step rates.
- Hub I/O is dispatched to bounded background lanes by default; synchronous
  mode remains available for deterministic debugging.

## Explicit non-goals

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

## Recommended production posture

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

Related: [DumbDiLoCo](../distributed/overview), [Checkpointing](../reference/checkpointing), and [Troubleshooting](./troubleshooting).
