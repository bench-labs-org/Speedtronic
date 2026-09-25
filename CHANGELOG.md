# Changelog

All notable Speedtronic changes are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## 2.0.0 - 2026-09-25

### Added

- **Hybrid Muon optimizer** with pure-PyTorch Newton–Schulz orthogonalization.
  Hidden 2-D linear/attention/MLP matrices route to Muon; embeddings, heads,
  normalization parameters, biases, and other tensors remain on AdamW.
- **Muon+** post-orthogonalization normalization, enabled with
  `optimizer.muon_plus: true`.
- **Cautious optimizer wrapper**, enabled with `optimizer.cautious: true`,
  compatible with AdamW and Muon.
- **Conservative CUDA stream scheduling** for opt-in out-of-order backprop,
  with dependency events and sequential CPU/MPS fallback. Configure with
  `ooo_backprop: true`.
- **Startup shape-efficiency validation** with hardware/dtype-aware,
  non-fatal warnings and nearby alignment suggestions.
- **Non-blocking DumbDiLoCo delta dispatch** by default, with bounded queue
  size, skip-and-log overflow behavior, asynchronous global polling, and
  bounded shutdown.
- New `optimizers.py`, `shapes.py`, and `scheduling.py` modules.
- New CPU and fake-Hub tests covering the v2 optimizer, systems, and
  asynchronous distributed paths.

### Changed

- The distribution is now version `2.0.0`.
- The reference model's causal loss consumes the already-shifted labels emitted
  by Speedtronic datasets, avoiding a double target shift.
- Runtime builds the model on the resolved device before optimizer creation,
  making fused-AdamW probing consistent with actual parameter placement.
- Trainer explicitly sets `model.train()` before fitting.
- Configuration validation keeps the default scheduler horizon synchronized
  with a shorter explicit `run.max_steps` target.
- `speedtronic validate` redacts distributed tokens in printed output.
- Metric emission is serialized for safe asynchronous event handling.
- DumbDiLoCo defaults to asynchronous delta upload and global polling; the
  synchronous path remains available with `async_delta_upload: false` and
  `async_global_poll: false`.

### Documentation decisions

- Muon may change the selected solution's simplicity bias as well as its
  convergence speed; it is not a guaranteed free lunch.
- Ahead-of-time kernel scheduling is treated as **subsumed/deferred** for this
  release. `torch.compile` and Inductor already perform graph capture and
  kernel scheduling; Speedtronic does not add a duplicate private-API flag.
- Sophia, mixture-of-experts layers, FP8 training, and multi-GPU
  pipeline/tensor parallelism remain deferred or out of scope as specified in
  `v2_additions.md`.

## 0.1.0 - 2026-09-25

- Initial public PyPI release of the configuration-driven PyTorch training
  engine, reference transformer, checkpoints, metrics, and DumbDiLoCo.
