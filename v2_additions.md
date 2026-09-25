# Speedtronic — v2 Addendum: Novel Optimization Techniques

This section extends the v1 spec with additions for the next release. Everything here follows the same rule as v1: no feature exists because it's novel for its own sake — it exists because it measurably improves training-phase speed or efficiency, and it must not break GPU-agnosticism (must degrade gracefully on CPU/MPS/any CUDA card, no hardware-locked dependency).

Two categories: **optimizer algorithms** (what update rule is used) and **systems/scheduling techniques** (how the same computation is executed faster on the hardware). Both matter; they're independent and compose with each other.

## 9. Optimizer additions

### 9.1 Muon (primary addition)

Muon is a matrix-structured optimizer that orthogonalizes the momentum via Newton-Schulz iteration, rather than AdamW's element-wise scaling. It set new training speed records on early benchmarks and has since been scaled to multi-billion-parameter LLMs, with reported roughly 2x training efficiency improvements over AdamW.

**Implementation requirements:**
- Muon applies to 2D weight matrices (linear/attention/MLP layers) only. Embeddings, layer norms, and biases must stay on AdamW. This means Speedtronic needs a **hybrid optimizer setup**: parameters are routed to Muon or AdamW based on shape/role, not a single global optimizer swap.
- Config flag to select optimizer: `optimizer: adamw` (v1 default, unchanged) or `optimizer: muon` (routes eligible params to Muon, rest to AdamW automatically).
- Newton-Schulz iteration must be implemented in pure PyTorch ops (no custom CUDA kernel) to preserve GPU-agnosticism.

### 9.2 Muon+ (cheap refinement, opt-in alongside Muon)

Adds one additional normalization step after orthogonalization. Reported as a consistent improvement over plain Muon across model scales, at near-zero extra implementation cost once Muon exists.

- Config flag: `muon_plus: true` (only valid when `optimizer: muon`).

### 9.3 Cautious optimizer wrapper (cheap, composable with anything)

A one-line-of-code modification pattern: mask parameter updates that don't align in sign with the current gradient. Composes with AdamW or Muon as a wrapper, not a separate optimizer implementation.

- Config flag: `cautious: true`, valid regardless of which base optimizer is selected.

### 9.4 Considered and deferred

- **Sophia** (second-order, Hessian-estimate-based optimizer): real speedups reported, but adds meaningful implementation complexity and has seen less consistent adoption than Muon. Deferred, not rejected — candidate for a future version if Muon integration goes well and there's appetite for a second-order option.

### 9.5 Known tradeoff to document

Active research raises the question of whether Muon's speedup comes with a loss of "simplicity bias" relative to AdamW — i.e., it may change *what* solution the model converges to, not just how fast it gets there. This should be one line in Speedtronic's docs (not a reason to withhold the feature, but users should know it's not a strict free lunch).

## 10. Systems / scheduling additions

### 10.1 Out-of-order backprop scheduling (highest-leverage addition in this section)

Standard backprop computes gradients strictly in reverse layer order, even though many gradient computations don't actually depend on each other. This leaves GPUs underutilized waiting on kernel launch overhead and unnecessary sequential dependencies. Out-of-order backprop reorders gradient computation according to the actual dependency graph and uses multi-stream execution to mask kernel launch overhead; reported single-GPU throughput improvements in the original work ranged roughly 1.03 to 1.58x depending on model.

**Implementation requirements:**
- Build the gradient dependency graph from the model's computation graph (standard autograd already tracks this — the work is in extracting a schedulable DAG from it, not rebuilding autograd).
- Use multiple CUDA streams (or the equivalent no-op on CPU/MPS — must degrade to standard sequential backprop where streams aren't meaningfully supported) to run independent gradient computations concurrently.
- Config flag: `ooo_backprop: true`. Off by default in v2 (opt-in) until it's been validated against the reference model and a few user model shapes, given this is the most implementation-heavy item in the addendum.
- This is genuinely novel *to accessible training frameworks* — it's published research, not a from-scratch invention, but it is not a default feature in mainstream frameworks (PyTorch, Hugging Face `transformers`/`trl`, etc.), so shipping it as a first-class flag is a real differentiator for Speedtronic.

### 10.2 Ahead-of-time (AoT) kernel scheduling

Related idea: resolve scheduling decisions before kernel execution rather than paying scheduling overhead every step. Composes naturally with `torch.compile`'s graph capture.

- Treat as a config flag layered on top of `compile: true` rather than a separate system — smaller implementation lift than §10.1, lower priority.
- Status: investigate feasibility on top of `torch.compile` before committing; may be subsumed by what `torch.compile` already does internally, in which case this item is dropped rather than duplicated.

### 10.3 Batch/dimension shape validation (cheap, high value-to-effort)

Not a novel technique — a correctness/efficiency guard. Tensor Core GEMM efficiency depends on batch size and layer dimensions being multiples of 8 (or higher, hardware/dtype-dependent); awkward shapes silently tank throughput with no error.

**Implementation requirement:**
- At run startup, Speedtronic inspects the configured batch size and model dimensions and emits a warning (not an error) if they're not well-aligned for the detected hardware/dtype, with a suggested nearby value.
- This directly serves "be fast, be efficient" by catching a common, easy-to-miss throughput bug automatically instead of requiring the user to discover it.

### 10.4 Non-blocking DumbDiLoCo sync (extends v1 §3, DumbDiLoCo-specific)

Currently (v1 spec §3.2) each node's delta push to HF happens as a discrete step in the inner loop. In v2, the push should be **non-blocking relative to the next inner-loop step** — i.e. the upload happens in a background thread/async call while local training continues, rather than stalling on network I/O.

**Implementation requirements:**
- Node continues its next local optimizer step immediately after dispatching the delta upload, not after the upload completes.
- Must handle the case where an upload is still in flight when the next sync point arrives — queue or skip-and-log, don't block (consistent with the "master doesn't wait" philosophy already in v1 §3.5).
- This is not from external research — it's closing a gap in Speedtronic's own architecture, and is in scope because it's a direct, low-risk throughput win specific to DumbDiLoCo.

### 10.5 Considered and deferred

- **MoE layers**: real throughput/parameter-efficiency gains reported, but this is an architecture decision affecting the reference model, not a training-engine technique, and adds significant memory/complexity. Deferred to a possible future reference-model variant, not the core engine.
- **FP8 training**: meaningful speedup on supported hardware, but requires Hopper-class (or newer) GPUs — breaks GPU-agnosticism as a default. Could be added later as a hardware-detected opt-in (same pattern as bf16 detection in v1 §2.1), not in this addendum.
- **Multi-GPU pipeline/tensor parallelism**: explicitly out of scope per v1 §8 non-goals; unchanged in v2.

## 11. Addendum summary — what's actually new

| Addition | Category | Effort | Priority |
|---|---|---|---|
| Muon | Optimizer | Medium | High — flagship v2 feature |
| Muon+ | Optimizer | Low | High — cheap once Muon exists |
| Cautious wrapper | Optimizer | Low | Medium |
| Out-of-order backprop | Systems | High | High — most differentiating addition |
| AoT kernel scheduling | Systems | Medium (pending investigation) | Low |
| Batch/dimension validation | Systems | Low | High — cheap, catches real bugs |
| Non-blocking DumbDiLoCo sync | Systems (DumbDiLoCo-specific) | Low-Medium | Medium |
