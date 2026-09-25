---
id: tensor-format
title: Tensor and Wire Format
sidebar_label: Tensor Format
description: CPU tensor normalization, safetensors I/O, pseudo-gradients, averaging, path parsing, and distributed file schema.
---

# Tensor and wire format

## `cpu_tensor(value)`

```python
value.detach().to(device="cpu").contiguous().clone()
```

This removes autograd history, transfers to CPU, makes contiguous layout, and clones storage. Cloning allows tied embedding/LM-head parameters to be serialized together, which safetensors rejects when they share memory.

## `cpu_state_dict(state)`

Applies `cpu_tensor()` to every value and stringifies keys.

## `save_safetensors(state, path, metadata=None)`

1. Creates parent directories.
2. Clones/normalizes every tensor.
3. Calls `safetensors.torch.save_file`.
4. Writes string metadata when supplied.

Integer and boolean state can be saved in global model files, but deltas exclude them.

## `load_safetensors(path)`

Calls `safetensors.torch.load_file(path, device="cpu")`.

## `read_metadata(path)`

Uses `safetensors.safe_open` to return safetensors header metadata as a string dictionary.

## `floating_state(state)`

Returns keys whose tensors are floating-point or complex.

## `compute_pseudo_gradient(baseline, current)`

Requirements:

- exact key-set equality;
- exact shape equality.

Output:

```text
float32(baseline) - float32(current)
```

for floating/complex keys. Non-floating keys are omitted. Any key or shape mismatch rejects the entire delta.

The sign is intentionally baseline minus current: local optimization moves from baseline toward current, so this is the outer direction to add to global weights.

## `average_deltas(deltas)`

- Rejects an empty list.
- Requires identical key sets.
- Uses the first delta's shapes as references.
- Accumulates in FP32.
- Divides by the number of deltas.

Dictionary key order is not significant.

## `parse_delta_path(path)`

Accepts exactly:

```text
nodes/<node_id>/delta_<integer>.safetensors
```

It splits on `/`, requires three segments, and parses the filename after removing `delta_` and `.safetensors`.

It does not validate that the file's embedded node ID and local step match the path.

## `load_delta(path)`

1. Reads safetensors metadata.
2. Loads tensor state.
3. Converts tensor-loading failures to `ValueError` with a stable message.

The outer loop catches that error, logs the filename, and skips the candidate.

## `metadata_json(metadata)`

Attempts to JSON-decode every metadata string. If any value is not valid JSON, it returns the original string dictionary. The current outer loop does not use this helper.

## Global model file

`global/latest.safetensors` contains the complete cloned `state_dict`:

- trainable parameters;
- floating buffers;
- integer/boolean buffers;
- other registered state entries.

No safetensors metadata accompanies the global model. Version information is in `global/step_count.json`.

## Global metadata file

```json
{
  "algorithm": "dumb_diloco",
  "model_file": "global/latest.safetensors",
  "optimizer": "nesterov_sgd",
  "outer_step": 12,
  "updated_at": "2026-01-01T12:00:00+00:00"
}
```

Bootstrap metadata adds:

```json
{"bootstrap": "true"}
```

Extra values are converted to strings.

## Node delta file

Path:

```text
nodes/<node_id>/delta_<local_step>.safetensors
```

Header:

```json
{
  "algorithm": "dumb_diloco",
  "base_outer_step": "11",
  "local_step": "500",
  "node_id": "worker-1"
}
```

Tensor payload:

```text
baseline - current
```

in FP32 for floating/complex state keys.

## Full remote tree

```text
<repo_id>/
├── global/
│   ├── latest.safetensors
│   └── step_count.json
└── nodes/
    ├── master-1/
    │   └── delta_500.safetensors
    ├── worker-1/
    │   ├── delta_500.safetensors
    │   └── delta_1000.safetensors
    └── worker-2/
        └── delta_500.safetensors
```

## Local coordinator tree

```text
<state_root>/<node_id>/
├── global/
│   └── latest-<outer_step>.safetensors
├── outgoing/
│   ├── latest.safetensors
│   └── <node_id>_delta_<step>.safetensors
└── cache/
    └── deltas/
        └── delta_<step>.safetensors
```

Master additionally stores:

```text
outer_state.pt
outer_metadata.json
```

## Compatibility rules

### Model installation

Global installation is permissive:

- matching key/shape entries load;
- missing local/global keys are tolerated;
- shape mismatches are silently ignored.

### Delta validation

Delta aggregation is strict:

- floating key set must match exactly;
- every shape must match;
- one bad delta is skipped in full.

There is no architecture/configuration fingerprint.

## Non-floating state

Non-floating entries:

- are published in global model files;
- are installed on workers;
- do not appear in deltas;
- do not receive outer updates.

Models with running integer counters or model-specific non-floating synchronization semantics need custom handling.

## Complex tensors

Complex keys pass floating-state checks, but arithmetic converts to FP32. Imaginary components are not preserved as true complex optimization. Real-valued models are the safe path.

## Safetensors safety

Safetensors protects the tensor container from arbitrary pickle code execution, but it does not authenticate writers, encrypt data, or validate numerical values.

Related: [Hub Transport](./hub-transport), [Outer Loop](./outer-loop), and [Security](../operations/security-and-limitations).
