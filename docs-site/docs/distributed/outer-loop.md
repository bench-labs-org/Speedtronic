---
id: outer-loop
title: Outer Optimizer and Master Loop
sidebar_label: Outer Loop
description: Nesterov outer SGD, delta discovery, validation, aggregation, rollback, persistence, snapshots, and background polling.
---

# Outer optimizer and master loop

## `NesterovOuterOptimizer`

```python
NesterovOuterOptimizer(
    state,
    lr,
    momentum=0.9,
)
```

Maintains CPU FP32 momentum buffers for floating/complex state keys.

### `step(state, delta)`

For every delta key:

1. Require the key to exist in global state.
2. Require matching shape.
3. Skip non-floating/non-complex state.
4. Convert gradient to CPU FP32.
5. Update momentum:

```text
buffer = momentum * buffer + gradient
```

6. Compute:

```text
effective = gradient + momentum * buffer
```

7. Update in the global state's native dtype:

```text
state -= lr * effective
```

This matches PyTorch's Nesterov SGD formulation used by the repository test.

### `state_dict()`

Returns cloned CPU momentum tensors only. Learning rate and momentum coefficient are not stored.

### `load_state_dict(state)`

Replaces momentum buffers with cloned CPU FP32 tensors.

## `OuterState`

A dataclass exists:

```python
OuterState(
    global_state,
    outer_step=0,
    processed_deltas=set(),
    momentum={},
    last_error=None,
)
```

The current `MasterOuterLoop` stores equivalent fields directly and does not use this dataclass.

## `MasterOuterLoop`

```python
MasterOuterLoop(
    hub,
    state,
    *,
    node_state_dir,
    outer_lr=0.7,
    outer_momentum=0.9,
    poll_interval=60.0,
    on_global_update=None,
    logger=None,
)
```

Immediately clones the initial full state to CPU. Loading `outer_state.pt` replaces only matching same-shape tensors.

Properties:

```python
loop.processed_filenames
loop.state_path
loop.metadata_path
```

## Processed identity

```text
(node_id, local_step, base_outer_step)
```

Node and local step come from the path; base step comes from safetensors metadata.

`processed_filenames` reconstructs only node/local paths and cannot distinguish different base-step metadata for the same file.

## Candidate discovery

`_delta_candidates()` calls `hub.delta_paths()`, accepts paths parsed as:

```text
nodes/<node>/delta_<integer>.safetensors
```

and sorts them by node, local step, and full path.

## Delta download cache

Downloads are cached under:

```text
<state_dir>/cache/deltas/<basename>
```

The basename omits the node ID, so two nodes with the same local step share a local path. Current code downloads immediately before reading, limiting immediate collision impact, but the cache is not safely node-namespaced.

## One `sync_once()` round

Under one outer lock:

1. List and sort candidates.
2. Download every candidate.
3. Parse base outer step and construct identity.
4. Skip already processed identities.
5. Compute the expected floating/complex key set.
6. Require exact delta key-set equality.
7. Require matching shapes.
8. Log and skip unreadable/corrupt/incompatible files.
9. If no valid delta remains, emit an event and return.
10. Average all valid deltas in FP32.
11. Snapshot old global state, processed set, step, and momentum.
12. Apply one Nesterov outer update.
13. Increment outer step.
14. Add valid identities to the processed set.
15. Publish global state and metadata.
16. Save local outer state.
17. On publication/save failure, restore all in-memory snapshots and re-raise.
18. Invoke `on_global_update` with a cloned full state.
19. Emit `outer_step`.

The lock covers all network I/O, so snapshots from the training thread can block for a full round.

## Validation

A delta is accepted when:

- safetensors and metadata load;
- keys exactly match all floating/complex global-state keys;
- each shape matches.

The implementation does not validate:

- NaN/infinity;
- magnitude;
- staleness against the current outer step;
- path/header node identity agreement;
- algorithm marker;
- model/config hash;
- dtype compatibility;
- cryptographic publisher identity.

## Average

```python
average = sum(valid_deltas) / len(valid_deltas)
```

Every delta has equal weight. There is no participant, sample, token, age, or quality weighting.

## In-process rollback

If publishing or local persistence fails after in-memory mutation, the loop restores:

```text
global_state
outer_step
processed_deltas
Nesterov momentum
```

The remote publication may already have succeeded before a later local failure, so the rollback cannot undo remote side effects.

## Local master state

`outer_state.pt`:

```text
global_state
outer_step
processed_deltas
momentum
```

`outer_metadata.json`:

```text
outer_step
processed_deltas
updated_at
```

Both are written atomically through local temp-and-rename helpers. The JSON is diagnostic; the `.pt` file is authoritative.

## Remote recovery

`adopt_global_state()` installs a complete remote state and outer step. It can reset momentum. It does not learn the remote processed-delta ledger or Nesterov momentum.

A hard crash after remote publication but before local state persistence can cause the restarted master to aggregate already-published deltas again.

## Snapshot API

### `snapshot()`

Under the lock, returns the outer step and a clone of the complete global state.

### `start()`

Starts one daemon thread named `speedtronic-diloco-master` unless one is already alive.

### Background loop

The thread calls `sync_once()` immediately, waits `poll_interval`, and repeats. Exceptions are logged and retried on the next interval.

### `stop()`

Sets an event and joins for at most 10 seconds. It does not force a final synchronization and can return while network work is still running.

## `state_dict()` and `load_state_dict()`

`state_dict()` returns:

```text
outer_step
processed_deltas
momentum
global_state
```

`load_state_dict()` restores these values and writes the local state file as a side effect.

## Unbounded growth

The remote delta tree, processed ledger, per-version global cache, and redownload workload grow over time. There is no delta deletion, compaction, or protocol-level processed list.

## Multiple masters

No leader election, lease, or compare-and-swap protects global files. Multiple masters maintain independent momentum/processed sets and can overwrite the same paths with conflicting versions.

Related: [Overview](./overview), [Hub Transport](./hub-transport), and [Tensor Format](./tensor-format).
