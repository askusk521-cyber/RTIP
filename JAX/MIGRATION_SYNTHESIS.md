# Synthesis Migration Notes

Rust source: `src/pes_exploration/synthesis.rs`.

Python target: `src/rtip_jax/workflows/synthesis.py`.

## Rust Semantics

- Supports exactly 2, 3, or 4 molecules.
- Extracts each molecule's coordinates from `System`.
- Recenters each molecule at its geometric center.
- Applies a random rotation matrix to each molecule.
- Places molecule centers on fixed 2-body, triangular 3-body, or tetrahedral
  4-body offsets scaled by `dist`.
- Mutates the input `System` coordinates in place.
- Uses MPI broadcast so all ranks receive the root rank's random rotations.

## Python Semantics

- `synthesize_layout(system, mol_index, dist, key=...)` returns a new `System`.
- `rotations=...` can be supplied for deterministic tests or external control.
- Exactly one of `key` or `rotations` must be provided.
- Random rotations use explicit JAX PRNG keys.
- No MPI behavior is implemented in this pure helper.

## Inputs and Outputs

- `system`: `System` with `(natom, 3)` coordinates.
- `mol_index`: 2, 3, or 4 non-empty atom-index groups.
- `dist`: scalar placement distance.
- Output: new `System` with updated coordinates and preserved metadata.

## State and Side Effects

- Pure function returning a replacement `System`.
- No IO, CP2K, MPI, or global random state.

## Numerical Behavior

- Fixed placement constants are copied from Rust.
- Random streams do not match Rust `ndarray-rand`; explicit-key
  reproducibility is the Python contract.

## JAX Support

- The layout arithmetic uses `jax.numpy`.
- The function includes Python loops over 2/3/4 molecule groups and is a
  workflow helper, not a primary `jit` target.

## Tests

Tests are in `tests/test_synthesis.py` and cover:

- 2/3/4 molecule offsets.
- Identity-rotation deterministic layouts.
- Explicit-key reproducibility.
- Input validation.

