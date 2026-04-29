# RTIP Migration Notes

Rust source: `src/pes_exploration/rtip.rs`.

Python target: `src/rtip_jax/core/rtip.py`.

## Rust Semantics

- Coordinates are centered by subtracting each structure's geometric center.
- The quaternion-method 4x4 matrix is built by summing `A_i.T @ A_i` across
  atoms.
- `rti_dist` returns the square root of the smallest eigenvalue.
- `rti_dists` returns square roots of all four eigenvalues.
- `rti_dist_vec` and `rti_dists_vecs` convert eigenvectors to rotation matrices
  and return aligned residual vectors.
- `rti_rot_tran` returns the rotation matrix and translation vector that align
  `coord1` to `coord2`.
- `rti_pot` computes a four-distance weighted Gaussian potential with
  `f(x) = 1 / x^7`.
- `rti_pot_force` computes the Rust force expression on `coord2`.
- `Rtip0PES` evaluates local-minimum and nearby-transition-state RTIP
  contributions, optionally using `atom_add_pot` fragments on the reference and
  target systems.

## Input and Output

- Coordinate input shape: `(natom, 3)`, internal unit Bohr.
- `rti_dist` returns a scalar JAX array.
- `rti_dists` returns a JAX array of shape `(4,)`.
- `rti_dist_vec` returns `(distance, vector)` where vector shape is `(natom, 3)`.
- `rti_dists_vecs` returns `(distances, vectors)` where vector shape is
  `(4, natom, 3)`.
- `rti_rot_tran` returns `(rotation, translation)` with shapes `(3, 3)` and
  `(3,)`.
- `rti_pot_force` returns `(energy, force)` where force shape is `(natom, 3)`.
- `Rtip0PES.get_energy(system)` returns a Python `float`.
- `Rtip0PES.get_energy_force(system)` returns `(float, JAX array)`.

## State and Side Effects

- Core functions are pure and side-effect free.
- `Rtip0PES` is a frozen dataclass.
- Fragment force scatter uses JAX `.at[indices].set`.
- No IO, CP2K, MPI, or random state is used in this module.

## Numerical Behavior

- Eigenvalue and eigenvector behavior depends on the backend eigensolver.
- Eigenvector signs are not stable across implementations, but the derived
  rotation should preserve alignment behavior.
- Degenerate or near-degenerate eigenvalues can change eigenvector selection.
- The Rust singularity in `rti_weight(x) = 1 / x^7` is preserved; exact zero
  RTI distances can produce inf/NaN in weighted potentials and forces.
- No eigenvalue clamping or epsilon stabilization has been added in Stage 4.2.

## JAX Support

- `rti_dist`, `rti_dists`, `rti_dist_vec`, `rti_dists_vecs`,
  `rti_rot_tran`, `rti_pot`, and `rti_pot_force` use `jax.numpy` and should be
  compatible with `jit` for fixed shapes, subject to eigensolver support.
- `vmap` should work over wrappers with fixed atom counts.
- `Rtip0PES` methods return Python floats and include Python loops over
  transition states, so they are workflow-level wrappers rather than primary
  `jit` targets.

## Key Differences From Rust

- Rust mutable loops are replaced by vectorized `jax.numpy` operations.
- Rust `Vec<Array2<f64>>` output from `rti_dists_vecs` becomes a stacked JAX
  array of shape `(4, natom, 3)`.
- Rust's implicit shape panics are explicit Python `ValueError`s.
- The PES wrapper stores transition states and sigma values as immutable tuples.

## Tests

Tests are in `tests/test_rtip.py` and cover:

- RTIP weighting formulas.
- Quaternion-to-rotation identity case.
- Translation and rotation/translation invariance of `rti_dist`.
- `rti_rot_tran` reconstruction.
- RTI vector norm consistency.
- Weighted Gaussian potential formula.
- Force direction against finite differences.
- Fragment force scatter for `atom_add_pot`.

