# IDWM Migration Notes

Rust source: `src/pes_exploration/idwm.rs`.

Python target: `src/rtip_jax/core/idwm.py`.

## Rust Semantics

- `w(x)` computes `exp(-(x / 3)^5) + 1`.
- `dw(x)` computes the derivative of `w(x)`.
- `wei_dist_mat(coord)` fills only the upper triangle of an `natom x natom`
  weighted-distance matrix. The diagonal and lower triangle stay zero.
- `idw_dist(w_mat0, coord)` compares a reference weighted matrix to the
  weighted distances of `coord`, using only the upper triangle.
- `idw_dist1(w_mat0, w_mat1)` compares two weighted matrices, using only the
  upper triangle.
- `idw_dist_vec(w_mat0, coord)` returns the IDW distance and the normalized
  coordinate vector used by the Rust force expression.
- `idw_pot_force` computes `a * exp(-dist^2 / (2*sigma^2))` and force
  `vec * pot * dist / sigma^2`.
- `Idwm0PES` stores weighted matrices for a local minimum and nearby transition
  states, then evaluates energies and forces on either the full system or
  `system.atom_add_pot` fragments.

## Input and Output

- Coordinate input shape: `(natom, 3)`, internal unit Bohr.
- Weighted matrix input shape: `(natom, natom)`, upper triangle meaningful.
- `idw_dist` and `idw_dist1` return scalar JAX arrays.
- `idw_dist_vec` returns `(distance, vector)` where vector shape is `(natom, 3)`.
- `idw_pot_force` returns `(energy, force)` where force shape is `(natom, 3)`.
- `Idwm0PES.get_energy(system)` returns a Python `float`.
- `Idwm0PES.get_energy_force(system)` returns `(float, JAX array)`.

## State and Side Effects

- Core functions are pure and side-effect free.
- `Idwm0PES` is a frozen dataclass and normalizes matrices in `__post_init__`.
- Fragment force scatter uses JAX `.at[indices].set`.
- No IO, CP2K, MPI, or random state is used in this module.

## Numerical Behavior

- The zero-distance singularity from Rust is preserved: if `idw_dist_vec`
  receives a configuration identical to its reference, normalization by zero can
  produce NaN values.
- No epsilon stabilization has been added in Stage 4.1.
- Lower-triangle data in reference matrices is ignored, matching Rust loops.

## JAX Support

- `weight`, `weight_derivative`, `wei_dist_mat`, `idw_dist`, `idw_dist1`,
  `idw_dist_vec`, `idw_pot`, and `idw_pot_force` are written with `jax.numpy`
  and should be compatible with `jit` for fixed shapes.
- `vmap` should work over external wrappers that pass fixed-size coordinates.
- `Idwm0PES` methods return Python floats and include Python loops over
  transition-state matrices, so they are workflow-level wrappers rather than
  primary `jit` targets.

## Key Differences From Rust

- Rust mutable loops are replaced by vectorized `jax.numpy` operations.
- Rust `Array2<Array1<f64>>` temporary storage is replaced by a single pairwise
  vector tensor.
- Rust `panic!` shape errors are Python `ValueError`s.
- The PES wrapper stores tuples instead of mutable `Vec` state.

## Tests

Tests are in `tests/test_idwm.py` and cover:

- Weight and derivative formulas.
- Upper-triangular weighted matrix behavior.
- Matrix-vs-coordinate and matrix-vs-matrix distances.
- Gaussian energy formula.
- Force direction against finite differences.
- Fragment force scatter for `atom_add_pot`.

