# Numerical Differences

This file records known or expected numerical differences between Rust and JAX.

## Current Status

No core RTIP or IDWM algorithm has been migrated yet.

Stage 3 added constants, configuration, system IO, and random rotations.
Constants and unit conversions are copied from Rust. Random rotations preserve
the Rust angle-to-matrix formula, but not Rust's `ndarray-rand` random stream.

Stage 4.1 added IDWM. It keeps Rust's upper-triangle convention and preserves
the zero-distance normalization singularity in `idw_dist_vec`.

Stage 4.2 added RTIP. It preserves the Rust quaternion matrix and
quaternion-to-rotation formula, but backend eigensolver differences can change
eigenvector signs or choices in degenerate cases. Stage 6 validation found that
JAX can return a tiny negative eigenvalue for an exactly rotation/translation
equivalent structure; RTIP distance square roots now clamp the PSD eigenspectrum
at zero before `sqrt`.

Stage 4.3 added optimization. It preserves Rust's one-dimensional search
strategy, but explicitly rejects zero total-force directions instead of letting
division-by-zero/NaN behavior proceed.

Stage 4.4 added bias PES containers and a Python-only `SumPES` helper. The
containers do not introduce numerical behavior.

Stage 4.5 added synthesis layout. It preserves Rust placement constants but
uses explicit JAX PRNG keys instead of Rust `ndarray-rand` and MPI broadcast.

Stage 4.6 added pathway helper logic. Initial perturbations preserve Rust's
normalization shape but use explicit JAX PRNG keys.

Stage 4.7 added MD helpers. Single-atom temperature is explicitly rejected
instead of dividing by zero through `3 * (natom - 1)`.

Stage 5 added Python workflow runners. The runners preserve the Rust update
order at the orchestration level, but use generic PES providers, explicit JAX
PRNG keys, and `ZeroPES` when a bias is disabled. Workflow-level equivalence
still depends on future Rust scalar-log snapshots.

Stage 6 added DeePMD as the production replacement for CP2K. DeePMD native
outputs are eV, eV/Angstrom, and eV virial; the provider converts them to
Hartree, Hartree/Bohr, and Hartree before returning data to RTIP workflows.

## Expected Difference Areas

- JAX and OpenBLAS/LAPACK may choose different eigenvector signs for RTIP
  eigensolver outputs.
- Degenerate or near-degenerate eigenvalues in RTIP may change selected vectors.
- JAX x64 should reduce, but not eliminate, differences from Rust `f64`.
- IDWM and RTIP force formulas divide by distances. Zero-distance cases need
  explicit tests and documented behavior.
- Random rotations and initial perturbations will not reproduce Rust
  `ndarray-rand` streams. The JAX version will instead guarantee explicit-key
  reproducibility.
- XYZ/PDB formatting is intended to preserve units and field intent. Exact
  byte-for-byte output compatibility is not a Stage 3 goal unless later
  fixtures prove it is necessary.
- IDWM `Idwm0PES` wrapper returns Python floats for workflow convenience; core
  IDWM functions return JAX arrays.
- RTIP `Rtip0PES` wrapper returns Python floats for workflow convenience; core
  RTIP functions return JAX arrays.
- Optimization functions use Python control flow and scalar conversion; they
  are not intended for `jit` in the initial migration.
- Stage 5 workflow runners use Python control flow, file IO, and injected PES
  providers; they are not intended for `jit`.
- Mock `HarmonicPES` is Python-only and exists for testing/smoke runs, not as a
  Rust behavior match.
- DeePMD model energy references may differ from CP2K reference energies. This
  is a model/scientific validation issue rather than a unit-conversion issue.

## Tolerance Policy Draft

- Pure algebraic helpers: strict or near machine precision.
- RTIP/IDWM energy and force: start with `rtol=1e-10`, `atol=1e-12` for simple
  fixtures, then revise after Rust snapshot comparisons.
- Workflow trajectories: compare invariants and logged scalar sequences within
  documented tolerances, not bitwise equality.
