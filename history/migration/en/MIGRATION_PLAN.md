# Migration Plan

This document tracks the staged Rust to Python + JAX rewrite.

## Scope Rules

- Migrate incrementally, one small module or stage at a time.
- Keep Rust source unchanged unless a later stage explicitly needs comparison
  tests or fixtures.
- Prefer behavior equivalence and testability over line-by-line translation.
- Enable JAX x64 by default to match Rust `f64` behavior as closely as possible.
- Use explicit JAX PRNG keys for all new random behavior.
- Keep orchestration, IO, CLI, and logging as ordinary Python unless there is a
  clear numerical reason to use JAX primitives.

## Stage Status

### Stage 1: Migration Audit

Status: complete.

Findings:

- Top-level Rust entry points are static-library `extern fn main` functions in
  `src/lib.rs`, `src/lib_2mol.rs`, and `src/lib_3mol.rs`.
- Core numerical modules are `rtip.rs`, `idwm.rs`, and `optimization.rs`.
- Workflow modules are `pathway_sampling.rs`, `md.rs`, and `synthesis.rs`.
- CP2K is currently the real PES provider, but it will not be ported.
- No Rust tests, fixtures, or CP2K input examples were found.

### Stage 2: JAX Project Skeleton

Status: complete.

Deliverables:

- Python package under `src/rtip_jax`.
- `pyproject.toml` with JAX and pytest dependencies.
- Import-time JAX x64 configuration.
- Placeholder CLI.
- CP2K boundary stub documenting the old contract.
- Initial tests for package configuration and CP2K boundary behavior.
- Migration documentation skeleton.

### Stage 3: Basic Public Modules

Status: complete.

- Constants and unit conversions.
- `Para` configuration dataclass.
- `System` dataclass and XYZ/PDB IO.
- Small math helpers and explicit PRNG patterns.

Local dependency-dependent checks are intentionally deferred because JAX and
pytest are available on the remote validation server, not in the current local
environment. Stage work still adds tests and records the commands to run later.

### Stage 4: Core Algorithm Modules

Status: complete.

Planned order:

1. IDWM metric and potential.
2. RTIP distance, transform, potential, and force.
3. One-dimensional optimization.
4. Bias PES composition.

#### Stage 4.1: IDWM

Status: implemented.

- Rust source: `src/pes_exploration/idwm.rs`.
- Python target: `src/rtip_jax/core/idwm.py`.
- Module notes: `MIGRATION_IDWM.md`.
- Dependency-based tests are authored but deferred to the remote validation
  environment.

#### Stage 4.2: RTIP

Status: implemented.

- Rust source: `src/pes_exploration/rtip.rs`.
- Python target: `src/rtip_jax/core/rtip.py`.
- Module notes: `MIGRATION_RTIP.md`.
- Dependency-based tests are authored but deferred to the remote validation
  environment.

#### Stage 4.3: Optimization

Status: implemented.

- Rust source: `src/pes_exploration/optimization.rs`.
- Python target: `src/rtip_jax/core/optimization.py`.
- Module notes: `MIGRATION_OPTIMIZATION.md`.
- Dependency-based tests are authored but deferred to the remote validation
  environment.

#### Stage 4.4: Bias PES Containers

Status: implemented.

- Rust source: `src/pes_exploration/potential.rs`.
- Python targets: `src/rtip_jax/pes/bias.py`, `src/rtip_jax/pes/base.py`.
- Module notes: `MIGRATION_BIAS_PES.md`.
- Dependency-based tests are authored but deferred to the remote validation
  environment.

#### Stage 4.5: Synthesis Layout

Status: implemented.

- Rust source: `src/pes_exploration/synthesis.rs`.
- Python target: `src/rtip_jax/workflows/synthesis.py`.
- Module notes: `MIGRATION_SYNTHESIS.md`.
- Dependency-based tests are authored but deferred to the remote validation
  environment.

#### Stage 4.6: Pathway Sampling Helpers

Status: implemented.

- Rust source: `src/pes_exploration/pathway_sampling.rs`.
- Python target: `src/rtip_jax/workflows/pathway_sampling.py`.
- Module notes: `MIGRATION_PATHWAY_HELPERS.md`.
- Only pure helper logic is migrated here; full sampling loops remain Stage 5.

#### Stage 4.7: MD Helpers

Status: implemented.

- Rust source: `src/pes_exploration/md.rs`.
- Python target: `src/rtip_jax/workflows/md.py`.
- Module notes: `MIGRATION_MD_HELPERS.md`.
- Only pure numerical helper logic is migrated here; full MD loops remain Stage
  5.

### Stage 5: Workflow and Entry Layers

Status: implemented.

- JSON/TOML parameter loading and default-config CLI output.
- Legacy output-path helpers without MPI dependency.
- Generic PES-driven RTIP/IDWM pathway runners.
- Generic PES-driven RTIP NVT MD runner.
- Synthesis target-state helper for runtime pathway loops.
- CLI subcommands for CP2K boundary inspection, synthesis layout generation,
  and mock pathway/MD smoke runs.
- Module notes: `MIGRATION_STAGE5_WORKFLOWS.md`.
Dependency-based tests are authored but deferred to the remote validation
environment.

### Stage 6: Full Validation

Status: engineering validation complete as of 2026-05-24.

- Unit and integration tests.
- Numerical-difference records.
- DeePMD provider integration as the production replacement for CP2K.

Remote validation:

- `n5:/home/lhshen/RTIP/JAX`
- `pytest`: 87 passed.

Remaining scientific validation:

- Rust vs JAX behavior snapshots where Rust fixtures are available.
- DeePMD model benchmark systems or labeled comparison structures.

#### Stage 6.1: DeePMD Provider

Status: implemented.

- CP2K remains documentation-only.
- `DeepMDPES` implements the generic `PES` contract for real energy/force
  data.
- Coordinates/cell are converted from Bohr to Angstrom before DeePMD inference.
- Energy/force/virial are converted from eV/eV-Angstrom units to RTIP internal
  Hartree/Hartree-Bohr units.
- Module notes: `DEEPMD_INTO.md`.

### Stage 7: JAX Native Optimization

Planned only after behavior is stable:

- Evaluate `jit`, `vmap`, `lax.scan`, and `lax.fori_loop`.
- Optimize only measured bottlenecks.

## Archive Note

The migration documentation was archived after the remote engineering test pass.
Stage 7 JAX-native optimization and additional scientific benchmark validation
remain future work, but they are not blockers for the completed migration
archive.
