# Test Strategy

## Stage 2 Tests

- Package import configures JAX x64.
- CP2K boundary records original input/output/unit contract.
- CP2K provider methods fail explicitly because CP2K is not ported.

## Local vs Remote Execution

The current local environment does not contain JAX or pytest. Dependency-based
checks are therefore recorded but deferred to the remote validation server and
to the final unified test pass. Local stage validation is limited to syntax and
metadata checks that do not import third-party packages.

## Stage 3 Tests

- Constants and unit conversions match Rust values.
- Element parsing and supported masses match Rust behavior.
- `Para` defaults match Rust `Para::new()`.
- XYZ read/write converts Angstrom to Bohr and back.
- PDB/XYZ formatting is stable enough for snapshot tests.
- Random rotation helper returns orthogonal matrices and is reproducible with a
  fixed JAX key.

## Stage 4 Tests

- IDWM weighted matrix and distance on small hand-computed systems.
- IDWM force direction checked by finite differences.
- IDWM fragment force scatter for `atom_add_pot`.
- RTIP distance invariance under translation and rotation.
- RTIP transform reconstructs aligned structures within tolerance.
- RTIP force direction checked by finite differences.
- RTIP fragment force scatter for `atom_add_pot`.
- Optimization tested against analytic one-dimensional functions and mock PES.
- Optimization zero-force and force-shape guards.
- Bias PES dataclass normalization and validation.
- Simple PES summation helper with mock PES.
- Synthesis 2/3/4 molecule layouts with deterministic rotations.
- Synthesis explicit-key reproducibility.
- Pathway helper perturbation reproducibility and stopping conditions.
- MD helper kinetic energy, temperature, thermostat, acceleration, and
  leapfrog formulas.

## Stage 5 Tests

- JSON parameter loading and unknown-key validation.
- Output helper paths for RTIP and CP2K legacy filenames.
- CLI smoke tests for default config and CP2K boundary inspection.
- Pathway sampling runners with deterministic mock PES and disabled
  line-search motion where appropriate.
- Synthesis target-state construction with environment atoms.
- RTIP NVT MD runner with deterministic mock PES and explicit PRNG key.

## Required Baselines

- Small XYZ fixtures for one, two, and three molecule cases.
- Rust-generated scalar snapshots for IDWM and RTIP once fixtures are agreed.
- Mock PES fixtures independent of CP2K.
