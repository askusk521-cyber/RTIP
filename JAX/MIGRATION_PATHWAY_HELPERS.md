# Pathway Sampling Helper Migration Notes

Rust source: shared helper logic from `src/pes_exploration/pathway_sampling.rs`.

Python target: `src/rtip_jax/workflows/pathway_sampling.py`.

## Scope

This stage migrates only pure, reusable helper logic:

- Mean-centered random perturbation of a full system or atom subset.
- Force norm helpers.
- Repulsive/IDWM stopping condition state update.
- Attractive stopping condition state update.
- Synthesis stopping condition state update.

It does not migrate the full sampling loop, PES construction inside each loop,
PDB/OUT writing, MPI behavior, or CP2K-backed real PES calls.

## Rust Semantics Preserved

- Initial perturbation is sampled from `[0, 1)`, mean-centered, normalized, and
  scaled by `0.1` by default.
- Repulsive and IDWM stop logic tracks `pot_real_max`, `pot_real_min`, and
  `add_bias`.
- Attractive stop logic disables bias when `sigma_min < 1.0` or `f_bias >
  1000.0`.
- Synthesis stop logic additionally records `pot_real_initial` on step 1.

## Key Differences From Rust

- Python helpers return new immutable state dataclasses.
- Random perturbation uses explicit JAX PRNG keys instead of `ndarray-rand` and
  MPI broadcast.
- Full workflow side effects are intentionally absent.

## Tests

Tests are in `tests/test_pathway_sampling_helpers.py` and cover:

- Perturbation reproducibility and norm scaling.
- Full-system and subset perturbation.
- Invalid atom-subset inputs.
- Repulsive/IDWM stop conditions.
- Attractive stop conditions.
- Synthesis initial-energy bookkeeping.

