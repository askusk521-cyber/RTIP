# MD Helper Migration Notes

Rust source: shared numerical logic from `src/pes_exploration/md.rs`.

Python target: `src/rtip_jax/workflows/md.py`.

## Scope

This stage migrates only pure MD helper logic:

- Atom mass extraction.
- Twice kinetic energy `sum_i m_i |v_i|^2`.
- Kinetic energy `0.5 * sum_i m_i |v_i|^2`.
- Temperature formula.
- Berendsen thermostat scaling factor.
- Acceleration from force/mass.
- First and second leapfrog half-steps.

It does not migrate the full RTIP NVT MD loop, PES construction, file output, or
MPI behavior.

## Rust Semantics Preserved

- Velocity and acceleration are in atomic units.
- Time step `dt` is in femtoseconds and converted by `FEMTOSECOND_TO_AU`.
- Rust temperature uses `sum_i m_i |v_i|^2`, not the half kinetic energy.
- Rust applies a temperature floor of `1.0 K` before Berendsen scaling.
- The second leapfrog half-step applies the thermostat scaling after updating
  velocity.

## Key Differences From Rust

- Python helpers return arrays/values and do not mutate `System`.
- `temperature` rejects single-atom systems because Rust divides by
  `3 * (natom - 1)`, which is undefined for one atom.
- Full MD loop side effects are intentionally absent.

## Tests

Tests are in `tests/test_md_helpers.py` and cover:

- Atom mass extraction.
- Kinetic energy and temperature formulas.
- Single-atom temperature rejection.
- Berendsen temperature floor.
- Acceleration calculation.
- Leapfrog first and second half-step formulas.

