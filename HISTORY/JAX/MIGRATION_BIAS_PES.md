# Bias PES Container Migration Notes

Rust source: `src/pes_exploration/potential.rs`.

Python targets:

- `src/rtip_jax/pes/bias.py`
- `src/rtip_jax/pes/base.py`

## Rust Semantics

- `RepulsivePot` stores:
  - local-minimum `System`
  - nearby transition-state `System` values
  - `Para`
  - structure-output filename
  - scalar-output filename
- `AttractivePot` stores:
  - initial `System`
  - final `System`
  - `Para`
  - output filenames
- `SynthesisPot` stores:
  - initial `System`
  - molecule atom-index groups
  - `Para`
  - output filenames

These Rust structs are workflow configuration containers. They do not implement
the `PES` trait directly; pathway sampling and MD implementations are attached
in other modules.

## Input and Output

- `RepulsivePot(local_min, nearby_ts, para, str_output_file, output_file)`.
- `AttractivePot(initial_state, final_state, para, str_output_file, output_file)`.
- `SynthesisPot(initial_state, mol_index, para, str_output_file, output_file)`.
- `nearby_ts` is normalized to a tuple.
- `mol_index` is normalized to nested tuples of integers.

## State and Side Effects

- The Python containers are frozen dataclasses.
- They perform validation/normalization only.
- No IO, CP2K, MPI, random state, or numerical algorithm is run in this module.

## JAX Support

- These are orchestration-level containers and are not intended as `jit` targets.
- Contained `System.coord` values remain JAX arrays.

## Key Differences From Rust

- Rust lifetime references to `Para` become ordinary Python dataclass fields.
- Default `Para()` and output filenames are provided for ergonomic construction.
- `SynthesisPot` validates empty/out-of-range `mol_index` values early.
- `SumPES` is new Python infrastructure. It sums multiple PES providers and is
  not a direct Rust type.

## Tests

Tests are in `tests/test_bias_pes.py` and cover:

- `SumPES` energy and force summation.
- Rejection of empty `SumPES`.
- `RepulsivePot` transition-state normalization and defaults.
- `AttractivePot` field storage.
- `SynthesisPot` molecule-index normalization and validation.

