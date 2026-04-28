# Stage 5 Workflow and Entry Migration

## Scope

Stage 5 wires the migrated core modules into ordinary Python workflow runners
and CLI entry points. CP2K is still not ported; workflows accept a generic
`PES` provider, and tests/CLI smoke paths use a deterministic `HarmonicPES`.

## Rust to JAX Mapping

- `src/io/input.rs::Para` loading/use -> `rtip_jax.config.load_para`,
  `Para.from_mapping`, `format_default_para`.
- `src/io/output.rs::output_rtip` -> `rtip_jax.io.outputs.output_rtip`.
- `src/io/output.rs::output_cp2k` -> `rtip_jax.io.outputs.output_cp2k`.
- `src/pes_exploration/pathway_sampling.rs::RepulsivePot::rtip_path_sampling`
  -> `run_rtip_repulsive_path_sampling`.
- `src/pes_exploration/pathway_sampling.rs::RepulsivePot::idwm_path_sampling`
  -> `run_idwm_repulsive_path_sampling`.
- `src/pes_exploration/pathway_sampling.rs::AttractivePot::rtip_path_sampling`
  -> `run_rtip_attractive_path_sampling`.
- `src/pes_exploration/pathway_sampling.rs::SynthesisPot::rtip_path_sampling`
  -> `run_rtip_synthesis_path_sampling`.
- `src/pes_exploration/md.rs::RepulsivePot::rtip_nvt_md`
  -> `run_rtip_nvt_md`.
- `src/lib.rs`, `src/lib_2mol.rs`, `src/lib_3mol.rs`
  -> `rtip_jax.cli` subcommands.

## Inputs and Outputs

- Pathway runners take a bias config (`RepulsivePot`, `AttractivePot`, or
  `SynthesisPot`) and a generic real `PES` provider.
- Repulsive pathway and MD runners require an explicit JAX PRNG key for the
  Rust-like initial perturbation.
- Runners return `PathwayResult` or `MDResult` with final state and scalar
  history.
- When `write_outputs=True`, runners write the legacy-style `rtip.pdb` and
  `rtip.out` files through Python IO helpers.

## State and Side Effects

- Core numerical state remains immutable: each coordinate update returns a new
  `System`.
- File output is isolated behind `write_outputs`; tests can disable it.
- MPI root-rank behavior is not reproduced in core runners. Distributed
  orchestration should wrap these functions later if needed.
- CP2K lifecycle is represented only by `Cp2kBoundary` and remains outside the
  JAX core.

## Numerical Notes

- Workflow loops use Python control flow and are not intended for `jit`.
- Bias-disabled steps use `ZeroPES` rather than evaluating RTIP/IDWM formulas
  with zero amplitudes. This avoids avoidable singular arithmetic while
  preserving the intended zero-bias contribution.
- Random perturbations use explicit JAX keys, so trajectories are reproducible
  within the JAX implementation but not bitwise-equivalent to Rust
  `ndarray-rand`.
- Full trajectory equivalence still requires Rust-generated scalar snapshots.

## JIT / VMAP / Scan

- Full pathway and MD runners are ordinary Python orchestration and are not
  `jit` targets.
- Individual core functions remain better candidates for `jit`, `vmap`, or
  `scan` in Stage 7 after behavior validation.

## Tests Added

- `tests/test_stage5_config_outputs_cli.py`
- `tests/test_pathway_workflows.py`
- `tests/test_md_workflow.py`

The tests use mock PES providers and dependency execution is deferred to the
remote validation environment.

## Open Risks

- No external real PES provider has replaced CP2K yet.
- Rust vs JAX workflow scalar logs have not been compared against fixtures.
- MPI directory/rank fan-out is not implemented.
- CLI supports mock/demo workflows and layout generation; production workflow
  execution will need the future external PES provider.
