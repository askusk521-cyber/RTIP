# Assumptions

## Confirmed

- CP2K will not be ported in this migration.
- CP2K input/output semantics must remain documented because a later provider
  will replace CP2K as the source of real PES energies and forces.
- Core numerical code should prefer `jax.numpy` and JAX primitives.
- JAX x64 should be enabled by default.
- Random behavior must use explicit JAX PRNG keys.
- Workflow and IO code may remain normal Python.

## Working Assumptions

- Initial Python package name is `rtip_jax`; distribution name is `rtip-jax`.
- The first useful core PoC should be IDWM because it is pure numerical code and
  does not depend on eigensolver sign conventions.
- MPI is not required for early stages. If needed later, it should be isolated
  in workflow/orchestration code rather than core numerical functions.
- CP2K-compatible units remain the internal baseline unless a later stage
  explicitly changes the unit model.
- Local dependency-dependent checks are deferred until the remote validation
  environment is used. Tests are still authored during each stage.
- IDWM zero-distance behavior should remain Rust-compatible until a later
  validation stage explicitly chooses a stabilized convention.
- RTIP eigensolver outputs should be tested by invariant behavior, not by raw
  eigenvector signs, unless Rust snapshots prove a stricter comparison is safe.
- A zero total-force direction is treated as invalid in Python optimization
  code, because Rust's behavior is not meaningfully defined there.
- `RepulsivePot`, `AttractivePot`, and `SynthesisPot` remain workflow
  configuration containers. Full workflow behavior belongs to later
  `pathway_sampling`, `md`, and `synthesis` migrations.
- Synthesis MPI broadcast behavior is not part of the pure layout helper. If
  distributed orchestration is needed later, it belongs in Stage 5 workflow
  code.
- Stage 4 pathway sampling migration is limited to pure helpers. File output,
  logging cadence, PES construction inside loops, and MPI orchestration remain
  Stage 5 responsibilities.
- Stage 4 MD migration is limited to pure helpers. Full RTIP NVT MD loops and
  logging remain Stage 5 responsibilities.
- Stage 5 workflow runners use generic Python PES providers and normal Python
  control flow. They are intentionally not forced into `jit`.
- Bias-disabled pathway steps use an explicit zero PES to avoid evaluating
  singular RTIP/IDWM formulas with zero amplitudes.
- Stage 5 CLI production runs remain limited until the replacement external PES
  provider is selected. Mock workflows are provided for smoke tests and usage
  checks.
- Output directory creation uses practical Python `mkdir(..., exist_ok=True)`
  behavior by default, while Rust `create_dir` would error if the directory
  already existed.

## Open Questions

- Which external PES provider will replace CP2K?
- Should production CLI workflows preserve the three separate 2-molecule and
  3-molecule entry variants, or stay with one configurable CLI?
- What Rust fixtures or scientific examples should become numerical baselines?
