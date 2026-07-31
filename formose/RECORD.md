# Formose replication record

## Decisions (2026-08-01)

* Engine: DeePMD-kit 3.1.1 + DPA-3.2-5M (RTX 5090 node5); CP2K avoided by
  decision (its source exists at `/home/lhshen/cp2k` but is not compiled).
* Branch `formose/replicate` created from `size-scaling-lite`
  (upstream fork HEAD 72c9746).  History is not rewritten.
* Removed from the branch: `research/ic5c02384/`, `rtipmd/jax/examples/`,
  reaction-coordinate (RCMD) machinery, `size_scaling`, sinusoidal
  oscillation, temperature feedback, early stopping, multi-round synthesis
  MD, and the `rust_compat` switch (the code now always follows the Rust
  algorithm).
* MD implementation now ports the Rust `md.rs` directly: `repulsive_md`
  (`RepulsivePot`) and `evolution_md` (`EvolutionPot`, bond-variation
  controlled Increasing/Decreasing/Falling amplitude machine).
* `Para` defaults now match Rust `Para::new()` (`io/input.rs`), including
  `a0=0.0005`, `tau=10`, `temp_bath=1500`, `decreasing_multiple=2`,
  `decreasing_bound=0.5`, `split_step=100`, and the paper's ignored bond
  pairs (C-O, H-O, and all Ca pairs).

## Known discrepancies

* Microkinetic temperature: the repository's example `input` uses
  `T = 328.15 K` (55 C); the paper/Table S1 state 65 C (338.15 K).  This
  branch documents the paper value (338.15 K) as the production setting for
  microkinetic re-runs and records the code example as-is.
* The paper's real PES was B97-3c DFT (CP2K); DeePMD replaces it here, so
  energies/barriers are not directly comparable.  Acceptance is limited to
  mechanism-level events.

## Progress log

* 2026-08-01: SI downloaded from Europe PMC (PMC12933356); 55 structures +
  Table S1 parsed into `formose/data/`.
* 2026-08-01: JAX package refactored to upstream-only algorithms;
  `pytest`: 90 passed.
* 2026-08-01: initial cells built (seeds 0-2); first DeePMD evolution-MD
  smoke run submitted (slurm job, 200 steps) and verified: Berendsen
  heating, linear RTIP growth, DeePMD energies/forces OK.
* 2026-08-01: microkinetic replication (`simulate.py`, same ODE model as the
  paper's Microkinetics program, adaptive LSODA instead of explicit Euler;
  the Euler dt=5e-12 solution converges to this ODE solution).  Results match
  paper Figure 4 quantitatively: formyl anion 3.62e-14 M (paper 3.6e-14),
  formaldehyde dimerization rate 1.90e-9 (paper 1.9e-9 mol L-1 s-1),
  aldotetrose retroaldol net rate turns positive at 0.763 s (paper 0.76 s),
  ribose negligible vs linear tetroses (0.027 M) -> low-ribose-yield
  rationale reproduced.
* 2026-08-01: full 10000-step DeePMD evolution-MD runs submitted as slurm
  array seeds 0-2; bond-variation cycles (Increasing/Decreasing) already
  observed in seed 0 by step ~3900 (9 cycles).
