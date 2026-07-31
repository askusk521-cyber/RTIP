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
* 2026-08-01: seed-0 run (20 A box, zero initial velocity, a0=0.0005) lost
  stability at step ~1440: temperature runaway to 1e6 K and an atom escaped
  to ~-84 A.  Diagnosis (rtip.out): the attractive RTIP amplitude had grown
  to ~-0.4 Ha without an early C-C event; sigma (= rti_dist) collapsed to
  ~0.5 Bohr, producing runaway forces.  Inputs were retuned without touching
  the algorithm: denser 14 A cell, Maxwell-Boltzmann initial velocities at
  1500 K (upstream `initial_velocity` support), a0 kept at the Rust default
  0.0005 for the first stability test.  The dense box + hot start makes
  molecular collisions (and hence bond events) occur earlier, before the
  amplitude grows too deep.
* 2026-08-01: a0=0.0005 still produced runaway heating (T ~ 3800 K at step
  990) in the 14 A cell before any monitored bond event (the Rust default
  was tuned for the CP2K/B97-3c engine).  With the DeePMD engine the
  production `a0` is set to 0.0001 (input parameter; the EvolutionPot
  algorithm itself is unchanged): temperature now stays at 1500-1700 K and
  the system condenses monotonically (E -11.8 -> -12.2 Ha over 900 steps).
* 2026-08-01: FORMULA AUDIT (user request) - the RTIP potential/force
  formulas are identical across paper (SI Eq. 1-6), Rust `rtip.rs`, and the
  JAX port: weight f(d)=1/d^7, weighted Gaussian sum, sigma = rti_dist
  (dynamic), and the force term `pot/sigma^2 + dw * sum_j w_j (u_j - u_i) /
  (w^2 d)` are implemented literally in both languages.  DeePMD unit
  conversions verified: Bohr<->Angstrom, eV<->Hartree, eV/Angstrom->
  Hartree/Bohr are correct at the boundary.
* 2026-08-01: SIZE-SCALING-LITE re-enabled (user request): `Para.size_scaling`
  multiplies the effective amplitude by the biased-atom count, cancelling the
  ~1/N per-atom force dilution (RTI distance ~ sqrt(N)).  Tuning scan
  (8 combos x 600 steps, DPA-3.2-5M) showed the stable regime is a dense
  10 A cell with all 66 atoms biased, `size_scaling=True`, `a0=0.00001`
  (effective 0.00066, the Rust default scale) and a cold start (zero initial
  velocity = Rust default; the earlier --init-temp 1500 hot start caused
  runaway heating before the first bond event).  In this regime the
  Increasing/Decreasing machine cycles continuously (149 cycles in 2000
  steps, mostly H-H / H2 chemistry), T stays 1300-2600 K, no atom escape,
  and the minimum C-C distance decreases 6.05 -> 4.4 A over 2000 steps.
* 2026-08-01: full 10000-step production runs submitted (slurm array seeds
  0-2, ~50 min each, serial on the single GPU).  Open question: whether C-C
  bond events (formose condensation) appear within 10000 steps; the box
  evolution MD clusters molecules but individual C-C contacts at <= 1.52 A
  had not yet occurred in the 2000-step test.
* 2026-08-01: ANALYSIS-PARSER BUG FOUND AND FIXED: the PDB readers in
  analyze_trajectory.py / analyze_run.py used regex decimals [1:4] as x,y,z,
  but atom serials (integers) are not captured, so coordinates were shifted
  by one column.  All earlier "no C-C event" / "min C-C ~4-7 A" conclusions
  were computed on corrupted coordinates and are RETRACTED.  With the fix:
  - r2pair (formyl anion + CH2O at 5.5 A, evolution MD, 2000 steps): C-C
    bond forms and persists at 1.42 A by step 2000 (acceptance item 1:
    formaldehyde dimerization via formyl-anion umpolung).
  - box seed0 (10 A cell, 4951 steps before cancel): C-C formed events at
    1.48 A / 1.25 A (steps 2760/2790), i.e. condensation happens but during
    an amplitude-driven temperature spike (see below).
* 2026-08-01: the remaining issue is NOT whether C-C bonds form (they do)
  but the dynamics regime: the RTIP crush (rti_dist -> 0.02-0.4 Bohr)
  produces brief temperature spikes (10k-1e6 K) before the bond resets.  A
  10000-step r2pair run is in progress to characterize the post-bond
  dynamics and check whether a glycolaldehyde-type product persists.
* 2026-08-01: ROOT-CAUSE FIX for the instability: the upstream Rust code
  sets the Gaussian width sigma = rti_dist *every step* (dynamic), so as
  molecules approach the centroid-coincident target sigma collapses
  (0.4 -> 0.02 Bohr) and the Gaussian force explodes.  The paper's Eq. 6
  (sigma = d_des) is a FIXED width: the distance to the destination at the
  start of the search.  Implemented `Para.fixed_sigma` (default False =
  Rust behavior preserved; True = paper semantics, sigma captured once from
  the initial distance to the destination).  With `fixed_sigma=True` the
  10 A box run is stable: mean temperature 1488 K (max 1832 K) over 2000
  steps, 201 RTIP cycles, monotonic condensation (E -11.6 -> -12.15 Ha,
  rti_dist 3.56 -> 2.88 Bohr), no spikes, no escapes.  Full 10000-step
  production runs launched with this configuration.
