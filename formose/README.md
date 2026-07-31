# Formose RTIP-MD replication

This directory contains the formose-reaction replication workflow for
*Conserved-Potential-Driven Molecular Dynamics Deciphers Formose Reaction
Mechanisms* (JACS Au 2026, 6, 922; DOI 10.1021/jacsau.5c01359).

The MD engine is the upstream RTIP-MD algorithm (`EvolutionPot::rtip_nvt_md`
in the Rust reference), ported 1:1 to Python/JAX in `rtipmd/jax`; the real
PES is provided by DeePMD-kit (DPA-3.2-5M), replacing the paper's CP2K
B97-3c engine.  Acceptance is mechanism-level: the trajectories must
reproduce the paper's key reaction events and bond-monitoring signatures.

## Data

* `data/species/`, `data/ts/` – structures parsed from the paper SI
  (coordinates in Angstrom; energies in Hartree).  `data/manifest.csv`
  lists all 55 structures (27 species + 28 transition states).
* `data/table_S1.csv` – the 28-step reaction network with forward/reverse
  Gibbs barriers (kcal/mol) from Table S1, used by the microkinetic
  replication (see `Microkinetics` at github.com/MillenniumDream/Microkinetics).
* `data/box_seedN.xyz` – initial cells with 8 H2O + 8 CH2O + 2 Ca + 4 OH-
  (66 atoms) built by `build_box.py`.

Source of the SI: Europe PMC supplementary files for PMC12933356
(`au5c01359_si_001.pdf`, videos `au5c01359_si_002..008.mp4`); raw files are
kept outside the repository at `/home/lhshen/si_formose/` on n5.

## How to run

```bash
# 1. Build the initial cell (already committed for seeds 0-2)
PYTHONPATH=rtipmd/jax/src JAX/.venv/bin/python formose/build_box.py \
  --molecules-dir molecules --output formose/data/box_seed0.xyz --seed 0

# 2. Run evolution MD with DeePMD on the GPU node (slurm)
cd /home/lhshen/RTIP
sbatch formose/run_formose.slurm          # full 10000 steps, seed 0
SEED=1 MAX_STEP=10000 sbatch formose/run_formose.slurm

# 3. Analyze bond events against the paper's monitoring scheme
PYTHONPATH=rtipmd/jax/src JAX/.venv/bin/python formose/analyze_trajectory.py \
  formose/runs/seed0/rtip.pdb --output formose/runs/seed0/bond_events.csv
```

Parameters are in `para_formose.json` (Rust `Para::new()` defaults:
`a0=0.0005`, `dt=0.5 fs`, `tau=10`, `temp_bath=1500 K`,
`decreasing_multiple=2`, `decreasing_bound=0.5`, `split_step=100`).

## Acceptance criteria (paper standard)

For each seed, the RTIP-MD trajectory must show, via bond monitoring of
C-C / C-H / H-H / O-O pairs only:

1. Formaldehyde dimerization through the formyl anion (1 -> 2 -> 3 -> 4),
   i.e. a new C-C bond between two CH2O-derived fragments;
2. Glycolaldehyde -> glyceraldehyde aldol growth (C3 + C1);
3. Ribose-pathway C3 + C2 -> C5 coupling (if reachable in the trajectory);
4. Breslow-cycle retroaldol cleavage of aldotetrose (26 -> 4 + 5).

The check is qualitative/mechanism-level: DFT re-validation
(TS search / omegaB97M-V / Shermo) is out of scope on this branch; the SI
structures are provided for RMSD-style reference checks of intermediate
geometries.
