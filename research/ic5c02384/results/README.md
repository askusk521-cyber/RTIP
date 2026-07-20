# Simulation Results Index

All simulation results for the ic5c02384 validation study are organized here.

## Directory Layout

- rc-md/          Reaction-coordinate MD parameter scans (k=0.0005-0.050, all 11 reactions)
- attractive/     Attractive RTIP bias runs (1-tBu+CO2, various a0/T/steps)
- synthesis/      Synthesis MD runs (1-tBu+CO2)
- rust-align/     Post Rust-alignment validation (v1-v3, 1-tBu+CO2)
- archive/        Superseded or early exploratory runs

## Naming Convention

Each result directory follows: {method}_{reaction}_{target}_k{force}_{T}temp_{steps}

Example: bias_1-tBu__CO2_rc-md_product_k0020_T300_2000
  - method: rc-md (reaction-coordinate MD)
  - reaction: 1-tBu__CO2
  - target: product (restraint target structure)
  - k0020: force constant 0.020 Ha/Bohr^2
  - T300: bath temperature 300 K
  - 2000: max steps

## Key Results (1-tBu + CO2)

| Run | k | Steps | Best B-O | Best N-C | Bond? | Notes |
|-----|---|-------|----------|----------|:---:|-------|
| rust-align/v3 | 0.020 | 2000 | 1.19 A | 1.64 A | Yes | Post-alignment, fixed atom mapping |
| rc-md/k0006 | 0.006 | 1000 | 1.39 A | 1.29 A | Yes | Original best (pre-alignment) |
| Paper ref | - | - | 1.49 A | 1.49 A | - | DFT omegaB97X-D/cc-pVTZ |

## Per-Run Contents

Each result directory contains:
- *.out       Scalar trajectory log (step, energy, forces, state)
- *.pdb       Structure trajectory
- *_final.xyz Final structure
- *_best_bond_forming.xyz  Frame with shortest forming bonds
- *_best_ts_like.xyz       Frame closest to TS geometry
- *_best_product_like.xyz  Frame closest to product geometry
- summary.json  Machine-readable metrics
- summary.md    Human-readable summary table
