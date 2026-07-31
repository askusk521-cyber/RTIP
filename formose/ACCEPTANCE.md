# Formose replication acceptance summary

Branch `formose/replicate`, engine: DeePMD-kit 3.1.1 + DPA-3.2-5M
(replacing the paper's CP2K/B97-3c), algorithm: upstream Rust
`EvolutionPot::rtip_nvt_md` ported 1:1 to Python/JAX.

## Verified against the paper (JACS Au 2026, 6, 922)

### 1. RTIP-MD mechanism: formaldehyde dimerization via formyl anion (R2)

**VERIFIED.** Evolution MD on the reactive pair formyl anion (HCO) + CH2O
(5.5 A initial separation, `formose/data/r2pair.xyz`):

* C-C bond forms and persists: min C-C = 1.42 A at step 2000
  (`formose/runs/r2pair/`, report: `formose/analyze_run.py`);
* product connectivity O=C(H)-C(=O) (C-C 1.42 A, C=O 1.27/1.19 A) is the
  umpolung condensation product (glycolaldehyde-alkoxide-like core);
* RTIP amplitude machine cycled 31 times (increasing/decreasing) driven by
  bond-variation detection, exactly the paper's Section 3.1 control scheme.

Box runs (10 A cell, all atoms biased) additionally produced C-C formation
events at 1.48 A / 1.25 A (seed 0, steps 2760/2790), confirming that
condensation occurs in the full cell as well.

### 2. Microkinetic simulation (paper Section 3.5 / Figure 4)

**VERIFIED QUANTITATIVELY.** `formose/microkinetics/simulate.py` solves the
same 28-step Table S1 ODE (mass-action + kT/h prefactor, 65 C, paper initial
concentrations) with scipy LSODA (the paper's explicit-Euler dt=5e-12
solution converges to this ODE):

| quantity | simulation | paper (Figure 4) |
|---|---|---|
| formyl anion c(2) | 3.62e-14 M | 3.6e-14 M |
| CH2O dimerization rate | 1.90e-9 mol L-1 s-1 | 1.9e-9 |
| retroaldol net rate > 0 at | 0.763 s | ~0.76 s |
| ribose vs linear tetroses | 3.8e-15 vs 0.027 M | ribose minor |

### 3. Code / algorithm fidelity

* RTIP potential/force formulas identical across paper, Rust and JAX
  (weight 1/d^7, weighted Gaussian, sigma = distance to destination;
  force term `pot/sigma^2 + dw*sum_j w_j(u_j-u_i)/(w^2*d)`).
* Gaussian-width semantics: the Rust code updates sigma = rti_dist every
  step (dynamic); the paper's Eq. 6 fixes sigma = d_des.  The production
  config enables `Para.fixed_sigma` (paper semantics; Rust behavior remains
  the default).  This was the root cause of the earlier temperature
  explosions and is recorded in RECORD.md.
* DeePMD unit conversions verified (Bohr/Angstrom, eV/Hartree,
  eV/Angstrom -> Hartree/Bohr).
* `pytest`: 90 passed.  SI data: 55 structures + Table S1 parsed.

## Documented limitations (recorded in RECORD.md)

1. DeePMD + upstream EvolutionPot is not long-term stable for this system:
   the RTIP crush (rti_dist -> 0.02-0.4 Bohr, sigma collapses) drives brief
   temperature spikes (10k-1e6 K) and eventually NaN beyond ~2000-4000
   steps.  Mechanism-level C-C formation is demonstrated in the stable
   window; paper-scale 5 ps box trajectories were not reproduced.
2. The full 28-step network via RTIP-MD is not reproduced; only the key C-C
   step (R2) and H-H/C-H chemistry were demonstrated.  DFT/TS re-validation
   is out of scope by decision (Q2).
3. A PDB-analysis off-by-one bug (coordinates shifted by one column) was
   found and fixed on 2026-08-01; conclusions based on the fixed parser.

## How to reproduce

```bash
cd /home/lhshen/RTIP
# mechanism (r2pair):
sbatch formose/run_formose.slurm   # BOX=formose/data/r2pair.xyz, 2000 steps
PYTHONPATH=rtipmd/jax/src JAX/.venv/bin/python formose/analyze_run.py \
  formose/runs/r2pair --label r2pair
# microkinetics:
source /group/software/deepmd-kit-3.1.1/bin/activate
python formose/microkinetics/simulate.py \
  --table formose/data/table_S1.csv --output-dir formose/microkinetics/runs
```
