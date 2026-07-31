# Microkinetic replication

The paper (Section 3.5 / Figure 4) simulates the 28-step formose network
with transition-state-theory rate constants
(`k = kT/h * exp(-E/rt)`) and the program at
github.com/MillenniumDream/Microkinetics.

`simulate.py` implements the identical ODE model in Python:

* the 28 Table S1 reactions (barriers already elevated to >= 5 kcal/mol),
  read from `formose/data/table_S1.csv`;
* the Table S1 initial concentrations: c(CH2O)=0.35 M, c(HOCH2CHO)=0.05 M,
  c(OH-)=0.06 M, c(H2O)=55.5 M;
* T = 338.15 K (65 C, paper value; the upstream example input used
  328.15 K - see RECORD.md).

The paper integrates the ODE with explicit Euler at dt=5e-12 s (~2e11
steps).  Here the same ODE is integrated with scipy's adaptive LSODA
solver; Euler's solution converges to this ODE solution, so results are
equivalent within solver tolerance.  The integration-method deviation is
recorded in `formose/RECORD.md`.

Run:

```bash
cd /home/lhshen/RTIP
source /group/software/deepmd-kit-3.1.1/bin/activate   # provides scipy
python formose/microkinetics/simulate.py \
  --table formose/data/table_S1.csv \
  --output-dir formose/microkinetics/runs
```

Outputs: `concentrations.csv` (time + 28 species), `summary.json` with the
paper Figure 4 checks: formyl-anion level (~1e-14 M), formaldehyde
dimerization rate (~1e-9 mol L-1 s-1), the time at which the aldotetrose
retroaldol net rate turns positive (paper: ~0.76 s), and the final
ribose vs linear-tetrose concentrations (ribose should be the minor
product).
