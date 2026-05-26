# Week 1 Code-Reading Conclusions

## 1. Project Goal

Use RTIP/JAX plus DeePMD (DPA-3.2-5M) to simulate the `1-tBu + CO2` coupling
reaction and validate the reaction path reported in the ACS `ic5c02384` paper.

## 2. Code Architecture

```
rtip_jax/
├── constants.py     — physical constants + periodic table + atomic masses
├── _config.py       — JAX initialization (enable x64)
├── config.py        — Para parameter bundle (steps, temperature, force strength, etc.)
├── system.py        — System data structure (coordinates / elements / potential energy)
├── core/
│   └── rtip.py      — core algorithm: RTI distance + RTIP Gaussian bias potential
├── pes/
│   ├── base.py      — PES interface (PES / SumPES / ZeroPES)
│   └── bias.py      — four bias containers + ReactionCoordinatePES
├── workflows/
│   ├── pathway_sampling.py  — Pathway (no temperature, force + line search)
│   ├── md.py                — MD (with temperature, leapfrog + thermostat)
│   └── synthesis.py         — fragment placement (synthesize / synthesis_target_state)
├── external/
│   └── deepmd.py   — DeePMD boundary (Angstrom/eV ↔ Bohr/Hartree unit conversion)
└── cli.py           — command-line entry point
```

Important modules:

- `constants.py`: physical constants, periodic table, and atomic masses.
- `_config.py`: JAX initialization, including x64.
- `config.py`: the `Para` parameter bundle.
- `system.py`: coordinates, element types, energy fields, and system helpers.
- `core/rtip.py`: RTI distance and RTIP Gaussian bias.
- `pes/base.py`, `pes/bias.py`: PES interfaces and bias containers.
- `workflows/pathway_sampling.py`: zero-temperature pathway workflows.
- `workflows/md.py`: finite-temperature RTIP NVT MD.
- `workflows/synthesis.py`: fragment placement and synthesis target state.
- `external/deepmd.py`: DeePMD boundary and unit conversions.
- `cli.py`: command-line entry points.

## 3. Core Concepts

### 3.1 RTI Distance (`rti_dist`)

Rotation-translation-invariant distance. A quaternion method finds the optimal
rotation alignment between two molecular structures; the residual is the RTI
distance. `rti_dist = 0` means the two structures are isomorphs; larger values
mean larger differences.

### 3.2 RTIP Bias Potential (`Rtip0PES`)

Places a Gaussian around reference structures:

- **a > 0**: repulsive (Gaussian "hill", pushes away from reference), suitable
  for dissociation reactions.
- **a < 0**: attractive (Gaussian "valley", pulls toward reference), suitable
  for association reactions.
- `|a|` grows with step number (`a = a0 * step`), strengthening progressively.

The Gaussian width sigma is the current RTI distance to the reference.

### 3.3 Four Bias Types

| Type | Sign of a | Reference | Use case | Rust origin |
|------|-----------|-----------|----------|-------------|
| RepulsivePot | + (repulsive) | local_min | dissociation AB→A+B | Rust pathway + MD |
| AttractivePot | - (attractive) | final_state (product) | association A+B→AB | Rust pathway yes, MD no |
| SynthesisPot | - (attractive) | fragment-shared centroid | multi-fragment synthesis | Rust pathway yes, MD no |
| ReactionCoordinatePot | harmonic k | specified atom-pair distances | constraint key bond distance | Rust: none, JAX new |

### 3.4 Two Workflows

**Pathway (`pathway_sampling.py`)**: zero temperature, pure force-driven. Each
step computes `force_total = force_real + force_bias` and does a
one-dimensional line search along the combined force direction. Uses a state
machine for stopping (`pot_drop`/`pot_climb`/`f_epsilon`). `temp_K` and
`kin_Ha` showing `nan` is expected.

**MD (`md.py`)**: finite temperature, thermal motion. Each step uses leapfrog
integration to advance coordinates and velocities, with a Berendsen thermostat.
Four bias branches: repulsive / attractive / synthesis /
reaction-coordinate.

## 4. Validation Metrics

| Metric | Meaning | How to judge |
|--------|---------|--------------|
| pot_real | Real PES energy (DeePMD) | **Primary metric** — should decrease for a successful reaction |
| pot_bias | Bias potential (RTIP) | Artificial energy; do not judge by this alone |
| pot_total | pot_real + pot_bias | HISTORY repeatedly warns: do not rely on total alone |
| rti_dist / sigma_min | RTI distance | Bias width in pathway; monitoring quantity in MD |
| min(B-O) | Shortest B-O distance | Key bond-formation metric; product reference ~1.49 A |
| min(N-C) | Shortest N-C distance | Key bond-formation metric; product reference ~1.49 A |
| RMSD | Deviation from reference | Smaller = closer to target structure |
| temp_K | Temperature | Meaningful only in MD |
| f_real | Real force norm | Smaller = closer to a stationary point |

**Core lesson**: judge reaction progress by `pot_real`, key bond distances, and
RMSD to reference structures. Do not rely only on `rti_dist` or `pot_total`,
because `pot_total` includes artificial bias energy.

## 5. Rust Original Design Investigation

Compared against Rust source under
`/home/slh/4_28/RTIP/src/pes_exploration/`.

### Pathway layer (`pathway_sampling.rs`)

All three modes exist in Rust:

- `impl RtipPathSampling for RepulsivePot` — line 21
- `impl RtipPathSampling for AttractivePot` — line 629
- `impl RtipPathSampling for SynthesisPot` — line 832

The JAX migration faithfully reproduces all three.

### MD layer (`md.rs`)

Rust has **only** the repulsive mode:

- `impl RtipNVTMD for RepulsivePot` — lines 18 to 307
- AttractivePot / SynthesisPot `RtipNVTMD` **do not exist**
- ReactionCoordinatePot **completely absent from Rust**

### Conclusion

| Layer | Repulsive | Attractive | Synthesis | ReactionCoordinate |
|-------|:---:|:---:|:---:|:---:|
| Rust pathway | yes | yes | yes | no |
| Rust MD | yes | **no** | **no** | no |
| JAX pathway | yes | yes | yes | no |
| JAX MD | yes | **added in this study** | **added in this study** | **added in this study** |
| CLI entry | yes | no | no | no |

CLI only exposes repulsive — this is not an omission. The original Rust MD
layer only had repulsive. Attractive/synthesis MD and ReactionCoordinate were
extended on the JAX side during this research after discovering that repulsive
does not drive association reactions.

## 6. Key Conclusions from HISTORY Documents

1. The DPA-3.2-5M model gives the correct qualitative ordering
   (`TS > reactant > product` energy ranking) but has quantitative deviations
   (TS barrier ~3 kcal/mol too low, product exothermicity ~7.5 kcal/mol too
   shallow).

2. Stock repulsive RTIP MD failed across all 5 seeds for this association
   reaction — it is not suitable for bond-formation reactions.

3. Among the four bias explorations, ReactionCoordinate (`k=0.002`, product
   target) gave the only promising frame, but `pot_real` still rose by
   ~24 kcal/mol and the bond distance (2.5 A) remained far from the target
   (1.49 A) — not a success.

4. **Update (2026-05-24 evening): RC_K scan (k=0.001–0.010, 1000 steps)
   confirmed that the earlier failure was primarily insufficient steps.**
   `k=0.006` product-target successfully held the system in the product basin
   region for ~800 steps (400 fs), with final B-O=1.44 A, N-C=1.66 A
   (reference 1.49, 1.49 A). `pot_real` remained elevated
   (Erel ~+43 kcal/mol) and needs unbiased relaxation for confirmation. See
   `HISTORY/20260524.md` RC_K scan section for details.

5. Next steps: run unbiased relaxation on the `k=0.006` trajectory; reproduce
   with `k=0.006` for other substituent systems.
