# AGENT.md — RTIP Project Context

## Project Goal

RTIP (Roto-Translationally Invariant Potential) is a computational chemistry tool for exploring molecular reaction pathways using biased potential energy surfaces. It places Gaussian bias potentials in roto-translationally invariant distance (RTI) space to drive molecular systems from reactants over transition states to products.

Current research target: validating the ACS paper `ic5c02384` — methyleneborane-N2 complex coupling with CO2 and other small molecules.

## Repository Layout

```
rtipmd/rust/          — Original Rust implementation (reference only, frozen)
rtipmd/jax/           — Python + JAX rewrite (primary development target)
  src/rtip_jax/       — Package source (canonical)
  tests/              — pytest suite (91 tests)
  examples/ic5c02384/ — Research runner scripts
research/ic5c02384/
  reactions/          — Input structures (11 reactions, reactants/product/ts)
  scripts/            — Runner scripts (run_rcmd.py, rcmd_lib.py, slurm/)
  results/            — ALL simulation results (see results/README.md)
    rc-md/            — RC-MD parameter scans (73 runs)
    attractive/       — Attractive bias runs (16 runs)
    synthesis/        — Synthesis MD runs
    rust-align/       — Post-alignment validation (v1-v3)
    archive/          — Superseded/early exploratory runs
molecules/            — Small molecule XYZ library
history/              — Frozen archive (migration docs, daily logs, legacy docs, slurm logs)
JAX/.venv/            — Active Python 3.10 virtualenv (rtip-jax editable install)
```

## Architecture (rtip_jax)

```
rtip_jax/
├── constants.py       — Physical constants, Element enum (103), atomic masses (H,B,C,N,O,Si,P,S)
├── config.py          — Para dataclass (a0, max_step, dt, temp_bath, pot_drop, etc.)
├── system.py          — System: coord (Bohr), cell, atom_type, atom_add_pot, pot
├── core/
│   ├── rtip.py        — RTI distance (quaternion eigensystem) + Rtip0PES Gaussian bias
│   ├── idwm.py        — Interatomic distance weighted metric + Idwm0PES
│   └── optimization.py — 1D golden-section line search
├── pes/
│   ├── base.py        — PES protocol, SumPES, ZeroPES, HarmonicPES
│   └── bias.py        — RepulsivePot, AttractivePot, SynthesisPot, ReactionCoordinatePot
├── workflows/
│   ├── pathway_sampling.py — Force-driven pathway (no temperature, line search)
│   ├── md.py               — NVT MD (leapfrog + Berendsen thermostat, 4 bias types)
│   └── synthesis.py        — Fragment layout (2-4 molecules, random rotation)
├── external/
│   ├── deepmd.py      — DeePMD-kit PES provider (production)
│   └── cp2k.py        — Legacy boundary doc (not functional)
├── io/                — XYZ (Angstrom boundary), PDB, output paths
├── math/rotations.py  — Euler-angle random rotation
└── cli.py             — 8 subcommands
```

## Key Conventions

- Internal units: Bohr, Hartree, Hartree/Bohr. XYZ files use Angstrom. DeePMD uses eV/eV/Angstrom.
- JAX float64 (x64) is enabled on import to match Rust f64 behavior.
- System.coord is always (natom, 3) in Bohr.
- Para.bias_amplitude(step) returns non-negative envelope; sign applied by caller.
- Pathway: no temperature, temp_K/kin_Ha are NaN by design.
- MD: leapfrog integration, Berendsen thermostat, phase machine (growing → reducing → off).

## Development Commands

```bash
# Activate the project virtualenv
source /home/lhshen/RTIP/JAX/.venv/bin/activate

# Run tests (from the package directory)
cd /home/lhshen/RTIP/rtipmd/jax
pytest

# Syntax check
python -m compileall src tests

# Reinstall editable (if source layout changes)
pip install -e /home/lhshen/RTIP/rtipmd/jax --no-deps --no-build-isolation

# CLI smoke
rtip-jax show-default-config
rtip-jax synthesize --inputs /home/lhshen/RTIP/rtipmd/jax/examples/ic5c02384/reactions/1-tBu__CO2/1.xyz /home/lhshen/RTIP/rtipmd/jax/examples/ic5c02384/reactions/1-tBu__CO2/2.xyz --output /tmp/IS.xyz --dist 5.0 --seed 0
rtip-jax mock-pathway --input /tmp/IS.xyz --output-dir /tmp/mock --max-step 5 --seed 0
rtip-jax mock-md --input /tmp/IS.xyz --output-dir /tmp/mock_md --max-step 5 --seed 0
```

## DeePMD Production Environment (n5)

```bash
export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source /group/software/deepmd-kit-3.1.1/bin/activate
export PYTHONPATH=/home/lhshen/RTIP/rtipmd/jax/src${PYTHONPATH:+:${PYTHONPATH}}

# Model
/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt
```

## Research Context (ic5c02384)

- 11 reactions: 7 substituents (CN, H, Me, Ph, PMe2, SiMe3, tBu) + CO2, plus 4 small molecules with 1-tBu (CS2, H2CO, MeCN, MeCH=NMe).
- Key bonding metrics: B-O/B-S/B-N distance and N-C distance (product reference ~1.49 A).
- Judgment criteria: pot_real trend, key bond distances, RMSD to reference. Do NOT judge by pot_total or rti_dist alone.
- RC-MD (ReactionCoordinate MD) with k=0.006 product-target is the most promising approach so far.
- Repulsive RTIP fails for association reactions (5 seeds all failed).

## Important Gotchas

- atomic_mass() only supports 8 elements (H, B, C, N, O, Si, P, S). Other elements raise UnsupportedElementMassError.
- DeePMD --type-map order must match model training order exactly.
- MD temperature formula uses 3*(natom-1) DOF; single-atom systems are explicitly rejected.
- JAX eigh may return tiny negative eigenvalues for PSD matrices; code clips to zero before sqrt.
- .gitignore blocks *.xyz and *.pdf globally except under research/ and molecules/.
- JAX/src/rtip_jax/ is a STALE incomplete copy (only subdirectories, missing __init__.py, cli.py, config.py, etc.). The canonical source is rtipmd/jax/src/rtip_jax/. Do not edit JAX/src/.
- The virtualenv lives at JAX/.venv (not rtipmd/jax/.venv). The editable install points to rtipmd/jax.

## Git

- Remote: git@github.com:askusk521-cyber/RTIP.git
- Main branch: main
- Current working branch: dev/reorganize

## Known Issues & Fixes (dev/rust-align)

### Atom-index mapping bug (FIXED in e564ba2)

 was using reactant (IS) atom
indices to compute target distances in the product structure. Product XYZ files
have DIFFERENT atom ordering than the concatenated reactant IS. This caused
restraints to pull toward wrong atoms (H instead of O/C).

Fix: use  indices (element+proximity detected in target)
for computing target distances; keep config indices for restraint atom_i/atom_j.

The same bug exists in  reference distance reporting
(cosmetic only, does not affect simulation).

### RC-MD parameter guidance

- k=0.006 with 1000 steps is insufficient for 1-tBu+CO2 with the current IS.xyz
- k=0.02 with 2000 steps reliably forms B-O bond (1.19 A) and approaches N-C target
- The previous successful run (k=0.006, 1000 steps) used a different IS orientation
- Temperature rises to ~1000K during strong restraint pulling; thermostat manages it

### Rust alignment changes (855f605)

- : quaternion matrix symmetrized before eigh (eliminates UPLO diff)
- : attractive/synthesis line search passes full system (matches Rust)
- :  field added
- : rust_compat=True disables temp feedback, early stopping, phase transitions
- DeePMD-related code unchanged; JAX extensions (attractive/synthesis/RC bias) preserved


## Known Issues & Fixes (dev/rust-align)

### Atom-index mapping bug (FIXED in e564ba2)

rcmd_lib.py reaction_coordinate_restraints() was using reactant (IS) atom
indices to compute target distances in the product structure. Product XYZ files
have DIFFERENT atom ordering than the concatenated reactant IS. This caused
restraints to pull toward wrong atoms (H instead of O/C).

Fix: use detect_reference_core() indices (element+proximity detected in target)
for computing target distances; keep config indices for restraint atom_i/atom_j.

The same bug exists in analyze_generalized_job() reference distance reporting
(cosmetic only, does not affect simulation).

### RC-MD parameter guidance

- k=0.006 with 1000 steps is insufficient for 1-tBu+CO2 with the current IS.xyz
- k=0.02 with 2000 steps reliably forms B-O bond (1.19 A) and approaches N-C target
- The previous successful run (k=0.006, 1000 steps) used a different IS orientation
- Temperature rises to ~1000K during strong restraint pulling; thermostat manages it

### Rust alignment changes (855f605)

- core/rtip.py: quaternion matrix symmetrized before eigh (eliminates UPLO diff)
- pathway_sampling.py: attractive/synthesis line search passes full system (matches Rust)
- config.py: Para.rust_compat=False field added
- md.py: rust_compat=True disables temp feedback, early stopping, phase transitions
- DeePMD-related code unchanged; JAX extensions (attractive/synthesis/RC bias) preserved
