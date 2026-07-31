# AGENTS.md — RTIP Project Context

## Project Goal

RTIP (Roto-Translationally Invariant Potential) is a computational chemistry tool for exploring molecular reaction pathways using biased potential energy surfaces. It places Gaussian bias potentials in roto-translationally invariant distance (RTI) space to drive molecular systems from reactants over transition states to products.

Research target: validating the ACS paper `ic5c02384` — methyleneborane-N2 complex coupling with CO2 and other small molecules. The `reactions/` dataset holds the corresponding input structures.

> **Branch note:** `size-scaling-lite` is a slimmed, PR-friendly branch containing only the JAX
> implementation plus minimal input data. The Rust reference (`rtipmd/rust/`), all simulation
> results (`research/ic5c02384/results/`), and daily/migration history have been stripped —
> the full project lives on `feat/size-scaling`.

## Repository Layout

```
rtipmd/jax/           — Python + JAX implementation (the whole codebase)
  src/rtip_jax/       — Package source (canonical)
  tests/              — pytest suite (101 tests)
  examples/ic5c02384/ — Research runner entry points (symlinks into research/)
research/ic5c02384/
  reactions/          — Input structures (11 reactions: reactants / product / ts)
  scripts/            — Runner scripts (run_rcmd.py, rcmd_lib.py, run_size_scaling_synth.py, slurm/)
  manifests/          — Reaction & structure manifests (CSV)
  paper/              — Source paper PDF + SI coordinates
  references/         — Reaction reference notes
  structures/         — Extracted reference structures
molecules/            — Small molecule XYZ library
history/slurm-logs/   — Archived SLURM job logs
JAX/.venv/            — Active Python 3.10 virtualenv (rtip-jax editable install)
```

## Architecture (rtip_jax)

```
rtip_jax/
├── constants.py       — Physical constants, Element enum (103), atomic masses (H,B,C,N,O,Si,P,S)
├── config.py          — Para dataclass (a0, max_step, dt, temp_bath, pot_drop, size_scaling, etc.)
├── _config.py         — Enables JAX float64 (x64) on import
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
└── cli.py             — 9 subcommands
```

## Key Conventions

- Internal units: Bohr, Hartree, Hartree/Bohr. XYZ files use Angstrom. DeePMD uses eV/eV/Angstrom.
- JAX float64 (x64) is enabled on import (see `_config.py`).
- System.coord is always (natom, 3) in Bohr.
- `Para.bias_amplitude(step, n_bias=...)` returns a non-negative envelope; sign applied by caller.
- Pathway: no temperature, temp_K/kin_Ha are NaN by design.
- MD: leapfrog integration, Berendsen thermostat, phase machine (growing → reducing → off).

## Size Scaling (this branch's feature)

RTI distance grows as √N with the number of biased atoms, so sigma² ∝ N while the amplitude
`a0·step` is a fixed scalar. The dominant per-atom bias force therefore scales as ~a0/N and is
diluted in large systems. `Para.size_scaling` cancels this dilution by multiplying the effective
amplitude by `n_bias` (power = 1).

- `Para.size_scaling: bool = False` — opt-in; when False, output is identical to the pre-feature behavior.
- `bias_amplitude(step, ..., n_bias=N)` multiplies the growing/reducing envelope by N only when size_scaling is enabled (the Gaussian shape is unchanged; the √N factors cancel).
- MD and pathway bias builders pass `n_bias = len(atom_add_pot)` (or `natom` when unset), matching the atom set used for sigma.
- When size_scaling is on, the MD `f_bias` stop threshold scales by √n_bias so it stays size-insensitive.
- Enable via CLI `--size-scaling` (deepmd-md / deepmd-synthesis-md) or `size_scaling = true` in a config file.
- Not applied to ReactionCoordinatePES (distance restraints, not a Gaussian RTIP amplitude).

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

## Important Gotchas

- atomic_mass() only supports 8 elements (H, B, C, N, O, Si, P, S). Other elements raise UnsupportedElementMassError.
- DeePMD --type-map order must match model training order exactly.
- MD temperature formula uses 3*(natom-1) DOF; single-atom systems are explicitly rejected.
- JAX eigh may return tiny negative eigenvalues for PSD matrices; code clips to zero before sqrt.
- .gitignore blocks *.xyz and *.pdf globally except under research/ and molecules/.
- The virtualenv lives at JAX/.venv (not rtipmd/jax/.venv). The editable install points to rtipmd/jax.

## Git

- Remote: git@github.com:askusk521-cyber/RTIP.git
- Main branch: main
- Current working branch: size-scaling-lite
