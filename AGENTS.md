# AGENTS.md - RTIP formose branch

## Project goal

Replicate the formose-reaction RTIP-MD study (JACS Au 2026, 6, 922) using a
Python/JAX port of the upstream Rust RTIP-MD algorithm, with DeePMD-kit as
the real PES provider.  This branch (`formose/replicate`) intentionally
contains only the upstream-consistent algorithm: no reaction-coordinate
bias (RCMD), no ic5c02384 workflow, no size-scaling / oscillation /
early-stopping engineering.

## Layout

```
rtipmd/jax/            Python + JAX implementation of RTIP-MD
  src/rtip_jax/        Package source
  tests/               pytest suite
molecules/             Small-molecule XYZ library (formose species)
formose/
  data/                SI-parsed structures, Table S1, initial cells
  build_box.py         Initial-cell builder (8 H2O + 8 CH2O + 2 Ca + 4 OH-)
  run_formose.slurm    Slurm runner (DeePMD evolution MD)
  analyze_trajectory.py  Bond-event analysis (paper monitoring scheme)
  para_formose.json    Rust-default MD parameters
  README.md / RECORD.md  Workflow docs and decision record
```

## Architecture (rtip_jax)

* `constants.py` - constants, Element enum, masses, covalent radii (Rust
  `STR_TO_ATOMIC_MASS` / `STR_TO_ATOMIC_RADIUS` tables).
* `config.py` - `Para` matching Rust `Para::new()` (io/input.rs).
* `system.py` - System + bond-connectivity helpers (`get_adj_mat`,
  `split_into_mol`, `judge_adj_of_mol`, `judge_variation_of_bonding`).
* `core/rtip.py` - RTI distance (quaternion) + `Rtip0PES` Gaussian bias.
* `core/idwm.py`, `core/optimization.py` - IDWM + line search.
* `pes/` - PES protocol, `RepulsivePot`, `AttractivePot`, `EvolutionPot`,
  `SynthesisPot`.
* `workflows/md.py` - `repulsive_md` and `evolution_md`, ported 1:1 from
  Rust `md.rs`.
* `workflows/pathway_sampling.py` - RTIP/IDWM pathway sampling.
* `workflows/synthesis.py` - fragment layout.
* `external/deepmd.py` - DeePMD-kit PES provider (unit conversion at the
  boundary).
* `cli.py` - `deepmd-md`, `deepmd-evolution-md`, `deepmd-pathway`,
  `mock-md`, `mock-pathway`, `synthesize`, boundary/config commands.

## Environment (n5)

* Runtime: activate `/group/software/deepmd-kit-3.1.1/bin/activate`
  (Python 3.12, jax 0.7.2 + CUDA, deepmd-kit 3.1.1), then
  `export PYTHONPATH=/home/lhshen/RTIP/rtipmd/jax/src`.
* Tests: `/home/lhshen/RTIP/JAX/.venv/bin/python -m pytest` (Python 3.10,
  jax 0.6.2; run from `rtipmd/jax` with `PYTHONPATH=src`).
* GPU runs must go through slurm (login node has no CUDA device):
  `sbatch formose/run_formose.slurm`.
* Model: `/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt` (full type map,
  includes Ca).  `--type-map` can be omitted; the model's own map is used.

## Conventions

* Internal units: Bohr, Hartree, Hartree/Bohr.  XYZ files use Angstrom
  (conversion at IO boundary).  DeePMD uses eV / eV/Angstrom.
* JAX x64 enabled on import (`_config.py`).
* `System.coord` is always (natom, 3) in Bohr.
* Evolution MD amplitude: `Increasing: a_min -= a0`; `Decreasing:
  a_min += a0 * decreasing_multiple`; `Falling: a_min += a0`; bond-change
  trigger uses `judge_variation_of_bonding(..., 1.0, 1.6)`; reset at
  `a_min > a_min * decreasing_bound`.

## Git

* Branch: `formose/replicate` (local on n5; not pushed upstream).
* Base: `size-scaling-lite` (fork HEAD 72c9746); history is preserved.
