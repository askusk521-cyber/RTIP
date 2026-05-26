# RTIP/JAX Usage

This document describes the current Rust-to-Python/JAX migration status,
environment setup, testing, CLI usage, Slurm usage, and output formats.

Chinese companion: [USAGE_ZH.md](USAGE_ZH.md).

## Completion Status

The JAX version has completed the main structural migration and core functional
rewrite of the Rust RTIP project:

- Python package structure: complete, package name `rtip_jax`.
- JAX x64: enabled by default to match Rust `f64` behavior.
- Public modules: constants, elements, masses, `Para`, `System`, XYZ/PDB IO.
- Core algorithms: IDWM, RTIP, local one-dimensional optimization, RTIP/IDWM
  PES wrappers.
- Workflows: repulsive RTIP/IDWP pathway, attractive RTIP pathway, synthesis
  pathway, and RTIP NVT MD.
- DeePMD as CP2K replacement: provider interface and tests are complete. CP2K
  is kept as a legacy boundary document only; real energy/force comes from
  DeePMD.
- CLI: config output, synthesis layout, mock runs, DeePMD pathway, and DeePMD
  MD.
- Tests: the full suite passed on `n5` with `87 passed` (2026-05-24).

Still pending or needing further validation:

- Real scientific benchmark runs with a validated DeePMD model.
- Full Rust-vs-JAX trajectory and scalar-log comparison.
- MPI rank fan-out; current orchestration is plain Python (wrap separately if
  parallel scheduling is needed later).
- Stage 7 JAX-native performance optimization (current priority is clear
  behavior and testability).

In summary:

- Engineering migration: essentially complete.
- Unit/integration tests: complete for the currently testable scope.
- DeePMD provider replacement: interface and unit-contract tests complete;
  real-model validation still needed.
- Scientific numerical validation: still needs real-model benchmarks and
  Rust/JAX comparison samples.

## Directory Layout

```text
rtipmd/jax/
  pyproject.toml
  README.md
  README_ZH.md
  USAGE.md
  USAGE_ZH.md
  DEEPMD_INTO.md
  DEEPMD_INTO_ZH.md
  src/rtip_jax/
    constants.py
    config.py
    system.py
    core/
      idwm.py
      rtip.py
      optimization.py
    pes/
      base.py
      bias.py
    workflows/
      synthesis.py
      pathway_sampling.py
      md.py
    external/
      cp2k.py
      deepmd.py
    io/
      xyz.py
      pdb.py
      outputs.py
  tests/
```

## Environment Setup

### 1. Create a virtual environment

Use Python 3.10 or newer. The remote validation used Python 3.10.

From `rtipmd/jax`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On `n5`, an existing conda Python can create the venv:

```bash
cd /home/lhshen/RTIP/rtipmd/jax
/home/lhshen/miniconda3/envs/flower/bin/python -m venv .venv
source .venv/bin/activate
```

### 2. Install development dependencies

```bash
python -m pip install -e ".[dev]"
```

This installs:

- `jax[cpu]`
- `numpy`
- `pytest`
- `pytest-cov`

### 3. Install DeePMD support

For real DeePMD model runs:

```bash
python -m pip install -e ".[deepmd]"
```

For both development tests and DeePMD:

```bash
python -m pip install -e ".[dev,deepmd]"
```

Unit tests do not require real `deepmd-kit`; they use fake `DeepPot` objects to
validate the boundary and unit conversions.

## n5 DeePMD Production Environment

The current `n5` installation provides:

```text
DeePMD-kit: /group/software/deepmd-kit-3.1.1
CUDA:       /group/software/cuda-12.9.1
Model:      /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt
```

Activate CUDA and DeePMD:

```bash
export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source /group/software/deepmd-kit-3.1.1/bin/activate
```

Because this environment is not installed through `pip install -e`, set
`PYTHONPATH` when running this project:

```bash
export PYTHONPATH=/home/lhshen/RTIP/rtipmd/jax/src${PYTHONPATH:+:${PYTHONPATH}}
```

Then call the CLI as a module:

```bash
python -m rtip_jax.cli deepmd-boundary \
  --model /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt
```

The DPA-3.2-5M model exposes a full-periodic-table `type_map`, so `--type-map`
may be omitted for this model. If a type map is given explicitly, its element
order must match the model training order.

Login nodes may report no CUDA device and fall back to CPU. Production CUDA
runs should use a GPU job or GPU compute node.

Minimal real-model smoke result (verified 2026-05-26 on `n5`):

```text
energy_hartree -7.599025170124905
force_shape (36, 3)
force_norm 0.05569718105175971
```

This only verifies provider loading and unit conversion, not scientific
benchmark validity.

## Testing

Run the full suite:

```bash
cd RTIP/rtipmd/jax
source .venv/bin/activate
pytest
```

Remote `n5` validation result:

```text
87 passed
```

Syntax-level check:

```bash
python -m compileall src tests
```

## Units

Internal RTIP/JAX units follow the Rust/CP2K-style atomic-unit convention:

- Coordinates: Bohr.
- Energy: Hartree.
- Force: Hartree/Bohr.

Text IO boundaries:

- XYZ input/output uses Angstrom.
- XYZ input is converted to internal Bohr.
- XYZ/PDB output is converted back to Angstrom.

DeePMD boundary:

- Input coordinates: Angstrom.
- Input cell: Angstrom, or `None` for isolated systems.
- Output energy: eV, converted to Hartree.
- Output force: eV/Angstrom, converted to Hartree/Bohr.
- Output virial: eV, converted to Hartree and stored in `DeepMDResult`.

## Configuration

Default parameters correspond to Rust `Para::new()`:

```bash
rtip-jax show-default-config
```

Save a JSON file such as `para.json`:

```json
{
  "a0": 0.002,
  "scale_ts_a0": 1.0,
  "scale_ts_sigma": 0.25,
  "max_step": 1500,
  "print_step": 1,
  "pot_climb": 0.185,
  "pot_drop": 0.02,
  "pot_epsilon": 0.00005,
  "f_epsilon": 0.001,
  "dt": 0.5,
  "tau": 500.0,
  "temp_bath": 1000.0
}
```

Use it with `--config para.json`.

## CLI

### Inspect the legacy CP2K boundary

CP2K is no longer a real provider. This command documents the legacy
input/output contract only:

```bash
rtip-jax cp2k-boundary --input cp2k.inp --output cp2k.out
```

### Inspect the DeePMD boundary

```bash
rtip-jax deepmd-boundary --model model.pth --type-map O,H
```

On `n5` with the system DeePMD environment:

```bash
python -m rtip_jax.cli deepmd-boundary \
  --model /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt
```

`--type-map` must match the element order used during DeePMD model training.
For example, if the model has `type_map = ["O", "H"]`:

- O -> `atype = 0`
- H -> `atype = 1`

### Generate a synthesis initial layout

Default reads `1.xyz` and `2.xyz` from the current directory:

```bash
rtip-jax synthesize --output IS.xyz --dist 5.0 --seed 0
```

Or pass explicit molecule files (2, 3, or 4 files):

```bash
rtip-jax synthesize \
  --inputs 1.xyz 2.xyz 3.xyz \
  --output IS.xyz \
  --dist 5.0 \
  --seed 0
```

This concatenates files in order and auto-generates `mol_index`.

For a pre-combined XYZ file, provide `mol_index` manually:

```bash
rtip-jax synthesize \
  --input input.xyz \
  --output IS.xyz \
  --mol-index "0,1;2,3" \
  --dist 5.0 \
  --seed 0
```

`--mol-index` uses semicolons to separate molecules and commas to separate
atom indices. The example above means:

- Molecule 1: atoms 0, 1
- Molecule 2: atoms 2, 3

The current synthesis layout supports 2, 3, or 4 molecules.

### Run mock PES pathway smoke tests

```bash
rtip-jax mock-pathway \
  --input IS.xyz \
  --output-dir run_mock_pathway \
  --method rtip \
  --max-step 5 \
  --seed 0
```

IDWM pathway:

```bash
rtip-jax mock-pathway \
  --input IS.xyz \
  --output-dir run_mock_idwm \
  --method idwm \
  --max-step 5 \
  --seed 0
```

Mock PES is only for software workflow testing; it does not represent a real
potential energy surface.

### Run a DeePMD pathway

```bash
rtip-jax deepmd-pathway \
  --input IS.xyz \
  --model model.pth \
  --type-map O,H \
  --output-dir run_deepmd_pathway \
  --method rtip \
  --config para.json \
  --seed 0
```

Use `--method idwm` for IDWM instead of RTIP.

On `n5` with the system DeePMD environment:

```bash
python -m rtip_jax.cli deepmd-pathway \
  --input IS.xyz \
  --model /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt \
  --output-dir run_deepmd_pathway \
  --method rtip \
  --config para.json \
  --seed 0
```

### Run RTIP NVT MD with DeePMD

```bash
rtip-jax deepmd-md \
  --input IS.xyz \
  --model model.pth \
  --type-map O,H \
  --output-dir run_deepmd_md \
  --config para.json \
  --seed 0
```

On `n5` with the system DeePMD environment:

```bash
python -m rtip_jax.cli deepmd-md \
  --input IS.xyz \
  --model /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt \
  --output-dir run_deepmd_md \
  --config para.json \
  --seed 0
```

Disable the initial random perturbation with `--no-perturb`:

```bash
rtip-jax deepmd-md \
  --input IS.xyz \
  --model model.pth \
  --type-map O,H \
  --output-dir run_deepmd_md \
  --no-perturb
```

## Slurm on n5

The Slurm script is at:

```text
research/ic5c02384/scripts/slurm/run_deepmd_rtip.slurm
```

Submit from `rtipmd/jax`:

```bash
cd /home/lhshen/RTIP/rtipmd/jax
sbatch ../../research/ic5c02384/scripts/slurm/run_deepmd_rtip.slurm
```

Default settings:

- Partition: `main`.
- GPU: `--gres=gpu:1`.
- CPU: `-c 10`.
- Time: `24:00:00`.
- DeePMD environment: `/group/software/deepmd-kit-3.1.1`.
- CUDA: `/group/software/cuda-12.9.1`.
- Model: `/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt`.

Each `sbatch` command submits an independent job. By default, the job
synthesizes an initial structure, loads the DeePMD model, runs sampling or MD,
and writes trajectory plus scalar outputs.

Output is organized by Slurm job ID. For job `123456`:

```text
123456/123456-IS.xyz
123456/123456.pdb
123456/123456.out
```

`123456-IS.xyz` is the auto-synthesized initial structure. `123456.pdb` is the
structure trajectory. `123456.out` is the scalar log. When using the CLI
directly the files are `rtip.pdb`/`rtip.out`; the Slurm script renames them
with the job-ID prefix at the end of the job.

Default mode is `MODE=pathway` with `METHOD=rtip`. A single submit completes
the full workflow:

```bash
cd /home/lhshen/RTIP/rtipmd/jax
MAX_STEP=100 sbatch ../../research/ic5c02384/scripts/slurm/run_deepmd_rtip.slurm
```

By default, synthesis reads `1.xyz` and `2.xyz` from the current directory. If
those are absent and the directory contains exactly 2-4 `.xyz` files, the
script auto-selects them. To specify 3 or 4 molecules, or to fix the input
order, use `SYNTH_INPUTS`:

```bash
SYNTH_INPUTS="1.xyz 2.xyz 3.xyz" MAX_STEP=100 \
  sbatch ../../research/ic5c02384/scripts/slurm/run_deepmd_rtip.slurm
```

Override synthesis distance and seed:

```bash
SYNTH_INPUTS="1.xyz 2.xyz" SYNTH_DIST=6.0 SEED=12 MAX_STEP=100 \
  sbatch ../../research/ic5c02384/scripts/slurm/run_deepmd_rtip.slurm
```

Run IDWM pathway:

```bash
METHOD=idwm MAX_STEP=100 \
  sbatch ../../research/ic5c02384/scripts/slurm/run_deepmd_rtip.slurm
```

Run RTIP NVT MD:

```bash
MODE=md MAX_STEP=1000 \
  sbatch ../../research/ic5c02384/scripts/slurm/run_deepmd_rtip.slurm
```

Use a config file:

```bash
CONFIG=para.json \
  sbatch ../../research/ic5c02384/scripts/slurm/run_deepmd_rtip.slurm
```

Use a prepared input and skip synthesis:

```bash
INPUT=IS.xyz MAX_STEP=100 \
  sbatch ../../research/ic5c02384/scripts/slurm/run_deepmd_rtip.slurm
```

The script skips synthesis when `INPUT` is set explicitly. To force synthesis
that writes to a custom `INPUT` path, set `SYNTHESIZE=1`.

Override model, output directory, or prefix:

```bash
INPUT=/path/to/input.xyz \
MODEL=/path/to/model.pt \
OUTPUT_DIR=run_custom \
  sbatch ../../research/ic5c02384/scripts/slurm/run_deepmd_rtip.slurm
```

The examples above represent different experiment choices; do not submit all of
them sequentially unless you intend to run separate RTIP, IDWP, MD, or
different-config jobs.

## Output Files

Direct CLI workflows write:

```text
rtip.pdb
rtip.out
```

With `--output-dir run1`:

```text
run1/rtip.pdb
run1/rtip.out
```

The Slurm script creates a numeric directory named from the job ID and prefixes
files with that ID (see Slurm section above).

Slurm stdout/stderr logs are written as `slurm-%j.out` and `slurm-%j.err`
according to the script header.

`rtip.out` records step, time, distance, temperature or kinetic energy, real
PES energy, bias energy, total energy, force norms, and state-machine
decisions.

RTIP pathway header:

```text
step  time_fs  wall_time_s  rti_dist  temp_K  kin_Ha  pot_real_Ha  pot_rtip_Ha  pot_total_Ha  f_real  f_rtip  f_total  rms_f_real  bias_used  next_add_bias  stopped  state_decision
```

IDWM pathway header:

```text
step  time_fs  wall_time_s  idw_dist  temp_K  kin_Ha  pot_real_Ha  pot_idwm_Ha  pot_total_Ha  f_real  f_idwm  f_total  rms_f_real  bias_used  next_add_bias  stopped  state_decision
```

RTIP NVT MD header:

```text
step  time_fs  wall_time_s  rti_dist  temp_K  temp_bath_K  thermo_lambda  kin_Ha  pot_real_Ha  pot_rtip_Ha  pot_total_Ha  f_real  f_rtip  f_total  rms_f_real  state_decision
```

- `time_fs`: cumulative sampling/dynamics step time in fs.
- `wall_time_s`: elapsed wall-clock time in seconds.
- Energy columns: Hartree.
- Force-norm columns: Hartree/Bohr.
- `state_decision`: per-step state-machine judgment, e.g. `bias_on`,
  `bias_off_after_pot_drop`, `stop_converged`, `stop_pot_climb`,
  `stop_large_bias_force`, or MD's `md_running`/`max_step`.

Pathway runs do not have velocities, so `temp_K` and `kin_Ha` are `nan`. RTIP
NVT MD computes real instantaneous values from velocities and masses.

`rtip.pdb` is a text trajectory with `TITLE`, per-frame `REMARK`, `ATOM`
records, and `END`. It does not write `CONECT` or bond-order data;
visualization software may infer bonds from element-distance heuristics.

## Pathway State Machines

### Repulsive RTIP and IDWM pathway

Shared stop logic:

- Start with `add_bias = True`.
- Track historical maximum (`pot_real_max`) and minimum (`pot_real_min`) real
  PES energies.
- If `pot_real < pot_real_max - pot_drop`: the system has crossed the peak and
  is descending. Disable bias from the next step (`add_bias = False`).
- After bias is disabled, stop when `f_real / sqrt(natom) < f_epsilon` (local
  convergence).
- Stop if `pot_real > pot_real_min + pot_climb` (real energy has climbed too
  far above the historical minimum).
- Stop if `f_bias > 1000.0` (bias force abnormally large).
- Otherwise run until `max_step`.

### Synthesis pathway

Similar to repulsive pathway, but also records the first-step real energy
`pot_real_initial`. Bias is disabled only when both:

- `pot_real < pot_real_max - pot_drop`
- `pot_real > pot_real_initial`

After bias is disabled, the same convergence, `pot_climb`, and bias-force stop
rules apply.

### Attractive pathway

Simpler state machine:

- Start with `add_bias = True`.
- Disable bias when `rti_dist < 1.0` or `f_bias > 1000.0`.
- After bias is disabled, stop when `f_real / sqrt(natom) < f_epsilon`.
- Otherwise run until `max_step`.

### RTIP NVT MD

Currently runs a fixed number of steps (`max_step`) without the pathway
early-stop state machine. Each step performs leapfrog integration, Berendsen
thermostat, DeePMD real PES evaluation, and RTIP bias calculation.

## Python API

### Read a structure

```python
from rtip_jax.system import System

system = System.read_xyz("IS.xyz")
```

### Construct a DeePMD PES

```python
from rtip_jax.external import DeepMDBoundary, DeepMDPES

pes = DeepMDPES(
    DeepMDBoundary(
        model="model.pth",
        type_map=("O", "H"),
    )
)

energy, force = pes.get_energy_force(system)
```

### Run a repulsive RTIP pathway

```python
from jax import random
from rtip_jax.config import Para
from rtip_jax.pes import RepulsivePot
from rtip_jax.workflows import run_rtip_repulsive_path_sampling

config = RepulsivePot(
    local_min=system,
    nearby_ts=(),
    para=Para(max_step=100),
    str_output_file="rtip.pdb",
    output_file="rtip.out",
)

result = run_rtip_repulsive_path_sampling(
    config,
    pes,
    key=random.PRNGKey(0),
)

final_system = result.system
history = result.history
```

### Run RTIP NVT MD

```python
from jax import random
from rtip_jax.workflows import run_rtip_nvt_md

result = run_rtip_nvt_md(
    config,
    pes,
    key=random.PRNGKey(0),
)
```

## DeePMD Replaces CP2K

Rust workflows required:

- `get_energy(system) -> f64`
- `get_energy_force(system) -> (f64, Array2<f64>)`

JAX now supplies these through `DeepMDPES`:

- `get_energy(system) -> float`
- `get_energy_force(system) -> tuple[float, force]`

Tests cover DeePMD input shapes, coordinate/cell conversion, non-periodic
`cell=None`, element-to-`atype` mapping, energy/force/virial conversions, and
using the provider in the workflow slot formerly occupied by CP2K.

## FAQ

### 1. `ModuleNotFoundError: No module named 'deepmd'`

DeePMD support is not installed. Run:

```bash
python -m pip install -e ".[deepmd]"
```

### 2. Element not in `type_map`

If the structure contains O/H/Ca but the command only specifies:

```bash
--type-map O,H
```

the program will error on Ca. Fix by including all elements:

```bash
--type-map Ca,O,H
```

The order must match the DeePMD model training order.

### 3. Single-atom MD temperature error

The temperature formula uses `3 * (natom - 1)` degrees of freedom. The Python
version explicitly rejects single-atom temperature calculation to avoid the
division-by-zero behavior present in the Rust version.

### 4. RTIP distance should be zero but became NaN

This has been fixed by clipping tiny negative eigenvalues from JAX eigensolver
output before square root.

## Follow-Up

1. Provide validated real DeePMD models and their type maps.
2. Prepare two or three small official smoke-example XYZ systems.
3. Generate scalar baselines from Rust or known references.
4. Compare JAX `rtip.out`, final structures, and energy/force statistics.
5. Start JAX-native optimization after behavior is stable.
