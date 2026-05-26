# DeePMD Integration Notes

This document records the replacement of the old CP2K real-PES provider with
DeePMD-kit inference.

The filename intentionally follows the requested `DEEPMD_INTO.md` spelling.
Chinese companion: [DEEPMD_INTO_ZH.md](DEEPMD_INTO_ZH.md).

## Goal

DeePMD is the production replacement for the CP2K provider used by the original
Rust workflows. The RTIP/JAX core still depends only on the generic `PES`
protocol:

- `get_energy(system) -> float`
- `get_energy_force(system) -> tuple[float, force]`

The external provider that satisfies this protocol is now
`rtip_jax.external.deepmd.DeepMDPES`.

## Rust CP2K Contract Replaced

The Rust `Cp2kPES` supplied:

- coordinates from `System.coord`
- optional cell
- potential energy
- atomic forces

The JAX DeePMD provider supplies the same data required by the workflows:

- input coordinates: converted from internal Bohr to DeePMD Angstrom
- input cell: converted from internal Bohr to Angstrom, or `None` for isolated
  systems
- atom types: converted from `System.atom_type` symbols to DeePMD `atype`
  indices using the model `type_map`
- output energy: converted from eV to Hartree
- output force: converted from eV/Angstrom to Hartree/Bohr
- output virial: converted from eV to Hartree and retained on
  `DeepMDResult`, although current RTIP workflows do not consume virial

## Python API

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

If `type_map` is omitted, `DeepMDPES` attempts to call
`deep_pot.get_type_map()`. Supplying `type_map` explicitly is preferred because
it makes the atom-index contract visible and testable.

## CLI

The CLI now exposes DeePMD-oriented entry points:

```bash
rtip-jax deepmd-boundary --model model.pth --type-map O,H
rtip-jax deepmd-pathway --input IS.xyz --model model.pth --type-map O,H --method rtip
rtip-jax deepmd-md --input IS.xyz --model model.pth --type-map O,H
```

The older `cp2k-boundary` command remains as documentation of the legacy
contract only. It is not a production provider.

## n5 Environment

The current `n5` installation provides:

- DeePMD-kit: `/group/software/deepmd-kit-3.1.1`
- CUDA: `/group/software/cuda-12.9.1`
- Model: `/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt`

Activate it with:

```bash
export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source /group/software/deepmd-kit-3.1.1/bin/activate
export PYTHONPATH=/home/lhshen/RTIP/rtipmd/jax/src${PYTHONPATH:+:${PYTHONPATH}}
```

Then run the project through the module entry point:

```bash
python -m rtip_jax.cli deepmd-boundary \
  --model /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt
```

The DPA-3.2-5M model has been smoke-tested on `n5` and exposes a full periodic
table `type_map`, so `--type-map` can be omitted for this model. Login nodes may
fall back to CPU if no CUDA device is allocated; production CUDA runs should use
a GPU job or GPU compute node.

## Dependency

DeePMD-kit is optional because tests can validate the boundary with injected
fake models:

```bash
python -m pip install -e ".[deepmd]"
```

The provider imports `deepmd.infer.DeepPot` only when a real model-backed
`DeepMDPES` is constructed without an injected `deep_pot`.

## Tests

Added tests in `tests/test_external_deepmd_provider.py` verify:

- Bohr -> Angstrom coordinate conversion
- Bohr -> Angstrom cell conversion
- non-periodic `cell=None` behavior
- `System.atom_type` -> DeePMD `atype` mapping through `type_map`
- eV -> Hartree energy conversion
- eV/Angstrom -> Hartree/Bohr force conversion
- eV -> Hartree virial conversion
- use of DeePMD-backed `PES` in the same workflow slot previously occupied by
  CP2K

CLI smoke tests in `tests/test_stage5_config_outputs_cli.py` verify the
`deepmd-boundary` command and type-map parsing.

## Assumptions and Risks

- The DeePMD model must be compatible with the chemical symbols in
  `System.atom_type`.
- DeePMD model energy reference conventions may differ from CP2K; full
  scientific validation needs labeled structures or old CP2K comparison data.
- The current workflows do not consume virial, but the provider stores converted
  virial in `DeepMDResult` for future extensions.
- DeePMD and RTIP/JAX may run with different array backends. The boundary uses
  NumPy arrays for DeepPot input and JAX arrays for returned forces.
