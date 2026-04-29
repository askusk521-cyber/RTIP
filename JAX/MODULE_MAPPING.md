# Module Mapping

This file maps Rust modules to their planned Python + JAX counterparts.

| Rust source | Python/JAX target | Stage | Notes |
| --- | --- | --- | --- |
| `src/common/constants.rs` | `src/rtip_jax/constants.py` | 3 | Physical constants, unit conversion, elements, masses. |
| `src/common/error.rs` | `src/rtip_jax/errors.py` | 3 | Replace panic strings with Python exceptions. |
| `src/io/input.rs` | `src/rtip_jax/config.py` | 3, 5 | `Para` becomes a dataclass; Stage 5 adds JSON/TOML loading. |
| `src/io/output.rs` | `src/rtip_jax/io/outputs.py` | 5 | Implemented output path helpers, no MPI dependency. |
| `src/pes_exploration/system.rs` | `src/rtip_jax/system.py`, `src/rtip_jax/io/xyz.py`, `src/rtip_jax/io/pdb.py` | 3 | System data, XYZ/PDB IO, units. CP2K methods become external provider boundary notes. |
| `src/matrix/mod.rs` | `src/rtip_jax/math/rotations.py` | 3 | Random rotations must take explicit PRNG keys. |
| `src/pes_exploration/traits.rs` | `src/rtip_jax/pes/base.py` | 2, 4 | Rust traits become Python protocols. |
| `src/pes_exploration/potential.rs` | `src/rtip_jax/pes/bias.py` | 4 | Implemented in Stage 4.4. |
| `src/pes_exploration/idwm.rs` | `src/rtip_jax/core/idwm.py` | 4 | Implemented in Stage 4.1. |
| `src/pes_exploration/rtip.rs` | `src/rtip_jax/core/rtip.py` | 4 | Implemented in Stage 4.2. |
| `src/pes_exploration/optimization.rs` | `src/rtip_jax/core/optimization.py` | 4 | Implemented in Stage 4.3. |
| `src/pes_exploration/synthesis.rs` | `src/rtip_jax/workflows/synthesis.py` | 4, 5 | Pure layout and runtime target-state helper implemented. |
| `src/pes_exploration/pathway_sampling.rs` | `src/rtip_jax/workflows/pathway_sampling.py` | 4, 5 | Pure helpers and full generic-PES loops implemented. |
| `src/pes_exploration/md.rs` | `src/rtip_jax/workflows/md.py` | 4, 5 | Pure helpers and RTIP NVT MD loop implemented. |
| `src/external/cp2k.rs` | `src/rtip_jax/external/cp2k.py` | 2 | Not ported. Contract is documented as a replaceable external PES boundary. |
| `src/lib.rs`, `src/lib_2mol.rs`, `src/lib_3mol.rs` | `src/rtip_jax/cli.py`, `examples/` | 5 | CLI/config-driven entry points; production external-PES CLI remains provider-dependent. |

## CP2K Boundary Mapping

Rust `Cp2kPES` currently provides:

- `input_file`: CP2K input path such as `cp2k.inp`.
- `output_file`: CP2K output path such as `cp2k.out`.
- `mpi_comm`: communicator integer passed to CP2K.
- `get_energy(system) -> f64`.
- `get_energy_force(system) -> (f64, Array2<f64>)`.
- Position units: Bohr.
- Energy units: Hartree.
- Force units: Hartree/Bohr.
- Lifecycle: create force environment, set positions, calculate energy or
  energy/force, read results, destroy force environment.

The JAX rewrite preserves this as a documented external provider contract, not
as an implemented CP2K binding.

## Stage 3 Implemented Mapping

- `Element::from_str` -> `Element.from_str`.
- `Element::get_mass` -> `Element.get_mass` / `atomic_mass`.
- `Para::new()` -> `Para()`.
- `System::read_xyz` -> `System.read_xyz` / `rtip_jax.io.xyz.read_xyz`.
- `System::write_xyz` -> `System.write_xyz` / `rtip_jax.io.xyz.write_xyz`.
- `System::write_pdb` -> `System.write_pdb` / `rtip_jax.io.pdb.write_pdb`.
- `matrix::rand_rot` -> `rtip_jax.math.rotations.random_rotation` / `rand_rot`.

## Stage 4.1 Implemented Mapping

- `w` -> `weight`.
- `dw` -> `weight_derivative`.
- `wei_dist_mat` -> `wei_dist_mat`.
- `idw_dist` -> `idw_dist`.
- `idw_dist1` -> `idw_dist1`.
- `idw_dist_vec` -> `idw_dist_vec`.
- `idw_pot` -> `idw_pot`.
- `idw_pot_force` -> `idw_pot_force`.
- `Idwm0PES` -> `Idwm0PES`.

## Stage 4.2 Implemented Mapping

- `rti_dist` -> `rti_dist`.
- `rti_dists` -> `rti_dists`.
- `rti_dist_vec` -> `rti_dist_vec`.
- `rti_dists_vecs` -> `rti_dists_vecs`.
- `rti_rot_tran` -> `rti_rot_tran`.
- `f` -> `rti_weight`.
- `df` -> `rti_weight_derivative`.
- `rti_pot` -> `rti_pot`.
- `rti_pot_force` -> `rti_pot_force`.
- `Rtip0PES` -> `Rtip0PES`.

## Stage 4.3 Implemented Mapping

- `min_1d` -> `min_1d`.
- `min_1d_real_bias` -> `min_1d_real_bias`.

## Stage 4.4 Implemented Mapping

- `RepulsivePot` -> `RepulsivePot`.
- `AttractivePot` -> `AttractivePot`.
- `SynthesisPot` -> `SynthesisPot`.
- Python-only helper: `SumPES` for energy/force summation.

## Stage 4.5 Implemented Mapping

- `synthesis` pure layout logic -> `synthesize_layout` / `synthesis`.
- Fixed molecule offsets -> `synthesis_offsets`.

## Stage 4.6 Implemented Mapping

- Initial random displacement blocks -> `perturb_system`.
- Force norm calculation -> `force_norm` / `rms_force_norm`.
- Repulsive/IDWM stop block -> `repulsive_stop_update`.
- Attractive stop block -> `attractive_stop_update`.
- Synthesis stop block -> `synthesis_stop_update`.

## Stage 4.7 Implemented Mapping

- Atom mass collection -> `atom_masses`.
- Kinetic energy block -> `twice_kinetic_energy` / `kinetic_energy`.
- Temperature block -> `temperature`.
- Berendsen lambda block -> `berendsen_lambda`.
- Force-to-acceleration loop -> `accelerations`.
- Leapfrog first half-step -> `leapfrog_first`.
- Leapfrog second half-step -> `leapfrog_second`.

## Stage 5 Implemented Mapping

- `Para` file loading -> `load_para`.
- Default parameter output -> `format_default_para`.
- `output_rtip` -> `rtip_jax.io.outputs.output_rtip`.
- `output_cp2k` -> `rtip_jax.io.outputs.output_cp2k`.
- `RepulsivePot::rtip_path_sampling` -> `run_rtip_repulsive_path_sampling`.
- `RepulsivePot::idwm_path_sampling` -> `run_idwm_repulsive_path_sampling`.
- `AttractivePot::rtip_path_sampling` -> `run_rtip_attractive_path_sampling`.
- `SynthesisPot::rtip_path_sampling` -> `run_rtip_synthesis_path_sampling`.
- `RepulsivePot::rtip_nvt_md` -> `run_rtip_nvt_md`.
- Runtime synthesis final-state construction -> `synthesis_target_state`.
- Rust staticlib `main` variants -> `rtip_jax.cli` subcommands:
  `show-default-config`, `cp2k-boundary`, `synthesize`, `mock-pathway`,
  `mock-md`.
- Python-only mock provider for tests/smoke runs -> `HarmonicPES`.
