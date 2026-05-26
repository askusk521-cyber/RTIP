"""Generalized RC-MD runner for ic5c02384 reactions.

Usage:
  python examples/ic5c02384/run_rcmd.py \
    --reaction-dir examples/ic5c02384/reactions/1-Me__CO2 \
    --mode rc-md --target product --rc-k 0.006 \
    --max-step 1000 --temp-bath 300 --initial-temp 300 \
    --model /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import random

from rcmd_lib import (
    ReactionConfig,
    ReferenceCore,
    _closest_pair,
    _system_coord_angstrom,
    _system_symbols,
    analyze_generalized_job,
    detect_reaction_config,
    reaction_coordinate_restraints,
)
from rtip_jax.config import Para
from rtip_jax.constants import BOLTZMANN, HARTREE_TO_JOULE, element_symbol
from rtip_jax.external import DeepMDBoundary, DeepMDPES
from rtip_jax.pes import AttractivePot, ReactionCoordinatePot, SynthesisPot
from rtip_jax.system import System
from rtip_jax.workflows import (
    atom_masses,
    run_rtip_nvt_md,
    run_rtip_path_sampling,
    synthesize_layout,
    synthesis_target_state,
    temperature,
)

MODE_CHOICES = (
    "attractive-md",
    "synthesis-md",
    "rc-md",
    "attractive-pathway",
    "synthesis-pathway",
)


def _read_xyz_molecules(filenames: list[Path]) -> tuple[System, tuple[tuple[int, ...], ...]]:
    molecules = tuple(System.read_xyz(f) for f in filenames)
    coord = jnp.concatenate([m.coord for m in molecules], axis=0)
    atom_type: tuple | None = tuple(atom for m in molecules for atom in (m.atom_type or ()))
    if len(atom_type) != coord.shape[0]:
        atom_type = None

    mol_index: list[tuple[int, ...]] = []
    start = 0
    for m in molecules:
        stop = start + m.natom
        mol_index.append(tuple(range(start, stop)))
        start = stop
    return System(coord=coord, atom_type=atom_type, pot=0.0), tuple(mol_index)


def _resolve_reaction_dir(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    fallback = Path(__file__).resolve().parent / "reactions" / path.name
    if fallback.exists():
        return fallback
    return path


def _ensure_input_system(
    input_path: Path,
    *,
    reaction_dir: Path,
    synth_dist: float,
    seed: int,
) -> tuple[System, tuple[tuple[int, ...], ...]]:
    reactant_files = (reaction_dir / "1.xyz", reaction_dir / "2.xyz")
    combined, mol_index = _read_xyz_molecules(list(reactant_files))
    if input_path.exists():
        return System.read_xyz(input_path), mol_index

    if input_path.name != "IS.xyz":
        raise FileNotFoundError(f"input XYZ not found: {input_path}")

    input_path.parent.mkdir(parents=True, exist_ok=True)
    initial = synthesize_layout(combined, mol_index, synth_dist, key=random.PRNGKey(seed))
    initial.write_xyz(input_path, create_new_file=True, step=0)
    return initial, mol_index


def _load_target(reaction_dir: Path, target: str) -> System:
    return System.read_xyz(reaction_dir / f"{target}.xyz")


def _direction_velocity(system: System, direction: np.ndarray, initial_temp: float) -> jnp.ndarray:
    if initial_temp <= 0.0:
        return jnp.zeros_like(system.coord)
    masses = np.asarray(atom_masses(system), dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction - np.average(direction, axis=0, weights=masses)
    norm2 = float(np.sum(masses[:, np.newaxis] * direction * direction))
    if norm2 <= 0.0:
        raise ValueError("initial velocity direction has zero mass-weighted norm")
    twice_kinetic = initial_temp * BOLTZMANN * 3.0 * float(system.natom - 1) / HARTREE_TO_JOULE
    return jnp.asarray(direction * np.sqrt(twice_kinetic / norm2), dtype=jnp.float64)


def _align_mobile_to_reference(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    ref_c = reference.mean(axis=0)
    mob_c = mobile.mean(axis=0)
    ref0 = reference - ref_c
    mob0 = mobile - mob_c
    u, _s, vt = np.linalg.svd(mob0.T @ ref0)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return mob0 @ rot + ref_c


def _target_velocity(initial: System, target: System, initial_temp: float) -> jnp.ndarray:
    reference = np.asarray(initial.coord, dtype=np.float64)
    mobile = np.asarray(target.coord, dtype=np.float64)
    aligned = _align_mobile_to_reference(reference, mobile)
    return _direction_velocity(initial, aligned - reference, initial_temp)


def _synthesis_velocity(
    initial: System, mol_index: tuple[tuple[int, ...], ...], initial_temp: float
) -> jnp.ndarray:
    target, indices = synthesis_target_state(initial, mol_index)
    if indices is not None:
        direction = np.zeros_like(np.asarray(initial.coord, dtype=np.float64))
        direction[np.asarray(indices, dtype=np.int64)] = np.asarray(target.coord, dtype=np.float64) - np.asarray(
            initial.coord[jnp.asarray(indices)], dtype=np.float64,
        )
    else:
        direction = np.asarray(target.coord, dtype=np.float64) - np.asarray(initial.coord, dtype=np.float64)
    return _direction_velocity(initial, direction, initial_temp)


def _output_prefix(reaction_name: str, mode: str, target: str, rc_k: float | None, temp_bath: float, max_step: int) -> str:
    """Auto-generate output prefix from reaction parameters."""
    k_int = int(round(rc_k * 1000)) if rc_k is not None else 0
    temp_int = int(temp_bath)
    return f"bias_{reaction_name}_{mode}_{target}_k{k_int:04d}_T{temp_int}_{max_step}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generalized RC-MD for ic5c02384 reactions with DeePMD."
    )
    parser.add_argument("--mode", choices=MODE_CHOICES, default="rc-md")
    parser.add_argument("--target", choices=("ts", "product"), default="product")
    parser.add_argument("--reaction-dir", required=True, help="Path to reaction directory (e.g. reactions/1-Me__CO2)")
    parser.add_argument("--input", default=None, help="Optional path to IS.xyz (auto-generated if missing)")
    parser.add_argument("--model", default="/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--max-step", type=int, default=1000)
    parser.add_argument("--a0", type=float, default=0.0005)
    parser.add_argument("--rc-k", type=float, default=0.006)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--temp-bath", type=float, default=300.0)
    parser.add_argument("--initial-temp", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synth-dist", type=float, default=5.0)
    parser.add_argument("--print-step", type=int, default=1)
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reaction_dir = _resolve_reaction_dir(args.reaction_dir)
    reaction_name = reaction_dir.name

    input_path = (
        Path(args.input) if args.input else reaction_dir / "IS.xyz"
    )
    initial, mol_index = _ensure_input_system(
        input_path, reaction_dir=reaction_dir, synth_dist=args.synth_dist, seed=0,
    )

    # Detect reaction configuration
    config = detect_reaction_config(reaction_dir, mol_index)

    target_system = _load_target(reaction_dir, args.target)

    output_prefix = args.output_prefix or _output_prefix(
        reaction_name, args.mode, args.target, args.rc_k, args.temp_bath, args.max_step,
    )
    output_dir = Path(args.output_dir or output_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    para = Para(
        a0=args.a0,
        max_step=args.max_step,
        print_step=args.print_step,
        dt=args.dt,
        temp_bath=args.temp_bath,
    )
    structure_file = output_dir / f"{output_prefix}.pdb"
    table_file = output_dir / f"{output_prefix}.out"

    # Build bias config
    if args.mode.startswith("attractive"):
        bias_config = AttractivePot(
            initial_state=initial,
            final_state=target_system,
            para=para,
            str_output_file=str(structure_file),
            output_file=str(table_file),
        )
    elif args.mode.startswith("synthesis"):
        bias_config = SynthesisPot(
            initial_state=initial,
            mol_index=mol_index,
            para=para,
            str_output_file=str(structure_file),
            output_file=str(table_file),
        )
    else:
        rc_k = args.rc_k
        bias_config = ReactionCoordinatePot(
            initial_state=initial,
            restraints=reaction_coordinate_restraints(target_system, config, rc_k),
            para=para,
            str_output_file=str(structure_file),
            output_file=str(table_file),
        )

    pes = DeepMDPES(DeepMDBoundary(model=args.model))
    print(f"reaction={reaction_name}")
    print(f"small_molecule={config.small_molecule.value}")
    print(f"mode={args.mode}")
    print(f"target={args.target}")
    print(f"input={input_path}")
    print(f"reaction_dir={reaction_dir}")
    print(f"model={args.model}")
    print(f"output_dir={output_dir}")
    print(f"output_prefix={output_prefix}")
    print(f"max_step={args.max_step}")
    print(f"a0={args.a0}")
    print(f"rc_k={args.rc_k}")
    print(f"dt={args.dt}")
    print(f"temp_bath={args.temp_bath}")
    print(f"initial_temp={args.initial_temp}")
    print(f"seed={args.seed}")
    print(f"config: b_index={config.b_index}, n2_indices={config.n2_indices}, "
          f"small_c_index={config.small_c_index}, small_x_indices={config.small_x_indices}, "
          f"b_x_label={config.b_x_label}")

    if isinstance(bias_config, ReactionCoordinatePot):
        for restraint in bias_config.restraints:
            print(
                f"  restraint={restraint.label}: atoms {restraint.atom_i}-{restraint.atom_j}, "
                f"target_Bohr={restraint.target:.8f}, k={restraint.k:.8f}"
            )

    if args.mode.endswith("-md"):
        if args.mode.startswith("synthesis"):
            initial_velocity = _synthesis_velocity(initial, mol_index, args.initial_temp)
        else:
            initial_velocity = _target_velocity(initial, target_system, args.initial_temp)
        masses = atom_masses(initial)
        print(f"initial_temp_actual={float(temperature(initial_velocity, masses)):.8f}")
        result = run_rtip_nvt_md(
            bias_config, pes, perturb=False, initial_velocity=initial_velocity, write_outputs=True,
        )
        print(f"history_steps={len(result.history)}")
        print(f"final_energy_Ha={float(result.system.pot):.12f}")
        print(f"final_temp_K={float(temperature(result.velocity, masses)):.8f}")
    else:
        result = run_rtip_path_sampling(bias_config, pes, key=random.PRNGKey(args.seed), write_outputs=True)
        print(f"history_steps={len(result.history)}")
        print(f"stopped={result.stopped}")
        print(f"final_energy_Ha={float(result.system.pot):.12f}")

    if not args.skip_analysis:
        summary = analyze_generalized_job(
            pdb_file=structure_file,
            out_file=table_file,
            reaction_dir=reaction_dir,
            input_file=input_path,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
        print("final_metrics=" + str(summary["final_metrics"]))
        print("best_bond_forming=" + str(summary["best_bond_forming"]))
        print(f"summary_json={output_dir / 'summary.json'}")
        print(f"summary_md={output_dir / 'summary.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
