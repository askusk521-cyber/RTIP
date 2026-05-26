from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from jax import random

from analyze_bias_exploration_1tbu_co2 import (
    B_INDEX,
    CO2_C_INDEX,
    CO2_O_INDICES,
    N2_INDICES,
    REFERENCE_REORDER_1TBU_CO2,
    analyze_job,
)
from rtip_jax.config import Para
from rtip_jax.constants import BOLTZMANN, HARTREE_TO_JOULE, element_symbol
from rtip_jax.external import DeepMDBoundary, DeepMDPES
from rtip_jax.pes import AttractivePot, DistanceRestraint, ReactionCoordinatePot, SynthesisPot
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


def _system_symbols(system: System) -> tuple[str, ...]:
    if system.atom_type is None:
        return tuple("XX" for _ in range(system.natom))
    return tuple(element_symbol(element) for element in system.atom_type)


def _read_xyz_molecules(filenames: Sequence[Path]) -> tuple[System, tuple[tuple[int, ...], ...]]:
    molecules = tuple(System.read_xyz(filename) for filename in filenames)
    coord = jnp.concatenate([molecule.coord for molecule in molecules], axis=0)
    atom_type = tuple(atom for molecule in molecules for atom in (molecule.atom_type or ()))
    if len(atom_type) != coord.shape[0]:
        atom_type = None

    mol_index: list[tuple[int, ...]] = []
    start = 0
    for molecule in molecules:
        stop = start + molecule.natom
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


def _resolve_input_path(value: str, reaction_dir: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = reaction_dir / path.name
    if candidate.exists() or path.name == "IS.xyz":
        return candidate
    return path


def _ensure_input_system(
    input_path: Path,
    *,
    reaction_dir: Path,
    synth_dist: float,
    seed: int,
) -> tuple[System, tuple[tuple[int, ...], ...]]:
    reactant_files = (reaction_dir / "1.xyz", reaction_dir / "2.xyz")
    combined, mol_index = _read_xyz_molecules(reactant_files)
    if input_path.exists():
        return System.read_xyz(input_path), mol_index

    if input_path.name != "IS.xyz":
        raise FileNotFoundError(f"input XYZ not found: {input_path}")

    input_path.parent.mkdir(parents=True, exist_ok=True)
    initial = synthesize_layout(combined, mol_index, synth_dist, key=random.PRNGKey(seed))
    initial.write_xyz(input_path, create_new_file=True, step=0)
    return initial, mol_index


def _reorder_reference_to_input(reference: System, input_system: System) -> System:
    ref_symbols = _system_symbols(reference)
    input_symbols = _system_symbols(input_system)
    if ref_symbols == input_symbols:
        return reference

    order = REFERENCE_REORDER_1TBU_CO2
    if len(order) == reference.natom == input_system.natom:
        ordered_symbols = tuple(ref_symbols[index] for index in order)
        if ordered_symbols == input_symbols:
            atom_type = None if reference.atom_type is None else tuple(reference.atom_type[index] for index in order)
            mutable = None if reference.mutable is None else reference.mutable[jnp.asarray(order)]
            return replace(
                reference,
                coord=reference.coord[jnp.asarray(order)],
                atom_type=atom_type,
                mutable=mutable,
            )

    raise ValueError("reference atom order cannot be matched to the input atom order")


def _load_target(reaction_dir: Path, target: str, input_system: System) -> System:
    return _reorder_reference_to_input(System.read_xyz(reaction_dir / f"{target}.xyz"), input_system)


def _align_mobile_to_reference(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    ref_center = reference.mean(axis=0)
    mob_center = mobile.mean(axis=0)
    ref0 = reference - ref_center
    mob0 = mobile - mob_center
    u, _s, vt = np.linalg.svd(mob0.T @ ref0)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return mob0 @ rotation + ref_center


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


def _target_velocity(initial: System, target: System, initial_temp: float) -> jnp.ndarray:
    reference = np.asarray(initial.coord, dtype=np.float64)
    mobile = np.asarray(target.coord, dtype=np.float64)
    aligned = _align_mobile_to_reference(reference, mobile)
    return _direction_velocity(initial, aligned - reference, initial_temp)


def _synthesis_velocity(initial: System, mol_index: tuple[tuple[int, ...], ...], initial_temp: float) -> jnp.ndarray:
    target, indices = synthesis_target_state(initial, mol_index)
    if indices is not None:
        direction = np.zeros_like(np.asarray(initial.coord, dtype=np.float64))
        direction[np.asarray(indices, dtype=np.int64)] = np.asarray(target.coord, dtype=np.float64) - np.asarray(
            initial.coord[jnp.asarray(indices)],
            dtype=np.float64,
        )
    else:
        direction = np.asarray(target.coord, dtype=np.float64) - np.asarray(initial.coord, dtype=np.float64)
    return _direction_velocity(initial, direction, initial_temp)


def _closest_pair(reference: System, left: Sequence[int], right: Sequence[int]) -> tuple[int, int, float]:
    coord = np.asarray(reference.coord, dtype=np.float64)
    best: tuple[int, int, float] | None = None
    for atom_i in left:
        for atom_j in right:
            distance = float(np.linalg.norm(coord[atom_i] - coord[atom_j]))
            if best is None or distance < best[2]:
                best = (int(atom_i), int(atom_j), distance)
    if best is None:
        raise ValueError("empty atom set for reaction-coordinate pair")
    return best


def _reaction_coordinate_restraints(target: System, k: float) -> tuple[DistanceRestraint, ...]:
    n_atom, c_atom, n_c_target = _closest_pair(target, N2_INDICES, (CO2_C_INDEX,))
    b_atom, o_atom, b_o_target = _closest_pair(target, (B_INDEX,), CO2_O_INDICES)
    return (
        DistanceRestraint(n_atom, c_atom, n_c_target, k, "N-CO2(C)"),
        DistanceRestraint(b_atom, o_atom, b_o_target, k, "B-O"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 1-tBu__CO2 RTIP bias exploration with DeePMD.")
    parser.add_argument("--mode", choices=MODE_CHOICES, default="attractive-md")
    parser.add_argument("--target", choices=("ts", "product"), default="ts")
    parser.add_argument("--input", default="examples/ic5c02384/reactions/1-tBu__CO2/IS.xyz")
    parser.add_argument("--reaction-dir", default="examples/ic5c02384/reactions/1-tBu__CO2")
    parser.add_argument("--model", default="/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--max-step", type=int, default=200)
    parser.add_argument("--a0", type=float, default=0.0005)
    parser.add_argument("--rc-k", type=float, default=None)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--temp-bath", type=float, default=300.0)
    parser.add_argument("--initial-temp", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zero-velocity", action="store_true")
    parser.add_argument("--synth-dist", type=float, default=5.0)
    parser.add_argument("--print-step", type=int, default=1)
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reaction_dir = _resolve_reaction_dir(args.reaction_dir)
    input_path = _resolve_input_path(args.input, reaction_dir)
    initial, mol_index = _ensure_input_system(
        input_path,
        reaction_dir=reaction_dir,
        synth_dist=args.synth_dist,
        seed=0,
    )
    target = _load_target(reaction_dir, args.target, initial)

    output_prefix = args.output_prefix or f"bias_1tBu_CO2_{args.mode}_{args.target}_seed{args.seed}_step{args.max_step}"
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

    if args.mode.startswith("attractive"):
        config = AttractivePot(
            initial_state=initial,
            final_state=target,
            para=para,
            str_output_file=str(structure_file),
            output_file=str(table_file),
        )
    elif args.mode.startswith("synthesis"):
        config = SynthesisPot(
            initial_state=initial,
            mol_index=mol_index,
            para=para,
            str_output_file=str(structure_file),
            output_file=str(table_file),
        )
    else:
        rc_k = args.a0 if args.rc_k is None else args.rc_k
        config = ReactionCoordinatePot(
            initial_state=initial,
            restraints=_reaction_coordinate_restraints(target, rc_k),
            para=para,
            str_output_file=str(structure_file),
            output_file=str(table_file),
        )

    pes = DeepMDPES(DeepMDBoundary(model=args.model))
    print(f"mode={args.mode}")
    print(f"target={args.target}")
    print(f"input={input_path}")
    print(f"reaction_dir={reaction_dir}")
    print(f"model={args.model}")
    print(f"output_dir={output_dir}")
    print(f"output_prefix={output_prefix}")
    print(f"max_step={args.max_step}")
    print(f"a0={args.a0}")
    print(f"rc_k={args.a0 if args.rc_k is None else args.rc_k}")
    print(f"dt={args.dt}")
    print(f"temp_bath={args.temp_bath}")
    print(f"initial_temp={args.initial_temp}")
    print(f"seed={args.seed}")
    if isinstance(config, ReactionCoordinatePot):
        for restraint in config.restraints:
            print(
                "restraint="
                f"{restraint.label}: atoms {restraint.atom_i}-{restraint.atom_j}, "
                f"target_Bohr={restraint.target:.8f}, k={restraint.k:.8f}"
            )

    if args.mode.endswith("-md"):
        masses = atom_masses(initial)
        if args.zero_velocity:
            initial_velocity = None
            print("initial_temp_actual=0.0 (zero_velocity)")
            result = run_rtip_nvt_md(
                config,
                pes,
                key=random.PRNGKey(args.seed),
                perturb=True,
                initial_velocity=None,
                write_outputs=True,
            )
        else:
            if args.mode.startswith("synthesis"):
                initial_velocity = _synthesis_velocity(initial, mol_index, args.initial_temp)
            else:
                initial_velocity = _target_velocity(initial, target, args.initial_temp)
            print(f"initial_temp_actual={float(temperature(initial_velocity, masses)):.8f}")
            result = run_rtip_nvt_md(
                config,
                pes,
                perturb=False,
                initial_velocity=initial_velocity,
                write_outputs=True,
            )
        print(f"history_steps={len(result.history)}")
        print(f"final_energy_Ha={float(result.system.pot):.12f}")
        print(f"final_temp_K={float(temperature(result.velocity, masses)):.8f}")
    else:
        result = run_rtip_path_sampling(config, pes, key=random.PRNGKey(args.seed), write_outputs=True)
        print(f"history_steps={len(result.history)}")
        print(f"stopped={result.stopped}")
        print(f"final_energy_Ha={float(result.system.pot):.12f}")

    if not args.skip_analysis:
        summary = analyze_job(
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
