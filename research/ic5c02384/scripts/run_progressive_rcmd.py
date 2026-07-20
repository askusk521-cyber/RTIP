"""Progressive RC-MD: start with weak k, tighten stepwise to avoid temp runaway.

Usage:
  python run_progressive_rcmd.py \
    --reaction-dir reactions/1-Me__CO2 \
    --target product \
    --k-stages 0.006,0.01,0.02 \
    --steps-per-stage 500 \
    --temp-bath 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import random

from rcmd_lib import (
    detect_reaction_config,
    reaction_coordinate_restraints,
    analyze_generalized_job,
    _system_coord_angstrom,
    _min_b_x,
    _min_n_c,
    read_pdb_frames,
    read_output_rows,
    frame_metrics,
    detect_reference_core,
)
from rtip_jax.config import Para
from rtip_jax.constants import BOHR_TO_ANGSTROM
from rtip_jax.external import DeepMDBoundary, DeepMDPES
from rtip_jax.pes import ReactionCoordinatePot
from rtip_jax.system import System
from rtip_jax.workflows import (
    atom_masses,
    run_rtip_nvt_md,
    synthesize_layout,
    temperature,
)


def _read_xyz_molecules(filenames):
    molecules = tuple(System.read_xyz(f) for f in filenames)
    coord = jnp.concatenate([m.coord for m in molecules], axis=0)
    atom_type = tuple(atom for m in molecules for atom in (m.atom_type or ()))
    if len(atom_type) != coord.shape[0]:
        atom_type = None

    mol_index = []
    start = 0
    for m in molecules:
        stop = start + m.natom
        mol_index.append(tuple(range(start, stop)))
        start = stop
    return System(coord=coord, atom_type=atom_type, pot=0.0), tuple(mol_index)


def _direction_velocity(system, direction, initial_temp):
    if initial_temp <= 0.0:
        return jnp.zeros_like(system.coord)
    masses = np.asarray(atom_masses(system), dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    from rtip_jax.constants import BOLTZMANN, HARTREE_TO_JOULE
    direction = direction - np.average(direction, axis=0, weights=masses)
    norm2 = float(np.sum(masses[:, np.newaxis] * direction * direction))
    if norm2 <= 0.0:
        raise ValueError("zero mass-weighted norm")
    twice_kinetic = initial_temp * BOLTZMANN * 3.0 * float(system.natom - 1) / HARTREE_TO_JOULE
    return jnp.asarray(direction * np.sqrt(twice_kinetic / norm2), dtype=np.float64)


def _align_mobile_to_reference(reference, mobile):
    ref_c = reference.mean(axis=0)
    mob_c = mobile.mean(axis=0)
    ref0 = reference - ref_c
    mob0 = mobile - mob_c
    u, _s, vt = np.linalg.svd(mob0.T @ ref0)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return mob0 @ rot + ref_c


def _target_velocity(initial, target, initial_temp):
    reference = np.asarray(initial.coord, dtype=np.float64)
    mobile = np.asarray(target.coord, dtype=np.float64)
    aligned = _align_mobile_to_reference(reference, mobile)
    return _direction_velocity(initial, aligned - reference, initial_temp)


def main():
    parser = argparse.ArgumentParser(description="Progressive restraint RC-MD")
    parser.add_argument("--reaction-dir", required=True)
    parser.add_argument("--target", choices=("ts", "product"), default="product")
    parser.add_argument("--k-stages", default="0.006,0.01,0.02",
                        help="Comma-separated k values for each stage")
    parser.add_argument("--steps-per-stage", type=int, default=500)
    parser.add_argument("--temp-bath", type=float, default=300.0)
    parser.add_argument("--initial-temp", type=float, default=300.0)
    parser.add_argument("--model", default="/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    reaction_dir = Path(args.reaction_dir)
    if not reaction_dir.exists():
        reaction_dir = Path(__file__).resolve().parent.parent / "reactions" / reaction_dir.name

    k_stages = [float(k) for k in args.k_stages.split(",")]
    print(f"Progressive RC-MD: {len(k_stages)} stages, k = {k_stages}")
    print(f"steps_per_stage = {args.steps_per_stage}")
    print(f"reaction_dir = {reaction_dir}")
    print(f"target = {args.target}")

    # Prepare initial system
    reactant_files = (reaction_dir / "1.xyz", reaction_dir / "2.xyz")
    combined, mol_index = _read_xyz_molecules(list(reactant_files))
    is_path = reaction_dir / "IS.xyz"
    if is_path.exists():
        initial = System.read_xyz(is_path)
    else:
        initial = synthesize_layout(combined, mol_index, 5.0, key=random.PRNGKey(args.seed))
        initial.write_xyz(is_path, create_new_file=True, step=0)

    config = detect_reaction_config(reaction_dir, mol_index)
    target_system = System.read_xyz(reaction_dir / f"{args.target}.xyz")

    output_prefix = args.output_prefix or f"prog_{reaction_dir.name}_{args.target}_{len(k_stages)}stage"
    output_dir = Path(args.output_dir or f"/home/lhshen/RTIP/JAX/{output_prefix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pes = DeepMDPES(DeepMDBoundary(model=args.model))

    # Prepare combined output files
    combined_pdb = output_dir / f"{output_prefix}.pdb"
    combined_out = output_dir / f"{output_prefix}.out"

    current_system = initial
    current_velocity = _target_velocity(initial, target_system, args.initial_temp)

    total_step_offset = 0
    all_results = []

    for stage_idx, k in enumerate(k_stages):
        stage_label = f"stage{stage_idx + 1}_k{k}"
        print(f"\n=== Stage {stage_idx + 1}/{len(k_stages)}: k={k} ===")

        restraints = reaction_coordinate_restraints(target_system, config, k)
        for r in restraints:
            print(f"  restraint {r.label}: atoms({r.atom_i},{r.atom_j}) target={r.target:.6f} Bohr")

        para = Para(
            a0=0.0005,
            max_step=args.steps_per_stage,
            print_step=1,
            dt=0.5,
            temp_bath=args.temp_bath,
        )

        stage_pdb = str(output_dir / f"{output_prefix}_{stage_label}.pdb")
        stage_out = str(output_dir / f"{output_prefix}_{stage_label}.out")

        bias_config = ReactionCoordinatePot(
            initial_state=current_system,
            restraints=restraints,
            para=para,
            str_output_file=stage_pdb,
            output_file=stage_out,
        )

        print(f"  starting temp={float(temperature(current_velocity, atom_masses(current_system))):.2f} K")

        result = run_rtip_nvt_md(
            bias_config, pes,
            perturb=False,
            initial_velocity=current_velocity,
            write_outputs=True,
        )

        masses = atom_masses(result.system)
        final_temp = float(temperature(result.velocity, masses))
        print(f"  final temp={final_temp:.2f} K")
        print(f"  history_steps={len(result.history)}")

        current_system = result.system
        current_velocity = result.velocity

        # Analyze stage
        if not args.skip_analysis:
            stage_analysis = analyze_generalized_job(
                pdb_file=stage_pdb,
                out_file=stage_out,
                reaction_dir=reaction_dir,
                output_dir=output_dir,
                output_prefix=f"{output_prefix}_{stage_label}",
            )
            all_results.append(stage_analysis)
            print(f"  final B-X={stage_analysis['final_metrics'].get('min_B_O_A', stage_analysis['final_metrics'].get('min_B_S_A', stage_analysis['final_metrics'].get('min_B_N_A', 'n/a'))):.4f} A")
            print(f"  best N-C={stage_analysis['best_bond_forming'].get('min_N_C_A', 'n/a')}")

        total_step_offset += args.steps_per_stage

    # Write combined summary
    summary = {
        "reaction": reaction_dir.name,
        "target": args.target,
        "k_stages": k_stages,
        "steps_per_stage": args.steps_per_stage,
        "stage_results": [
            {
                "stage": i + 1,
                "k": k,
                "final_metrics": r.get("final_metrics"),
                "best_bond_forming": r.get("best_bond_forming"),
            }
            for i, (k, r) in enumerate(zip(k_stages, all_results))
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\nProgressive RC-MD complete. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
