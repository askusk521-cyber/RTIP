from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rtip_jax.config import Para
from rtip_jax.constants import BOLTZMANN, HARTREE_TO_JOULE
from rtip_jax.external import DeepMDBoundary, DeepMDPES
from rtip_jax.pes import RepulsivePot
from rtip_jax.system import System
from rtip_jax.workflows.md import atom_masses, run_rtip_nvt_md, temperature


def _reorder_reference_to_is(system: System) -> System:
    order = tuple(range(20)) + (23, 24, 28, 32, 29, 30, 31, 33, 34, 35, 25, 26, 27, 21, 20, 22)
    atom_type = None if system.atom_type is None else tuple(system.atom_type[i] for i in order)
    mutable = None if system.mutable is None else system.mutable[jnp.asarray(order)]
    return replace(
        system,
        coord=system.coord[jnp.asarray(order)],
        atom_type=atom_type,
        mutable=mutable,
    )


def _align_mobile_to_reference(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    ref_center = reference.mean(axis=0)
    mob_center = mobile.mean(axis=0)
    ref0 = reference - ref_center
    mob0 = mobile - mob_center
    u, _s, vt = np.linalg.svd(mob0.T @ ref0)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return mob0 @ rotation + ref_center


def _target_velocity(local_min: System, target: System, initial_temp: float) -> jnp.ndarray:
    reference = np.asarray(local_min.coord, dtype=np.float64)
    mobile = np.asarray(target.coord, dtype=np.float64)
    aligned = _align_mobile_to_reference(reference, mobile)
    direction = aligned - reference
    masses = np.asarray(atom_masses(local_min), dtype=np.float64)
    direction = direction - np.average(direction, axis=0, weights=masses)
    norm2 = float(np.sum(masses[:, np.newaxis] * direction * direction))
    if norm2 <= 0.0:
        raise ValueError("target direction has zero mass-weighted norm")
    twice_kinetic = initial_temp * BOLTZMANN * 3.0 * float(local_min.natom - 1) / HARTREE_TO_JOULE
    velocity = direction * np.sqrt(twice_kinetic / norm2)
    return jnp.asarray(velocity, dtype=jnp.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run guided RTIPMD for ic5c02384 1-tBu__CO2.")
    parser.add_argument("--input", default="examples/ic5c02384/reactions/1-tBu__CO2/IS.xyz")
    parser.add_argument("--target", choices=("ts", "product"), default="ts")
    parser.add_argument("--model", default="/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt")
    parser.add_argument("--output-dir", default="guided_rtipmd_1tBu_CO2")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--max-step", type=int, default=200)
    parser.add_argument("--a0", type=float, default=0.002)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--temp-bath", type=float, default=1000.0)
    parser.add_argument("--initial-temp", type=float, default=1000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix or f"guided_rtipmd_{args.target}"

    local_min = System.read_xyz(args.input)
    target_path = Path("examples/ic5c02384/reactions/1-tBu__CO2") / f"{args.target}.xyz"
    target = _reorder_reference_to_is(System.read_xyz(target_path))
    initial_velocity = _target_velocity(local_min, target, args.initial_temp)
    masses = atom_masses(local_min)

    para = Para(
        a0=args.a0,
        max_step=args.max_step,
        print_step=1,
        dt=args.dt,
        temp_bath=args.temp_bath,
    )
    config = RepulsivePot(
        local_min=local_min,
        nearby_ts=(),
        para=para,
        str_output_file=str(output_dir / f"{output_prefix}.pdb"),
        output_file=str(output_dir / f"{output_prefix}.out"),
    )
    pes = DeepMDPES(DeepMDBoundary(model=args.model))
    initial_energy, _force = pes.get_energy_force(local_min)
    print(f"input={args.input}")
    print(f"target={target_path}")
    print(f"output_dir={output_dir}")
    print(f"output_prefix={output_prefix}")
    print(f"a0={args.a0}")
    print(f"dt={args.dt}")
    print(f"temp_bath={args.temp_bath}")
    print(f"initial_temp_requested={args.initial_temp}")
    print(f"initial_temp_actual={float(temperature(initial_velocity, masses)):.8f}")
    print(f"initial_energy_Ha={float(initial_energy):.12f}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
