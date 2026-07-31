"""Build the formose RTIP-MD initial cell from the molecule library.

Composition (paper Section 3.1): 8 H2O + 8 CH2O + 2 Ca + 4 OH-.
Molecule geometries come from ``molecules/*.xyz`` (Angstrom).
Output is a single XYZ file in Angstrom (rtip_jax converts to Bohr on read).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
from jax import random

from rtip_jax.constants import ANGSTROM_TO_BOHR
from rtip_jax.math.rotations import random_rotation  # type: ignore[attr-defined]
from rtip_jax.system import System


SPECIES = [
    ("H2O.xyz", 8),
    ("CH2O.xyz", 8),
    ("Ca.xyz", 2),
    ("OH.xyz", 4),
]


def _read_species(molecules_dir: Path, name: str) -> System:
    return System.read_xyz(molecules_dir / name)


def build_box(
    molecules_dir: Path,
    box_side: float,
    min_dist: float,
    seed: int,
) -> System:
    key = random.PRNGKey(seed)
    placements: list[tuple[System, jnp.ndarray, jnp.ndarray]] = []  # (molecule, coord, rot)
    box_half = 0.5 * box_side * ANGSTROM_TO_BOHR
    min_dist_bohr = min_dist * ANGSTROM_TO_BOHR

    for name, count in SPECIES:
        molecule = _read_species(molecules_dir, name)
        for _ in range(count):
            key, rot_key, pos_key = random.split(key, 3)
            rotation = random_rotation(rot_key)
            accepted = False
            for _attempt in range(200):
                center = random.uniform(pos_key, (3,), minval=-box_half, maxval=box_half)
                if _no_overlap(placements, molecule, center, rotation, min_dist_bohr):
                    accepted = True
                    break
                key, pos_key = random.split(key)
            if not accepted:
                raise RuntimeError(f"could not place {name} without overlap")
            placements.append((molecule, center, rotation))

    coords: list[jnp.ndarray] = []
    atom_type: list[str] = []
    for molecule, center, rotation in placements:
        coords.append((molecule.coord @ rotation.T) + center)
        atom_type.extend(str(el.value) for el in (molecule.atom_type or ()))
    coord = jnp.concatenate(coords, axis=0)
    return System(coord=coord, atom_type=tuple(atom_type))


def _no_overlap(
    placements: list[tuple[System, jnp.ndarray, jnp.ndarray]],
    molecule: System,
    center: jnp.ndarray,
    rotation: jnp.ndarray,
    min_dist: float,
) -> bool:
    mol_coord = (molecule.coord @ rotation.T) + center
    for other, other_center, _other_rot in placements:
        other_coord = other.coord + other_center
        for a in range(mol_coord.shape[0]):
            d2 = jnp.min(jnp.sum((mol_coord[a] - other_coord) ** 2, axis=1))
            if float(d2) < min_dist * min_dist:
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecules-dir", default="molecules", type=Path)
    parser.add_argument("--output", default="formose_box.xyz", type=Path)
    parser.add_argument("--box-side", type=float, default=20.0, help="Cubic box side in Angstrom")
    parser.add_argument("--min-dist", type=float, default=2.2, help="Minimum interatomic distance in Angstrom")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    system = build_box(args.molecules_dir, args.box_side, args.min_dist, args.seed)
    natom = system.natom
    counts = {el: system.atom_type.count(el) for el in ("H", "C", "O", "Ca")}
    print(f"natom={natom} counts={counts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    system.write_xyz(str(args.output), create_new_file=True, step=0)
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
