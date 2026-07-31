"""Analyze a formose RTIP-MD run against the paper's bond-monitoring scheme.

The paper (JACS Au 2026, 6, 922) tracks C-C, C-H, H-H and O-O bond changes
(ignoring C-O, H-O and all Ca pairs).  This script replays the PDB trajectory,
rebuilds the adjacency matrix with the Rust rules, and reports the first steps
at which each monitored bond type forms or breaks.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from rtip_jax.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, Element, atomic_radius
from rtip_jax.system import (
    dist_mat_bohr,
    get_adj_mat,
    judge_variation_of_bonding,
    split_into_mol,
)

MONITORED = {("C", "C"), ("C", "H"), ("H", "H"), ("O", "O")}
IGNORED = [
    (Element.H, Element.O), (Element.C, Element.O),
    (Element.H, Element.Ca), (Element.C, Element.Ca),
    (Element.O, Element.Ca), (Element.Ca, Element.Ca),
]


def read_pdb_frames(path: Path) -> list[tuple[list[str], np.ndarray]]:
    """Return (atom_type, coord_angstrom) frames from a multi-frame PDB."""

    frames: list[tuple[list[str], np.ndarray]] = []
    atom_type: list[str] = []
    coord: list[list[float]] = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            # Robust parse: the 8.3f PDB columns collide for |coord| >= 1000,
            # so collect signed floats by regex instead of fixed columns.
            numbers = re.findall(r"-?\d+\.\d+", line)
            if len(numbers) < 4:
                continue
            element = line[76:78].strip() or line[12:16].strip().lstrip("0123456789")
            atom_type.append(element)
            coord.append([float(numbers[1]), float(numbers[2]), float(numbers[3])])
        elif line.startswith(("ENDMDL", "END")) and atom_type:
            frames.append((atom_type, np.asarray(coord, dtype=np.float64)))
            atom_type, coord = [], []
    if atom_type:
        frames.append((atom_type, np.asarray(coord, dtype=np.float64)))
    return frames


def bond_events(frames) -> list[dict]:
    """Detect bond-variation events relative to the first frame."""

    events: list[dict] = []
    reference_adj = None
    radii_bohr: list[float] | None = None
    atom_type: list[str] | None = None
    step = 0
    for atom_type, coord_ang in frames:
        coord_bohr = coord_ang * ANGSTROM_TO_BOHR
        radii = np.asarray([atomic_radius(el) for el in atom_type], dtype=np.float64)
        dist_mat = dist_mat_bohr(coord_bohr)
        adj = get_adj_mat(tuple(Element.from_str(el) for el in atom_type), radii, dist_mat, 1.25, tuple(IGNORED))
        if reference_adj is None:
            reference_adj = adj
            radii_bohr = radii
        else:
            if judge_variation_of_bonding(radii_bohr, dist_mat, reference_adj, 1.0, 1.6):
                # Identify which monitored pair types changed.
                formed, broken = [], []
                for i in range(len(atom_type)):
                    for j in range(i + 1, len(atom_type)):
                        pair = tuple(sorted((atom_type[i], atom_type[j])))
                        if pair not in MONITORED:
                            continue
                        d = float(dist_mat[i, j])
                        rsum = float(radii[i] + radii[j])
                        if reference_adj[i, j] == -1 and d < rsum * 1.0:
                            formed.append((i, j, pair, round(d * BOHR_TO_ANGSTROM, 3)))
                        if reference_adj[i, j] == 1 and d > rsum * 1.6:
                            broken.append((i, j, pair, round(d * BOHR_TO_ANGSTROM, 3)))
                events.append({
                    "frame": step,
                    "formed": formed,
                    "broken": broken,
                })
                reference_adj = adj
        step += 1
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    frames = read_pdb_frames(args.pdb)
    events = bond_events(frames)
    rows = []
    for event in events:
        for i, j, pair, d in event["formed"]:
            rows.append([event["frame"], f"{pair[0]}-{pair[1]}", "formed", d])
        for i, j, pair, d in event["broken"]:
            rows.append([event["frame"], f"{pair[0]}-{pair[1]}", "broken", d])

    print(f"frames={len(frames)} bond_events={len(events)}")
    for row in rows[:40]:
        print(row)
    if args.output is not None:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "bond_type", "event", "distance_angstrom"])
            writer.writerows(rows)


if __name__ == "__main__":
    main()
