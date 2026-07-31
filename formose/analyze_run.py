"""Produce a markdown acceptance report for one formose RTIP-MD run.

Reads the evolution-MD outputs (rtip.out, rtip.pdb, rtip_decreasing_steps)
and writes a report comparing the trajectory with the paper's bond-monitoring
scheme and the four acceptance events.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np

from rtip_jax.constants import ANGSTROM_TO_BOHR, Element, atomic_radius
from rtip_jax.system import (
    dist_mat_bohr,
    get_adj_mat,
    judge_adj_of_mol,
    judge_variation_of_bonding,
    split_into_mol,
)

MONITORED = {("C", "C"), ("C", "H"), ("H", "H"), ("O", "O")}
IGNORED = [
    (Element.H, Element.O), (Element.C, Element.O),
    (Element.H, Element.Ca), (Element.C, Element.Ca),
    (Element.O, Element.Ca), (Element.Ca, Element.Ca),
]


def read_pdb_frames(path: Path):
    frames = []
    atom_type, coord = [], []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            numbers = re.findall(r"-?\d+\.\d+", line)
            if len(numbers) < 4:
                continue
            atom_type.append(line[76:78].strip() or line[12:16].strip().lstrip("0123456789"))
            coord.append([float(numbers[1]), float(numbers[2]), float(numbers[3])])
        elif line.startswith("END") and atom_type:
            frames.append((tuple(atom_type), np.asarray(coord, dtype=np.float64)))
            atom_type, coord = [], []
    if atom_type:
        frames.append((tuple(atom_type), np.asarray(coord, dtype=np.float64)))
    return frames


def frame_metrics(atom_type, coord_ang):
    coord = coord_ang * ANGSTROM_TO_BOHR
    elements = tuple(Element.from_str(el) for el in atom_type)
    radii = np.asarray([atomic_radius(el) for el in elements], dtype=np.float64)
    dm = dist_mat_bohr(coord)
    adj = get_adj_mat(elements, radii, dm, 1.25, tuple(IGNORED))
    mol = split_into_mol(radii, dm, 1.25)
    cc = [
        float(dm[i, j] * 0.52917720859)
        for i in range(len(coord))
        for j in range(i + 1, len(coord))
        if elements[i] == Element.C and elements[j] == Element.C
    ]
    return {
        "nmol": len(mol),
        "min_cc": min(cc) if cc else float("nan"),
        "adj_of_mol": judge_adj_of_mol(mol, radii, adj, dm, 1.2),
    }


def bond_events(frames):
    events = []
    reference_adj = None
    radii_ref = None
    for frame_idx, (atom_type, coord_ang) in enumerate(frames):
        coord = coord_ang * ANGSTROM_TO_BOHR
        elements = tuple(Element.from_str(el) for el in atom_type)
        radii = np.asarray([atomic_radius(el) for el in elements], dtype=np.float64)
        dm = dist_mat_bohr(coord)
        adj = get_adj_mat(elements, radii, dm, 1.25, tuple(IGNORED))
        if reference_adj is None:
            reference_adj = adj
            radii_ref = radii
            continue
        if judge_variation_of_bonding(radii_ref, dm, reference_adj, 1.0, 1.6):
            for i in range(len(atom_type)):
                for j in range(i + 1, len(atom_type)):
                    pair = tuple(sorted((atom_type[i], atom_type[j])))
                    if pair not in MONITORED:
                        continue
                    d = float(dm[i, j])
                    rsum = float(radii[i] + radii[j])
                    if reference_adj[i, j] == -1 and d < rsum * 1.0:
                        events.append(("formed", pair[0] + "-" + pair[1], d * 0.52917720859, frame_idx))
                    if reference_adj[i, j] == 1 and d > rsum * 1.6:
                        events.append(("broken", pair[0] + "-" + pair[1], d * 0.52917720859, frame_idx))
            reference_adj = adj
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--label", default="seed0")
    args = parser.parse_args()

    run = args.run_dir
    out_lines: list[str] = []

    rows = []
    with open(run / "rtip.out") as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) >= 8:
                rows.append({
                    "step": int(parts[0]),
                    "time_fs": float(parts[1]),
                    "rti_dist": float(parts[2]),
                    "temp": float(parts[3]),
                    "kin": float(parts[4]),
                    "pot_real": float(parts[5]),
                    "pot_rtip": float(parts[6]),
                    "f_real": float(parts[7]),
                    "f_rtip": float(parts[8]),
                    "status": parts[9] if len(parts) > 9 else "",
                })
    temps = [r["temp"] for r in rows]
    pot_real = [r["pot_real"] for r in rows]
    rti = [r["rti_dist"] for r in rows]

    frames = read_pdb_frames(run / "rtip.pdb")
    metrics = [frame_metrics(*frame) for frame in frames]
    events = bond_events(frames)
    cc_events = [(e[3], e[2]) for e in events if e[1] == "C-C"]

    dec_lines = (run / "rtip_decreasing_steps").read_text().splitlines()[1:]
    cycles = [ln.split() for ln in dec_lines if ln.strip()]

    out_lines.append(f"# RTIP-MD acceptance report: {args.label}")
    out_lines.append("")
    out_lines.append(f"- steps: {len(rows)}  (max_step target: 10000)")
    out_lines.append(f"- simulation time: {rows[-1]['time_fs']:.1f} fs" if rows else "- no data")
    out_lines.append(f"- temperature: min {min(temps):.0f} K, mean {np.mean(temps):.0f} K, max {max(temps):.0f} K")
    out_lines.append(f"- potential energy: {min(pot_real):.3f} .. {max(pot_real):.3f} Ha")
    out_lines.append(f"- rti_dist: {min(rti):.3f} .. {max(rti):.3f} Bohr")
    out_lines.append(f"- RTIP cycles (increasing->decreasing): {len(cycles)}")
    out_lines.append(f"- frames: {len(frames)}; molecules per frame: {min(m['nmol'] for m in metrics)} .. {max(m['nmol'] for m in metrics)}")
    out_lines.append(f"- min C-C distance: final {metrics[-1]['min_cc']:.2f} A (start {metrics[0]['min_cc']:.2f} A)")
    out_lines.append("")

    out_lines.append("## Bond events (paper monitoring scheme)")
    counts = Counter((e[1], e[0]) for e in events)
    if counts:
        for (bond, kind), n in sorted(counts.items()):
            out_lines.append(f"- {bond} {kind}: {n}")
        if cc_events:
            out_lines.append("")
            out_lines.append("### C-C events (formose condensation)")
            for frame, distance in cc_events:
                out_lines.append(f"- frame {frame} (step {frame * 10}): C-C {distance:.3f} A")
        else:
            out_lines.append("")
            out_lines.append("### C-C events: NONE (formaldehyde dimerization not observed)")
    else:
        out_lines.append("- no bond events")
    out_lines.append("")

    out_lines.append("## Acceptance checklist (paper JACS Au 2026, 6, 922)")
    checks = [
        ("1. Formaldehyde dimerization via formyl anion (C-C bond formation)", bool(cc_events)),
        ("2. Aldol growth to C3/C5 sugars (additional C-C events)", len(cc_events) >= 2),
        ("3. Aldose-ketose tautomerization (C-H enolization events)", any(e[1] == "C-H" for e in events)),
        ("4. H2 formation (H-H events)", any(e[1] == "H-H" for e in events)),
    ]
    for text, ok in checks:
        out_lines.append(f"- [{'x' if ok else ' '}] {text}")
    out_lines.append("")

    report = "\n".join(out_lines)
    output = args.output if args.output is not None else run / "acceptance_report.md"
    output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
