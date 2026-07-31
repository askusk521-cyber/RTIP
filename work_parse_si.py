"""Parse the formose paper SI (pdftotext output) into per-structure XYZ files.

Input : pdftotext -layout output of au5c01359_si_001.pdf
Output: <outdir>/species/species_<NNN>.xyz
        <outdir>/ts/ts_<NNN>.xyz
        <outdir>/manifest.csv
        <outdir>/table_S1.csv

The SI text is structured as:
  ... "Coordinates of all species and TS structures in the reaction network"
  <natom>
  i= <freq>, E = <energy Ha>
  <atom> x y z   (natom lines, coordinates in Angstrom)
  ... next species ...
  TS<k>
  <natom>
  Step = 0, E = 0.00000000
  <atom> x y z
  ... "REFERENCES"
"""

import csv
import os
import re
import sys


ATOM_RE = re.compile(
    r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)
META_RE = re.compile(r"i=\s*([-\d.]+)\s*,\s*E\s*=\s*(-?\d+\.\d+)")
TS_RE = re.compile(r"^\s*TS(\d+)\s*$")
NUM_RE = re.compile(r"^\s*(\d+)\s*$")


def _next_atom(body, i):
    """Return (atoms_tuple, new_i), skipping blank/form-feed lines."""
    guard = 0
    while i < len(body):
        line = body[i].replace("\x0c", "").strip()
        if line:
            m = ATOM_RE.match(line)
            if not m:
                raise RuntimeError(f"bad atom line: {body[i]!r}")
            return (m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))), i + 1
        i += 1
        guard += 1
        if guard > 20:
            raise RuntimeError("too many blank lines in atom block")
    raise RuntimeError("unexpected end of coordinate section")


def parse(text: str):
    lines = text.splitlines()
    start = None
    end = None
    for idx, line in enumerate(lines):
        if "Coordinates of all species and TS structures" in line:
            start = idx + 1
        if line.strip().startswith("REFERENCES"):
            end = idx
            break
    if start is None or end is None:
        raise RuntimeError("coordinate section not found")
    body = lines[start:end]

    species = []   # list of (natom, meta, atoms)
    ts = []
    i = 0
    while i < len(body):
        stripped = body[i].strip()
        m_ts = TS_RE.match(stripped)
        m_num = NUM_RE.match(stripped)
        if m_ts:
            label = int(m_ts.group(1))
            i += 1
            if i >= len(body):
                break
            natom = int(NUM_RE.match(body[i].strip()).group(1))
            i += 1
            # skip blank lines and the "Step = 0, E = ..." line
            while i < len(body) and not body[i].replace("\x0c", "").strip():
                i += 1
            i += 1
            atoms = []
            for _ in range(natom):
                atom, i = _next_atom(body, i)
                atoms.append(atom)
            ts.append((label, natom, atoms))
        elif m_num:
            # Species block: <index> <natom> then a meta line.
            i += 1
            if i < len(body) and NUM_RE.match(body[i].strip()):
                natom = int(NUM_RE.match(body[i].strip()).group(1))
                i += 1
            else:
                natom = int(m_num.group(1))
            meta = None
            if i < len(body):
                mm = META_RE.search(body[i])
                if mm:
                    meta = (float(mm.group(1)), float(mm.group(2)))
                    i += 1
            if meta is None:
                raise RuntimeError(f"missing meta line for species at line {i}: {body[i-1]!r}")
            atoms = []
            for _ in range(natom):
                atom, i = _next_atom(body, i)
                atoms.append(atom)
            species.append((len(species) + 1, natom, meta, atoms))
        else:
            i += 1
    return species, ts


def write_xyz(path: str, atoms, comment: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for el, x, y, z in atoms:
            f.write(f"{el:<3s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        text = f.read()
    species, ts = parse(text)

    sp_dir = os.path.join(outdir, "species")
    ts_dir = os.path.join(outdir, "ts")
    os.makedirs(sp_dir, exist_ok=True)
    os.makedirs(ts_dir, exist_ok=True)

    rows = []
    for idx, natom, meta, atoms in species:
        freq, energy = meta
        fname = f"species_{idx:03d}.xyz"
        write_xyz(
            os.path.join(sp_dir, fname),
            atoms,
            f"species {idx}; E = {energy:.10f} Ha; i = {freq:.0f} cm-1",
        )
        rows.append(["species", idx, natom, energy, freq, f"species/{fname}"])
    for label, natom, atoms in ts:
        fname = f"ts_{label:03d}.xyz"
        write_xyz(os.path.join(ts_dir, fname), atoms, f"TS{label}; E = 0.0 Ha")
        rows.append(["ts", label, natom, 0.0, 0.0, f"ts/{fname}"])

    with open(os.path.join(outdir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "index", "natom", "energy_ha", "i_value", "file"])
        w.writerows(rows)

    print(f"species={len(species)} ts={len(ts)} total_structures={len(rows)}")

    table_rows = parse_table_s1(text)
    if table_rows:
        with open(os.path.join(outdir, "table_S1.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "equation", "forward_barrier_kcal_mol", "reverse_barrier_kcal_mol"])
            w.writerows(table_rows)
        print(f"table_S1 rows={len(table_rows)}")
    else:
        print("table_S1: not found")


def parse_table_s1(text: str) -> list[list[str]]:
    """Parse the 28-row reaction table (Index, Chemical equation, barriers)."""
    rows = []
    in_table = False
    for line in text.splitlines():
        if "Index" in line and "Chemical equation" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if "Coordinates of all species" in line:
            break
        m = re.match(r"^\s*(\d+)\s+(.+?)\s+([-\d.]+)\s+([-\d.]+)\s*$", line)
        if m:
            rows.append([int(m.group(1)), m.group(2).strip(), float(m.group(3)), float(m.group(4))])
    return rows


if __name__ == "__main__":
    main()
