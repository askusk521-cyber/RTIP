"""XYZ reading and writing.

Rust source: `System::read_xyz` and `System::write_xyz`.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import jax.numpy as jnp

from rtip_jax.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, Element, element_symbol
from rtip_jax.errors import InputFormatError, error_file, error_read_xyz
from rtip_jax.system import System


def read_xyz(filename: str | PathLike[str]) -> System:
    path = Path(filename)
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise OSError(error_file("reading", str(path))) from exc

    try:
        natom = int(lines[0].strip())
    except (IndexError, ValueError) as exc:
        raise InputFormatError(error_read_xyz(str(path))) from exc

    atom_type: list[Element] = []
    coord_angstrom: list[list[float]] = []
    atom_lines = lines[2 : 2 + natom]
    if len(atom_lines) != natom:
        raise InputFormatError(error_read_xyz(str(path)))

    for line in atom_lines:
        fields = line.split()
        if len(fields) < 4:
            raise InputFormatError(error_read_xyz(str(path)))
        atom_type.append(Element.from_str(fields[0]))
        try:
            coord_angstrom.append([float(fields[1]), float(fields[2]), float(fields[3])])
        except ValueError as exc:
            raise InputFormatError(error_read_xyz(str(path))) from exc

    coord = jnp.asarray(coord_angstrom, dtype=jnp.float64) * ANGSTROM_TO_BOHR
    return System(coord=coord, atom_type=tuple(atom_type), pot=0.0)


def format_xyz(system: System, step: int) -> str:
    lines = [
        f"{system.natom:8d}",
        f"Step = {step:8d}, E = {system.pot:15.8f}",
    ]
    for index in range(system.natom):
        symbol = "XX" if system.atom_type is None else element_symbol(system.atom_type[index])
        coord_angstrom = system.coord[index] * BOHR_TO_ANGSTROM
        lines.append(
            f"{symbol:>4} "
            f"{float(coord_angstrom[0]):20.10f} "
            f"{float(coord_angstrom[1]):20.10f} "
            f"{float(coord_angstrom[2]):20.10f}"
        )
    return "\n".join(lines) + "\n"


def write_xyz(
    system: System,
    filename: str | PathLike[str],
    *,
    create_new_file: bool,
    step: int,
) -> None:
    path = Path(filename)
    mode = "w" if create_new_file else "a"
    try:
        with path.open(mode) as handle:
            handle.write(format_xyz(system, step))
    except OSError as exc:
        operation = "creating" if create_new_file else "opening"
        raise OSError(error_file(operation, str(path))) from exc
