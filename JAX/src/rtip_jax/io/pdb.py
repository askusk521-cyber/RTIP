"""PDB writing helpers.

Rust source: `System::write_pdb`.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from rtip_jax.constants import BOHR_TO_ANGSTROM, element_symbol
from rtip_jax.errors import error_file
from rtip_jax.system import System


def format_pdb(system: System, step: int, *, include_title: bool = False) -> str:
    lines: list[str] = []
    if include_title:
        lines.append("TITLE     PDB file created by RTIP")
    lines.append(f"REMARK    , Step = {step:8d}, E = {system.pot:15.8f}")
    for index in range(system.natom):
        symbol = "XX" if system.atom_type is None else element_symbol(system.atom_type[index])
        coord_angstrom = system.coord[index] * BOHR_TO_ANGSTROM
        lines.append(
            f"ATOM  {index + 1:>5} {symbol:<4}              "
            f"{float(coord_angstrom[0]):8.3f}"
            f"{float(coord_angstrom[1]):8.3f}"
            f"{float(coord_angstrom[2]):8.3f}"
            f"  0.00  0.00          {symbol:>2}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_pdb(
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
            handle.write(format_pdb(system, step, include_title=create_new_file))
    except OSError as exc:
        operation = "creating" if create_new_file else "opening"
        raise OSError(error_file(operation, str(path))) from exc

