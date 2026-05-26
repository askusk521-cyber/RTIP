"""System data structure.

Rust source: `src/pes_exploration/system.rs`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from os import PathLike
from typing import Any

import jax.numpy as jnp

from .constants import Element, coerce_element


@dataclass(frozen=True)
class System:
    """Atomic system state used by RTIP workflows.

    Coordinates are stored internally in Bohr, matching the Rust code and CP2K
    boundary. Text IO helpers handle Angstrom conversion at the boundary.
    """

    coord: Any
    cell: Any | None = None
    atom_type: tuple[Element, ...] | None = None
    atom_add_pot: tuple[int, ...] | None = None
    mutable: Any | None = None
    pot: float = 0.0

    def __post_init__(self) -> None:
        coord = jnp.asarray(self.coord, dtype=jnp.float64)
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(f"coord must have shape (natom, 3), got {coord.shape}")
        object.__setattr__(self, "coord", coord)

        if self.cell is not None:
            cell = jnp.asarray(self.cell, dtype=jnp.float64)
            if cell.shape != (3, 3):
                raise ValueError(f"cell must have shape (3, 3), got {cell.shape}")
            object.__setattr__(self, "cell", cell)

        if self.mutable is not None:
            mutable = jnp.asarray(self.mutable, dtype=bool)
            if mutable.shape != coord.shape:
                raise ValueError(f"mutable must have shape {coord.shape}, got {mutable.shape}")
            object.__setattr__(self, "mutable", mutable)

        if self.atom_type is not None:
            atom_type = tuple(coerce_element(element) for element in self.atom_type)
            if len(atom_type) != coord.shape[0]:
                raise ValueError("atom_type length must match natom")
            object.__setattr__(self, "atom_type", atom_type)

        if self.atom_add_pot is not None:
            atom_add_pot = tuple(int(index) for index in self.atom_add_pot)
            if any(index < 0 or index >= coord.shape[0] for index in atom_add_pot):
                raise ValueError("atom_add_pot contains an index outside the system")
            object.__setattr__(self, "atom_add_pot", atom_add_pot)

    @property
    def natom(self) -> int:
        return int(self.coord.shape[0])

    def with_coord(self, coord: Any) -> "System":
        return replace(self, coord=coord)

    def with_pot(self, pot: float) -> "System":
        return replace(self, pot=float(pot))

    def fragment(self, indices: tuple[int, ...] | list[int]) -> "System":
        indices_tuple = tuple(int(index) for index in indices)
        atom_type = None
        if self.atom_type is not None:
            atom_type = tuple(self.atom_type[index] for index in indices_tuple)
        mutable = None
        if self.mutable is not None:
            mutable = self.mutable[jnp.asarray(indices_tuple)]
        return System(
            coord=self.coord[jnp.asarray(indices_tuple)],
            cell=None,
            atom_type=atom_type,
            atom_add_pot=None,
            mutable=mutable,
            pot=self.pot,
        )

    @classmethod
    def read_xyz(cls, filename: str | PathLike[str]) -> "System":
        from .io.xyz import read_xyz

        return read_xyz(filename)

    def write_xyz(self, filename: str | PathLike[str], create_new_file: bool, step: int) -> None:
        from .io.xyz import write_xyz

        write_xyz(self, filename, create_new_file=create_new_file, step=step)

    def write_pdb(self, filename: str | PathLike[str], create_new_file: bool, step: int) -> None:
        from .io.pdb import write_pdb

        write_pdb(self, filename, create_new_file=create_new_file, step=step)

