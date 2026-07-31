"""System data structure.

Rust source: `src/pes_exploration/system.rs`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from os import PathLike
from typing import Any

import jax.numpy as jnp

from .constants import Element, atomic_radius, coerce_element


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


# ---------------------------------------------------------------------------
# Bond-connectivity helpers (Rust `System::{get_dist_mat, split_into_mol,
# get_adj_mat, judge_adj_of_mol, judge_variation_of_bonding}`).
#
# All distances are in Bohr, matching `System.coord`.  Pure NumPy
# implementations of the Rust algorithms; used by the evolution MD loop.
# ---------------------------------------------------------------------------


def dist_mat_bohr(coord: Any) -> Any:
    """Return the pairwise distance matrix (Bohr) of a coordinate array."""

    import numpy as np

    coord_np = np.asarray(coord, dtype=np.float64)
    natom = coord_np.shape[0]
    dist_mat = np.zeros((natom, natom), dtype=np.float64)
    for i in range(natom - 1):
        diff = coord_np[i + 1:] - coord_np[i]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        dist_mat[i, i + 1:] = d
        dist_mat[i + 1:, i] = d
    return dist_mat


def split_into_mol(
    atomic_radii: Any,
    dist_mat: Any,
    transition_multiple: float,
) -> tuple[tuple[int, ...], ...]:
    """Split atoms into molecules by the transition-multiple radius criterion.

    Atoms i, j belong to the same molecule when
    ``dist[i, j] < (r_i + r_j) * transition_multiple``.
    """

    import numpy as np

    radii = np.asarray(atomic_radii, dtype=np.float64)
    dist_mat_np = np.asarray(dist_mat, dtype=np.float64)
    untreated = list(range(radii.size))
    molecules: list[tuple[int, ...]] = []
    while untreated:
        index = [untreated.pop(0)]
        n = 0
        while n < len(index):
            m = 0
            while m < len(untreated):
                if dist_mat_np[index[n], untreated[m]] < (radii[index[n]] + radii[untreated[m]]) * transition_multiple:
                    index.append(untreated.pop(m))
                else:
                    m += 1
            n += 1
        molecules.append(tuple(index))
    return tuple(molecules)


def get_adj_mat(
    atom_type: tuple[Element, ...],
    atomic_radii: Any,
    dist_mat: Any,
    transition_multiple: float,
    ignored_pair: tuple[tuple[Element, Element], ...] = (),
) -> Any:
    """Return the adjacency matrix: 1 bonded, -1 unbonded, 0 ignored pair."""

    import numpy as np

    radii = np.asarray(atomic_radii, dtype=np.float64)
    dist_mat_np = np.asarray(dist_mat, dtype=np.float64)
    natom = radii.size
    adj_mat = np.zeros((natom, natom), dtype=np.int8)
    ignored = {tuple(sorted((coerce_element(a), coerce_element(b)))) for a, b in ignored_pair}
    for i in range(natom - 1):
        for j in range(i + 1, natom):
            pair = tuple(sorted((atom_type[i], atom_type[j])))
            if pair in ignored:
                continue
            if dist_mat_np[i, j] < (radii[i] + radii[j]) * transition_multiple:
                adj_mat[i, j] = 1
                adj_mat[j, i] = 1
            else:
                adj_mat[i, j] = -1
                adj_mat[j, i] = -1
    return adj_mat


def judge_adj_of_mol(
    mol_index: tuple[tuple[int, ...], ...],
    atomic_radii: Any,
    adj_mat: Any,
    dist_mat: Any,
    multiple: float,
) -> bool:
    """Return True when two molecules are closer than the radius criterion."""

    import numpy as np

    radii = np.asarray(atomic_radii, dtype=np.float64)
    adj_mat_np = np.asarray(adj_mat, dtype=np.int8)
    dist_mat_np = np.asarray(dist_mat, dtype=np.float64)
    if len(mol_index) == 1:
        return True
    for i in range(len(mol_index) - 1):
        for j in range(i + 1, len(mol_index)):
            for n in mol_index[i]:
                for m in mol_index[j]:
                    if (
                        adj_mat_np[n, m] == -1
                        and dist_mat_np[n, m] < (radii[n] + radii[m]) * multiple
                    ):
                        return True
    return False


def judge_variation_of_bonding(
    atomic_radii: Any,
    dist_mat: Any,
    adj_mat: Any,
    bonded_multiple: float,
    unbonded_multiple: float,
) -> bool:
    """Return True when the bonding pattern changed with respect to `adj_mat`.

    A bonded pair (1) breaking requires ``dist > (r_i + r_j) * unbonded_multiple``;
    an unbonded pair (-1) forming requires ``dist < (r_i + r_j) * bonded_multiple``.
    """

    import numpy as np

    radii = np.asarray(atomic_radii, dtype=np.float64)
    dist_mat_np = np.asarray(dist_mat, dtype=np.float64)
    adj_mat_np = np.asarray(adj_mat, dtype=np.int8)
    natom = radii.size
    for i in range(natom - 1):
        for j in range(i + 1, natom):
            if adj_mat_np[i, j] == 1:
                if dist_mat_np[i, j] > (radii[i] + radii[j]) * unbonded_multiple:
                    return True
            elif adj_mat_np[i, j] == -1:
                if dist_mat_np[i, j] < (radii[i] + radii[j]) * bonded_multiple:
                    return True
    return False
