"""Molecule synthesis layout helpers.

Rust source: `src/pes_exploration/synthesis.rs`.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp
from jax import random

from rtip_jax.math.rotations import random_rotation
from rtip_jax.system import System


def _normalize_mol_index(system: System, mol_index: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    normalized = tuple(tuple(int(index) for index in molecule) for molecule in mol_index)
    if len(normalized) not in (2, 3, 4):
        raise ValueError("synthesis supports exactly 2, 3, or 4 molecules")
    for molecule in normalized:
        if len(molecule) == 0:
            raise ValueError("mol_index cannot contain an empty molecule")
        if any(index < 0 or index >= system.natom for index in molecule):
            raise ValueError("mol_index contains an atom index outside the system")
    return normalized


def synthesis_offsets(n_molecules: int, dist: float) -> Any:
    """Return molecule-center offsets used by the Rust synthesis routine."""

    dist = float(dist)
    if n_molecules == 2:
        return jnp.asarray([[-dist, 0.0, 0.0], [dist, 0.0, 0.0]], dtype=jnp.float64)
    if n_molecules == 3:
        return jnp.asarray(
            [
                [0.0, dist, 0.0],
                [dist * 0.866025403784439, -dist * 0.5, 0.0],
                [-dist * 0.866025403784439, -dist * 0.5, 0.0],
            ],
            dtype=jnp.float64,
        )
    if n_molecules == 4:
        return jnp.asarray(
            [
                [0.0, 0.0, dist],
                [0.0, dist * 0.942809041582063, -dist * 0.333333333333333],
                [dist * 0.816496580927726, -dist * 0.471404520791032, -dist * 0.333333333333333],
                [-dist * 0.816496580927726, -dist * 0.471404520791032, -dist * 0.333333333333333],
            ],
            dtype=jnp.float64,
        )
    raise ValueError("synthesis supports exactly 2, 3, or 4 molecules")


def _rotations_from_key(key: Any, n_molecules: int) -> tuple[Any, ...]:
    keys = random.split(key, n_molecules)
    return tuple(random_rotation(subkey) for subkey in keys)


def synthesize_layout(
    system: System,
    mol_index: Sequence[Sequence[int]],
    dist: float,
    *,
    key: Any | None = None,
    rotations: Sequence[Any] | None = None,
) -> System:
    """Return a new system with separated, randomly rotated molecule groups.

    Pass `key` for Rust-like random rotations, or `rotations` for deterministic
    tests and reproducible callers. Exactly one of them must be provided.
    """

    mol_index = _normalize_mol_index(system, mol_index)
    n_molecules = len(mol_index)
    if (key is None) == (rotations is None):
        raise ValueError("provide exactly one of key or rotations")
    if rotations is None:
        rotations = _rotations_from_key(key, n_molecules)
    rotations = tuple(jnp.asarray(rotation, dtype=jnp.float64) for rotation in rotations)
    if len(rotations) != n_molecules:
        raise ValueError("number of rotations must match number of molecules")
    for rotation in rotations:
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")

    offsets = synthesis_offsets(n_molecules, dist)
    coord = system.coord
    for molecule_index, molecule in enumerate(mol_index):
        indices = jnp.asarray(molecule)
        molecule_coord = system.coord[indices]
        centered = molecule_coord - jnp.mean(molecule_coord, axis=0)
        placed = centered @ rotations[molecule_index] + offsets[molecule_index]
        coord = coord.at[indices].set(placed)
    return system.with_coord(coord)


def flattened_mol_indices(system: System, mol_index: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Return molecule indices flattened in Rust synthesis-loop order."""

    normalized = _normalize_mol_index(system, mol_index)
    return tuple(index for molecule in normalized for index in molecule)


def synthesis_target_state(system: System, mol_index: Sequence[Sequence[int]]) -> tuple[System, tuple[int, ...] | None]:
    """Return the per-step synthesis RTIP target.

    The target places each selected molecule at the common geometric center.
    If the selected molecules cover the whole system, the returned target has
    the original atom ordering and no active-index subset. Otherwise, the target
    contains only the selected molecules in flattened `mol_index` order and the
    active indices are returned for force scattering onto the full system.
    """

    normalized = _normalize_mol_index(system, mol_index)
    flat = tuple(index for molecule in normalized for index in molecule)
    covers_whole_system = len(flat) == system.natom and set(flat) == set(range(system.natom))

    if covers_whole_system:
        coord = system.coord
        origin_all = jnp.mean(coord, axis=0)
        for molecule in normalized:
            idx = jnp.asarray(molecule)
            molecule_coord = system.coord[idx]
            origin_i = jnp.mean(molecule_coord, axis=0)
            coord = coord.at[idx].set(molecule_coord - origin_i + origin_all)
        return system.with_coord(coord), None

    selected_coord = system.coord[jnp.asarray(flat)]
    origin_all = jnp.mean(selected_coord, axis=0)
    final_coord = jnp.zeros_like(selected_coord)
    offset = 0
    for molecule in normalized:
        idx = jnp.asarray(molecule)
        molecule_coord = system.coord[idx]
        origin_i = jnp.mean(molecule_coord, axis=0)
        moved = molecule_coord - origin_i + origin_all
        n_atom = len(molecule)
        final_coord = final_coord.at[offset : offset + n_atom].set(moved)
        offset += n_atom
    return System(coord=final_coord, pot=system.pot), flat


synthesis = synthesize_layout


__all__ = [
    "flattened_mol_indices",
    "synthesis",
    "synthesis_offsets",
    "synthesis_target_state",
    "synthesize_layout",
]
