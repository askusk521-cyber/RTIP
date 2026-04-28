"""Roto-translationally invariant potential helpers.

Rust source: `src/pes_exploration/rtip.rs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp

from rtip_jax.system import System


def _as_coord(coord: Any, name: str = "coord") -> Any:
    coord = jnp.asarray(coord, dtype=jnp.float64)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"{name} must have shape (natom, 3), got {coord.shape}")
    if coord.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one atom")
    return coord


def _as_matching_coords(coord1: Any, coord2: Any) -> tuple[Any, Any]:
    coord1 = _as_coord(coord1, "coord1")
    coord2 = _as_coord(coord2, "coord2")
    if coord1.shape != coord2.shape:
        raise ValueError(f"coord1 and coord2 must have matching shape, got {coord1.shape} and {coord2.shape}")
    return coord1, coord2


def _center_coords(coord1: Any, coord2: Any) -> tuple[Any, Any, Any, Any]:
    coord1, coord2 = _as_matching_coords(coord1, coord2)
    origin1 = jnp.mean(coord1, axis=0)
    origin2 = jnp.mean(coord2, axis=0)
    return coord1 - origin1, coord2 - origin2, origin1, origin2


def _quaternion_system_matrix(centered1: Any, centered2: Any) -> Any:
    add = centered1 + centered2
    sub = centered1 - centered2
    zeros = jnp.zeros((centered1.shape[0],), dtype=jnp.float64)

    matrices = jnp.stack(
        [
            jnp.stack([zeros, sub[:, 0], sub[:, 1], sub[:, 2]], axis=1),
            jnp.stack([-sub[:, 0], zeros, -add[:, 2], add[:, 1]], axis=1),
            jnp.stack([-sub[:, 1], add[:, 2], zeros, -add[:, 0]], axis=1),
            jnp.stack([-sub[:, 2], -add[:, 1], add[:, 0], zeros], axis=1),
        ],
        axis=1,
    )
    return jnp.einsum("nri,nrj->ij", matrices, matrices)


def _eigh_for_coords(coord1: Any, coord2: Any) -> tuple[Any, Any, Any, Any, Any]:
    centered1, centered2, origin1, origin2 = _center_coords(coord1, coord2)
    system_matrix = _quaternion_system_matrix(centered1, centered2)
    eigvals, eigvecs = jnp.linalg.eigh(system_matrix)
    return eigvals, eigvecs, centered1, centered2, origin1, origin2


def quaternion_to_rotation(quaternion: Any) -> Any:
    """Convert Rust quaternion/eigenvector ordering to a 3x3 rotation matrix."""

    q = jnp.asarray(quaternion, dtype=jnp.float64)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {q.shape}")
    q0, q1, q2, q3 = q
    return jnp.asarray(
        [
            [
                q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
                2.0 * (q1 * q2 + q0 * q3),
                2.0 * (q1 * q3 - q0 * q2),
            ],
            [
                2.0 * (q1 * q2 - q0 * q3),
                q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
                2.0 * (q2 * q3 + q0 * q1),
            ],
            [
                2.0 * (q1 * q3 + q0 * q2),
                2.0 * (q2 * q3 - q0 * q1),
                q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
            ],
        ],
        dtype=jnp.float64,
    )


def _rotations_from_eigvecs(eigvecs: Any) -> Any:
    return jnp.stack([quaternion_to_rotation(eigvecs[:, index]) for index in range(4)], axis=0)


def rti_dist(coord1: Any, coord2: Any) -> Any:
    """Return the minimum roto-translationally invariant distance."""

    eigvals, _eigvecs, _centered1, _centered2, _origin1, _origin2 = _eigh_for_coords(coord1, coord2)
    return jnp.sqrt(eigvals[0])


def rti_dists(coord1: Any, coord2: Any) -> Any:
    """Return the four RTI distances from the quaternion eigensystem."""

    eigvals, _eigvecs, _centered1, _centered2, _origin1, _origin2 = _eigh_for_coords(coord1, coord2)
    return jnp.sqrt(eigvals)


def rti_dist_vec(coord1: Any, coord2: Any) -> tuple[Any, Any]:
    """Return minimum RTI distance and vector from aligned `coord1` to `coord2`."""

    eigvals, eigvecs, centered1, centered2, _origin1, _origin2 = _eigh_for_coords(coord1, coord2)
    rotation = quaternion_to_rotation(eigvecs[:, 0])
    vector = centered2 - centered1 @ rotation
    return jnp.sqrt(eigvals[0]), vector


def rti_dists_vecs(coord1: Any, coord2: Any) -> tuple[Any, Any]:
    """Return all four RTI distances and their corresponding vectors."""

    eigvals, eigvecs, centered1, centered2, _origin1, _origin2 = _eigh_for_coords(coord1, coord2)
    rotations = _rotations_from_eigvecs(eigvecs)
    aligned = jnp.einsum("ni,kij->knj", centered1, rotations)
    vectors = centered2[jnp.newaxis, :, :] - aligned
    return jnp.sqrt(eigvals), vectors


def rti_rot_tran(coord1: Any, coord2: Any) -> tuple[Any, Any]:
    """Return rotation and translation that best align `coord1` to `coord2`."""

    _eigvals, eigvecs, _centered1, _centered2, origin1, origin2 = _eigh_for_coords(coord1, coord2)
    rotation = quaternion_to_rotation(eigvecs[:, 0])
    translation = origin2 - origin1 @ rotation
    return rotation, translation


def rti_weight(distance: Any) -> Any:
    """Rust private `f(x) = 1 / x^7` used for RTIP distance weighting."""

    distance = jnp.asarray(distance, dtype=jnp.float64)
    return 1.0 / (distance**7)


def rti_weight_derivative(distance: Any) -> Any:
    """Derivative of `rti_weight` with respect to distance."""

    distance = jnp.asarray(distance, dtype=jnp.float64)
    return -7.0 / (distance**8)


def rti_pot(coord1: Any, coord2: Any, a: float, sigma: float) -> Any:
    """Return RTIP Gaussian potential energy."""

    distances = rti_dists(coord1, coord2)
    weights = rti_weight(distances)
    total_weight = jnp.sum(weights)
    sigma = float(sigma)
    u = float(a) * jnp.exp(-(distances * distances) / (2.0 * sigma * sigma))
    return jnp.sum((weights / total_weight) * u)


def rti_pot_force(coord1: Any, coord2: Any, a: float, sigma: float) -> tuple[Any, Any]:
    """Return RTIP Gaussian potential energy and force on `coord2`."""

    distances, vectors = rti_dists_vecs(coord1, coord2)
    weights = rti_weight(distances)
    total_weight = jnp.sum(weights)
    dweights = rti_weight_derivative(distances)
    sigma = float(sigma)
    u = float(a) * jnp.exp(-(distances * distances) / (2.0 * sigma * sigma))
    pot_terms = (weights / total_weight) * u

    weighted_u_sum = jnp.sum(weights * u)
    interactions = weighted_u_sum - u * total_weight
    coeff = pot_terms / (sigma * sigma) + dweights * interactions / (
        total_weight * total_weight * distances
    )
    force = jnp.sum(vectors * coeff[:, jnp.newaxis, jnp.newaxis], axis=0)
    return jnp.sum(pot_terms), force


def _extract_coord(system: System, indices: tuple[int, ...] | None) -> Any:
    if indices is None:
        return system.coord
    return system.coord[jnp.asarray(indices)]


@dataclass(frozen=True)
class Rtip0PES:
    """RTIP repulsive potential-energy surface.

    This mirrors Rust `Rtip0PES`, including the four combinations of reference
    and current-system `atom_add_pot` fragment selection.
    """

    local_min: System
    nearby_ts: Sequence[System]
    a_min: float
    a_ts: float
    sigma_min: float
    sigma_ts: Sequence[float]

    def __post_init__(self) -> None:
        nearby_ts = tuple(self.nearby_ts)
        sigma_ts = tuple(float(value) for value in self.sigma_ts)
        if len(nearby_ts) != len(sigma_ts):
            raise ValueError("nearby_ts and sigma_ts must have the same length")
        object.__setattr__(self, "nearby_ts", nearby_ts)
        object.__setattr__(self, "a_min", float(self.a_min))
        object.__setattr__(self, "a_ts", float(self.a_ts))
        object.__setattr__(self, "sigma_min", float(self.sigma_min))
        object.__setattr__(self, "sigma_ts", sigma_ts)

    @property
    def _reference_indices(self) -> tuple[int, ...] | None:
        return self.local_min.atom_add_pot

    def _reference_coords(self) -> tuple[Any, tuple[Any, ...]]:
        indices = self._reference_indices
        local_min_coord = _extract_coord(self.local_min, indices)
        nearby_ts_coord = tuple(_extract_coord(system, indices) for system in self.nearby_ts)
        return local_min_coord, nearby_ts_coord

    def _target_coord(self, system: System) -> tuple[Any, tuple[int, ...] | None]:
        if system.atom_add_pot is None:
            return system.coord, None
        indices = tuple(system.atom_add_pot)
        return _extract_coord(system, indices), indices

    def get_energy(self, system: System) -> float:
        target_coord, _target_indices = self._target_coord(system)
        local_min_coord, nearby_ts_coord = self._reference_coords()
        energy = rti_pot(local_min_coord, target_coord, self.a_min, self.sigma_min)
        for ts_coord, sigma in zip(nearby_ts_coord, self.sigma_ts, strict=True):
            energy = energy + rti_pot(ts_coord, target_coord, self.a_ts, sigma)
        return float(energy)

    def get_energy_force(self, system: System) -> tuple[float, Any]:
        target_coord, target_indices = self._target_coord(system)
        local_min_coord, nearby_ts_coord = self._reference_coords()
        energy, force_fragment = rti_pot_force(local_min_coord, target_coord, self.a_min, self.sigma_min)
        for ts_coord, sigma in zip(nearby_ts_coord, self.sigma_ts, strict=True):
            ts_energy, ts_force = rti_pot_force(ts_coord, target_coord, self.a_ts, sigma)
            energy = energy + ts_energy
            force_fragment = force_fragment + ts_force

        if target_indices is None:
            return float(energy), force_fragment

        force = jnp.zeros_like(system.coord)
        force = force.at[jnp.asarray(target_indices)].set(force_fragment)
        return float(energy), force


__all__ = [
    "Rtip0PES",
    "quaternion_to_rotation",
    "rti_dist",
    "rti_dist_vec",
    "rti_dists",
    "rti_dists_vecs",
    "rti_pot",
    "rti_pot_force",
    "rti_rot_tran",
    "rti_weight",
    "rti_weight_derivative",
]

