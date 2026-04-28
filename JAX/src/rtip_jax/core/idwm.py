"""Interatomic distance weighted metric.

Rust source: `src/pes_exploration/idwm.rs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp

from rtip_jax.system import System

IDWM_WEIGHT_N = 5
IDWM_WEIGHT_SIGMA = 3.0


def weight(distance: Any) -> Any:
    """Rust `w(x) = exp(-(x / 3)^5) + 1`."""

    distance = jnp.asarray(distance, dtype=jnp.float64)
    return jnp.exp(-((distance / IDWM_WEIGHT_SIGMA) ** IDWM_WEIGHT_N)) + 1.0


def weight_derivative(distance: Any) -> Any:
    """Derivative of `weight` with respect to distance."""

    distance = jnp.asarray(distance, dtype=jnp.float64)
    return jnp.exp(-((distance / IDWM_WEIGHT_SIGMA) ** IDWM_WEIGHT_N)) * (
        -float(IDWM_WEIGHT_N) * (distance ** (IDWM_WEIGHT_N - 1)) / (IDWM_WEIGHT_SIGMA**IDWM_WEIGHT_N)
    )


def _as_coord(coord: Any) -> Any:
    coord = jnp.asarray(coord, dtype=jnp.float64)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"coord must have shape (natom, 3), got {coord.shape}")
    return coord


def _as_square_matrix(matrix: Any, natom: int | None = None) -> Any:
    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must have shape (natom, natom), got {matrix.shape}")
    if natom is not None and matrix.shape[0] != natom:
        raise ValueError(f"matrix shape {matrix.shape} does not match natom={natom}")
    return matrix


def _upper_mask(natom: int) -> Any:
    return jnp.triu(jnp.ones((natom, natom), dtype=bool), k=1)


def _pairwise_vectors(coord: Any) -> Any:
    return coord[:, None, :] - coord[None, :, :]


def _pairwise_distances(coord: Any) -> Any:
    return jnp.linalg.norm(_pairwise_vectors(coord), axis=-1)


def wei_dist_mat(coord: Any) -> Any:
    """Return the upper-triangular weighted distance matrix.

    The lower triangle and diagonal are zero, matching the Rust implementation.
    """

    coord = _as_coord(coord)
    natom = coord.shape[0]
    mask = _upper_mask(natom)
    return jnp.where(mask, weight(_pairwise_distances(coord)), 0.0)


def idw_dist(w_mat0: Any, coord: Any) -> Any:
    """Return IDW distance between a reference weighted matrix and coordinates."""

    coord = _as_coord(coord)
    w_mat0 = _as_square_matrix(w_mat0, coord.shape[0])
    mask = _upper_mask(coord.shape[0])
    current = wei_dist_mat(coord)
    return jnp.sqrt(jnp.sum(jnp.where(mask, (current - w_mat0) ** 2, 0.0)))


def idw_dist1(w_mat0: Any, w_mat1: Any) -> Any:
    """Return IDW distance between two weighted distance matrices."""

    w_mat0 = _as_square_matrix(w_mat0)
    w_mat1 = _as_square_matrix(w_mat1, w_mat0.shape[0])
    mask = _upper_mask(w_mat0.shape[0])
    return jnp.sqrt(jnp.sum(jnp.where(mask, (w_mat1 - w_mat0) ** 2, 0.0)))


def idw_dist_vec(w_mat0: Any, coord: Any) -> tuple[Any, Any]:
    """Return IDW distance and the corresponding coordinate vector.

    This preserves Rust's singular behavior: if the IDW distance is zero, the
    final normalization produces NaN values instead of applying an epsilon.
    """

    coord = _as_coord(coord)
    w_mat0 = _as_square_matrix(w_mat0, coord.shape[0])
    natom = coord.shape[0]
    mask = _upper_mask(natom)

    vec_mat = _pairwise_vectors(coord)
    dist_mat = _pairwise_distances(coord)
    w_mat = jnp.where(mask, weight(dist_mat), 0.0)
    dw_mat = jnp.where(mask, weight_derivative(dist_mat), 0.0)
    delta = jnp.where(mask, w_mat - w_mat0, 0.0)
    dist = jnp.sqrt(jnp.sum(delta**2))

    safe_dist_mat = jnp.where(mask, dist_mat, 1.0)
    coeff = jnp.where(mask, delta * dw_mat / safe_dist_mat, 0.0)
    pair_vec = coeff[:, :, None] * vec_mat
    vec = pair_vec.sum(axis=1) - pair_vec.sum(axis=0)
    vec = vec / dist

    return dist, vec


def idw_pot(w_mat0: Any, coord: Any, a: float, sigma: float) -> Any:
    """Return IDWM Gaussian potential energy."""

    dist = idw_dist(w_mat0, coord)
    return float(a) * jnp.exp(-(dist * dist) / (2.0 * float(sigma) * float(sigma)))


def idw_pot_force(w_mat0: Any, coord: Any, a: float, sigma: float) -> tuple[Any, Any]:
    """Return IDWM Gaussian potential energy and force."""

    dist, vec = idw_dist_vec(w_mat0, coord)
    pot = float(a) * jnp.exp(-(dist * dist) / (2.0 * float(sigma) * float(sigma)))
    force = vec * pot * dist / (float(sigma) * float(sigma))
    return pot, force


@dataclass(frozen=True)
class Idwm0PES:
    """IDWM repulsive potential-energy surface.

    `local_min` and `nearby_ts` are weighted distance matrices, not coordinates,
    matching Rust `Idwm0PES`.
    """

    local_min: Any
    nearby_ts: Sequence[Any]
    a_min: float
    a_ts: float
    sigma_min: float
    sigma_ts: Sequence[float]

    def __post_init__(self) -> None:
        local_min = _as_square_matrix(self.local_min)
        nearby_ts = tuple(_as_square_matrix(item, local_min.shape[0]) for item in self.nearby_ts)
        sigma_ts = tuple(float(value) for value in self.sigma_ts)
        if len(nearby_ts) != len(sigma_ts):
            raise ValueError("nearby_ts and sigma_ts must have the same length")
        object.__setattr__(self, "local_min", local_min)
        object.__setattr__(self, "nearby_ts", nearby_ts)
        object.__setattr__(self, "a_min", float(self.a_min))
        object.__setattr__(self, "a_ts", float(self.a_ts))
        object.__setattr__(self, "sigma_min", float(self.sigma_min))
        object.__setattr__(self, "sigma_ts", sigma_ts)

    def _active_coord(self, system: System) -> tuple[Any, tuple[int, ...] | None]:
        if system.atom_add_pot is None:
            return system.coord, None
        indices = tuple(system.atom_add_pot)
        return system.coord[jnp.asarray(indices)], indices

    def get_energy(self, system: System) -> float:
        coord, _indices = self._active_coord(system)
        energy = idw_pot(self.local_min, coord, self.a_min, self.sigma_min)
        for ts_matrix, sigma in zip(self.nearby_ts, self.sigma_ts, strict=True):
            energy = energy + idw_pot(ts_matrix, coord, self.a_ts, sigma)
        return float(energy)

    def get_energy_force(self, system: System) -> tuple[float, Any]:
        coord, indices = self._active_coord(system)
        energy, force_fragment = idw_pot_force(self.local_min, coord, self.a_min, self.sigma_min)
        for ts_matrix, sigma in zip(self.nearby_ts, self.sigma_ts, strict=True):
            ts_energy, ts_force = idw_pot_force(ts_matrix, coord, self.a_ts, sigma)
            energy = energy + ts_energy
            force_fragment = force_fragment + ts_force

        if indices is None:
            return float(energy), force_fragment

        force = jnp.zeros_like(system.coord)
        force = force.at[jnp.asarray(indices)].set(force_fragment)
        return float(energy), force


__all__ = [
    "IDWM_WEIGHT_N",
    "IDWM_WEIGHT_SIGMA",
    "Idwm0PES",
    "idw_dist",
    "idw_dist1",
    "idw_dist_vec",
    "idw_pot",
    "idw_pot_force",
    "wei_dist_mat",
    "weight",
    "weight_derivative",
]

