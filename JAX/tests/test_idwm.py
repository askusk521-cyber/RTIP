from __future__ import annotations

import jax.numpy as jnp

from rtip_jax.core.idwm import (
    Idwm0PES,
    idw_dist,
    idw_dist1,
    idw_pot,
    idw_pot_force,
    wei_dist_mat,
    weight,
    weight_derivative,
)
from rtip_jax.system import System


def test_weight_and_derivative_match_rust_formula() -> None:
    x = 2.0

    assert jnp.allclose(weight(x), jnp.exp(-((x / 3.0) ** 5)) + 1.0)
    assert jnp.allclose(
        weight_derivative(x),
        jnp.exp(-((x / 3.0) ** 5)) * (-5.0 * x**4 / 3.0**5),
    )


def test_wei_dist_mat_uses_upper_triangle_only() -> None:
    coord = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=jnp.float64,
    )

    matrix = wei_dist_mat(coord)

    expected = jnp.asarray(
        [
            [0.0, weight(1.0), weight(2.0)],
            [0.0, 0.0, weight(jnp.sqrt(5.0))],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    assert jnp.allclose(matrix, expected)


def test_idw_dist_and_idw_dist1_match_for_weighted_matrices() -> None:
    ref_coord = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
    coord = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64)
    ref_matrix = wei_dist_mat(ref_coord)
    matrix = wei_dist_mat(coord)

    assert jnp.allclose(idw_dist(ref_matrix, coord), abs(weight(2.0) - weight(1.0)))
    assert jnp.allclose(idw_dist(ref_matrix, coord), idw_dist1(ref_matrix, matrix))


def test_idw_pot_is_gaussian_of_idw_distance() -> None:
    ref_coord = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
    coord = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64)
    ref_matrix = wei_dist_mat(ref_coord)
    dist = idw_dist(ref_matrix, coord)

    assert jnp.allclose(idw_pot(ref_matrix, coord, a=0.5, sigma=2.0), 0.5 * jnp.exp(-(dist**2) / 8.0))


def test_idw_force_matches_negative_finite_difference_gradient() -> None:
    ref_coord = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
    coord = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64)
    ref_matrix = wei_dist_mat(ref_coord)
    eps = 1e-5

    energy, force = idw_pot_force(ref_matrix, coord, a=0.5, sigma=2.0)
    plus = coord.at[1, 0].add(eps)
    minus = coord.at[1, 0].add(-eps)
    derivative = (idw_pot(ref_matrix, plus, 0.5, 2.0) - idw_pot(ref_matrix, minus, 0.5, 2.0)) / (2.0 * eps)

    assert jnp.isfinite(energy)
    assert jnp.allclose(force[1, 0], -derivative, rtol=1e-7, atol=1e-9)


def test_idwm0_pes_scatters_fragment_force_to_full_system() -> None:
    ref_fragment = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
    system = System(
        coord=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        atom_add_pot=(0, 2),
    )
    pes = Idwm0PES(
        local_min=wei_dist_mat(ref_fragment),
        nearby_ts=(),
        a_min=0.5,
        a_ts=0.0,
        sigma_min=2.0,
        sigma_ts=(),
    )

    energy, force = pes.get_energy_force(system)

    assert energy == pes.get_energy(system)
    assert force.shape == system.coord.shape
    assert jnp.allclose(force[1], jnp.zeros(3, dtype=jnp.float64))
    assert jnp.any(jnp.abs(force[0]) > 0.0)
    assert jnp.any(jnp.abs(force[2]) > 0.0)

