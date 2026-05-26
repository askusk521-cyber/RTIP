from __future__ import annotations

import jax.numpy as jnp

from rtip_jax.core.rtip import (
    Rtip0PES,
    quaternion_to_rotation,
    rti_dist,
    rti_dist_vec,
    rti_dists,
    rti_pot,
    rti_pot_force,
    rti_rot_tran,
    rti_weight,
    rti_weight_derivative,
)
from rtip_jax.math.rotations import rotation_from_angles
from rtip_jax.system import System


def _reference_coord():
    return jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.3, 0.1, 0.0],
            [-0.2, 1.1, 0.4],
        ],
        dtype=jnp.float64,
    )


def _distorted_coord():
    coord = _reference_coord()
    rot = rotation_from_angles(0.2, 0.4, 0.7)
    translated = coord @ rot + jnp.asarray([2.0, -1.0, 0.5], dtype=jnp.float64)
    return translated.at[2, 1].add(0.15)


def test_rti_weight_and_derivative_match_rust_formula() -> None:
    x = 2.0

    assert jnp.allclose(rti_weight(x), 1.0 / x**7)
    assert jnp.allclose(rti_weight_derivative(x), -7.0 / x**8)


def test_quaternion_to_rotation_matches_identity_quaternion() -> None:
    rotation = quaternion_to_rotation(jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64))

    assert jnp.allclose(rotation, jnp.eye(3, dtype=jnp.float64))


def test_rti_distance_is_translation_invariant() -> None:
    coord = _reference_coord()
    shifted = coord + jnp.asarray([3.0, -2.0, 0.7], dtype=jnp.float64)

    assert jnp.allclose(rti_dist(coord, shifted), 0.0, atol=1e-10)


def test_rti_distance_is_rotation_translation_invariant() -> None:
    coord = _reference_coord()
    rotation = rotation_from_angles(0.3, 0.6, 1.1)
    transformed = coord @ rotation + jnp.asarray([1.0, -4.0, 0.25], dtype=jnp.float64)

    assert jnp.allclose(rti_dist(coord, transformed), 0.0, atol=1e-10)


def test_rti_rot_tran_reconstructs_best_alignment() -> None:
    coord = _reference_coord()
    expected_rotation = rotation_from_angles(0.3, 0.6, 1.1)
    transformed = coord @ expected_rotation + jnp.asarray([1.0, -4.0, 0.25], dtype=jnp.float64)

    rotation, translation = rti_rot_tran(coord, transformed)
    aligned = coord @ rotation + translation

    assert jnp.allclose(aligned, transformed, atol=1e-10)


def test_rti_dist_vec_norm_matches_distance() -> None:
    distance, vector = rti_dist_vec(_reference_coord(), _distorted_coord())

    assert jnp.allclose(jnp.sqrt(jnp.sum(vector * vector)), distance)
    assert rti_dists(_reference_coord(), _distorted_coord()).shape == (4,)


def test_rti_pot_is_weighted_gaussian_sum() -> None:
    coord1 = _reference_coord()
    coord2 = _distorted_coord()
    distances = rti_dists(coord1, coord2)
    weights = rti_weight(distances)
    u = 0.7 * jnp.exp(-(distances * distances) / (2.0 * 1.8 * 1.8))

    assert jnp.allclose(rti_pot(coord1, coord2, a=0.7, sigma=1.8), jnp.sum((weights / jnp.sum(weights)) * u))


def test_rti_force_matches_negative_finite_difference_gradient() -> None:
    coord1 = _reference_coord()
    coord2 = _distorted_coord()
    eps = 1e-5

    energy, force = rti_pot_force(coord1, coord2, a=0.7, sigma=1.8)
    plus = coord2.at[2, 1].add(eps)
    minus = coord2.at[2, 1].add(-eps)
    derivative = (rti_pot(coord1, plus, 0.7, 1.8) - rti_pot(coord1, minus, 0.7, 1.8)) / (2.0 * eps)

    assert jnp.isfinite(energy)
    assert jnp.allclose(force[2, 1], -derivative, rtol=1e-5, atol=1e-7)


def test_rtip0_pes_scatters_fragment_force_to_full_system() -> None:
    local_min = System(coord=_reference_coord())
    system = System(
        coord=[
            [0.0, 0.0, 0.0],
            [9.0, 9.0, 9.0],
            [1.3, 0.1, 0.0],
            [-0.2, 1.25, 0.4],
        ],
        atom_add_pot=(0, 2, 3),
    )
    pes = Rtip0PES(
        local_min=local_min,
        nearby_ts=(),
        a_min=0.7,
        a_ts=0.0,
        sigma_min=1.8,
        sigma_ts=(),
    )

    energy, force = pes.get_energy_force(system)

    assert energy == pes.get_energy(system)
    assert force.shape == system.coord.shape
    assert jnp.allclose(force[1], jnp.zeros(3, dtype=jnp.float64))
    assert jnp.any(jnp.abs(force[0]) > 0.0)
    assert jnp.any(jnp.abs(force[2]) > 0.0)
    assert jnp.any(jnp.abs(force[3]) > 0.0)

