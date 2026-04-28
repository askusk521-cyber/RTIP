from __future__ import annotations

import jax.numpy as jnp
import pytest

from rtip_jax.config import Para
from rtip_jax.constants import BOLTZMANN, FEMTOSECOND_TO_AU, HARTREE_TO_JOULE, Element
from rtip_jax.system import System
from rtip_jax.workflows.md import (
    accelerations,
    atom_masses,
    berendsen_lambda,
    kinetic_energy,
    leapfrog_first,
    leapfrog_second,
    temperature,
    twice_kinetic_energy,
)


def test_atom_masses_match_system_atom_types() -> None:
    system = System(coord=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atom_type=("H", "O"))

    masses = atom_masses(system)

    assert jnp.allclose(masses, jnp.asarray([Element.H.get_mass(), Element.O.get_mass()], dtype=jnp.float64))


def test_atom_masses_requires_atom_types() -> None:
    with pytest.raises(ValueError):
        atom_masses(System(coord=[[0.0, 0.0, 0.0]]))


def test_kinetic_energy_and_temperature_match_rust_formula() -> None:
    velocity = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=jnp.float64)
    masses = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    twice_kin = 14.0

    assert jnp.allclose(twice_kinetic_energy(velocity, masses), twice_kin)
    assert jnp.allclose(kinetic_energy(velocity, masses), 0.5 * twice_kin)
    assert jnp.allclose(
        temperature(velocity, masses),
        twice_kin * HARTREE_TO_JOULE / (BOLTZMANN * 3.0),
    )


def test_temperature_requires_at_least_two_atoms() -> None:
    with pytest.raises(ValueError):
        temperature(jnp.zeros((1, 3), dtype=jnp.float64), jnp.ones((1,), dtype=jnp.float64))


def test_berendsen_lambda_applies_temperature_floor() -> None:
    para = Para(dt=0.5, tau=500.0, temp_bath=1000.0)

    assert jnp.allclose(berendsen_lambda(0.5, para), jnp.sqrt(1.0 + (0.5 / 500.0) * (1000.0 / 1.0 - 1.0)))


def test_accelerations_divide_force_by_mass() -> None:
    force = jnp.asarray([[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], dtype=jnp.float64)
    masses = jnp.asarray([2.0, 3.0], dtype=jnp.float64)

    assert jnp.allclose(accelerations(force, masses), jnp.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))


def test_leapfrog_steps_match_rust_updates() -> None:
    para = Para(dt=0.5)
    coord = jnp.zeros((1, 3), dtype=jnp.float64)
    velocity = jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float64)
    acceleration = jnp.asarray([[0.2, 0.0, 0.0]], dtype=jnp.float64)

    coord_new, velocity_half = leapfrog_first(coord, velocity, acceleration, para)
    expected_velocity_half = velocity + 0.5 * para.dt * FEMTOSECOND_TO_AU * acceleration
    expected_coord = coord + para.dt * FEMTOSECOND_TO_AU * expected_velocity_half

    assert jnp.allclose(velocity_half, expected_velocity_half)
    assert jnp.allclose(coord_new, expected_coord)
    assert jnp.allclose(
        leapfrog_second(velocity_half, acceleration, 2.0, para),
        (velocity_half + 0.5 * para.dt * FEMTOSECOND_TO_AU * acceleration) * 2.0,
    )

