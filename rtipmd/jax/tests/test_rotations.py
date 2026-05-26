from __future__ import annotations

import jax.numpy as jnp
from jax import random

from rtip_jax.math.rotations import random_rotation, rotation_from_angles


def test_rotation_from_zero_angles_is_identity() -> None:
    rot = rotation_from_angles(0.0, 0.0, 0.0)

    assert jnp.allclose(rot, jnp.eye(3, dtype=jnp.float64))


def test_random_rotation_is_reproducible_with_explicit_key() -> None:
    key = random.PRNGKey(123)

    assert jnp.allclose(random_rotation(key), random_rotation(key))


def test_random_rotation_is_orthogonal_with_unit_determinant() -> None:
    rot = random_rotation(random.PRNGKey(5))

    assert jnp.allclose(rot.T @ rot, jnp.eye(3, dtype=jnp.float64), atol=1e-12)
    assert jnp.allclose(jnp.linalg.det(rot), 1.0, atol=1e-12)

