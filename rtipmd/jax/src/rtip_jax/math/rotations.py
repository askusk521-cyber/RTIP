"""Rotation matrix helpers.

Rust source: `src/matrix/mod.rs`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import random

from rtip_jax.constants import PI


def random_rotation(key: Any) -> Any:
    """Return a random 3x3 rotation matrix from an explicit JAX PRNG key.

    The formula matches Rust `matrix::rand_rot`, replacing implicit
    `ndarray-rand` global randomness with explicit JAX key usage.
    """

    alpha, beta, gamma = random.uniform(
        key,
        shape=(3,),
        minval=0.0,
        maxval=2.0 * PI,
        dtype=jnp.float64,
    )
    return rotation_from_angles(alpha, beta, gamma)


def rotation_from_angles(alpha: Any, beta: Any, gamma: Any) -> Any:
    """Construct the rotation matrix used by the Rust implementation."""

    return jnp.asarray(
        [
            [
                jnp.cos(alpha) * jnp.cos(gamma)
                - jnp.cos(beta) * jnp.sin(alpha) * jnp.sin(gamma),
                -jnp.cos(beta) * jnp.cos(gamma) * jnp.sin(alpha)
                - jnp.cos(alpha) * jnp.sin(gamma),
                jnp.sin(alpha) * jnp.sin(beta),
            ],
            [
                jnp.cos(gamma) * jnp.sin(alpha)
                + jnp.cos(alpha) * jnp.cos(beta) * jnp.sin(gamma),
                jnp.cos(alpha) * jnp.cos(beta) * jnp.cos(gamma)
                - jnp.sin(alpha) * jnp.sin(gamma),
                -jnp.cos(alpha) * jnp.sin(beta),
            ],
            [
                jnp.sin(beta) * jnp.sin(gamma),
                jnp.cos(gamma) * jnp.sin(beta),
                jnp.cos(beta),
            ],
        ],
        dtype=jnp.float64,
    )


rand_rot = random_rotation

