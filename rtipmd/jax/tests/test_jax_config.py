from __future__ import annotations

import jax.numpy as jnp

import rtip_jax


def test_jax_x64_is_enabled_on_import() -> None:
    assert rtip_jax.is_x64_enabled()
    assert jnp.asarray([1.0]).dtype == jnp.float64

