from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import random

from rtip_jax.system import System
from rtip_jax.workflows.synthesis import synthesis_offsets, synthesize_layout


def _identity_rotations(n_molecules: int):
    return tuple(jnp.eye(3, dtype=jnp.float64) for _ in range(n_molecules))


def _system(n_atoms: int) -> System:
    return System(coord=[[float(index), 0.0, 0.0] for index in range(n_atoms)])


def test_synthesis_offsets_match_rust_layouts() -> None:
    assert jnp.allclose(synthesis_offsets(2, 5.0), jnp.asarray([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    assert jnp.allclose(jnp.sum(synthesis_offsets(3, 2.0), axis=0), jnp.zeros(3, dtype=jnp.float64))
    assert jnp.allclose(jnp.sum(synthesis_offsets(4, 2.0), axis=0), jnp.zeros(3, dtype=jnp.float64))


def test_synthesize_layout_for_two_molecules_with_identity_rotations() -> None:
    system = _system(4)

    out = synthesize_layout(system, [[0, 1], [2, 3]], 5.0, rotations=_identity_rotations(2))

    assert jnp.allclose(jnp.mean(out.coord[jnp.asarray([0, 1])], axis=0), jnp.asarray([-5.0, 0.0, 0.0]))
    assert jnp.allclose(jnp.mean(out.coord[jnp.asarray([2, 3])], axis=0), jnp.asarray([5.0, 0.0, 0.0]))


def test_synthesize_layout_for_three_and_four_molecules_centers() -> None:
    system3 = _system(3)
    out3 = synthesize_layout(system3, [[0], [1], [2]], 3.0, rotations=_identity_rotations(3))
    assert jnp.allclose(out3.coord, synthesis_offsets(3, 3.0))

    system4 = _system(4)
    out4 = synthesize_layout(system4, [[0], [1], [2], [3]], 3.0, rotations=_identity_rotations(4))
    assert jnp.allclose(out4.coord, synthesis_offsets(4, 3.0))


def test_synthesize_layout_uses_explicit_random_key() -> None:
    system = _system(2)
    key = random.PRNGKey(17)

    out1 = synthesize_layout(system, [[0], [1]], 2.0, key=key)
    out2 = synthesize_layout(system, [[0], [1]], 2.0, key=key)

    assert jnp.allclose(out1.coord, out2.coord)


def test_synthesize_layout_validates_inputs() -> None:
    system = _system(2)

    with pytest.raises(ValueError):
        synthesize_layout(system, [[0]], 1.0, rotations=_identity_rotations(1))
    with pytest.raises(ValueError):
        synthesize_layout(system, [[0], []], 1.0, rotations=_identity_rotations(2))
    with pytest.raises(ValueError):
        synthesize_layout(system, [[0], [2]], 1.0, rotations=_identity_rotations(2))
    with pytest.raises(ValueError):
        synthesize_layout(system, [[0], [1]], 1.0)
    with pytest.raises(ValueError):
        synthesize_layout(system, [[0], [1]], 1.0, key=random.PRNGKey(0), rotations=_identity_rotations(2))

