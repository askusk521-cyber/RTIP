from __future__ import annotations

import jax.numpy as jnp
from jax import random

from rtip_jax.config import Para
from rtip_jax.pes import HarmonicPES, RepulsivePot
from rtip_jax.system import System
from rtip_jax.workflows import run_rtip_nvt_md


def test_rtip_nvt_md_runs_with_mock_pes() -> None:
    local_min = System(
        coord=jnp.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=jnp.float64),
        atom_type=("H", "H"),
    )
    para = Para(max_step=2, print_step=1, dt=0.1, tau=100.0, temp_bath=300.0)
    config = RepulsivePot(local_min=local_min, nearby_ts=(), para=para)
    pes = HarmonicPES(k=0.1, center=local_min.coord)

    result = run_rtip_nvt_md(config, pes, key=random.PRNGKey(5), write_outputs=False)

    assert len(result.history) == 2
    assert result.system.coord.shape == local_min.coord.shape
    assert result.velocity.shape == local_min.coord.shape
    assert result.acceleration.shape == local_min.coord.shape
    assert result.history[0].time == para.dt


def test_rtip_nvt_md_requires_key_when_perturbing() -> None:
    local_min = System(coord=jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64), atom_type=("H", "H"))
    config = RepulsivePot(local_min=local_min, para=Para(max_step=1))

    import pytest

    with pytest.raises(ValueError):
        run_rtip_nvt_md(config, HarmonicPES(), write_outputs=False)
