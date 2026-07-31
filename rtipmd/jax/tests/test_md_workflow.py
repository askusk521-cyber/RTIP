from __future__ import annotations

import jax.numpy as jnp
from jax import random

from rtip_jax.config import Para
from rtip_jax.pes import EvolutionPot, HarmonicPES, RepulsivePot
from rtip_jax.system import System
from rtip_jax.workflows import evolution_md, repulsive_md


def test_repulsive_md_runs_with_mock_pes() -> None:
    local_min = System(
        coord=jnp.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=jnp.float64),
        atom_type=("H", "H"),
    )
    para = Para(max_step=2, print_step=1, dt=0.1, tau=100.0, temp_bath=300.0)
    config = RepulsivePot(local_min=local_min, nearby_ts=(), para=para)
    pes = HarmonicPES(k=0.1, center=local_min.coord)

    result = repulsive_md(config, pes, key=random.PRNGKey(5), perturb=True, write_outputs=False)

    assert len(result.history) == 2
    assert result.system.coord.shape == local_min.coord.shape
    assert result.velocity.shape == local_min.coord.shape
    assert result.acceleration.shape == local_min.coord.shape
    assert result.history[0].time == para.dt


def test_repulsive_md_writes_diagnostic_columns(tmp_path) -> None:
    local_min = System(
        coord=jnp.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=jnp.float64),
        atom_type=("H", "H"),
    )
    para = Para(max_step=1, print_step=1, dt=0.1, tau=100.0, temp_bath=300.0)
    config = RepulsivePot(
        local_min=local_min,
        nearby_ts=(),
        para=para,
        str_output_file=str(tmp_path / "rtip.pdb"),
        output_file=str(tmp_path / "rtip.out"),
    )
    pes = HarmonicPES(k=0.1, center=local_min.coord)

    repulsive_md(config, pes, key=random.PRNGKey(5), write_outputs=True)

    lines = (tmp_path / "rtip.out").read_text().splitlines()
    assert "time_fs" in lines[0]
    assert "temp_K" in lines[0]
    assert "pot_real_Ha" in lines[0]
    assert "pot_rtip_Ha" in lines[0]


def test_evolution_md_runs_with_mock_pes(tmp_path) -> None:
    # Two water molecules: O-H-H / O-H-H.
    initial = System(
        coord=jnp.asarray(
            [
                [0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.24, 0.93, 0.0],
                [5.0, 0.0, 0.0], [5.96, 0.0, 0.0], [5.24, 0.93, 0.0],
            ],
            dtype=jnp.float64,
        ),
        atom_type=("O", "H", "H", "O", "H", "H"),
    )
    para = Para(max_step=3, print_step=1, dt=0.1, tau=100.0, temp_bath=300.0)
    config = EvolutionPot(
        initial_state=initial,
        para=para,
        str_output_file=str(tmp_path / "evo.pdb"),
        output_file=str(tmp_path / "evo.out"),
        dec_output_file=str(tmp_path / "decreasing_steps"),
    )

    result = evolution_md(config, HarmonicPES(k=0.1), write_outputs=True)

    assert len(result.history) == 3
    assert result.system.coord.shape == initial.coord.shape
    assert result.history[0].rtip_status in ("Increasing", "Falling", "Decreasing")
    lines = (tmp_path / "evo.out").read_text().splitlines()
    assert "rti_dist" in lines[0]
    assert "rtip_status" in lines[0]
    assert (tmp_path / "decreasing_steps").exists()


def test_repulsive_md_requires_key_when_perturbing() -> None:
    local_min = System(coord=jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64), atom_type=("H", "H"))
    config = RepulsivePot(local_min=local_min, para=Para(max_step=1))

    import pytest

    with pytest.raises(ValueError):
        repulsive_md(config, HarmonicPES(), perturb=True, write_outputs=False)
