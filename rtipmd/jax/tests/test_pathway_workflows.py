from __future__ import annotations

import math

import jax.numpy as jnp
from jax import random

from rtip_jax.config import Para
from rtip_jax.pes import AttractivePot, HarmonicPES, RepulsivePot, SynthesisPot
from rtip_jax.system import System
from rtip_jax.workflows import (
    run_idwm_repulsive_path_sampling,
    run_rtip_attractive_path_sampling,
    run_rtip_repulsive_path_sampling,
    run_rtip_synthesis_path_sampling,
    synthesis_target_state,
)


def _two_atom_system() -> System:
    return System(coord=jnp.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=jnp.float64))


def test_rtip_repulsive_pathway_runs_with_mock_pes() -> None:
    local_min = _two_atom_system()
    para = Para(max_step=2, print_step=1, scale_ts_sigma=None)
    config = RepulsivePot(local_min=local_min, nearby_ts=(), para=para)
    pes = HarmonicPES(k=0.1, center=local_min.coord)

    result = run_rtip_repulsive_path_sampling(
        config,
        pes,
        key=random.PRNGKey(3),
        line_search=None,
        write_outputs=False,
    )

    assert len(result.history) == 2
    assert result.system.coord.shape == local_min.coord.shape
    assert result.history[0].sigma_min > 0.0
    assert result.history[0].time_fs == para.dt
    assert math.isnan(result.history[0].temp)
    assert result.history[0].state_decision == "bias_on"
    assert result.history[0].next_add_bias


def test_rtip_repulsive_pathway_writes_diagnostic_columns(tmp_path) -> None:
    local_min = _two_atom_system()
    para = Para(max_step=1, print_step=1, dt=0.5, scale_ts_sigma=None)
    config = RepulsivePot(
        local_min=local_min,
        nearby_ts=(),
        para=para,
        str_output_file=str(tmp_path / "rtip.pdb"),
        output_file=str(tmp_path / "rtip.out"),
    )
    pes = HarmonicPES(k=0.1, center=local_min.coord)

    run_rtip_repulsive_path_sampling(
        config,
        pes,
        key=random.PRNGKey(3),
        line_search=None,
        write_outputs=True,
    )

    lines = (tmp_path / "rtip.out").read_text().splitlines()
    assert "time_fs" in lines[0]
    assert "wall_time_s" in lines[0]
    assert "temp_K" in lines[0]
    assert "pot_total_Ha" in lines[0]
    assert "bias_used" in lines[0]
    assert "next_add_bias" in lines[0]
    assert "state_decision" in lines[0]
    assert lines[1].split()[-1] == "bias_on"


def test_idwm_repulsive_pathway_runs_with_mock_pes() -> None:
    local_min = _two_atom_system()
    config = RepulsivePot(local_min=local_min, nearby_ts=(), para=Para(max_step=2))
    pes = HarmonicPES(k=0.1, center=local_min.coord)

    result = run_idwm_repulsive_path_sampling(
        config,
        pes,
        key=random.PRNGKey(4),
        line_search=None,
        write_outputs=False,
    )

    assert len(result.history) == 2
    assert result.history[0].sigma_min > 0.0


def test_attractive_pathway_runs_with_mock_pes() -> None:
    initial = _two_atom_system()
    final = System(coord=jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64))
    config = AttractivePot(initial_state=initial, final_state=final, para=Para(max_step=1))
    pes = HarmonicPES(k=0.1)

    result = run_rtip_attractive_path_sampling(config, pes, line_search=None, write_outputs=False)

    assert len(result.history) == 1
    assert result.history[0].pot_bias < 0.0


def test_synthesis_target_state_handles_environment_atoms() -> None:
    system = System(coord=jnp.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=jnp.float64))

    target, indices = synthesis_target_state(system, ((0,), (2,)))

    assert indices == (0, 2)
    assert target.coord.shape == (2, 3)
    assert jnp.allclose(target.coord[0], target.coord[1])


def test_synthesis_pathway_runs_with_mock_pes() -> None:
    initial = System(coord=jnp.asarray([[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64))
    config = SynthesisPot(initial_state=initial, mol_index=((0,), (1,)), para=Para(max_step=1))
    pes = HarmonicPES(k=0.1)

    result = run_rtip_synthesis_path_sampling(config, pes, line_search=None, write_outputs=False)

    assert len(result.history) == 1
    assert result.history[0].pot_bias < 0.0
