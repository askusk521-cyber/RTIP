from __future__ import annotations

import jax.numpy as jnp
from jax import random

from rtip_jax.config import Para
from rtip_jax.pes import AttractivePot, HarmonicPES, RepulsivePot, SynthesisPot
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
    assert result.history[0].temp_bath == para.temp_bath
    assert result.history[0].pot_total == result.history[0].pot_real + result.history[0].pot_bias
    assert result.history[-1].state_decision == "max_step"


def test_rtip_nvt_md_writes_diagnostic_columns(tmp_path) -> None:
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

    run_rtip_nvt_md(config, pes, key=random.PRNGKey(5), write_outputs=True)

    lines = (tmp_path / "rtip.out").read_text().splitlines()
    assert "time_fs" in lines[0]
    assert "wall_time_s" in lines[0]
    assert "temp_K" in lines[0]
    assert "thermo_lambda" in lines[0]
    assert "pot_total_Ha" in lines[0]
    assert "state_decision" in lines[0]
    assert lines[1].split()[-1] == "max_step"


def test_rtip_nvt_md_supports_attractive_bias_with_mock_pes(tmp_path) -> None:
    initial = System(
        coord=jnp.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=jnp.float64),
        atom_type=("H", "H"),
    )
    final = System(
        coord=jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64),
        atom_type=("H", "H"),
    )
    para = Para(max_step=1, print_step=1, dt=0.1, tau=100.0, temp_bath=300.0)
    config = AttractivePot(
        initial_state=initial,
        final_state=final,
        para=para,
        str_output_file=str(tmp_path / "attractive.pdb"),
        output_file=str(tmp_path / "attractive.out"),
    )

    result = run_rtip_nvt_md(config, HarmonicPES(k=0.1), perturb=False, write_outputs=True)

    assert len(result.history) == 1
    assert result.history[0].pot_bias < 0.0
    lines = (tmp_path / "attractive.out").read_text().splitlines()
    assert "pot_real_Ha" in lines[0]
    assert "pot_rtip_Ha" in lines[0]
    assert "pot_total_Ha" in lines[0]
    assert lines[1].split()[-1] == "max_step"


def test_rtip_nvt_md_supports_synthesis_bias_with_mock_pes() -> None:
    initial = System(
        coord=jnp.asarray([[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64),
        atom_type=("H", "H"),
    )
    para = Para(max_step=1, print_step=1, dt=0.1, tau=100.0, temp_bath=300.0)
    config = SynthesisPot(initial_state=initial, mol_index=((0,), (1,)), para=para)

    result = run_rtip_nvt_md(config, HarmonicPES(k=0.1), perturb=False, write_outputs=False)

    assert len(result.history) == 1
    assert result.history[0].pot_bias < 0.0
    assert result.history[0].pot_total == result.history[0].pot_real + result.history[0].pot_bias


def test_rtip_nvt_md_requires_key_when_perturbing() -> None:
    local_min = System(coord=jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64), atom_type=("H", "H"))
    config = RepulsivePot(local_min=local_min, para=Para(max_step=1))

    import pytest

    with pytest.raises(ValueError):
        run_rtip_nvt_md(config, HarmonicPES(), write_outputs=False)
