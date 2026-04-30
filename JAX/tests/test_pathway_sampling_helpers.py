from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import random

from rtip_jax.config import Para
from rtip_jax.system import System
from rtip_jax.workflows.pathway_sampling import (
    AttractiveSamplingState,
    RepulsiveSamplingState,
    SynthesisSamplingState,
    repulsive_stop_decision,
    force_norm,
    perturb_system,
    repulsive_stop_update,
    synthesis_stop_update,
    attractive_stop_update,
)


def test_perturb_system_is_reproducible_and_norm_scaled() -> None:
    system = System(coord=jnp.zeros((3, 3), dtype=jnp.float64))
    key = random.PRNGKey(11)

    out1 = perturb_system(system, key, scale=0.1)
    out2 = perturb_system(system, key, scale=0.1)

    assert jnp.allclose(out1.coord, out2.coord)
    assert jnp.allclose(jnp.mean(out1.coord, axis=0), jnp.zeros(3, dtype=jnp.float64), atol=1e-12)
    assert jnp.allclose(force_norm(out1.coord), 0.1)


def test_perturb_system_can_target_atom_subset() -> None:
    system = System(coord=jnp.zeros((4, 3), dtype=jnp.float64))

    out = perturb_system(system, random.PRNGKey(12), atom_indices=(1, 3), scale=0.1)

    assert jnp.allclose(out.coord[0], jnp.zeros(3, dtype=jnp.float64))
    assert jnp.allclose(out.coord[2], jnp.zeros(3, dtype=jnp.float64))
    assert jnp.allclose(force_norm(out.coord[jnp.asarray([1, 3])]), 0.1)


def test_perturb_system_validates_atom_subset() -> None:
    system = System(coord=jnp.zeros((2, 3), dtype=jnp.float64))

    with pytest.raises(ValueError):
        perturb_system(system, random.PRNGKey(0), atom_indices=())
    with pytest.raises(ValueError):
        perturb_system(system, random.PRNGKey(0), atom_indices=(2,))


def test_repulsive_stop_update_matches_rust_conditions() -> None:
    para = Para(pot_drop=0.5, pot_climb=1.0, f_epsilon=0.1)
    state = RepulsiveSamplingState(pot_real_max=10.0, pot_real_min=8.0, add_bias=True)

    new_state, should_stop = repulsive_stop_update(
        state,
        pot_real=9.0,
        f_real=10.0,
        f_bias=0.0,
        natom=4,
        para=para,
    )
    assert not new_state.add_bias
    assert not should_stop

    new_state, should_stop = repulsive_stop_update(
        new_state,
        pot_real=8.5,
        f_real=0.1,
        f_bias=0.0,
        natom=4,
        para=para,
    )
    assert should_stop


def test_repulsive_stop_decision_names_state_machine_branch() -> None:
    para = Para(pot_drop=0.5, pot_climb=10.0, f_epsilon=0.1)

    state, should_stop, decision = repulsive_stop_decision(
        RepulsiveSamplingState(pot_real_max=10.0, pot_real_min=8.0, add_bias=True),
        pot_real=9.0,
        f_real=10.0,
        f_bias=0.0,
        natom=4,
        para=para,
    )
    assert not state.add_bias
    assert not should_stop
    assert decision == "bias_off_after_pot_drop"

    _state, should_stop, decision = repulsive_stop_decision(
        state,
        pot_real=8.5,
        f_real=0.1,
        f_bias=0.0,
        natom=4,
        para=para,
    )
    assert should_stop
    assert decision == "stop_converged"


def test_repulsive_stop_update_stops_on_climb_or_large_bias_force() -> None:
    para = Para(pot_climb=1.0)

    _, should_stop_climb = repulsive_stop_update(
        RepulsiveSamplingState(pot_real_max=1.0, pot_real_min=1.0),
        pot_real=2.5,
        f_real=10.0,
        f_bias=0.0,
        natom=1,
        para=para,
    )
    _, should_stop_force = repulsive_stop_update(
        RepulsiveSamplingState(),
        pot_real=0.0,
        f_real=10.0,
        f_bias=1001.0,
        natom=1,
        para=para,
    )
    assert should_stop_climb
    assert should_stop_force


def test_attractive_stop_update_matches_rust_conditions() -> None:
    para = Para(f_epsilon=0.1)

    state, should_stop = attractive_stop_update(
        AttractiveSamplingState(),
        sigma_min=0.5,
        f_real=1.0,
        f_bias=0.0,
        natom=4,
        para=para,
    )
    assert not state.add_bias
    assert not should_stop

    state, should_stop = attractive_stop_update(
        state,
        sigma_min=2.0,
        f_real=0.1,
        f_bias=0.0,
        natom=4,
        para=para,
    )
    assert should_stop


def test_synthesis_stop_update_records_initial_energy() -> None:
    para = Para(pot_drop=0.5, pot_climb=10.0, f_epsilon=0.1)
    state, should_stop = synthesis_stop_update(
        SynthesisSamplingState(),
        step=1,
        pot_real=5.0,
        f_real=10.0,
        f_bias=0.0,
        natom=1,
        para=para,
    )

    assert state.pot_real_initial == 5.0
    assert not should_stop
