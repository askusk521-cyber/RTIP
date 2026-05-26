from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from rtip_jax.config import Para
from rtip_jax.pes import (
    AttractivePot,
    DistanceRestraint,
    ReactionCoordinatePES,
    ReactionCoordinatePot,
    RepulsivePot,
    SumPES,
    SynthesisPot,
)
from rtip_jax.system import System


@dataclass(frozen=True)
class ConstantPES:
    energy: float
    force_value: float

    def get_energy(self, system: System) -> float:
        return self.energy

    def get_energy_force(self, system: System):
        return self.energy, jnp.full_like(system.coord, self.force_value)


def test_sum_pes_adds_energy_and_force() -> None:
    system = System(coord=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pes = SumPES([ConstantPES(energy=1.5, force_value=2.0), ConstantPES(energy=-0.25, force_value=0.5)])

    energy, force = pes.get_energy_force(system)

    assert pes.get_energy(system) == 1.25
    assert energy == 1.25
    assert jnp.allclose(force, jnp.full_like(system.coord, 2.5))


def test_sum_pes_rejects_empty_components() -> None:
    with pytest.raises(ValueError):
        SumPES([])


def test_repulsive_pot_normalizes_transition_states_and_defaults() -> None:
    local_min = System(coord=[[0.0, 0.0, 0.0]])
    ts = System(coord=[[1.0, 0.0, 0.0]])
    pot = RepulsivePot(local_min=local_min, nearby_ts=[ts])

    assert pot.local_min is local_min
    assert pot.nearby_ts == (ts,)
    assert pot.para == Para()
    assert pot.str_output_file == "rtip.pdb"
    assert pot.output_file == "rtip.out"


def test_attractive_pot_stores_initial_and_final_state() -> None:
    initial = System(coord=[[0.0, 0.0, 0.0]])
    final = System(coord=[[1.0, 0.0, 0.0]])
    para = Para(max_step=3)
    pot = AttractivePot(initial_state=initial, final_state=final, para=para)

    assert pot.initial_state is initial
    assert pot.final_state is final
    assert pot.para.max_step == 3


def test_synthesis_pot_normalizes_molecule_indices() -> None:
    system = System(coord=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pot = SynthesisPot(initial_state=system, mol_index=[[0, 1], [2]])

    assert pot.mol_index == ((0, 1), (2,))


def test_synthesis_pot_validates_molecule_indices() -> None:
    system = System(coord=[[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError):
        SynthesisPot(initial_state=system, mol_index=[])

    with pytest.raises(ValueError):
        SynthesisPot(initial_state=system, mol_index=[[]])

    with pytest.raises(ValueError):
        SynthesisPot(initial_state=system, mol_index=[[1]])


def test_reaction_coordinate_bias_energy_and_force_direction() -> None:
    system = System(coord=jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64))
    pes = ReactionCoordinatePES([DistanceRestraint(0, 1, target=1.0, k=2.0)])

    energy, force = pes.get_energy_force(system)

    assert energy == pytest.approx(1.0)
    assert force[0, 0] > 0.0
    assert force[1, 0] < 0.0
    assert jnp.allclose(force[0], -force[1])


def test_reaction_coordinate_pot_validates_restraints() -> None:
    system = System(coord=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    config = ReactionCoordinatePot(initial_state=system, restraints=[(0, 1, 1.2, 0.5, "N-C")])

    assert config.restraints[0].label == "N-C"
    assert config.restraints[0].target == pytest.approx(1.2)

    with pytest.raises(ValueError):
        ReactionCoordinatePot(initial_state=system, restraints=[(0, 2, 1.0, 0.5)])
