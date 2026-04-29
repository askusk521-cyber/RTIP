from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rtip_jax.constants import (
    BOHR_TO_ANGSTROM,
    EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR,
    EV_TO_HARTREE,
)
from rtip_jax.config import Para
from rtip_jax.external import DeepMDBoundary, DeepMDPES, deepmd_atype, deepmd_inputs
from rtip_jax.pes import RepulsivePot
from rtip_jax.system import System
from rtip_jax.workflows import run_rtip_repulsive_path_sampling


class FakeDeepPot:
    def __init__(self, *, energy_ev=2.0, force_ev_per_angstrom=3.0, virial_ev=4.0, type_map=("O", "H")) -> None:
        self.energy_ev = energy_ev
        self.force_ev_per_angstrom = force_ev_per_angstrom
        self.virial_ev = virial_ev
        self._type_map = tuple(type_map)
        self.calls = []

    def get_type_map(self):
        return self._type_map

    def eval(self, coord, cell, atype):
        self.calls.append((coord, cell, atype))
        natom = len(atype)
        energy = np.asarray([[self.energy_ev]], dtype=np.float64)
        force = np.full((1, natom, 3), self.force_ev_per_angstrom, dtype=np.float64)
        virial = np.full((1, 9), self.virial_ev, dtype=np.float64)
        return energy, force, virial


def test_deepmd_inputs_convert_system_to_deeppot_shapes_and_units() -> None:
    system = System(
        coord=jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float64),
        cell=jnp.eye(3, dtype=jnp.float64) * 20.0,
        atom_type=("O", "H"),
    )

    coord, cell, atype = deepmd_inputs(system, ("O", "H"))

    assert coord.shape == (1, 6)
    assert cell.shape == (1, 9)
    assert atype == [0, 1]
    assert np.allclose(coord.reshape(2, 3), np.asarray(system.coord) * BOHR_TO_ANGSTROM)
    assert np.allclose(cell.reshape(3, 3), np.asarray(system.cell) * BOHR_TO_ANGSTROM)


def test_deepmd_inputs_support_non_periodic_systems() -> None:
    system = System(coord=[[0.0, 0.0, 0.0]], atom_type=("H",))

    _coord, cell, atype = deepmd_inputs(system, ("H",))

    assert cell is None
    assert atype == [0]


def test_deepmd_pes_replaces_cp2k_energy_force_contract_with_internal_units() -> None:
    fake = FakeDeepPot(energy_ev=2.0, force_ev_per_angstrom=3.0, virial_ev=4.0)
    pes = DeepMDPES(DeepMDBoundary(model="model.pth"), deep_pot=fake)
    system = System(coord=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atom_type=("O", "H"))

    result = pes.evaluate(system)
    energy, force = pes.get_energy_force(system)

    assert energy == pytest.approx(2.0 * EV_TO_HARTREE)
    assert result.energy == pytest.approx(2.0 * EV_TO_HARTREE)
    assert jnp.allclose(force, jnp.full((2, 3), 3.0 * EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR))
    assert jnp.allclose(result.virial, jnp.full((9,), 4.0 * EV_TO_HARTREE))
    assert pes.get_energy(system) == pytest.approx(2.0 * EV_TO_HARTREE)


def test_deepmd_pes_uses_explicit_type_map_over_model_type_map() -> None:
    fake = FakeDeepPot(type_map=("wrong",))
    pes = DeepMDPES(DeepMDBoundary(model="model.pth", type_map=("Ca", "O", "H")), deep_pot=fake)
    system = System(coord=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atom_type=("O", "H"))

    pes.get_energy_force(system)

    _coord, _cell, atype = fake.calls[-1]
    assert atype == [1, 2]


def test_deepmd_rejects_missing_or_unknown_atom_types() -> None:
    with pytest.raises(ValueError):
        deepmd_atype(System(coord=[[0.0, 0.0, 0.0]]), ("H",))

    with pytest.raises(ValueError):
        deepmd_atype(System(coord=[[0.0, 0.0, 0.0]], atom_type=("O",)), ("H",))


def test_deepmd_pes_can_drive_workflow_where_cp2k_was_used() -> None:
    from jax import random

    fake = FakeDeepPot(energy_ev=1.0, force_ev_per_angstrom=0.1, type_map=("O", "H"))
    pes = DeepMDPES(DeepMDBoundary(model="model.pth"), deep_pot=fake)
    local_min = System(coord=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atom_type=("O", "H"))
    config = RepulsivePot(local_min=local_min, para=Para(max_step=1))

    result = run_rtip_repulsive_path_sampling(
        config,
        pes,
        key=random.PRNGKey(0),
        line_search=None,
        write_outputs=False,
    )

    assert len(result.history) == 1
    assert result.history[0].pot_real == pytest.approx(EV_TO_HARTREE)
    assert fake.calls
