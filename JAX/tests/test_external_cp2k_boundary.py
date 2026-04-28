from __future__ import annotations

import pytest

from rtip_jax.external import Cp2kBoundary, Cp2kPES, Cp2kPESUnavailable


def test_cp2k_boundary_records_original_io_contract() -> None:
    boundary = Cp2kBoundary("cp2k.inp", "cp2k.out", mpi_comm=4)

    assert boundary.input_file == "cp2k.inp"
    assert boundary.output_file == "cp2k.out"
    assert boundary.mpi_comm == 4
    assert boundary.position_units == "Bohr"
    assert boundary.energy_units == "Hartree"
    assert boundary.force_units == "Hartree/Bohr"


def test_cp2k_pes_is_explicitly_unavailable() -> None:
    pes = Cp2kPES(Cp2kBoundary("cp2k.inp", "cp2k.out"))

    with pytest.raises(Cp2kPESUnavailable):
        pes.get_energy(object())

