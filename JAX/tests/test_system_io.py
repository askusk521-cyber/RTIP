from __future__ import annotations

import jax.numpy as jnp
import pytest

from rtip_jax.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, Element
from rtip_jax.io.pdb import format_pdb
from rtip_jax.io.xyz import format_xyz, read_xyz
from rtip_jax.system import System


def test_system_normalizes_coord_and_metadata() -> None:
    system = System(coord=[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], atom_type=("H", "O"))

    assert system.natom == 2
    assert system.coord.dtype == jnp.float64
    assert system.atom_type == (Element.H, Element.O)


def test_system_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        System(coord=[1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        System(coord=[[0.0, 0.0, 0.0]], atom_type=("H", "O"))


def test_xyz_read_converts_angstrom_to_bohr(tmp_path) -> None:
    xyz_path = tmp_path / "molecule.xyz"
    xyz_path.write_text("2\ncomment\nH 0.0 0.0 0.0\nO 1.0 2.0 3.0\n")

    system = read_xyz(xyz_path)

    assert system.atom_type == (Element.H, Element.O)
    assert jnp.allclose(system.coord[1], jnp.asarray([1.0, 2.0, 3.0]) * ANGSTROM_TO_BOHR)


def test_xyz_format_converts_bohr_to_angstrom() -> None:
    system = System(coord=[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], atom_type=("H", "O"), pot=1.25)

    text = format_xyz(system, step=7)

    assert "Step =        7, E =      1.25000000" in text
    assert f"{1.0 * BOHR_TO_ANGSTROM:20.10f}" in text
    assert text.splitlines()[0] == "       2"


def test_system_write_xyz_appends_frames(tmp_path) -> None:
    system = System(coord=[[0.0, 0.0, 0.0]], atom_type=("H",))
    xyz_path = tmp_path / "traj.xyz"

    system.write_xyz(xyz_path, create_new_file=True, step=0)
    system.write_xyz(xyz_path, create_new_file=False, step=1)

    assert xyz_path.read_text().count("Step =") == 2


def test_pdb_format_matches_stage3_boundary() -> None:
    system = System(coord=[[0.0, 0.0, 0.0]], atom_type=("H",), pot=-0.5)

    text = format_pdb(system, step=3, include_title=True)

    assert text.startswith("TITLE     PDB file created by RTIP\n")
    assert "REMARK    , Step =        3, E =     -0.50000000" in text
    assert "ATOM      1 H" in text
    assert text.endswith("END\n")

