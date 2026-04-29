from __future__ import annotations

import json

import pytest

from rtip_jax.cli import main, parse_mol_index, parse_type_map
from rtip_jax.config import Para, load_para
from rtip_jax.io.outputs import output_cp2k, output_rtip


def test_para_loads_from_json(tmp_path) -> None:
    path = tmp_path / "para.json"
    path.write_text(json.dumps({"a0": 0.1, "max_step": 3, "scale_ts_sigma": None}))

    para = load_para(path)

    assert para.a0 == 0.1
    assert para.max_step == 3
    assert para.scale_ts_sigma is None


def test_para_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        Para.from_mapping({"unknown": 1.0})


def test_output_helpers_match_legacy_names(tmp_path) -> None:
    paths = output_rtip(index=7, base_dir=tmp_path)

    assert paths.structure == tmp_path / "7" / "rtip.pdb"
    assert paths.table == tmp_path / "7" / "rtip.out"
    assert output_cp2k(index=7, base_dir=tmp_path) == tmp_path / "7" / "cp2k.out"


def test_parse_mol_index_cli_format() -> None:
    assert parse_mol_index("0,1; 2,3") == ((0, 1), (2, 3))


def test_parse_type_map_cli_format() -> None:
    assert parse_type_map("Ca, O,H") == ("Ca", "O", "H")


def test_cli_show_default_config(capsys) -> None:
    assert main(["show-default-config"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["a0"] == Para().a0


def test_cli_cp2k_boundary(capsys) -> None:
    assert main(["cp2k-boundary", "--input", "cp2k.inp", "--output", "cp2k.out"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["input_file"] == "cp2k.inp"
    assert out["output_file"] == "cp2k.out"
    assert out["position_units"] == "Bohr"


def test_cli_deepmd_boundary(capsys) -> None:
    assert main(["deepmd-boundary", "--model", "model.pth", "--type-map", "O,H"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"] == "model.pth"
    assert out["type_map"] == ["O", "H"]
    assert out["position_units"] == "Angstrom"
    assert out["internal_position_units"] == "Bohr"
