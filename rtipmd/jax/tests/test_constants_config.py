from __future__ import annotations

import pytest

from rtip_jax.config import Para
from rtip_jax.constants import (
    ANGSTROM_TO_BOHR,
    AU_TO_FEMTOSECOND,
    BOHR_TO_ANGSTROM,
    Element,
    FEMTOSECOND_TO_AU,
    GOLDEN_RATIO1,
    GOLDEN_RATIO2,
)
from rtip_jax.errors import UnknownElementError, UnsupportedElementMassError


def test_para_defaults_match_rust_para_new() -> None:
    para = Para()

    assert para.a0 == 0.0005
    assert para.sigma == 0.75
    assert para.scale_ts_a0 == 1.0
    assert para.scale_ts_sigma == 0.25
    assert para.max_step == 10000
    assert para.print_step == 1
    assert para.pot_climb == 0.185
    assert para.pot_drop == 0.02
    assert para.pot_epsilon == 0.00005
    assert para.f_epsilon == 0.001
    assert para.dt == 0.5
    assert para.tau == 10.0
    assert para.temp_bath == 1500.0
    assert para.decreasing_multiple == 2.0
    assert para.decreasing_bound == 0.5
    assert para.split_step == 100
    assert (Element.H, Element.O) in para.ignored_pair
    assert (Element.O, Element.Ca) in para.ignored_pair


def test_constants_match_rust_values() -> None:
    assert GOLDEN_RATIO1 == 0.618033988749895
    assert GOLDEN_RATIO2 == 1.618033988749895
    assert BOHR_TO_ANGSTROM == 0.52917720859
    assert ANGSTROM_TO_BOHR == 1.0 / BOHR_TO_ANGSTROM
    assert FEMTOSECOND_TO_AU == 1.0 / AU_TO_FEMTOSECOND


def test_element_parsing_and_supported_masses_match_rust() -> None:
    assert Element.from_str("H") is Element.H
    assert Element.from_str("Cl") is Element.Cl
    assert Element.H.get_mass() == 1837.362218829611
    assert Element.B.get_mass() == 19707.247403384048
    assert Element.C.get_mass() == 21894.16671795623
    assert Element.O.get_mass() == 29165.12201514224
    assert Element.N.get_mass() == 25532.65213254827
    assert Element.Si.get_mass() == 51196.73452481201
    assert Element.S.get_mass() == 58450.91924794280
    assert Element.P.get_mass() == 56461.71406415092
    assert Element.Ca.get_mass() == pytest.approx(40.078 * 1837.362218829611 / 1.00794, rel=1e-12)


def test_atomic_radius_matches_rust_table() -> None:
    from rtip_jax.constants import atomic_radius

    assert atomic_radius("H") == pytest.approx(0.31 * ANGSTROM_TO_BOHR)
    assert atomic_radius("C") == pytest.approx(0.76 * ANGSTROM_TO_BOHR)
    assert atomic_radius("O") == pytest.approx(0.66 * ANGSTROM_TO_BOHR)
    assert atomic_radius("Ca") == pytest.approx(1.76 * ANGSTROM_TO_BOHR)


def test_unknown_element_and_unsupported_mass_are_explicit() -> None:
    with pytest.raises(UnknownElementError):
        Element.from_str("Xx")

    with pytest.raises(UnsupportedElementMassError):
        Element.He.get_mass()
