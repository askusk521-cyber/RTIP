"""Mathematical, physical, and chemical constants.

Rust source: `src/common/constants.rs`.
"""

from __future__ import annotations

from enum import Enum

from .errors import UnknownElementError, UnsupportedElementMassError, error_element, error_unsupported_mass

ROOT_RANK = 0

PI = 3.141592653589793
GOLDEN_RATIO1 = 0.618033988749895
GOLDEN_RATIO2 = 1.618033988749895

C_LIGHT = 299792458.0
H_PLANCK = 6.62606896e-34
BOLTZMANN = 1.3806504e-23
ELECTRONVOLT_TO_JOULE = 1.602176634e-19
RYBDERG = 10973731.568527

BOHR_TO_ANGSTROM = 0.52917720859
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

HARTREE_TO_JOULE = 2.0 * RYBDERG * H_PLANCK * C_LIGHT
JOULE_TO_HARTREE = 1.0 / HARTREE_TO_JOULE
HARTREE_TO_EV = HARTREE_TO_JOULE / ELECTRONVOLT_TO_JOULE
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR = EV_TO_HARTREE / ANGSTROM_TO_BOHR

AU_TO_FEMTOSECOND = 1.0e15 / (4.0 * PI * RYBDERG * C_LIGHT)
FEMTOSECOND_TO_AU = 1.0 / AU_TO_FEMTOSECOND


class Element(str, Enum):
    """Chemical elements accepted by the Rust parser."""

    H = "H"
    He = "He"
    Li = "Li"
    Be = "Be"
    B = "B"
    C = "C"
    N = "N"
    O = "O"
    F = "F"
    Ne = "Ne"
    Na = "Na"
    Mg = "Mg"
    Al = "Al"
    Si = "Si"
    P = "P"
    S = "S"
    Cl = "Cl"
    Ar = "Ar"
    K = "K"
    Ca = "Ca"
    Sc = "Sc"
    Ti = "Ti"
    V = "V"
    Cr = "Cr"
    Mn = "Mn"
    Fe = "Fe"
    Co = "Co"
    Ni = "Ni"
    Cu = "Cu"
    Zn = "Zn"
    Ga = "Ga"
    Ge = "Ge"
    As = "As"
    Se = "Se"
    Br = "Br"
    Kr = "Kr"
    Rb = "Rb"
    Sr = "Sr"
    Y = "Y"
    Zr = "Zr"
    Nb = "Nb"
    Mo = "Mo"
    Tc = "Tc"
    Ru = "Ru"
    Rh = "Rh"
    Pd = "Pd"
    Ag = "Ag"
    Cd = "Cd"
    In = "In"
    Sn = "Sn"
    Sb = "Sb"
    Te = "Te"
    I = "I"
    Xe = "Xe"
    Cs = "Cs"
    Ba = "Ba"
    Hf = "Hf"
    Ta = "Ta"
    W = "W"
    Re = "Re"
    Os = "Os"
    Ir = "Ir"
    Pt = "Pt"
    Au = "Au"
    Hg = "Hg"
    Tl = "Tl"
    Pb = "Pb"
    Bi = "Bi"
    Po = "Po"
    At = "At"
    Rn = "Rn"
    Fr = "Fr"
    Ra = "Ra"
    Rf = "Rf"
    Db = "Db"
    Sg = "Sg"
    Bh = "Bh"
    Hs = "Hs"
    Mt = "Mt"
    Ds = "Ds"
    Rg = "Rg"
    La = "La"
    Ce = "Ce"
    Pr = "Pr"
    Nd = "Nd"
    Pm = "Pm"
    Sm = "Sm"
    Eu = "Eu"
    Gd = "Gd"
    Tb = "Tb"
    Dy = "Dy"
    Ho = "Ho"
    Er = "Er"
    Tm = "Tm"
    Yb = "Yb"
    Lu = "Lu"
    Ac = "Ac"
    Th = "Th"
    Pa = "Pa"
    U = "U"
    Np = "Np"
    Pu = "Pu"
    Am = "Am"
    Cm = "Cm"
    Bk = "Bk"
    Cf = "Cf"
    Es = "Es"
    Fm = "Fm"
    Md = "Md"
    No = "No"
    Lr = "Lr"

    @classmethod
    def from_str(cls, symbol: str) -> "Element":
        try:
            return cls(symbol)
        except ValueError as exc:
            raise UnknownElementError(error_element(symbol)) from exc

    def get_mass(self) -> float:
        return atomic_mass(self)


_ATOMIC_MASS_AU = {
    Element.H: 1837.362218829611,
    Element.B: 19707.247403384048,
    Element.C: 21894.16671795623,
    Element.O: 29165.12201514224,
    Element.N: 25532.65213254827,
    Element.Si: 51196.73452481201,
    Element.S: 58450.91924794280,
    Element.P: 56461.71406415092,
}


def coerce_element(element: Element | str) -> Element:
    if isinstance(element, Element):
        return element
    return Element.from_str(element)


def element_symbol(element: Element | str) -> str:
    return coerce_element(element).value


def atomic_mass(element: Element | str) -> float:
    element = coerce_element(element)
    try:
        return _ATOMIC_MASS_AU[element]
    except KeyError as exc:
        raise UnsupportedElementMassError(error_unsupported_mass(element.value)) from exc
