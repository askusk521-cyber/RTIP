"""DeePMD-kit external PES provider.

This provider replaces the old CP2K real-PES boundary. RTIP keeps Bohr,
Hartree, and Hartree/Bohr internally; DeePMD-kit inference uses Angstrom, eV,
and eV/Angstrom, so conversion is handled here at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from rtip_jax.constants import (
    BOHR_TO_ANGSTROM,
    EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR,
    EV_TO_HARTREE,
    element_symbol,
)
from rtip_jax.system import System


class DeepMDPESUnavailable(ImportError):
    """Raised when DeePMD-kit is required but not importable."""


@dataclass(frozen=True)
class DeepMDBoundary:
    """Inputs, outputs, and unit contract for a DeePMD real PES provider."""

    model: str
    type_map: tuple[str, ...] | None = None
    position_units: str = "Angstrom"
    energy_units: str = "eV"
    force_units: str = "eV/Angstrom"
    virial_units: str = "eV"
    internal_position_units: str = "Bohr"
    internal_energy_units: str = "Hartree"
    internal_force_units: str = "Hartree/Bohr"
    lifecycle: str = "DeepPot(model) once -> eval(coord, cell, atype) per frame"

    def __post_init__(self) -> None:
        if self.type_map is not None:
            object.__setattr__(self, "type_map", tuple(str(item) for item in self.type_map))


@dataclass(frozen=True)
class DeepMDResult:
    """Single-frame DeePMD result converted to RTIP internal units."""

    energy: float
    force: Any
    virial: Any | None


def _load_deep_pot(model: str) -> Any:
    try:
        from deepmd.infer import DeepPot
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise DeepMDPESUnavailable("DeePMD-kit is not installed; install rtip-jax[deepmd]") from exc
    return DeepPot(model)


def _model_type_map(deep_pot: Any) -> tuple[str, ...] | None:
    get_type_map = getattr(deep_pot, "get_type_map", None)
    if get_type_map is None:
        return None
    type_map = get_type_map()
    if type_map is None:
        return None
    return tuple(str(item) for item in type_map)


def _normalize_type_map(type_map: Sequence[str] | None, deep_pot: Any) -> tuple[str, ...]:
    normalized = tuple(str(item) for item in type_map) if type_map is not None else _model_type_map(deep_pot)
    if normalized is None:
        raise ValueError("DeepMD type_map is required when the model object does not expose get_type_map()")
    if len(normalized) == 0:
        raise ValueError("DeepMD type_map cannot be empty")
    return normalized


def deepmd_atype(system: System, type_map: Sequence[str]) -> list[int]:
    """Return DeePMD integer atom types in model `type_map` order."""

    if system.atom_type is None:
        raise ValueError("system.atom_type is required for DeePMD inference")
    mapping = {str(symbol): index for index, symbol in enumerate(type_map)}
    atype: list[int] = []
    for element in system.atom_type:
        symbol = element_symbol(element)
        try:
            atype.append(mapping[symbol])
        except KeyError as exc:
            raise ValueError(f"element {symbol!r} is not present in DeepMD type_map {tuple(type_map)!r}") from exc
    return atype


def deepmd_inputs(system: System, type_map: Sequence[str]) -> tuple[np.ndarray, np.ndarray | None, list[int]]:
    """Convert a `System` into DeepPot `eval` inputs."""

    coord = np.asarray(system.coord, dtype=np.float64) * BOHR_TO_ANGSTROM
    coord = coord.reshape(1, system.natom * 3)
    cell = None
    if system.cell is not None:
        cell = (np.asarray(system.cell, dtype=np.float64) * BOHR_TO_ANGSTROM).reshape(1, 9)
    return coord, cell, deepmd_atype(system, type_map)


def _single_frame_energy(energy_ev: Any) -> float:
    energy = np.asarray(energy_ev, dtype=np.float64).reshape(-1)
    if energy.size == 0:
        raise ValueError("DeepMD returned an empty energy array")
    return float(energy[0] * EV_TO_HARTREE)


def _single_frame_force(force_ev_per_angstrom: Any, natom: int) -> Any:
    force = np.asarray(force_ev_per_angstrom, dtype=np.float64)
    force = force.reshape(-1, natom, 3)
    if force.shape[0] == 0:
        raise ValueError("DeepMD returned an empty force array")
    return jnp.asarray(force[0] * EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR, dtype=jnp.float64)


def _single_frame_virial(virial_ev: Any | None) -> Any | None:
    if virial_ev is None:
        return None
    virial = np.asarray(virial_ev, dtype=np.float64).reshape(-1, 9)
    if virial.shape[0] == 0:
        raise ValueError("DeepMD returned an empty virial array")
    return jnp.asarray(virial[0] * EV_TO_HARTREE, dtype=jnp.float64)


class DeepMDPES:
    """Real PES provider backed by DeePMD-kit `DeepPot` inference."""

    def __init__(self, boundary: DeepMDBoundary, deep_pot: Any | None = None) -> None:
        self.boundary = boundary
        self.deep_pot = deep_pot if deep_pot is not None else _load_deep_pot(boundary.model)
        self.type_map = _normalize_type_map(boundary.type_map, self.deep_pot)

    def evaluate(self, system: System) -> DeepMDResult:
        coord, cell, atype = deepmd_inputs(system, self.type_map)
        energy_ev, force_ev_per_angstrom, virial_ev = self.deep_pot.eval(coord, cell, atype)
        return DeepMDResult(
            energy=_single_frame_energy(energy_ev),
            force=_single_frame_force(force_ev_per_angstrom, system.natom),
            virial=_single_frame_virial(virial_ev),
        )

    def get_energy(self, system: System) -> float:
        return self.evaluate(system).energy

    def get_energy_force(self, system: System) -> tuple[float, Any]:
        result = self.evaluate(system)
        return result.energy, result.force


__all__ = [
    "DeepMDBoundary",
    "DeepMDPES",
    "DeepMDPESUnavailable",
    "DeepMDResult",
    "deepmd_atype",
    "deepmd_inputs",
]
