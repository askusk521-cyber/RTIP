"""Workflow-level bias-potential containers.

Rust source: `src/pes_exploration/potential.rs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import jax.numpy as jnp

from rtip_jax.config import Para
from rtip_jax.system import System


@dataclass(frozen=True)
class RepulsivePot:
    """Configuration for pushing a system away from a local minimum."""

    local_min: System
    nearby_ts: Sequence[System] = ()
    para: Para = field(default_factory=Para)
    str_output_file: str = "rtip.pdb"
    output_file: str = "rtip.out"

    def __post_init__(self) -> None:
        object.__setattr__(self, "nearby_ts", tuple(self.nearby_ts))


@dataclass(frozen=True)
class AttractivePot:
    """Configuration for pulling an initial state toward a final state."""

    initial_state: System
    final_state: System
    para: Para = field(default_factory=Para)
    str_output_file: str = "rtip.pdb"
    output_file: str = "rtip.out"


@dataclass(frozen=True)
class SynthesisPot:
    """Configuration for synthesizing a product from separated molecules."""

    initial_state: System
    mol_index: Sequence[Sequence[int]]
    para: Para = field(default_factory=Para)
    str_output_file: str = "rtip.pdb"
    output_file: str = "rtip.out"

    def __post_init__(self) -> None:
        mol_index = tuple(tuple(int(index) for index in molecule) for molecule in self.mol_index)
        if len(mol_index) == 0:
            raise ValueError("mol_index must contain at least one molecule")
        for molecule in mol_index:
            if len(molecule) == 0:
                raise ValueError("mol_index cannot contain an empty molecule")
            if any(index < 0 or index >= self.initial_state.natom for index in molecule):
                raise ValueError("mol_index contains an atom index outside initial_state")
        object.__setattr__(self, "mol_index", mol_index)


@dataclass(frozen=True)
class DistanceRestraint:
    """A harmonic distance restraint for reaction-coordinate biasing.

    `target` is in Bohr and `k` is in Hartree / Bohr^2.
    """

    atom_i: int
    atom_j: int
    target: float
    k: float
    label: str = ""

    def __post_init__(self) -> None:
        atom_i = int(self.atom_i)
        atom_j = int(self.atom_j)
        if atom_i == atom_j:
            raise ValueError("distance restraint atoms must be distinct")
        if self.target < 0.0:
            raise ValueError("distance restraint target must be non-negative")
        if self.k < 0.0:
            raise ValueError("distance restraint force constant must be non-negative")
        object.__setattr__(self, "atom_i", atom_i)
        object.__setattr__(self, "atom_j", atom_j)
        object.__setattr__(self, "target", float(self.target))
        object.__setattr__(self, "k", float(self.k))
        object.__setattr__(self, "label", str(self.label))


def _coerce_distance_restraint(value: DistanceRestraint | Sequence[object]) -> DistanceRestraint:
    if isinstance(value, DistanceRestraint):
        return value
    if len(value) == 4:
        atom_i, atom_j, target, k = value
        return DistanceRestraint(int(atom_i), int(atom_j), float(target), float(k))
    if len(value) == 5:
        atom_i, atom_j, target, k, label = value
        return DistanceRestraint(int(atom_i), int(atom_j), float(target), float(k), str(label))
    raise ValueError("distance restraints must be DistanceRestraint or (atom_i, atom_j, target, k[, label])")


@dataclass(frozen=True)
class ReactionCoordinatePES:
    """Harmonic distance-coordinate bias potential."""

    restraints: Sequence[DistanceRestraint]

    def __post_init__(self) -> None:
        restraints = tuple(_coerce_distance_restraint(restraint) for restraint in self.restraints)
        if len(restraints) == 0:
            raise ValueError("reaction-coordinate bias requires at least one distance restraint")
        object.__setattr__(self, "restraints", restraints)

    def get_energy(self, system: System) -> float:
        energy, _force = self.get_energy_force(system)
        return energy

    def get_energy_force(self, system: System) -> tuple[float, object]:
        coord = jnp.asarray(system.coord, dtype=jnp.float64)
        force = jnp.zeros_like(coord)
        energy = jnp.asarray(0.0, dtype=jnp.float64)
        for restraint in self.restraints:
            if restraint.atom_i < 0 or restraint.atom_i >= system.natom:
                raise ValueError("distance restraint atom_i is outside the system")
            if restraint.atom_j < 0 or restraint.atom_j >= system.natom:
                raise ValueError("distance restraint atom_j is outside the system")
            diff = coord[restraint.atom_j] - coord[restraint.atom_i]
            distance = jnp.linalg.norm(diff)
            safe_distance = jnp.maximum(distance, 1.0e-12)
            delta = distance - restraint.target
            energy = energy + 0.5 * restraint.k * delta * delta
            force_i = restraint.k * delta * diff / safe_distance
            force = force.at[restraint.atom_i].add(force_i)
            force = force.at[restraint.atom_j].add(-force_i)
        return float(energy), force

    def distance_error(self, system: System) -> float:
        coord = jnp.asarray(system.coord, dtype=jnp.float64)
        error2 = jnp.asarray(0.0, dtype=jnp.float64)
        for restraint in self.restraints:
            diff = coord[restraint.atom_j] - coord[restraint.atom_i]
            distance = jnp.linalg.norm(diff)
            error = distance - restraint.target
            error2 = error2 + error * error
        return float(jnp.sqrt(error2))


@dataclass(frozen=True)
class ReactionCoordinatePot:
    """Configuration for MD with harmonic reaction-coordinate restraints."""

    initial_state: System
    restraints: Sequence[DistanceRestraint | Sequence[object]]
    para: Para = field(default_factory=Para)
    str_output_file: str = "rtip.pdb"
    output_file: str = "rtip.out"

    def __post_init__(self) -> None:
        restraints = tuple(_coerce_distance_restraint(restraint) for restraint in self.restraints)
        if len(restraints) == 0:
            raise ValueError("reaction-coordinate bias requires at least one distance restraint")
        for restraint in restraints:
            if restraint.atom_i < 0 or restraint.atom_i >= self.initial_state.natom:
                raise ValueError("distance restraint atom_i is outside initial_state")
            if restraint.atom_j < 0 or restraint.atom_j >= self.initial_state.natom:
                raise ValueError("distance restraint atom_j is outside initial_state")
        object.__setattr__(self, "restraints", restraints)


__all__ = [
    "AttractivePot",
    "DistanceRestraint",
    "ReactionCoordinatePES",
    "ReactionCoordinatePot",
    "RepulsivePot",
    "SynthesisPot",
]
