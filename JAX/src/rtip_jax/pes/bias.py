"""Workflow-level bias-potential containers.

Rust source: `src/pes_exploration/potential.rs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

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


__all__ = ["AttractivePot", "RepulsivePot", "SynthesisPot"]

