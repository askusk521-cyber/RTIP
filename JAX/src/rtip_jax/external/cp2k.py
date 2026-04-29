"""Documented CP2K boundary from the Rust implementation.

CP2K is not ported in this migration. This module preserves the old contract as
legacy documentation. The production replacement provider is
`rtip_jax.external.deepmd.DeepMDPES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Cp2kPESUnavailable(NotImplementedError):
    """Raised when code tries to use the non-ported CP2K provider."""


@dataclass(frozen=True)
class Cp2kBoundary:
    """Original CP2K PES inputs, outputs, and unit contract.

    Rust source: `src/external/cp2k.rs`.
    """

    input_file: str
    output_file: str
    mpi_comm: int | None = None
    position_units: str = "Bohr"
    energy_units: str = "Hartree"
    force_units: str = "Hartree/Bohr"
    lifecycle: str = "create_force_env -> set_positions -> calc_energy(_force) -> destroy_force_env"


class Cp2kPES:
    """Placeholder for the Rust `Cp2kPES` type.

    The JAX project depends on a generic PES provider interface. Use
    `DeepMDPES` for production real-PES energy and force data.
    """

    def __init__(self, boundary: Cp2kBoundary) -> None:
        self.boundary = boundary

    def get_energy(self, system: Any) -> float:
        raise Cp2kPESUnavailable("CP2K is not implemented in the JAX migration")

    def get_energy_force(self, system: Any) -> tuple[float, Any]:
        raise Cp2kPESUnavailable("CP2K is not implemented in the JAX migration")
