"""Shared potential-energy surface interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax.numpy as jnp


@dataclass(frozen=True)
class EnergyForce:
    """Energy and force returned by a PES provider."""

    energy: float
    force: Any


class PES(Protocol):
    """Protocol matching the Rust `PES` trait."""

    def get_energy(self, system: Any) -> float:
        """Return potential energy for `system`."""

    def get_energy_force(self, system: Any) -> tuple[float, Any]:
        """Return potential energy and force for `system`."""


@dataclass(frozen=True)
class SumPES:
    """A PES that sums multiple PES providers."""

    components: tuple[PES, ...]

    def __init__(self, components: tuple[PES, ...] | list[PES]) -> None:
        if len(components) == 0:
            raise ValueError("SumPES requires at least one component")
        object.__setattr__(self, "components", tuple(components))

    def get_energy(self, system: Any) -> float:
        return float(sum(component.get_energy(system) for component in self.components))

    def get_energy_force(self, system: Any) -> tuple[float, Any]:
        total_energy = 0.0
        total_force = None
        for component in self.components:
            energy, force = component.get_energy_force(system)
            total_energy += float(energy)
            force = jnp.asarray(force, dtype=jnp.float64)
            total_force = force if total_force is None else total_force + force
        return total_energy, total_force


@dataclass(frozen=True)
class ZeroPES:
    """A zero-valued PES used when a workflow temporarily disables bias."""

    def get_energy(self, system: Any) -> float:
        return 0.0

    def get_energy_force(self, system: Any) -> tuple[float, Any]:
        return 0.0, jnp.zeros_like(system.coord)


@dataclass(frozen=True)
class HarmonicPES:
    """Deterministic harmonic PES for tests, examples, and CLI smoke runs."""

    k: float = 1.0
    center: Any | None = None

    def _displacement(self, system: Any) -> Any:
        coord = jnp.asarray(system.coord, dtype=jnp.float64)
        if self.center is None:
            center = jnp.zeros_like(coord)
        else:
            center = jnp.asarray(self.center, dtype=jnp.float64)
            if center.shape != coord.shape:
                raise ValueError(f"center must have shape {coord.shape}, got {center.shape}")
        return coord - center

    def get_energy(self, system: Any) -> float:
        displacement = self._displacement(system)
        return float(0.5 * float(self.k) * jnp.sum(displacement * displacement))

    def get_energy_force(self, system: Any) -> tuple[float, Any]:
        displacement = self._displacement(system)
        energy = 0.5 * float(self.k) * jnp.sum(displacement * displacement)
        force = -float(self.k) * displacement
        return float(energy), force
