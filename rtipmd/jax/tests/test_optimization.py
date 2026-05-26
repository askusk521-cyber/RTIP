from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from rtip_jax.core.optimization import min_1d, min_1d_real_bias
from rtip_jax.errors import OptimizationError
from rtip_jax.system import System


def test_min_1d_finds_nearby_quadratic_minimum() -> None:
    x_min, f_min = min_1d(lambda x: (x - 0.05) ** 2, f0=0.05**2, epsilon=1e-12)

    assert abs(x_min - 0.05) < 1e-4
    assert f_min < 1e-8


def test_min_1d_rejects_function_that_increases_from_zero() -> None:
    with pytest.raises(OptimizationError):
        min_1d(lambda x: (x + 1.0) ** 2, f0=1.0, epsilon=1e-12)


def test_min_1d_uses_rust_step_cap_when_function_keeps_decreasing() -> None:
    x_min, f_min = min_1d(lambda x: -x, f0=0.0, epsilon=1e-12)

    assert x_min >= 0.1
    assert f_min == -x_min


@dataclass(frozen=True)
class HarmonicPES:
    target: jnp.ndarray
    scale: float = 1.0

    def get_energy(self, system: System) -> float:
        diff = system.coord - self.target
        return float(0.5 * self.scale * jnp.sum(diff * diff))

    def get_energy_force(self, system: System):
        diff = system.coord - self.target
        return self.get_energy(system), -self.scale * diff


@dataclass(frozen=True)
class ZeroPES:
    def get_energy(self, system: System) -> float:
        return 0.0

    def get_energy_force(self, system: System):
        return 0.0, jnp.zeros_like(system.coord)


def test_min_1d_real_bias_returns_displacement_along_force_direction() -> None:
    system = System(coord=[[0.0, 0.0, 0.0]])
    target = jnp.asarray([[0.05, 0.0, 0.0]], dtype=jnp.float64)
    real_pes = HarmonicPES(target=target)
    bias_pes = ZeroPES()
    pot_real, force_real = real_pes.get_energy_force(system)

    delta = min_1d_real_bias(
        real_pes,
        bias_pes,
        system,
        force_real,
        pot_total=pot_real,
        epsilon=1e-12,
    )

    assert jnp.allclose(system.coord + delta, target, atol=1e-4)


def test_min_1d_real_bias_validates_force_shape() -> None:
    system = System(coord=[[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError):
        min_1d_real_bias(ZeroPES(), ZeroPES(), system, jnp.zeros((2, 3)), 0.0, 1e-12)


def test_min_1d_real_bias_rejects_zero_force_direction() -> None:
    system = System(coord=[[0.0, 0.0, 0.0]])

    with pytest.raises(OptimizationError):
        min_1d_real_bias(ZeroPES(), ZeroPES(), system, jnp.zeros_like(system.coord), 0.0, 1e-12)
