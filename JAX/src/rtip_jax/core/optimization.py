"""Local optimization helpers.

Rust source: `src/pes_exploration/optimization.rs`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from rtip_jax.constants import GOLDEN_RATIO1, GOLDEN_RATIO2
from rtip_jax.errors import OptimizationError, error_min_1d
from rtip_jax.system import System


def min_1d(fun: Callable[[float], float], f0: float, epsilon: float) -> tuple[float, float]:
    """One-dimensional minimization with Rust's extrapolation/golden search.

    The function is assumed to decrease along the positive x direction near
    zero. This mirrors Rust `min_1d` and raises `OptimizationError` when that
    precondition is not met.
    """

    f0 = float(f0)
    epsilon = float(epsilon)

    delta_x = 0.01
    x0 = 0.0
    x1 = delta_x
    f1 = float(fun(x1))

    while f1 > f0 and delta_x > 1.0e-15:
        delta_x /= 10.0
        x1 = delta_x
        f1 = float(fun(x1))

    if f1 > f0:
        raise OptimizationError(error_min_1d())

    delta_x *= GOLDEN_RATIO2
    x2 = x1 + delta_x
    f2 = float(fun(x2))

    while f2 < f1 and x1 < 0.1:
        delta_x *= GOLDEN_RATIO2
        x0 = x1
        x1 = x2
        x2 = x1 + delta_x
        f1 = f2
        f2 = float(fun(x2))

    if f2 < f1 and x1 >= 0.1:
        return x1, f1

    a = x0
    b = x2
    u = x1
    v = a + GOLDEN_RATIO1 * (b - a)
    fu = f1
    fv = float(fun(v))

    while True:
        if fu > fv:
            if (fu - fv) < epsilon:
                return v, fv
            a = u
            u = v
            v = a + GOLDEN_RATIO1 * (b - a)
            fu = fv
            fv = float(fun(v))
        else:
            if (fv - fu) < epsilon:
                return u, fu
            b = v
            v = u
            u = b - GOLDEN_RATIO1 * (b - a)
            fv = fu
            fu = float(fun(u))


def min_1d_real_bias(
    real_pes: Any,
    bias_pes: Any,
    system: System,
    force_total: Any,
    pot_total: float,
    epsilon: float,
) -> Any:
    """Search a minimum along the total-force direction.

    Returns the coordinate displacement `delta_coord`, matching Rust
    `min_1d_real_bias`.
    """

    force_total = jnp.asarray(force_total, dtype=jnp.float64)
    if force_total.shape != system.coord.shape:
        raise ValueError(f"force_total must have shape {system.coord.shape}, got {force_total.shape}")
    f_total = jnp.sqrt(jnp.sum(force_total * force_total))
    if float(f_total) == 0.0:
        raise OptimizationError("force_total norm is zero; cannot search along a direction")

    def objective(x: float) -> float:
        displaced = system.with_coord(system.coord + force_total * (float(x) / f_total))
        return float(real_pes.get_energy(displaced) + bias_pes.get_energy(displaced))

    x_min, _pot_min = min_1d(objective, pot_total, epsilon)
    return force_total * (x_min / f_total)


__all__ = ["min_1d", "min_1d_real_bias"]
