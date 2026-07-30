"""Tests for the size-consistency (``size_scaling``) bias amplitude fix.

Background: ``rti_dist`` grows as ``sqrt(N)`` with the number of biased atoms,
so ``sigma^2 proportional to N`` while the amplitude ``a = a0*step`` is a fixed
scalar.  The dominant bias force term ``~ pot_terms / sigma^2`` therefore makes
the per-atom bias force scale as ``a0 / N`` and dilutes the bias in large
systems.  With ``size_scaling`` enabled the effective amplitude is multiplied by
``n_bias`` (power = 1) to cancel the ``1/N`` dilution.  The Gaussian *shape* is
unchanged (the ``sqrt(N)`` factors in ``r^2 / 2 sigma^2`` cancel).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import random

from rtip_jax.config import Para
from rtip_jax.core.rtip import Rtip0PES, rti_dist
from rtip_jax.pes import HarmonicPES, RepulsivePot
from rtip_jax.system import System
from rtip_jax.workflows import run_rtip_nvt_md
from rtip_jax.workflows.md import _bias_force_limit


# ---------------------------------------------------------------------------
# bias_amplitude: envelope invariance / scaling
# ---------------------------------------------------------------------------


def test_bias_amplitude_unchanged_when_size_scaling_disabled() -> None:
    para = Para(a0=0.002, size_scaling=False)
    for n_bias in (None, 1, 5, 50):
        # growing
        assert para.bias_amplitude(100, n_bias=n_bias) == para.bias_amplitude(100)
        assert para.bias_amplitude(100, n_bias=n_bias) == para.a0 * 100.0
        # reducing
        reducing = para.bias_amplitude(
            120, bias_phase="reducing", step_reduction_started=100, n_bias=n_bias,
        )
        assert reducing == para.bias_amplitude(
            120, bias_phase="reducing", step_reduction_started=100,
        )


def test_bias_amplitude_scales_by_n_bias_when_enabled() -> None:
    base = Para(a0=0.002, size_scaling=False)
    scaled = Para(a0=0.002, size_scaling=True)

    for n_bias in (1, 5, 50):
        # growing phase
        g_base = base.bias_amplitude(100)
        g_scaled = scaled.bias_amplitude(100, n_bias=n_bias)
        assert math.isclose(g_scaled, g_base * n_bias, rel_tol=1e-12)

        # reducing phase
        r_base = base.bias_amplitude(120, bias_phase="reducing", step_reduction_started=100)
        r_scaled = scaled.bias_amplitude(
            120, bias_phase="reducing", step_reduction_started=100, n_bias=n_bias,
        )
        assert math.isclose(r_scaled, r_base * n_bias, rel_tol=1e-12)


def test_bias_amplitude_off_phase_is_zero_regardless_of_scaling() -> None:
    for size_scaling in (False, True):
        para = Para(size_scaling=size_scaling)
        assert para.bias_amplitude(100, bias_phase="off", n_bias=50) == 0.0


def test_bias_amplitude_scaling_noop_for_missing_or_zero_n_bias() -> None:
    para = Para(a0=0.002, size_scaling=True)
    # n_bias None or 0 must not scale (scale falls back to 1.0)
    assert para.bias_amplitude(100, n_bias=None) == para.a0 * 100.0
    assert para.bias_amplitude(100, n_bias=0) == para.a0 * 100.0


# ---------------------------------------------------------------------------
# Per-atom bias force: N-dependence with / without scaling
# ---------------------------------------------------------------------------


def _self_similar_pair(n: int, delta: float = 0.05, spacing: float = 2.0) -> tuple[System, System]:
    """Return (reference, current) systems with an N-independent per-atom RMSD.

    A zig-zag transverse deformation is an *internal* distortion that survives
    RTI alignment, so ``rti_dist ~ delta*sqrt(N)`` while the per-atom geometry
    stays self-similar as ``N`` grows.
    """
    ref = jnp.asarray([[i * spacing, 0.0, 0.0] for i in range(n)], dtype=jnp.float64)
    cur = jnp.asarray(
        [[i * spacing, delta * (-1.0) ** i, 0.0] for i in range(n)], dtype=jnp.float64,
    )
    return System(coord=ref), System(coord=cur)


def _mean_per_atom_bias_force(n: int, size_scaling: bool, *, step: int = 100, a0: float = 0.002) -> float:
    """Build the exact MD attractive bias for a self-similar pair and measure |f|.

    Mirrors ``_attractive_md_bias`` (whole system biased ``->`` ``n_bias = n``,
    ``sigma_min = rti_dist(reference, current)``).
    """
    reference, current = _self_similar_pair(n)
    sigma = float(rti_dist(reference.coord, current.coord))
    para = Para(a0=a0, size_scaling=size_scaling)
    amplitude = para.bias_amplitude(step, n_bias=n)
    pes = Rtip0PES(
        local_min=reference, nearby_ts=(), a_min=-amplitude, a_ts=0.0,
        sigma_min=sigma, sigma_ts=(),
    )
    _, force = pes.get_energy_force(current)
    return float(jnp.mean(jnp.linalg.norm(force, axis=1)))


def test_per_atom_bias_force_decays_like_one_over_n_without_scaling() -> None:
    small = _mean_per_atom_bias_force(5, size_scaling=False)
    large = _mean_per_atom_bias_force(50, size_scaling=False)
    # Without scaling the per-atom bias force falls off ~ 1/N (here 5/50 = 0.1).
    assert large < 0.5 * small
    assert math.isclose(large / small, 5.0 / 50.0, rel_tol=0.25)


def test_per_atom_bias_force_size_independent_with_scaling() -> None:
    small = _mean_per_atom_bias_force(5, size_scaling=True)
    large = _mean_per_atom_bias_force(50, size_scaling=True)
    # With scaling the per-atom bias force no longer decays with N.
    assert 0.5 < (large / small) < 2.0


# ---------------------------------------------------------------------------
# Stop-threshold size adaptation
# ---------------------------------------------------------------------------


def test_bias_force_limit_is_size_adaptive_only_when_enabled() -> None:
    off = Para(size_scaling=False)
    on = Para(size_scaling=True)
    for n_bias in (1, 4, 50):
        assert _bias_force_limit(off, n_bias) == 1000.0
        assert math.isclose(_bias_force_limit(on, n_bias), 1000.0 * math.sqrt(n_bias), rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Workflow regression + scaling propagation
# ---------------------------------------------------------------------------


def _repulsive_config(size_scaling: bool, max_step: int = 3) -> RepulsivePot:
    local_min = System(
        coord=jnp.asarray(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.4, 0.0], [0.0, 0.0, 1.4]],
            dtype=jnp.float64,
        ),
        atom_type=("H", "H", "H", "H"),
    )
    para = Para(max_step=max_step, print_step=1, dt=0.1, tau=100.0, temp_bath=300.0, size_scaling=size_scaling)
    return RepulsivePot(local_min=local_min, nearby_ts=(), para=para)


def test_md_repulsive_disabled_scaling_is_deterministic_baseline() -> None:
    key = random.PRNGKey(7)
    r1 = run_rtip_nvt_md(_repulsive_config(False), HarmonicPES(k=0.1), key=key, write_outputs=False)
    r2 = run_rtip_nvt_md(_repulsive_config(False), HarmonicPES(k=0.1), key=key, write_outputs=False)
    for a, b in zip(r1.history, r2.history):
        assert a.pot_bias == b.pot_bias
        assert a.f_bias == b.f_bias
        assert a.state_decision == b.state_decision


def test_md_size_scaling_multiplies_step1_bias_by_n_bias() -> None:
    key = random.PRNGKey(7)
    natom = 4  # whole system biased => n_bias = natom
    off = run_rtip_nvt_md(_repulsive_config(False), HarmonicPES(k=0.1), key=key, write_outputs=False)
    on = run_rtip_nvt_md(_repulsive_config(True), HarmonicPES(k=0.1), key=key, write_outputs=False)

    # Step 1 bias is computed in the "growing" phase for both runs, before any
    # phase transition or temperature feedback can diverge, so it is directly
    # comparable: energy and force both scale linearly with the amplitude.
    assert math.isclose(on.history[0].pot_bias, off.history[0].pot_bias * natom, rel_tol=1e-9)
    assert math.isclose(on.history[0].f_bias, off.history[0].f_bias * natom, rel_tol=1e-9)


def test_md_large_system_with_scaling_runs_without_premature_bias_force_stop() -> None:
    n = 12
    coord = jnp.asarray([[i * 1.6, 0.0, 0.0] for i in range(n)], dtype=jnp.float64)
    local_min = System(coord=coord, atom_type=tuple("H" for _ in range(n)))
    para = Para(max_step=3, print_step=1, dt=0.1, tau=100.0, temp_bath=300.0, size_scaling=True)
    config = RepulsivePot(local_min=local_min, nearby_ts=(), para=para)

    result = run_rtip_nvt_md(config, HarmonicPES(k=0.1), key=random.PRNGKey(3), write_outputs=False)

    assert len(result.history) == 3
    assert all(row.state_decision != "stop_large_bias_force" for row in result.history)
