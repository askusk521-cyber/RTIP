"""Pure molecular-dynamics helper functions.

Rust source: shared numerical logic from `src/pes_exploration/md.rs`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import jax.numpy as jnp

from rtip_jax.config import Para
from rtip_jax.constants import BOLTZMANN, FEMTOSECOND_TO_AU, HARTREE_TO_JOULE, atomic_mass
from rtip_jax.core.rtip import Rtip0PES, rti_dist
from rtip_jax.io.pdb import write_pdb
from rtip_jax.pes.base import PES
from rtip_jax.pes.bias import RepulsivePot
from rtip_jax.system import System
from rtip_jax.workflows.pathway_sampling import force_norm, perturb_system


def atom_masses(system: System) -> Any:
    """Return Rust-compatible atom masses for `system.atom_type`."""

    if system.atom_type is None:
        raise ValueError("system.atom_type is required for MD masses")
    return jnp.asarray([atomic_mass(element) for element in system.atom_type], dtype=jnp.float64)


def twice_kinetic_energy(velocity: Any, masses: Any) -> Any:
    """Return `sum_i m_i * |v_i|^2`, the value Rust uses before `kin *= 0.5`."""

    velocity = jnp.asarray(velocity, dtype=jnp.float64)
    masses = jnp.asarray(masses, dtype=jnp.float64)
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError(f"velocity must have shape (natom, 3), got {velocity.shape}")
    if masses.shape != (velocity.shape[0],):
        raise ValueError(f"masses must have shape ({velocity.shape[0]},), got {masses.shape}")
    return jnp.sum(masses[:, jnp.newaxis] * velocity * velocity)


def kinetic_energy(velocity: Any, masses: Any) -> Any:
    """Return kinetic energy in Hartree."""

    return 0.5 * twice_kinetic_energy(velocity, masses)


def temperature(velocity: Any, masses: Any) -> Any:
    """Return instantaneous temperature in K using the Rust formula."""

    velocity = jnp.asarray(velocity, dtype=jnp.float64)
    if velocity.shape[0] <= 1:
        raise ValueError("temperature requires at least two atoms")
    return twice_kinetic_energy(velocity, masses) * HARTREE_TO_JOULE / (
        BOLTZMANN * 3.0 * float(velocity.shape[0] - 1)
    )


def berendsen_lambda(temp: float, para: Para) -> Any:
    """Return the Berendsen thermostat scaling factor."""

    temp = jnp.maximum(jnp.asarray(temp, dtype=jnp.float64), 1.0)
    return jnp.sqrt(1.0 + (para.dt / para.tau) * (para.temp_bath / temp - 1.0))


def accelerations(force_total: Any, masses: Any) -> Any:
    """Return atomic accelerations from force and masses."""

    force_total = jnp.asarray(force_total, dtype=jnp.float64)
    masses = jnp.asarray(masses, dtype=jnp.float64)
    if force_total.ndim != 2 or force_total.shape[1] != 3:
        raise ValueError(f"force_total must have shape (natom, 3), got {force_total.shape}")
    if masses.shape != (force_total.shape[0],):
        raise ValueError(f"masses must have shape ({force_total.shape[0]},), got {masses.shape}")
    return force_total / masses[:, jnp.newaxis]


def leapfrog_first(coord: Any, velocity: Any, acceleration: Any, para: Para) -> tuple[Any, Any]:
    """First Rust leapfrog half-step: update velocity and coordinates."""

    coord = jnp.asarray(coord, dtype=jnp.float64)
    velocity = jnp.asarray(velocity, dtype=jnp.float64)
    acceleration = jnp.asarray(acceleration, dtype=jnp.float64)
    velocity_half = velocity + 0.5 * para.dt * FEMTOSECOND_TO_AU * acceleration
    coord_new = coord + para.dt * FEMTOSECOND_TO_AU * velocity_half
    return coord_new, velocity_half


def leapfrog_second(velocity_half: Any, acceleration: Any, lambda_scale: Any, para: Para) -> Any:
    """Second Rust leapfrog half-step followed by thermostat scaling."""

    velocity_half = jnp.asarray(velocity_half, dtype=jnp.float64)
    acceleration = jnp.asarray(acceleration, dtype=jnp.float64)
    return (velocity_half + 0.5 * para.dt * FEMTOSECOND_TO_AU * acceleration) * lambda_scale


@dataclass(frozen=True)
class MDStep:
    """One scalar-output row from an RTIP NVT MD workflow."""

    step: int
    time: float
    wall_time_s: float
    sigma_min: float
    temp: float
    temp_bath: float
    thermostat_lambda: float
    kin: float
    pot_real: float
    pot_bias: float
    pot_total: float
    f_real: float
    f_bias: float
    f_total: float
    rms_f_real: float
    state_decision: str = "md_running"


@dataclass(frozen=True)
class MDResult:
    """Final RTIP NVT MD state and scalar history."""

    system: System
    velocity: Any
    acceleration: Any
    history: tuple[MDStep, ...]


def _coords(system: System, indices: tuple[int, ...] | None) -> Any:
    if indices is None:
        return system.coord
    return system.coord[jnp.asarray(indices)]


def _with_atom_add_pot(system: System, indices: tuple[int, ...] | None) -> System:
    return replace(system, atom_add_pot=indices)


def _validate_md_para(para: Para) -> None:
    if para.max_step < 0:
        raise ValueError("max_step must be non-negative")
    if para.print_step <= 0:
        raise ValueError("print_step must be positive")
    if para.dt <= 0.0:
        raise ValueError("dt must be positive")
    if para.tau <= 0.0:
        raise ValueError("tau must be positive")


def _write_md_start(system: System, structure_file: str, output_file: str, write_outputs: bool) -> None:
    if not write_outputs:
        return
    Path(structure_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    write_pdb(system, structure_file, create_new_file=True, step=0)
    Path(output_file).write_text(
        f"{'step':>6s} {'time_fs':>15s} {'wall_time_s':>15s} {'rti_dist':>15s} "
        f"{'temp_K':>15s} {'temp_bath_K':>15s} {'thermo_lambda':>15s} {'kin_Ha':>15s} "
        f"{'pot_real_Ha':>15s} {'pot_rtip_Ha':>15s} {'pot_total_Ha':>15s} "
        f"{'f_real':>15s} {'f_rtip':>15s} {'f_total':>15s} {'rms_f_real':>15s} state_decision\n"
    )


def _write_md_step(system: System, row: MDStep, structure_file: str, output_file: str, para: Para, write_outputs: bool) -> None:
    if not write_outputs:
        return
    if row.step % para.print_step == 0:
        write_pdb(system, structure_file, create_new_file=False, step=row.step)
    with Path(output_file).open("a") as handle:
        handle.write(
            f"{row.step:6d} {row.time:15.8f} {row.wall_time_s:15.8f} {row.sigma_min:15.8f} "
            f"{row.temp:15.8f} {row.temp_bath:15.8f} {row.thermostat_lambda:15.8f} {row.kin:15.8f} "
            f"{row.pot_real:15.8f} {row.pot_bias:15.8f} {row.pot_total:15.8f} "
            f"{row.f_real:15.8f} {row.f_bias:15.8f} {row.f_total:15.8f} {row.rms_f_real:15.8f} "
            f"{row.state_decision}\n"
        )


def _rtip_md_bias(config: RepulsivePot, system: System, step: int) -> tuple[float, float, Any, float]:
    indices = config.local_min.atom_add_pot
    current = _with_atom_add_pot(system, indices)
    sigma_min = rti_dist(_coords(config.local_min, indices), _coords(system, indices))
    if config.para.scale_ts_sigma is None:
        sigma_ts = tuple(float(rti_dist(_coords(ts, indices), _coords(system, indices))) for ts in config.nearby_ts)
    else:
        sigma_ts = tuple(
            float(0.5 * config.para.scale_ts_sigma * rti_dist(_coords(ts, indices), _coords(config.local_min, indices)))
            for ts in config.nearby_ts
        )
    bias_pes = Rtip0PES(
        local_min=config.local_min,
        nearby_ts=config.nearby_ts,
        a_min=config.para.a0 * float(step),
        a_ts=config.para.a0 * float(step) * config.para.scale_ts_a0,
        sigma_min=float(sigma_min),
        sigma_ts=sigma_ts,
    )
    pot_bias, force_bias = bias_pes.get_energy_force(current)
    return float(sigma_min), float(pot_bias), force_bias, float(force_norm(force_bias))


def run_rtip_nvt_md(
    config: RepulsivePot,
    real_pes: PES,
    *,
    key: Any | None = None,
    perturb: bool = True,
    initial_velocity: Any | None = None,
    initial_acceleration: Any | None = None,
    write_outputs: bool = True,
) -> MDResult:
    """Run Rust-style RTIP NVT MD with a generic real PES provider."""

    _validate_md_para(config.para)
    if perturb:
        if key is None:
            raise ValueError("key is required when perturb=True")
        s = perturb_system(config.local_min, key, atom_indices=config.local_min.atom_add_pot)
    else:
        s = config.local_min

    masses = atom_masses(s)
    velocity = jnp.zeros_like(s.coord) if initial_velocity is None else jnp.asarray(initial_velocity, dtype=jnp.float64)
    acceleration = (
        jnp.zeros_like(s.coord) if initial_acceleration is None else jnp.asarray(initial_acceleration, dtype=jnp.float64)
    )
    if velocity.shape != s.coord.shape:
        raise ValueError(f"initial_velocity must have shape {s.coord.shape}, got {velocity.shape}")
    if acceleration.shape != s.coord.shape:
        raise ValueError(f"initial_acceleration must have shape {s.coord.shape}, got {acceleration.shape}")

    _write_md_start(config.local_min, config.str_output_file, config.output_file, write_outputs)
    history: list[MDStep] = []
    time = 0.0
    started_at = perf_counter()
    for step in range(1, config.para.max_step + 1):
        time += config.para.dt
        coord_new, velocity_half = leapfrog_first(s.coord, velocity, acceleration, config.para)
        s = s.with_coord(coord_new)
        lambda_scale = berendsen_lambda(temperature(velocity_half, masses), config.para)

        pot_real, force_real = real_pes.get_energy_force(s)
        s = s.with_pot(float(pot_real))
        f_real = float(force_norm(force_real))

        sigma_min, pot_bias, force_bias, f_bias = _rtip_md_bias(config, s, step)
        force_total = jnp.asarray(force_real, dtype=jnp.float64) + jnp.asarray(force_bias, dtype=jnp.float64)
        acceleration = accelerations(force_total, masses)
        velocity = leapfrog_second(velocity_half, acceleration, lambda_scale, config.para)
        temp = temperature(velocity, masses)
        kin = kinetic_energy(velocity, masses)
        state_decision = "max_step" if step == config.para.max_step else "md_running"

        row = MDStep(
            step=step,
            time=float(time),
            wall_time_s=perf_counter() - started_at,
            sigma_min=sigma_min,
            temp=float(temp),
            temp_bath=float(config.para.temp_bath),
            thermostat_lambda=float(lambda_scale),
            kin=float(kin),
            pot_real=float(pot_real),
            pot_bias=pot_bias,
            pot_total=float(pot_real) + float(pot_bias),
            f_real=f_real,
            f_bias=f_bias,
            f_total=float(force_norm(force_total)),
            rms_f_real=float(f_real) / float(jnp.sqrt(float(s.natom))),
            state_decision=state_decision,
        )
        history.append(row)
        _write_md_step(s, row, config.str_output_file, config.output_file, config.para, write_outputs)

    return MDResult(system=s, velocity=velocity, acceleration=acceleration, history=tuple(history))


__all__ = [
    "accelerations",
    "atom_masses",
    "berendsen_lambda",
    "kinetic_energy",
    "leapfrog_first",
    "leapfrog_second",
    "MDResult",
    "MDStep",
    "run_rtip_nvt_md",
    "temperature",
    "twice_kinetic_energy",
]
