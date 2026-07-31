"""RTIP-driven NVT MD, ported 1:1 from Rust `src/pes_exploration/md.rs`.

Two workflows are provided, matching the upstream implementations:

* ``repulsive_md``  – `RepulsivePot::rtip_nvt_md`; the RTIP amplitude grows
  linearly as ``a0 * step`` and pushes the system away from a local minimum
  (and optionally nearby transition states).
* ``evolution_md``  – `EvolutionPot::rtip_nvt_md`; an attractive RTIP pulls
  the molecules toward a virtual final state with coincident centroids.  The
  amplitude follows the three-state machine Increasing / Decreasing / Falling,
  driven by bond-variation detection and molecular-proximity checks.  This is
  the algorithm behind the formose-reaction RTIP-MD study.

The integrator is leapfrog with a Berendsen thermostat, exactly as in the Rust
implementation (units: fs, Bohr, Hartree).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jax.numpy as jnp

from rtip_jax.config import Para
from rtip_jax.constants import (
    BOLTZMANN,
    FEMTOSECOND_TO_AU,
    HARTREE_TO_JOULE,
    Element,
    atomic_mass,
    atomic_radius,
    element_symbol,
)
from rtip_jax.core.rtip import Rtip0PES, rti_dist
from rtip_jax.io.pdb import write_pdb
from rtip_jax.pes.base import PES
from rtip_jax.pes.bias import EvolutionPot, RepulsivePot
from rtip_jax.system import (
    System,
    dist_mat_bohr,
    get_adj_mat,
    judge_adj_of_mol,
    judge_variation_of_bonding,
    split_into_mol,
)
from rtip_jax.workflows.pathway_sampling import force_norm, perturb_system


# ---------------------------------------------------------------------------
# Numeric helpers (Rust md.rs, exact unit conventions)
# ---------------------------------------------------------------------------


def atom_masses(system: System) -> Any:
    """Return atomic masses in atomic units for `system.atom_type`."""

    if system.atom_type is None:
        raise ValueError("system.atom_type is required for MD masses")
    return jnp.asarray([atomic_mass(element) for element in system.atom_type], dtype=jnp.float64)


def twice_kinetic_energy(velocity: Any, masses: Any) -> Any:
    """Return ``sum_i m_i |v_i|^2`` (Hartree), the value before ``*0.5``."""

    velocity = jnp.asarray(velocity, dtype=jnp.float64)
    masses = jnp.asarray(masses, dtype=jnp.float64)
    return jnp.sum(masses[:, jnp.newaxis] * velocity * velocity)


def kinetic_energy(velocity: Any, masses: Any) -> Any:
    """Return kinetic energy in Hartree."""

    return 0.5 * twice_kinetic_energy(velocity, masses)


def temperature(velocity: Any, masses: Any) -> Any:
    """Return instantaneous temperature in K (Rust formula)."""

    velocity = jnp.asarray(velocity, dtype=jnp.float64)
    if velocity.shape[0] <= 1:
        raise ValueError("temperature requires at least two atoms")
    return twice_kinetic_energy(velocity, masses) * HARTREE_TO_JOULE / (
        BOLTZMANN * 3.0 * float(velocity.shape[0] - 1)
    )


def berendsen_lambda(temp: Any, para: Para) -> Any:
    """Return the Berendsen thermostat scaling factor (temp clamped to >= 1 K)."""

    temp = jnp.maximum(jnp.asarray(temp, dtype=jnp.float64), 1.0)
    return jnp.sqrt(1.0 + (para.dt / para.tau) * (para.temp_bath / temp - 1.0))


def accelerations(force_total: Any, masses: Any) -> Any:
    """Return atomic accelerations from total force and masses."""

    force_total = jnp.asarray(force_total, dtype=jnp.float64)
    masses = jnp.asarray(masses, dtype=jnp.float64)
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


# ---------------------------------------------------------------------------
# Output records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MDStep:
    """One scalar-output row from an RTIP NVT MD workflow (Rust rtip.out)."""

    step: int
    time: float
    rti_dist: float
    temp: float
    kin: float
    pot_real: float
    pot_rtip: float
    f_real: float
    f_rtip: float
    rtip_status: str = ""


@dataclass(frozen=True)
class MDResult:
    """Final RTIP NVT MD state and scalar history."""

    system: System
    velocity: Any
    acceleration: Any
    history: tuple[MDStep, ...]


def _write_header(system: System, structure_file: str, output_file: str, write_outputs: bool) -> None:
    if not write_outputs:
        return
    Path(structure_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    write_pdb(system, structure_file, create_new_file=True, step=0)
    Path(output_file).write_text(
        f"{'step':>6s} {'time_fs':>15s} {'rti_dist':>15s} {'temp_K':>15s} {'kin_Ha':>15s} "
        f"{'pot_real_Ha':>15s} {'pot_rtip_Ha':>15s} {'f_real':>15s} {'f_rtip':>15s} rtip_status\n"
    )


def _write_step(system: System, row: MDStep, structure_file: str, output_file: str, para: Para, write_outputs: bool) -> None:
    if not write_outputs:
        return
    if row.step % para.print_step == 0:
        write_pdb(system, structure_file, create_new_file=False, step=row.step)
    with Path(output_file).open("a") as handle:
        handle.write(
            f"{row.step:6d} {row.time:15.8f} {row.rti_dist:15.8f} {row.temp:15.8f} {row.kin:15.8f} "
            f"{row.pot_real:15.8f} {row.pot_rtip:15.8f} {row.f_real:15.8f} {row.f_rtip:15.8f} "
            f"{row.rtip_status}\n"
        )


def _validate_para(para: Para) -> None:
    if para.max_step < 0:
        raise ValueError("max_step must be non-negative")
    if para.print_step <= 0:
        raise ValueError("print_step must be positive")
    if para.dt <= 0.0:
        raise ValueError("dt must be positive")
    if para.tau <= 0.0:
        raise ValueError("tau must be positive")


def _atomic_radii(system: System) -> Any:
    """Return per-atom covalent radii in Bohr (Rust `get_atomic_radii`)."""

    import numpy as np

    if system.atom_type is None:
        raise ValueError("system.atom_type is required for bond detection")
    return np.asarray([atomic_radius(element) for element in system.atom_type], dtype=np.float64)


# ---------------------------------------------------------------------------
# RepulsivePot MD  (Rust `RepulsivePot::rtip_nvt_md`)
# ---------------------------------------------------------------------------


def repulsive_md(
    config: RepulsivePot,
    real_pes: PES,
    *,
    key: Any | None = None,
    perturb: bool = False,
    initial_velocity: Any | None = None,
    initial_acceleration: Any | None = None,
    write_outputs: bool = True,
) -> MDResult:
    """NVT MD driven by a repulsive RTIP with amplitude ``a0 * step``.

    The RTIP is built from the local minimum (and nearby TS, when provided)
    exactly as in Rust: ``a_min = a0 * i`` and ``a_ts = a_min * scale_ts_a0``.
    """

    _validate_para(config.para)
    para = config.para
    if perturb:
        if key is None:
            raise ValueError("key is required when perturb=True")
        s = perturb_system(config.local_min, key, atom_indices=config.local_min.atom_add_pot)
    else:
        s = config.local_min

    masses = atom_masses(s)
    velocity = jnp.zeros_like(s.coord) if initial_velocity is None else jnp.asarray(initial_velocity, dtype=jnp.float64)
    acceleration = (
        jnp.zeros_like(s.coord)
        if initial_acceleration is None
        else jnp.asarray(initial_acceleration, dtype=jnp.float64)
    )

    _write_header(config.local_min, config.str_output_file, config.output_file, write_outputs)
    history: list[MDStep] = []
    time = 0.0

    for step in range(1, para.max_step + 1):
        # First leapfrog half-step.
        time += para.dt
        coord_new, velocity_half = leapfrog_first(s.coord, velocity, acceleration, para)
        s = s.with_coord(coord_new)

        # Berendsen thermostat (half-step kinetic temperature).
        lambda_scale = berendsen_lambda(temperature(velocity_half, masses), para)

        # Real PES.
        pot_real, force_real = real_pes.get_energy_force(s)
        s = s.with_pot(float(pot_real))
        f_real = float(force_norm(force_real))

        # RTIP with linearly growing amplitude (a_min = a0 * step).
        indices = config.local_min.atom_add_pot
        sigma_min = float(rti_dist(_coords(config.local_min, indices), _coords(s, indices)))
        if para.scale_ts_sigma is None:
            sigma_ts = tuple(
                float(rti_dist(_coords(ts, indices), _coords(s, indices))) for ts in config.nearby_ts
            )
        else:
            sigma_ts = tuple(
                float(0.5 * para.scale_ts_sigma * rti_dist(_coords(ts, indices), _coords(config.local_min, indices)))
                for ts in config.nearby_ts
            )
        amplitude = para.a0 * float(step)
        bias_pes = Rtip0PES(
            local_min=config.local_min,
            nearby_ts=config.nearby_ts,
            a_min=amplitude,
            a_ts=amplitude * para.scale_ts_a0,
            sigma_min=sigma_min,
            sigma_ts=sigma_ts,
        )
        current = _with_atom_add_pot(s, indices)
        pot_bias, force_bias = bias_pes.get_energy_force(current)
        f_bias = float(force_norm(force_bias))

        # Forces -> acceleration; second leapfrog half-step + thermostat.
        force_total = jnp.asarray(force_real, dtype=jnp.float64) + jnp.asarray(force_bias, dtype=jnp.float64)
        acceleration = accelerations(force_total, masses)
        velocity = leapfrog_second(velocity_half, acceleration, lambda_scale, para)
        temp = float(temperature(velocity, masses))
        kin = float(kinetic_energy(velocity, masses))

        row = MDStep(
            step=step,
            time=time,
            rti_dist=sigma_min,
            temp=temp,
            kin=kin,
            pot_real=float(pot_real),
            pot_rtip=float(pot_bias),
            f_real=f_real,
            f_rtip=f_bias,
        )
        history.append(row)
        _write_step(s, row, config.str_output_file, config.output_file, para, write_outputs)

    return MDResult(system=s, velocity=velocity, acceleration=acceleration, history=tuple(history))


# ---------------------------------------------------------------------------
# EvolutionPot MD  (Rust `EvolutionPot::rtip_nvt_md`)
# ---------------------------------------------------------------------------


def _coords(system: System, indices: tuple[int, ...] | None) -> Any:
    if indices is None:
        return system.coord
    return system.coord[jnp.asarray(indices)]


def _with_atom_add_pot(system: System, indices: tuple[int, ...] | None) -> System:
    from dataclasses import replace

    return replace(system, atom_add_pot=indices)


def _final_state_coords(coord: Any, mol_index: tuple[tuple[int, ...], ...]) -> Any:
    """Virtual final state: move each molecule centroid toward the overall
    centroid by 0.01 of the current separation (Rust EvolutionPot)."""

    import numpy as np

    coord_np = np.asarray(coord, dtype=np.float64)
    o_all = coord_np.mean(axis=0)
    final_coord = coord_np.copy()
    for molecule in mol_index:
        if not molecule:
            continue
        o_i = coord_np[list(molecule)].mean(axis=0)
        final_coord[list(molecule)] += 0.01 * (o_all - o_i)
    return final_coord


def evolution_md(
    config: EvolutionPot,
    real_pes: PES,
    *,
    initial_velocity: Any | None = None,
    initial_acceleration: Any | None = None,
    write_outputs: bool = True,
) -> MDResult:
    """Formose-style evolution MD (Rust `EvolutionPot::rtip_nvt_md`).

    An attractive RTIP toward a coincident-centroid virtual final state drives
    the molecular assembly; bond-variation detection switches the RTIP from
    Increasing to Decreasing; the RTIP resets to Increasing once the depth
    returns to ``decreasing_bound`` of its peak.  Bond connectivity is
    re-derived at the beginning of every search round.
    """

    _validate_para(config.para)
    para = config.para
    s = config.initial_state
    indices = s.atom_add_pot

    if indices is not None:
        fragment = s.fragment(indices)
        system_for_bonds = fragment
        atomic_type: tuple[Element, ...] | None = fragment.atom_type
    else:
        system_for_bonds = s
        atomic_type = s.atom_type
    if atomic_type is None:
        raise ValueError("initial_state.atom_type is required for evolution MD")

    atomic_radii = _atomic_radii(system_for_bonds)
    dist_mat = dist_mat_bohr(system_for_bonds.coord)
    adj_mat = get_adj_mat(atomic_type, atomic_radii, dist_mat, 1.25, para.ignored_pair)
    mol_index = split_into_mol(atomic_radii, dist_mat, 1.25)

    masses = atom_masses(s)
    velocity = jnp.zeros_like(s.coord) if initial_velocity is None else jnp.asarray(initial_velocity, dtype=jnp.float64)
    acceleration = (
        jnp.zeros_like(s.coord)
        if initial_acceleration is None
        else jnp.asarray(initial_acceleration, dtype=jnp.float64)
    )

    _write_header(s, config.str_output_file, config.output_file, write_outputs)
    if write_outputs:
        Path(config.dec_output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(config.dec_output_file).write_text("      begin_step        end_step\n")

    a_min = 0.0
    a_min_threshold = 0.0
    rtip_status = "Increasing"
    history: list[MDStep] = []
    time = 0.0

    for step in range(1, para.max_step + 1):
        # --- RTIP control (bond-variation detection) ---
        if rtip_status != "Decreasing" and judge_variation_of_bonding(
            atomic_radii, dist_mat, adj_mat, 1.0, 1.6
        ):
            a_min_threshold = a_min * para.decreasing_bound
            rtip_status = "Decreasing"
            if write_outputs:
                with Path(config.dec_output_file).open("a") as handle:
                    handle.write(f"{step:16d}")
        if rtip_status == "Decreasing" and a_min > a_min_threshold:
            adj_mat = get_adj_mat(atomic_type, atomic_radii, dist_mat, 1.25, para.ignored_pair)
            mol_index = split_into_mol(atomic_radii, dist_mat, 1.25)
            rtip_status = "Increasing"
            if write_outputs:
                with Path(config.dec_output_file).open("a") as handle:
                    handle.write(f"{step - 1:16d}\n")
        if para.split_step is not None and step % para.split_step == 0:
            mol_index = split_into_mol(atomic_radii, dist_mat, 1.25)

        # --- RTIP amplitude update ---
        if rtip_status == "Increasing":
            a_min -= para.a0
        elif rtip_status == "Decreasing":
            a_min += para.a0 * para.decreasing_multiple
        else:  # Falling
            a_min += para.a0

        # --- First leapfrog half-step ---
        time += para.dt
        coord_new, velocity_half = leapfrog_first(s.coord, velocity, acceleration, para)
        s = s.with_coord(coord_new)

        # --- Berendsen thermostat ---
        lambda_scale = berendsen_lambda(temperature(velocity_half, masses), para)

        # --- Real PES ---
        pot_real, force_real = real_pes.get_energy_force(s)
        s = s.with_pot(float(pot_real))
        f_real = float(force_norm(force_real))

        # --- Update distance matrix and the RTIP status ---
        current_bond_coord = _coords(s, indices)
        dist_mat = dist_mat_bohr(current_bond_coord)
        if rtip_status != "Decreasing":
            if judge_adj_of_mol(mol_index, atomic_radii, adj_mat, dist_mat, 1.2):
                rtip_status = "Falling"
            else:
                rtip_status = "Increasing"

        # --- Virtual final state (coincident molecular centroids) ---
        final_coord = _final_state_coords(current_bond_coord, mol_index)
        if indices is not None:
            subset_type = (
                tuple(s.atom_type[index] for index in indices) if s.atom_type is not None else None
            )
            final_state = System(
                coord=jnp.asarray(final_coord, dtype=jnp.float64),
                atom_type=subset_type,
            )
        else:
            final_state = System(coord=jnp.asarray(final_coord, dtype=jnp.float64))
        rtip_dist_value = float(rti_dist(final_state.coord, current_bond_coord))
        bias_pes = Rtip0PES(
            local_min=final_state,
            nearby_ts=(),
            a_min=a_min,
            a_ts=0.0,
            sigma_min=rtip_dist_value,
            sigma_ts=(),
        )
        pot_bias, force_bias = bias_pes.get_energy_force(_with_atom_add_pot(s, indices))
        f_bias = float(force_norm(force_bias))

        # --- Forces -> acceleration; second leapfrog half-step + thermostat ---
        force_total = jnp.asarray(force_real, dtype=jnp.float64) + jnp.asarray(force_bias, dtype=jnp.float64)
        acceleration = accelerations(force_total, masses)
        velocity = leapfrog_second(velocity_half, acceleration, lambda_scale, para)
        temp = float(temperature(velocity, masses))
        kin = float(kinetic_energy(velocity, masses))

        row = MDStep(
            step=step,
            time=time,
            rti_dist=rtip_dist_value,
            temp=temp,
            kin=kin,
            pot_real=float(pot_real),
            pot_rtip=float(pot_bias),
            f_real=f_real,
            f_rtip=f_bias,
            rtip_status=rtip_status,
        )
        history.append(row)
        _write_step(s, row, config.str_output_file, config.output_file, para, write_outputs)

    if write_outputs:
        with Path(config.dec_output_file).open("a") as handle:
            handle.write("\n")

    return MDResult(system=s, velocity=velocity, acceleration=acceleration, history=tuple(history))


__all__ = [
    "MDResult",
    "MDStep",
    "accelerations",
    "atom_masses",
    "berendsen_lambda",
    "evolution_md",
    "kinetic_energy",
    "leapfrog_first",
    "leapfrog_second",
    "repulsive_md",
    "temperature",
    "twice_kinetic_energy",
]
