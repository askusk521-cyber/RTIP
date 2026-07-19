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
from rtip_jax.constants import (
    BOHR_TO_ANGSTROM,
    BOLTZMANN,
    FEMTOSECOND_TO_AU,
    HARTREE_TO_JOULE,
    Element,
    atomic_mass,
    element_symbol,
)
from rtip_jax.core.optimization import min_1d_real_bias
from rtip_jax.core.rtip import Rtip0PES, rti_dist
from rtip_jax.io.pdb import write_pdb
from rtip_jax.pes.base import PES, ZeroPES
from rtip_jax.pes.bias import (
    AttractivePot,
    ReactionCoordinatePES,
    ReactionCoordinatePot,
    RepulsivePot,
    SynthesisPot,
)
from rtip_jax.system import System
from rtip_jax.workflows.pathway_sampling import force_norm, perturb_system, rms_force_norm
from rtip_jax.workflows.synthesis import synthesize_layout, synthesis_target_state


MDConfig = RepulsivePot | AttractivePot | SynthesisPot | ReactionCoordinatePot


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


def _repulsive_md_bias(
    config: RepulsivePot, system: System, step: int,
    *, bias_phase: str = "growing", step_reduction_started: int | None = None,
) -> tuple[float, float, Any, float]:
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
    amplitude = config.para.bias_amplitude(
        step, bias_phase=bias_phase, step_reduction_started=step_reduction_started,
    )
    bias_pes = Rtip0PES(
        local_min=config.local_min,
        nearby_ts=config.nearby_ts,
        a_min=amplitude,
        a_ts=amplitude * config.para.scale_ts_a0,
        sigma_min=float(sigma_min),
        sigma_ts=sigma_ts,
    )
    pot_bias, force_bias = bias_pes.get_energy_force(current)
    return float(sigma_min), float(pot_bias), force_bias, float(force_norm(force_bias))


def _attractive_md_bias(
    config: AttractivePot, system: System, step: int,
    *, bias_phase: str = "growing", step_reduction_started: int | None = None,
) -> tuple[float, float, Any, float]:
    indices = config.final_state.atom_add_pot
    current = _with_atom_add_pot(system, indices)
    sigma_min = rti_dist(_coords(config.final_state, indices), _coords(system, indices))
    amplitude = config.para.bias_amplitude(
        step, bias_phase=bias_phase, step_reduction_started=step_reduction_started,
    )
    bias_pes = Rtip0PES(
        local_min=config.final_state,
        nearby_ts=(),
        a_min=-amplitude,  # attractive: negative sign
        a_ts=0.0,
        sigma_min=float(sigma_min),
        sigma_ts=(),
    )
    pot_bias, force_bias = bias_pes.get_energy_force(current)
    return float(sigma_min), float(pot_bias), force_bias, float(force_norm(force_bias))


def _synthesis_md_bias(
    config: SynthesisPot, system: System, step: int,
    *, bias_phase: str = "growing", step_reduction_started: int | None = None,
) -> tuple[float, float, Any, float]:
    final_state, indices = synthesis_target_state(system, config.mol_index)
    current = _with_atom_add_pot(system, indices)
    target_coord = _coords(system, indices)
    sigma_min = rti_dist(final_state.coord, target_coord)
    amplitude = config.para.bias_amplitude(
        step, bias_phase=bias_phase, step_reduction_started=step_reduction_started,
    )
    bias_pes = Rtip0PES(
        local_min=final_state,
        nearby_ts=(),
        a_min=-amplitude,  # attractive: negative sign
        a_ts=0.0,
        sigma_min=float(sigma_min),
        sigma_ts=(),
    )
    pot_bias, force_bias = bias_pes.get_energy_force(current)
    return float(sigma_min), float(pot_bias), force_bias, float(force_norm(force_bias))


def _reaction_coordinate_md_bias(config: ReactionCoordinatePot, system: System, _step: int) -> tuple[float, float, Any, float]:
    bias_pes = ReactionCoordinatePES(config.restraints)
    pot_bias, force_bias = bias_pes.get_energy_force(system)
    return bias_pes.distance_error(system), float(pot_bias), force_bias, float(force_norm(force_bias))


def _md_bias(
    config: MDConfig, system: System, step: int,
    *, bias_phase: str = "growing", step_reduction_started: int | None = None,
) -> tuple[float, float, Any, float]:
    if isinstance(config, RepulsivePot):
        return _repulsive_md_bias(config, system, step, bias_phase=bias_phase, step_reduction_started=step_reduction_started)
    if isinstance(config, AttractivePot):
        return _attractive_md_bias(config, system, step, bias_phase=bias_phase, step_reduction_started=step_reduction_started)
    if isinstance(config, SynthesisPot):
        return _synthesis_md_bias(config, system, step, bias_phase=bias_phase, step_reduction_started=step_reduction_started)
    if isinstance(config, ReactionCoordinatePot):
        return _reaction_coordinate_md_bias(config, system, step)
    raise TypeError(f"unsupported RTIP MD config: {type(config)!r}")


def _config_initial_state(config: MDConfig) -> System:
    if isinstance(config, RepulsivePot):
        return config.local_min
    if isinstance(config, (AttractivePot, SynthesisPot, ReactionCoordinatePot)):
        return config.initial_state
    raise TypeError(f"unsupported RTIP MD config: {type(config)!r}")


def run_rtip_nvt_md(
    config: MDConfig,
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
    initial_state = _config_initial_state(config)
    if perturb:
        if key is None:
            raise ValueError("key is required when perturb=True")
        s = perturb_system(initial_state, key, atom_indices=initial_state.atom_add_pot)
    else:
        s = initial_state

    masses = atom_masses(s)
    velocity = jnp.zeros_like(s.coord) if initial_velocity is None else jnp.asarray(initial_velocity, dtype=jnp.float64)
    acceleration = (
        jnp.zeros_like(s.coord) if initial_acceleration is None else jnp.asarray(initial_acceleration, dtype=jnp.float64)
    )
    if velocity.shape != s.coord.shape:
        raise ValueError(f"initial_velocity must have shape {s.coord.shape}, got {velocity.shape}")
    if acceleration.shape != s.coord.shape:
        raise ValueError(f"initial_acceleration must have shape {s.coord.shape}, got {acceleration.shape}")

    _write_md_start(initial_state, config.str_output_file, config.output_file, write_outputs)
    history: list[MDStep] = []
    time = 0.0
    started_at = perf_counter()

    # State tracking (paper basis: Eclimb / Edrop; engineering: phase machine)
    pot_real_max = -1.0e15
    pot_real_min = 1.0e15
    pot_real_initial = 0.0
    bias_phase = "growing"
    step_reduction_started: int | None = None
    step_off_started: int | None = None
    high_temp_streak = 0

    for step in range(1, config.para.max_step + 1):
        time += config.para.dt
        coord_new, velocity_half = leapfrog_first(s.coord, velocity, acceleration, config.para)
        s = s.with_coord(coord_new)
        temp_half = float(temperature(velocity_half, masses))
        lambda_scale = berendsen_lambda(temp_half, config.para)

        pot_real, force_real = real_pes.get_energy_force(s)
        s = s.with_pot(float(pot_real))
        f_real = float(force_norm(force_real))

        sigma_min, pot_bias, force_bias, f_bias = _md_bias(
            config, s, step, bias_phase=bias_phase, step_reduction_started=step_reduction_started,
        )
        # Temperature feedback: scale bias inversely with T above 1500 K
        if not config.para.rust_compat and temp_half > 1500.0:
            temp_scale = 1500.0 / temp_half
            pot_bias = float(pot_bias) * temp_scale
            force_bias = jnp.asarray(force_bias, dtype=jnp.float64) * temp_scale
            f_bias = float(f_bias) * temp_scale

        force_total = jnp.asarray(force_real, dtype=jnp.float64) + jnp.asarray(force_bias, dtype=jnp.float64)
        acceleration = accelerations(force_total, masses)
        velocity = leapfrog_second(velocity_half, acceleration, lambda_scale, config.para)
        temp = temperature(velocity, masses)
        kin = kinetic_energy(velocity, masses)

        # ---- phase state machine (paper basis + engineering safeguards) ----
        state_decision = "md_running"
        should_stop = False

        if not config.para.rust_compat:
            # --- safety checks (all phases) ---
            if temp > 3000.0:
                high_temp_streak += 1
            else:
                high_temp_streak = 0
            if high_temp_streak > 10:
                should_stop = True
                state_decision = "stop_temperature_explosion"
            if sigma_min > 500.0:
                should_stop = True
                state_decision = "stop_rti_distance_excessive"

            # --- phase transitions ---
            if not should_stop:
                if bias_phase == "growing":
                    pot_real_max = max(pot_real_max, float(pot_real))
                    pot_real_min = min(pot_real_min, float(pot_real))
                    if isinstance(config, SynthesisPot) and step == 1:
                        pot_real_initial = float(pot_real)

                    if isinstance(config, AttractivePot):
                        if sigma_min < 1.0 or f_bias > 1000.0:
                            state_decision = "bias_off_target_reached" if sigma_min < 1.0 else "bias_off_large_bias_force"
                            if rms_force_norm(f_real, s.natom) < config.para.f_epsilon:
                                should_stop = True
                                state_decision = "stop_converged"

                    elif isinstance(config, (RepulsivePot, SynthesisPot)):
                        # pot_drop detection (paper basis: Edrop = Emax - En)
                        if isinstance(config, SynthesisPot):
                            if pot_real < (pot_real_max - config.para.pot_drop) and pot_real > pot_real_initial:
                                bias_phase = "reducing"
                                step_reduction_started = step
                                state_decision = "bias_reducing_pot_drop"
                        elif pot_real < (pot_real_max - config.para.pot_drop):
                            bias_phase = "reducing"
                            step_reduction_started = step
                            state_decision = "bias_reducing_pot_drop"

                        if pot_real > (pot_real_min + config.para.pot_climb):
                            should_stop = True
                            state_decision = "stop_pot_climb"
                        if f_bias > 1000.0:
                            should_stop = True
                            state_decision = "stop_large_bias_force"

                elif bias_phase == "reducing":
                    pot_real_max = max(pot_real_max, float(pot_real))
                    pot_real_min = min(pot_real_min, float(pot_real))
                    # Check if amplitude reached zero (monotonic decrease, no oscillation)
                    amp = config.para.bias_amplitude(
                        step, bias_phase="reducing", step_reduction_started=step_reduction_started,
                    )
                    if amp <= 0.0:
                        bias_phase = "off"
                        step_off_started = step
                        state_decision = "bias_off"
                    else:
                        state_decision = "bias_reducing"

                    if pot_real > (pot_real_min + config.para.pot_climb):
                        should_stop = True
                        state_decision = "stop_pot_climb"
                    if f_bias > 1000.0:
                        should_stop = True
                        state_decision = "stop_large_bias_force"

                else:  # bias_phase == "off"
                    state_decision = "bias_off_pure_md"
                    # Convergence check on real PES (paper basis: local optimization)
                    if rms_force_norm(f_real, s.natom) < config.para.f_epsilon:
                        should_stop = True
                        state_decision = "stop_converged"
                    # Safety timeout after bias off (engineering: prevent infinite drift)
                    if step_off_started is not None and (step - step_off_started) >= 100:
                        should_stop = True
                        state_decision = "stop_max_pure_md"

        if step == config.para.max_step and not should_stop:
            state_decision = "max_step"

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
            rms_f_real=rms_force_norm(f_real, s.natom),
            state_decision=state_decision,
        )
        history.append(row)
        _write_md_step(s, row, config.str_output_file, config.output_file, config.para, write_outputs)
        if should_stop:
            break

    return MDResult(system=s, velocity=velocity, acceleration=acceleration, history=tuple(history))


# =============================================================================
# Bond connectivity detection (paper basis: bond-change detection)
# Engineering: covalent radii + tolerance factor
# =============================================================================

COVALENT_RADII_ANGSTROM: dict[str, float] = {
    "H": 0.37, "He": 0.32, "Li": 1.28, "Be": 0.96, "B": 0.84,
    "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
}


def _covalent_radius(element: str) -> float:
    """Covalent radius in Angstrom; falls back to 1.0 A for unknown elements."""
    return COVALENT_RADII_ANGSTROM.get(element, 1.0)


@dataclass(frozen=True)
class ConnectivityState:
    """Inter-molecular bond connectivity snapshot.

    Paper basis: bond connectivity is recorded before each round and updated
    after product formation (Figure 3 / Section 3.1).
    """

    bonds: frozenset[tuple[int, int]]
    connected_components: tuple[tuple[int, ...], ...]
    n_components: int


def detect_intermolecular_bonds(
    coord: Any,
    mol_index: tuple[tuple[int, ...], ...],
    atom_type: tuple[Element, ...] | None,
    tolerance: float = 1.3,
) -> frozenset[tuple[int, int]]:
    """Detect bonds between atoms belonging to different molecules.

    Criterion: ``distance_Angstrom < tolerance * (r_cov_i + r_cov_j)``.

    Paper basis: bond-change detection during RTIP-MD.
    Engineering: tolerance = 1.3 (typical for formation detection;
    not specified in the paper).  Uses numpy (called at phase
    transitions, not per MD step).
    """
    import numpy as np

    coord_np = np.asarray(coord, dtype=np.float64)
    coord_ang = coord_np * BOHR_TO_ANGSTROM

    atom_to_mol: dict[int, int] = {}
    for mol_id, indices in enumerate(mol_index):
        for idx in indices:
            atom_to_mol[idx] = mol_id

    if atom_type is not None:
        radii = np.asarray(
            [_covalent_radius(element_symbol(el)) for el in atom_type],
            dtype=np.float64,
        )
    else:
        radii = np.ones(coord_np.shape[0], dtype=np.float64)

    bonds: set[tuple[int, int]] = set()
    natom = coord_np.shape[0]
    for i in range(natom):
        for j in range(i + 1, natom):
            if atom_to_mol.get(i) == atom_to_mol.get(j):
                continue
            d = float(np.linalg.norm(coord_ang[i] - coord_ang[j]))
            threshold = tolerance * (radii[i] + radii[j])
            if d < threshold:
                bonds.add((i, j))
    return frozenset(bonds)


def _build_connectivity(
    coord: Any,
    mol_index: tuple[tuple[int, ...], ...],
    atom_type: tuple[Element, ...] | None,
    tolerance: float = 1.3,
) -> ConnectivityState:
    """Build ConnectivityState from current coordinates."""
    bonds = detect_intermolecular_bonds(coord, mol_index, atom_type, tolerance)

    # Union-find to compute connected components
    all_atoms: list[int] = []
    for indices in mol_index:
        all_atoms.extend(indices)
    natom = len(all_atoms)
    parent = {a: a for a in all_atoms}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j in bonds:
        union(i, j)
    # Also union atoms within each molecule
    for indices in mol_index:
        for a, b in zip(indices, indices[1:]):
            union(a, b)

    comp_map: dict[int, list[int]] = {}
    for a in sorted(parent):
        r = find(a)
        comp_map.setdefault(r, []).append(a)
    components = tuple(tuple(sorted(lst)) for lst in comp_map.values())
    return ConnectivityState(
        bonds=bonds,
        connected_components=components,
        n_components=len(components),
    )


# =============================================================================
# Real-PES geometry relaxation (paper basis: local optimization after RTIP off)
# Engineering: steepest-descent + golden-section line search
# =============================================================================


def relax_on_real_pes(
    system: System,
    real_pes: PES,
    f_epsilon: float = 0.001,
    max_steps: int = 200,
) -> tuple[System, bool, float]:
    """Local optimization on the real PES after RTIP removal.

    Paper basis: supplementary materials — after Edrop, remove RTIP and
    perform local optimization on the real PES to obtain product Ppro.

    Engineering: steepest-descent with golden-section line search reusing
    ``min_1d_real_bias`` from ``optimization.py`` (ZeroPES as bias so only
    real forces are used).

    Returns ``(relaxed_system, converged, final_rms_force)``.
    """
    s = system
    for _ in range(max_steps):
        energy, force = real_pes.get_energy_force(s)
        s = s.with_pot(float(energy))
        rms = rms_force_norm(float(force_norm(force)), s.natom)
        if rms < f_epsilon:
            return s, True, rms
        dcoord = min_1d_real_bias(real_pes, ZeroPES(), s, force, float(energy), f_epsilon)
        s = s.with_coord(s.coord + dcoord)
    energy, force = real_pes.get_energy_force(s)
    rms = rms_force_norm(float(force_norm(force)), s.natom)
    return s, rms < f_epsilon, rms


# =============================================================================
# Multi-round synthesis MD (paper basis: multi-round reaction search)
# =============================================================================


def run_multiround_synthesis_md(
    initial_state: System,
    mol_index: tuple[tuple[int, ...], ...],
    real_pes: PES,
    para: Para,
    *,
    synth_dist: float = 5.0,
    seed: int = 0,
    bond_tolerance: float = 1.3,
    write_outputs: bool = True,
    output_dir: str | None = None,
) -> tuple[MDResult, tuple[MDResult, ...]]:
    """Multi-round RTIP-MD for reaction product discovery.

    Paper basis: Figure 3 / Section 3.1 — multi-round search with
    oscillating attractive RTIP, bond-change detection, bias reduction,
    product dispersal, and bond-connectivity updates.

    Round 0: ``synthesize_layout()`` separates reactants at *synth_dist* Å.
    Each round: MD with SynthesisPot through growing → reducing → off.
    After each round: detect new inter-molecular bonds.
    If new bonds found: update connectivity, restart from current state.
    Stop: max rounds, no new bonds, convergence, or hard failure.

    Returns ``(final_result, all_round_results)``.
    """
    from jax import random

    actual_max_rounds = max(para.max_rounds, 1)
    round_results: list[MDResult] = []

    # Round 0: layout (paper basis: place reactants at distance, record bonds)
    synth_dist_bohr = synth_dist / BOHR_TO_ANGSTROM
    current_system = synthesize_layout(
        initial_state, mol_index, float(synth_dist_bohr),
        key=random.PRNGKey(seed),
    )
    connectivity = _build_connectivity(
        current_system.coord, mol_index, current_system.atom_type, bond_tolerance,
    )

    current_para = para
    for round_idx in range(actual_max_rounds):
        output_pdb = f"rtip_r{round_idx}.pdb"
        output_out = f"rtip_r{round_idx}.out"
        if output_dir is not None:
            output_pdb = str(Path(output_dir) / output_pdb)
            output_out = str(Path(output_dir) / output_out)
        if write_outputs:
            Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
            Path(output_out).parent.mkdir(parents=True, exist_ok=True)

        config = SynthesisPot(
            initial_state=current_system,
            mol_index=mol_index,
            para=current_para,
            str_output_file=output_pdb,
            output_file=output_out,
        )
        result = run_rtip_nvt_md(
            config, real_pes,
            perturb=False,
            write_outputs=write_outputs,
        )
        round_results.append(result)

        # Detect bond changes
        new_bonds = detect_intermolecular_bonds(
            result.system.coord, mol_index, result.system.atom_type, bond_tolerance,
        )
        added_bonds = new_bonds - connectivity.bonds

        # Check for hard failures
        last_state = result.history[-1].state_decision if result.history else "unknown"
        hard_failures = {
            "stop_temperature_explosion", "stop_rti_distance_excessive",
            "stop_pot_climb", "stop_large_bias_force",
        }
        if last_state in hard_failures:
            return result, tuple(round_results)

        if not added_bonds or round_idx == actual_max_rounds - 1:
            # Relax on real PES (paper basis: local optimization → Ppro)
            relaxed, converged, rms = relax_on_real_pes(
                result.system, real_pes, current_para.f_epsilon, current_para.relax_max_steps,
            )
            final_result = MDResult(
                system=relaxed,
                velocity=jnp.zeros_like(result.velocity),
                acceleration=jnp.zeros_like(result.acceleration),
                history=result.history,
            )
            if not converged:
                # Replace last step decision
                if final_result.history:
                    last = final_result.history[-1]
                    final_result = MDResult(
                        system=relaxed,
                        velocity=final_result.velocity,
                        acceleration=final_result.acceleration,
                        history=final_result.history[:-1] + (
                            MDStep(
                                step=last.step, time=last.time,
                                wall_time_s=last.wall_time_s,
                                sigma_min=last.sigma_min, temp=last.temp,
                                temp_bath=last.temp_bath,
                                thermostat_lambda=last.thermostat_lambda,
                                kin=last.kin, pot_real=last.pot_real,
                                pot_bias=last.pot_bias, pot_total=last.pot_total,
                                f_real=last.f_real, f_bias=last.f_bias,
                                f_total=last.f_total, rms_f_real=last.rms_f_real,
                                state_decision="stop_relax_failed",
                            ),
                        ),
                    )
            return final_result, tuple(round_results)

        # Update for next round (paper basis: update bond connectivity)
        connectivity = _build_connectivity(
            result.system.coord, mol_index, result.system.atom_type, bond_tolerance,
        )
        current_system = result.system
        # Engineering: halve a0 for gentler subsequent rounds
        current_para = replace(current_para, a0=current_para.a0 * 0.5)

    return round_results[-1], tuple(round_results)


__all__ = [
    "COVALENT_RADII_ANGSTROM",
    "ConnectivityState",
    "accelerations",
    "atom_masses",
    "berendsen_lambda",
    "detect_intermolecular_bonds",
    "kinetic_energy",
    "leapfrog_first",
    "leapfrog_second",
    "MDResult",
    "MDStep",
    "relax_on_real_pes",
    "run_multiround_synthesis_md",
    "run_rtip_nvt_md",
    "temperature",
    "twice_kinetic_energy",
]
