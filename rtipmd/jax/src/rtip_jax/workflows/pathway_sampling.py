"""Pathway sampling helpers and workflow runners.

Rust source: `src/pes_exploration/pathway_sampling.rs`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import jax.numpy as jnp
from jax import random

from rtip_jax.config import Para
from rtip_jax.core.idwm import Idwm0PES, idw_dist, idw_dist1, wei_dist_mat
from rtip_jax.core.optimization import min_1d_real_bias
from rtip_jax.core.rtip import Rtip0PES, rti_dist
from rtip_jax.io.pdb import write_pdb
from rtip_jax.pes.base import PES, ZeroPES
from rtip_jax.pes.bias import AttractivePot, RepulsivePot, SynthesisPot
from rtip_jax.system import System
from rtip_jax.workflows.synthesis import synthesis_target_state


LineSearch = Callable[[Any, Any, System, Any, float, float], Any] | None


@dataclass(frozen=True)
class RepulsiveSamplingState:
    pot_real_max: float = -1.0e15
    pot_real_min: float = 1.0e15
    add_bias: bool = True


@dataclass(frozen=True)
class SynthesisSamplingState:
    pot_real_max: float = -1.0e15
    pot_real_min: float = 1.0e15
    pot_real_initial: float = 0.0
    add_bias: bool = True


@dataclass(frozen=True)
class AttractiveSamplingState:
    add_bias: bool = True


@dataclass(frozen=True)
class PathwayStep:
    """One scalar-output row from a pathway sampling workflow."""

    step: int
    time_fs: float
    wall_time_s: float
    sigma_min: float
    temp: float
    kin: float
    pot_real: float
    pot_bias: float
    pot_total: float
    f_real: float
    f_bias: float
    f_total: float
    rms_f_real: float
    add_bias: bool
    next_add_bias: bool
    stopped: bool = False
    state_decision: str = "running"


@dataclass(frozen=True)
class PathwayResult:
    """Final pathway sampling state and scalar history."""

    system: System
    history: tuple[PathwayStep, ...]
    stopped: bool


def perturb_system(system: System, key: Any, atom_indices: Sequence[int] | None = None, scale: float = 0.1) -> System:
    """Return a nearby structure using Rust's mean-centered perturbation logic."""

    if atom_indices is None:
        dcoord = random.uniform(key, shape=system.coord.shape, minval=0.0, maxval=1.0, dtype=jnp.float64)
        dcoord = dcoord - jnp.mean(dcoord, axis=0)
        norm = jnp.sqrt(jnp.sum(dcoord * dcoord))
        return system.with_coord(system.coord + dcoord * (float(scale) / norm))

    indices = tuple(int(index) for index in atom_indices)
    if len(indices) == 0:
        raise ValueError("atom_indices cannot be empty")
    if any(index < 0 or index >= system.natom for index in indices):
        raise ValueError("atom_indices contains an atom index outside the system")
    idx = jnp.asarray(indices)
    dcoord = random.uniform(key, shape=(len(indices), 3), minval=0.0, maxval=1.0, dtype=jnp.float64)
    dcoord = dcoord - jnp.mean(dcoord, axis=0)
    norm = jnp.sqrt(jnp.sum(dcoord * dcoord))
    coord = system.coord.at[idx].add(dcoord * (float(scale) / norm))
    return system.with_coord(coord)


def force_norm(force: Any) -> Any:
    force = jnp.asarray(force, dtype=jnp.float64)
    return jnp.sqrt(jnp.sum(force * force))


def rms_force_norm(force_norm_value: float, natom: int) -> float:
    return float(force_norm_value) / float(jnp.sqrt(float(natom)))


def repulsive_stop_update(
    state: RepulsiveSamplingState,
    *,
    pot_real: float,
    f_real: float,
    f_bias: float,
    natom: int,
    para: Para,
) -> tuple[RepulsiveSamplingState, bool]:
    """Update Rust repulsive/IDWM stopping state and return `should_stop`."""

    new_state, should_stop, _decision = repulsive_stop_decision(
        state,
        pot_real=pot_real,
        f_real=f_real,
        f_bias=f_bias,
        natom=natom,
        para=para,
    )
    return new_state, should_stop


def repulsive_stop_decision(
    state: RepulsiveSamplingState,
    *,
    pot_real: float,
    f_real: float,
    f_bias: float,
    natom: int,
    para: Para,
) -> tuple[RepulsiveSamplingState, bool, str]:
    """Update repulsive/IDWM stopping state and describe the state-machine decision."""

    pot_real_max = max(float(state.pot_real_max), float(pot_real))
    pot_real_min = min(float(state.pot_real_min), float(pot_real))
    add_bias = bool(state.add_bias)
    decision = "bias_on" if add_bias else "bias_off"
    if pot_real < (pot_real_max - para.pot_drop):
        add_bias = False
        decision = "bias_off_after_pot_drop"
    should_stop = False
    if (not add_bias) and (rms_force_norm(f_real, natom) < para.f_epsilon):
        should_stop = True
        decision = "stop_converged"
    if pot_real > (pot_real_min + para.pot_climb) or f_bias > 1000.0:
        should_stop = True
        decision = "stop_pot_climb" if pot_real > (pot_real_min + para.pot_climb) else "stop_large_bias_force"
    return RepulsiveSamplingState(pot_real_max, pot_real_min, add_bias), should_stop, decision


def attractive_stop_update(
    state: AttractiveSamplingState,
    *,
    sigma_min: float,
    f_real: float,
    f_bias: float,
    natom: int,
    para: Para,
) -> tuple[AttractiveSamplingState, bool]:
    """Update Rust attractive stopping state and return `should_stop`."""

    new_state, should_stop, _decision = attractive_stop_decision(
        state,
        sigma_min=sigma_min,
        f_real=f_real,
        f_bias=f_bias,
        natom=natom,
        para=para,
    )
    return new_state, should_stop


def attractive_stop_decision(
    state: AttractiveSamplingState,
    *,
    sigma_min: float,
    f_real: float,
    f_bias: float,
    natom: int,
    para: Para,
) -> tuple[AttractiveSamplingState, bool, str]:
    """Update attractive stopping state and describe the state-machine decision."""

    add_bias = bool(state.add_bias)
    decision = "bias_on" if add_bias else "bias_off"
    if sigma_min < 1.0 or f_bias > 1000.0:
        add_bias = False
        decision = "bias_off_after_target_reached" if sigma_min < 1.0 else "bias_off_after_large_bias_force"
    should_stop = (not add_bias) and (rms_force_norm(f_real, natom) < para.f_epsilon)
    if should_stop:
        decision = "stop_converged"
    return AttractiveSamplingState(add_bias), should_stop, decision


def synthesis_stop_update(
    state: SynthesisSamplingState,
    *,
    step: int,
    pot_real: float,
    f_real: float,
    f_bias: float,
    natom: int,
    para: Para,
) -> tuple[SynthesisSamplingState, bool]:
    """Update Rust synthesis stopping state and return `should_stop`."""

    new_state, should_stop, _decision = synthesis_stop_decision(
        state,
        step=step,
        pot_real=pot_real,
        f_real=f_real,
        f_bias=f_bias,
        natom=natom,
        para=para,
    )
    return new_state, should_stop


def synthesis_stop_decision(
    state: SynthesisSamplingState,
    *,
    step: int,
    pot_real: float,
    f_real: float,
    f_bias: float,
    natom: int,
    para: Para,
) -> tuple[SynthesisSamplingState, bool, str]:
    """Update synthesis stopping state and describe the state-machine decision."""

    pot_real_max = max(float(state.pot_real_max), float(pot_real))
    pot_real_min = min(float(state.pot_real_min), float(pot_real))
    pot_real_initial = float(pot_real) if int(step) == 1 else float(state.pot_real_initial)
    add_bias = bool(state.add_bias)
    decision = "bias_on" if add_bias else "bias_off"
    if pot_real < (pot_real_max - para.pot_drop) and pot_real > pot_real_initial:
        add_bias = False
        decision = "bias_off_after_synthesis_drop"
    should_stop = False
    if (not add_bias) and (rms_force_norm(f_real, natom) < para.f_epsilon):
        should_stop = True
        decision = "stop_converged"
    if pot_real > (pot_real_min + para.pot_climb) or f_bias > 1000.0:
        should_stop = True
        decision = "stop_pot_climb" if pot_real > (pot_real_min + para.pot_climb) else "stop_large_bias_force"
    return SynthesisSamplingState(pot_real_max, pot_real_min, pot_real_initial, add_bias), should_stop, decision


def _validate_para_runtime(para: Para) -> None:
    if para.max_step < 0:
        raise ValueError("max_step must be non-negative")
    if para.print_step <= 0:
        raise ValueError("print_step must be positive")


def _coords(system: System, indices: tuple[int, ...] | None) -> Any:
    if indices is None:
        return system.coord
    return system.coord[jnp.asarray(indices)]


def _with_atom_add_pot(system: System, indices: tuple[int, ...] | None) -> System:
    return replace(system, atom_add_pot=indices)


def _write_pathway_start(system: System, structure_file: str, table_file: str, header: str, write_outputs: bool) -> None:
    if not write_outputs:
        return
    Path(structure_file).parent.mkdir(parents=True, exist_ok=True)
    Path(table_file).parent.mkdir(parents=True, exist_ok=True)
    write_pdb(system, structure_file, create_new_file=True, step=0)
    Path(table_file).write_text(header)


def _write_pathway_step(system: System, step: PathwayStep, structure_file: str, table_file: str, para: Para, write_outputs: bool) -> None:
    if not write_outputs:
        return
    if step.step % para.print_step == 0:
        write_pdb(system, structure_file, create_new_file=False, step=step.step)
    with Path(table_file).open("a") as handle:
        handle.write(
            f"{step.step:6d} {step.time_fs:15.8f} {step.wall_time_s:15.8f} "
            f"{step.sigma_min:15.8f} {step.temp:15.8f} {step.kin:15.8f} "
            f"{step.pot_real:15.8f} {step.pot_bias:15.8f} {step.pot_total:15.8f} "
            f"{step.f_real:15.8f} {step.f_bias:15.8f} {step.f_total:15.8f} "
            f"{step.rms_f_real:15.8f} {int(step.add_bias):9d} {int(step.next_add_bias):13d} {int(step.stopped):8d} "
            f"{step.state_decision}\n"
        )


def _pathway_header(distance_label: str, bias_label: str) -> str:
    return (
        f"{'step':>6s} {'time_fs':>15s} {'wall_time_s':>15s} {distance_label:>15s} "
        f"{'temp_K':>15s} {'kin_Ha':>15s} {'pot_real_Ha':>15s} {bias_label:>15s} "
        f"{'pot_total_Ha':>15s} {'f_real':>15s} {'f_bias':>15s} {'f_total':>15s} "
        f"{'rms_f_real':>15s} {'bias_used':>9s} {'next_add_bias':>13s} {'stopped':>8s} state_decision\n"
    )


def _pathway_row(
    *,
    step: int,
    started_at: float,
    para: Para,
    sigma_min: float,
    pot_real: float,
    pot_bias: float,
    force_total: Any,
    f_real: float,
    f_bias: float,
    natom: int,
    add_bias: bool,
    next_add_bias: bool,
    stopped: bool,
    state_decision: str,
) -> PathwayStep:
    return PathwayStep(
        step=step,
        time_fs=float(step) * float(para.dt),
        wall_time_s=perf_counter() - started_at,
        sigma_min=sigma_min,
        temp=float("nan"),
        kin=float("nan"),
        pot_real=pot_real,
        pot_bias=pot_bias,
        pot_total=pot_real + pot_bias,
        f_real=f_real,
        f_bias=f_bias,
        f_total=float(force_norm(force_total)),
        rms_f_real=rms_force_norm(f_real, natom),
        add_bias=add_bias,
        next_add_bias=next_add_bias,
        stopped=stopped,
        state_decision=state_decision,
    )


def _line_search_step(
    line_search: LineSearch,
    real_pes: PES,
    bias_pes: Any,
    system: System,
    force_total: Any,
    pot_total: float,
    epsilon: float,
) -> Any:
    if line_search is None:
        return jnp.zeros_like(system.coord)
    return line_search(real_pes, bias_pes, system, force_total, pot_total, epsilon)


def _repulsive_rtip_bias(config: RepulsivePot, system: System, step: int, add_bias: bool) -> tuple[float, float, Any, float, Any]:
    indices = config.local_min.atom_add_pot
    current = _with_atom_add_pot(system, indices)
    sigma_min = rti_dist(_coords(config.local_min, indices), _coords(system, indices))
    if not add_bias:
        return float(sigma_min), 0.0, jnp.zeros_like(system.coord), 0.0, ZeroPES()

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
    f_bias = force_norm(force_bias)
    return float(sigma_min), float(pot_bias), force_bias, float(f_bias), bias_pes


def _repulsive_idwm_reference(config: RepulsivePot) -> tuple[tuple[int, ...] | None, Any, tuple[Any, ...]]:
    indices = config.local_min.atom_add_pot
    local_min = wei_dist_mat(_coords(config.local_min, indices))
    nearby_ts = tuple(wei_dist_mat(_coords(ts, indices)) for ts in config.nearby_ts)
    return indices, local_min, nearby_ts


def _repulsive_idwm_bias(
    config: RepulsivePot,
    system: System,
    step: int,
    add_bias: bool,
    indices: tuple[int, ...] | None,
    local_min: Any,
    nearby_ts: tuple[Any, ...],
) -> tuple[float, float, Any, float, Any]:
    current = _with_atom_add_pot(system, indices)
    current_coord = _coords(system, indices)
    sigma_min = idw_dist(local_min, current_coord)
    if not add_bias:
        return float(sigma_min), 0.0, jnp.zeros_like(system.coord), 0.0, ZeroPES()

    if config.para.scale_ts_sigma is None:
        sigma_ts = tuple(float(idw_dist(ts_matrix, current_coord)) for ts_matrix in nearby_ts)
    else:
        sigma_ts = tuple(float(0.5 * config.para.scale_ts_sigma * idw_dist1(ts_matrix, local_min)) for ts_matrix in nearby_ts)
    bias_pes = Idwm0PES(
        local_min=local_min,
        nearby_ts=nearby_ts,
        a_min=config.para.a0 * float(step),
        a_ts=config.para.a0 * float(step) * config.para.scale_ts_a0,
        sigma_min=float(sigma_min),
        sigma_ts=sigma_ts,
    )
    pot_bias, force_bias = bias_pes.get_energy_force(current)
    f_bias = force_norm(force_bias)
    return float(sigma_min), float(pot_bias), force_bias, float(f_bias), bias_pes


def _attractive_rtip_bias(config: AttractivePot, system: System, step: int, add_bias: bool) -> tuple[float, float, Any, float, Any, System]:
    indices = config.final_state.atom_add_pot
    current = _with_atom_add_pot(system, indices)
    sigma_min = rti_dist(_coords(config.final_state, indices), _coords(system, indices))
    if not add_bias:
        return float(sigma_min), 0.0, jnp.zeros_like(system.coord), 0.0, ZeroPES(), current

    bias_pes = Rtip0PES(
        local_min=config.final_state,
        nearby_ts=(),
        a_min=-config.para.a0 * float(step),
        a_ts=0.0,
        sigma_min=float(sigma_min),
        sigma_ts=(),
    )
    pot_bias, force_bias = bias_pes.get_energy_force(current)
    f_bias = force_norm(force_bias)
    return float(sigma_min), float(pot_bias), force_bias, float(f_bias), bias_pes, current


def _synthesis_rtip_bias(config: SynthesisPot, system: System, step: int, add_bias: bool) -> tuple[float, float, Any, float, Any, System]:
    final_state, indices = synthesis_target_state(system, config.mol_index)
    current = _with_atom_add_pot(system, indices)
    target_coord = _coords(system, indices)
    sigma_min = rti_dist(final_state.coord, target_coord)
    if not add_bias:
        return float(sigma_min), 0.0, jnp.zeros_like(system.coord), 0.0, ZeroPES(), current

    bias_pes = Rtip0PES(
        local_min=final_state,
        nearby_ts=(),
        a_min=-config.para.a0 * float(step),
        a_ts=0.0,
        sigma_min=float(sigma_min),
        sigma_ts=(),
    )
    pot_bias, force_bias = bias_pes.get_energy_force(current)
    f_bias = force_norm(force_bias)
    return float(sigma_min), float(pot_bias), force_bias, float(f_bias), bias_pes, current


def run_rtip_repulsive_path_sampling(
    config: RepulsivePot,
    real_pes: PES,
    *,
    key: Any,
    line_search: LineSearch = min_1d_real_bias,
    write_outputs: bool = True,
) -> PathwayResult:
    """Run Rust-style RTIP repulsive pathway sampling with a generic PES."""

    _validate_para_runtime(config.para)
    s = perturb_system(config.local_min, key, atom_indices=config.local_min.atom_add_pot)
    _write_pathway_start(
        config.local_min,
        config.str_output_file,
        config.output_file,
        _pathway_header("rti_dist", "pot_rtip_Ha"),
        write_outputs,
    )

    state = RepulsiveSamplingState()
    history: list[PathwayStep] = []
    stopped = False
    started_at = perf_counter()
    for step in range(1, config.para.max_step + 1):
        pot_real, force_real = real_pes.get_energy_force(s)
        s = s.with_pot(float(pot_real))
        f_real = float(force_norm(force_real))

        used_bias = state.add_bias
        sigma_min, pot_bias, force_bias, f_bias, bias_pes = _repulsive_rtip_bias(config, s, step, used_bias)
        force_total = jnp.asarray(force_real, dtype=jnp.float64) + jnp.asarray(force_bias, dtype=jnp.float64)
        dcoord = _line_search_step(
            line_search,
            real_pes,
            bias_pes,
            s,
            force_total,
            float(pot_real) + float(pot_bias),
            config.para.pot_epsilon * s.natom,
        )

        state, stopped, state_decision = repulsive_stop_decision(
            state,
            pot_real=float(pot_real),
            f_real=f_real,
            f_bias=f_bias,
            natom=s.natom,
            para=config.para,
        )
        row = _pathway_row(
            step=step,
            started_at=started_at,
            para=config.para,
            sigma_min=sigma_min,
            pot_real=float(pot_real),
            pot_bias=float(pot_bias),
            force_total=force_total,
            f_real=f_real,
            f_bias=f_bias,
            natom=s.natom,
            add_bias=used_bias,
            next_add_bias=state.add_bias,
            stopped=stopped,
            state_decision=state_decision,
        )
        history.append(row)
        _write_pathway_step(s, row, config.str_output_file, config.output_file, config.para, write_outputs)
        s = s.with_coord(s.coord + dcoord)
        if stopped:
            break
    return PathwayResult(system=s, history=tuple(history), stopped=stopped)


def run_idwm_repulsive_path_sampling(
    config: RepulsivePot,
    real_pes: PES,
    *,
    key: Any,
    line_search: LineSearch = min_1d_real_bias,
    write_outputs: bool = True,
) -> PathwayResult:
    """Run Rust-style IDWM repulsive pathway sampling with a generic PES."""

    _validate_para_runtime(config.para)
    s = perturb_system(config.local_min, key, atom_indices=config.local_min.atom_add_pot)
    _write_pathway_start(
        config.local_min,
        config.str_output_file,
        config.output_file,
        _pathway_header("idw_dist", "pot_idwm_Ha"),
        write_outputs,
    )

    indices, local_min, nearby_ts = _repulsive_idwm_reference(config)
    state = RepulsiveSamplingState()
    history: list[PathwayStep] = []
    stopped = False
    started_at = perf_counter()
    for step in range(1, config.para.max_step + 1):
        pot_real, force_real = real_pes.get_energy_force(s)
        s = s.with_pot(float(pot_real))
        f_real = float(force_norm(force_real))

        used_bias = state.add_bias
        sigma_min, pot_bias, force_bias, f_bias, bias_pes = _repulsive_idwm_bias(
            config,
            s,
            step,
            used_bias,
            indices,
            local_min,
            nearby_ts,
        )
        force_total = jnp.asarray(force_real, dtype=jnp.float64) + jnp.asarray(force_bias, dtype=jnp.float64)
        dcoord = _line_search_step(
            line_search,
            real_pes,
            bias_pes,
            s,
            force_total,
            float(pot_real) + float(pot_bias),
            config.para.pot_epsilon * s.natom,
        )

        state, stopped, state_decision = repulsive_stop_decision(
            state,
            pot_real=float(pot_real),
            f_real=f_real,
            f_bias=f_bias,
            natom=s.natom,
            para=config.para,
        )
        row = _pathway_row(
            step=step,
            started_at=started_at,
            para=config.para,
            sigma_min=sigma_min,
            pot_real=float(pot_real),
            pot_bias=float(pot_bias),
            force_total=force_total,
            f_real=f_real,
            f_bias=f_bias,
            natom=s.natom,
            add_bias=used_bias,
            next_add_bias=state.add_bias,
            stopped=stopped,
            state_decision=state_decision,
        )
        history.append(row)
        _write_pathway_step(s, row, config.str_output_file, config.output_file, config.para, write_outputs)
        s = s.with_coord(s.coord + dcoord)
        if stopped:
            break
    return PathwayResult(system=s, history=tuple(history), stopped=stopped)


def run_rtip_attractive_path_sampling(
    config: AttractivePot,
    real_pes: PES,
    *,
    line_search: LineSearch = min_1d_real_bias,
    write_outputs: bool = True,
) -> PathwayResult:
    """Run Rust-style RTIP attractive pathway sampling with a generic PES."""

    _validate_para_runtime(config.para)
    s = config.initial_state
    _write_pathway_start(
        s,
        config.str_output_file,
        config.output_file,
        _pathway_header("rti_dist", "pot_rtip_Ha"),
        write_outputs,
    )

    state = AttractiveSamplingState()
    history: list[PathwayStep] = []
    stopped = False
    started_at = perf_counter()
    for step in range(1, config.para.max_step + 1):
        pot_real, force_real = real_pes.get_energy_force(s)
        s = s.with_pot(float(pot_real))
        f_real = float(force_norm(force_real))

        used_bias = state.add_bias
        sigma_min, pot_bias, force_bias, f_bias, bias_pes, current = _attractive_rtip_bias(config, s, step, used_bias)
        force_total = jnp.asarray(force_real, dtype=jnp.float64) + jnp.asarray(force_bias, dtype=jnp.float64)
        dcoord = _line_search_step(
            line_search,
            real_pes,
            bias_pes,
            current,
            force_total,
            float(pot_real) + float(pot_bias),
            config.para.pot_epsilon * s.natom,
        )

        state, stopped, state_decision = attractive_stop_decision(
            state,
            sigma_min=sigma_min,
            f_real=f_real,
            f_bias=f_bias,
            natom=s.natom,
            para=config.para,
        )
        row = _pathway_row(
            step=step,
            started_at=started_at,
            para=config.para,
            sigma_min=sigma_min,
            pot_real=float(pot_real),
            pot_bias=float(pot_bias),
            force_total=force_total,
            f_real=f_real,
            f_bias=f_bias,
            natom=s.natom,
            add_bias=used_bias,
            next_add_bias=state.add_bias,
            stopped=stopped,
            state_decision=state_decision,
        )
        history.append(row)
        _write_pathway_step(s, row, config.str_output_file, config.output_file, config.para, write_outputs)
        s = s.with_coord(s.coord + dcoord)
        if stopped:
            break
    return PathwayResult(system=s, history=tuple(history), stopped=stopped)


def run_rtip_synthesis_path_sampling(
    config: SynthesisPot,
    real_pes: PES,
    *,
    line_search: LineSearch = min_1d_real_bias,
    write_outputs: bool = True,
) -> PathwayResult:
    """Run Rust-style RTIP synthesis pathway sampling with a generic PES."""

    _validate_para_runtime(config.para)
    s = config.initial_state
    _write_pathway_start(
        s,
        config.str_output_file,
        config.output_file,
        _pathway_header("rti_dist", "pot_rtip_Ha"),
        write_outputs,
    )

    state = SynthesisSamplingState()
    history: list[PathwayStep] = []
    stopped = False
    started_at = perf_counter()
    for step in range(1, config.para.max_step + 1):
        pot_real, force_real = real_pes.get_energy_force(s)
        s = s.with_pot(float(pot_real))
        f_real = float(force_norm(force_real))

        used_bias = state.add_bias
        sigma_min, pot_bias, force_bias, f_bias, bias_pes, current = _synthesis_rtip_bias(config, s, step, used_bias)
        force_total = jnp.asarray(force_real, dtype=jnp.float64) + jnp.asarray(force_bias, dtype=jnp.float64)
        dcoord = _line_search_step(
            line_search,
            real_pes,
            bias_pes,
            current,
            force_total,
            float(pot_real) + float(pot_bias),
            config.para.pot_epsilon * s.natom,
        )

        state, stopped, state_decision = synthesis_stop_decision(
            state,
            step=step,
            pot_real=float(pot_real),
            f_real=f_real,
            f_bias=f_bias,
            natom=s.natom,
            para=config.para,
        )
        row = _pathway_row(
            step=step,
            started_at=started_at,
            para=config.para,
            sigma_min=sigma_min,
            pot_real=float(pot_real),
            pot_bias=float(pot_bias),
            force_total=force_total,
            f_real=f_real,
            f_bias=f_bias,
            natom=s.natom,
            add_bias=used_bias,
            next_add_bias=state.add_bias,
            stopped=stopped,
            state_decision=state_decision,
        )
        history.append(row)
        _write_pathway_step(s, row, config.str_output_file, config.output_file, config.para, write_outputs)
        s = s.with_coord(s.coord + dcoord)
        if stopped:
            break
    return PathwayResult(system=s, history=tuple(history), stopped=stopped)


def run_rtip_path_sampling(
    config: RepulsivePot | AttractivePot | SynthesisPot,
    real_pes: PES,
    *,
    key: Any | None = None,
    line_search: LineSearch = min_1d_real_bias,
    write_outputs: bool = True,
) -> PathwayResult:
    """Dispatch to the RTIP pathway workflow matching the bias config."""

    if isinstance(config, RepulsivePot):
        if key is None:
            raise ValueError("key is required for repulsive pathway perturbation")
        return run_rtip_repulsive_path_sampling(config, real_pes, key=key, line_search=line_search, write_outputs=write_outputs)
    if isinstance(config, AttractivePot):
        return run_rtip_attractive_path_sampling(config, real_pes, line_search=line_search, write_outputs=write_outputs)
    if isinstance(config, SynthesisPot):
        return run_rtip_synthesis_path_sampling(config, real_pes, line_search=line_search, write_outputs=write_outputs)
    raise TypeError(f"unsupported RTIP pathway config: {type(config)!r}")


__all__ = [
    "AttractiveSamplingState",
    "LineSearch",
    "PathwayResult",
    "PathwayStep",
    "RepulsiveSamplingState",
    "SynthesisSamplingState",
    "attractive_stop_update",
    "attractive_stop_decision",
    "force_norm",
    "perturb_system",
    "repulsive_stop_decision",
    "repulsive_stop_update",
    "rms_force_norm",
    "run_idwm_repulsive_path_sampling",
    "run_rtip_attractive_path_sampling",
    "run_rtip_path_sampling",
    "run_rtip_repulsive_path_sampling",
    "run_rtip_synthesis_path_sampling",
    "synthesis_stop_decision",
    "synthesis_stop_update",
]
