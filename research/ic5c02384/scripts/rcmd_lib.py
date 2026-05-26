"""Shared library for generalized RC-MD reaction validation across ic5c02384 reactions.

Auto-detects reactive atom indices, small molecule type, and key distances
so that the same runner/analyzer works for all 11 reactions without hardcoded
per-reaction constants.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rtip_jax.config import Para
from rtip_jax.constants import BOHR_TO_ANGSTROM, Element, element_symbol
from rtip_jax.pes.bias import DistanceRestraint
from rtip_jax.system import System


# ---------------------------------------------------------------------------
# Small molecule type detection
# ---------------------------------------------------------------------------

_SM_SIGNATURE: dict[tuple[int, frozenset[tuple[str, int]]], str] = {}
for _natom, _counts, _name in [
    (3, (("C", 1), ("O", 2)), "CO2"),
    (3, (("C", 1), ("S", 2)), "CS2"),
    (4, (("C", 1), ("O", 1), ("H", 2)), "H2CO"),
    (6, (("C", 2), ("N", 1), ("H", 3)), "MeCN"),
    (11, (("C", 3), ("N", 1), ("H", 7)), "MeCH=NMe"),
]:
    _SM_SIGNATURE[(_natom, frozenset(_counts))] = _name


class SmallMoleculeType(str, Enum):
    CO2 = "CO2"
    CS2 = "CS2"
    H2CO = "H2CO"
    MECN = "MeCN"
    MECHNME = "MeCH=NMe"

    @classmethod
    def detect(cls, system: System) -> SmallMoleculeType:
        symbols = _system_symbols(system)
        counts = Counter(symbols)
        natom = system.natom
        sig = frozenset(counts.items())
        name = _SM_SIGNATURE.get((natom, sig))
        if name is None:
            raise ValueError(f"unknown small molecule: {natom} atoms, elements {dict(counts)}")
        return cls(name)

    @property
    def b_bond_element(self) -> str:
        """Element symbol of the atom that bonds to B in the product."""
        return {"CO2": "O", "CS2": "S", "H2CO": "O", "MeCN": "N", "MeCH=NMe": "N"}[self.value]

    @property
    def n_bond_element(self) -> str:
        """Element symbol of the atom in the small molecule that N2 bonds to."""
        return "C"

    @property
    def b_bond_multiplicity(self) -> int:
        """How many small-molecule atoms of b_bond_element type bond to B."""
        if self in (SmallMoleculeType.CO2, SmallMoleculeType.CS2):
            return 2
        return 1


# ---------------------------------------------------------------------------
# Reaction configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReactionConfig:
    reaction_name: str
    reaction_dir: Path
    fragment1_natom: int
    fragment2_natom: int
    small_molecule: SmallMoleculeType
    b_index: int
    n2_indices: tuple[int, int]
    small_c_index: int
    small_x_indices: tuple[int, ...]
    b_x_label: str  # "B-O", "B-S", or "B-N"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _system_symbols(system: System) -> tuple[str, ...]:
    if system.atom_type is None:
        return tuple("XX" for _ in range(system.natom))
    return tuple(element_symbol(element) for element in system.atom_type)


def _system_coord_angstrom(system: System) -> np.ndarray:
    return np.asarray(system.coord, dtype=np.float64) * BOHR_TO_ANGSTROM


def _element_indices(system: System, element: Element) -> list[int]:
    """All indices in *system* whose atom_type equals *element*."""
    if system.atom_type is None:
        raise ValueError("system has no atom_type information")
    return [i for i, el in enumerate(system.atom_type) if el == element]


def _distance(coord: np.ndarray, atom_i: int, atom_j: int) -> float:
    return float(np.linalg.norm(coord[atom_i] - coord[atom_j]))


def _closest_pair(
    reference: System,
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[int, int, float]:
    coord = np.asarray(reference.coord, dtype=np.float64)
    best: tuple[int, int, float] | None = None
    for i in left:
        for j in right:
            d = _distance(coord, i, j)
            if best is None or d < best[2]:
                best = (int(i), int(j), d)
    if best is None:
        raise ValueError("empty atom set for closest-pair search")
    return best


def _align_mobile_to_reference(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    ref_c = reference.mean(axis=0)
    mob_c = mobile.mean(axis=0)
    ref0 = reference - ref_c
    mob0 = mobile - mob_c
    u, _s, vt = np.linalg.svd(mob0.T @ ref0)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return mob0 @ rot + ref_c


def _rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    aligned = _align_mobile_to_reference(reference, mobile)
    return float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# Reactive atom detection
# ---------------------------------------------------------------------------


def detect_reaction_config(
    reaction_dir: Path,
    mol_index: tuple[tuple[int, ...], ...] | None = None,
) -> ReactionConfig:
    """Auto-detect the reaction configuration from reactant and product files.

    Parameters
    ----------
    reaction_dir : Path
        Directory containing 1.xyz, 2.xyz, and optionally product.xyz / ts.xyz.
    mol_index : optional
        If provided, the fragment split from synthesizing IS.xyz.
        Used to locate fragment-1 and fragment-2 atoms in the combined system.

    Returns
    -------
    ReactionConfig
    """
    reaction_name = reaction_dir.name
    frag1 = System.read_xyz(reaction_dir / "1.xyz")
    frag2 = System.read_xyz(reaction_dir / "2.xyz")
    sm_type = SmallMoleculeType.detect(frag2)

    # Build combined system matching IS.xyz order: frag1 atoms first, then frag2
    combined_coord = np.concatenate([np.asarray(frag1.coord), np.asarray(frag2.coord)], axis=0)
    combined_type: tuple[Element, ...] | None = None
    if frag1.atom_type is not None and frag2.atom_type is not None:
        combined_type = tuple(frag1.atom_type) + tuple(frag2.atom_type)
    combined = System(coord=combined_coord, atom_type=combined_type, pot=0.0)

    b_index = _element_indices(combined, Element.B)[0]

    # N2 pair: among fragment-1 N atoms, the 2 closest to B
    frag1_n_indices = [
        i for i in range(frag1.natom)
        if combined.atom_type is not None and combined.atom_type[i] == Element.N
    ]
    frag1_n_sorted = sorted(frag1_n_indices, key=lambda i: _distance(np.asarray(combined.coord, dtype=np.float64), b_index, i))
    n2_indices = (frag1_n_sorted[0], frag1_n_sorted[1])

    # Fragment 2 atom indices in the combined system
    frag2_start = frag1.natom
    frag2_end = frag2_start + frag2.natom
    frag2_indices = tuple(range(frag2_start, frag2_end))

    # Small molecule C: the carbon in fragment 2 that bonds to N2.
    # For CO2/CS2/H2CO: only one C. For MeCN/MeCH=NMe: disambiguate using
    # the product reference.
    c_in_frag2 = [i for i in frag2_indices if combined.atom_type is not None and combined.atom_type[i] == Element.C]

    if len(c_in_frag2) == 1:
        small_c_index = c_in_frag2[0]
    else:
        # Disambiguate using product reference
        try:
            product_ref = System.read_xyz(reaction_dir / "product.xyz")
        except FileNotFoundError:
            product_ref = System.read_xyz(reaction_dir / "ts.xyz")
        # Find reference N2 pair and the C closest to them
        ref_b = _element_indices(product_ref, Element.B)[0]
        ref_n = sorted(
            _element_indices(product_ref, Element.N),
            key=lambda i: _distance(np.asarray(product_ref.coord, dtype=np.float64), ref_b, i),
        )[:2]
        ref_candidates = _element_indices(product_ref, Element.C)
        # The small molecule C is the one whose distance to the N2 pair
        # changes most dramatically between reactant and product.
        # In practice, the N2-terminal-N to small-molecule-C distance
        # is shortest among all inter-fragment C in the product.
        ref_n2_coords = np.asarray(product_ref.coord, dtype=np.float64)[ref_n]
        ref_c_by_n2 = sorted(
            ref_candidates,
            key=lambda i: min(np.linalg.norm(np.asarray(product_ref.coord, dtype=np.float64)[i] - n) for n in ref_n2_coords),
        )
        # The closest C to N2 in the reference is the electrophilic small-molecule C
        # Now find which frag2 C maps to it by checking which is farther from frag1 core
        small_c_index = max(
            c_in_frag2,
            key=lambda i: _distance(np.asarray(combined.coord, dtype=np.float64), b_index, i),
        )

    # Small molecule X (bonding to B): O/S/N atoms in fragment 2
    b_bond_elem = sm_type.b_bond_element
    x_elem = Element(b_bond_elem)
    x_in_frag2 = [i for i in frag2_indices if combined.atom_type is not None and combined.atom_type[i] == x_elem]

    # Sort by distance to B (in the IS geometry they're all far, but order is consistent)
    x_in_frag2_sorted = sorted(
        x_in_frag2,
        key=lambda i: _distance(np.asarray(combined.coord, dtype=np.float64), b_index, i),
    )
    small_x_indices: tuple[int, ...] = tuple(x_in_frag2_sorted[: sm_type.b_bond_multiplicity])

    b_x_label = f"B-{b_bond_elem}"

    return ReactionConfig(
        reaction_name=reaction_name,
        reaction_dir=reaction_dir,
        fragment1_natom=frag1.natom,
        fragment2_natom=frag2.natom,
        small_molecule=sm_type,
        b_index=b_index,
        n2_indices=n2_indices,
        small_c_index=small_c_index,
        small_x_indices=small_x_indices,
        b_x_label=b_x_label,
    )


# ---------------------------------------------------------------------------
# Reference atom detection (independent of IS ordering)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceCore:
    """Reactive-core atoms detected in a reference structure (product or TS)."""

    b_index: int
    n2_indices: tuple[int, int]
    small_c_index: int
    small_x_indices: tuple[int, ...]
    coord_angstrom: np.ndarray
    symbols: tuple[str, ...]


def detect_reference_core(reference: System, config: ReactionConfig) -> ReferenceCore:
    """Find reactive-core atoms in a reference structure independently of IS ordering.

    Uses element + proximity heuristics that work on product and TS geometries.
    """
    ref_symbols = _system_symbols(reference)
    ref_coord = np.asarray(reference.coord, dtype=np.float64)

    # B: unique
    b_index = _element_indices(reference, Element.B)[0]

    # N2 pair: identified by shortest N-N distance (N2 triple bond ~1.2 A,
    # much shorter than any other N-N pair in the system).  This is robust
    # even when the small molecule also contains N (MeCN, MeCH=NMe).
    n_indices = _element_indices(reference, Element.N)
    best_pair: tuple[int, int] | None = None
    best_dist = float("inf")
    for idx_i, i in enumerate(n_indices):
        for j in n_indices[idx_i + 1 :]:
            d = _distance(ref_coord, i, j)
            if d < best_dist:
                best_dist = d
                best_pair = (i, j)
    if best_pair is None:
        raise ValueError("cannot identify N2 pair in reference")
    n2_indices = best_pair

    # Small molecule C: among ALL C atoms, the one closest to the N2 pair
    # (the NHC-ligand C atoms are farther from N2 than the small molecule C in product/TS)
    c_indices = _element_indices(reference, Element.C)
    n2_coords = ref_coord[list(n2_indices)]
    c_by_n2 = sorted(
        c_indices,
        key=lambda i: min(np.linalg.norm(ref_coord[i] - n) for n in n2_coords),
    )
    small_c_index = c_by_n2[0]

    # Small molecule X: the O/S/N atoms closest to B, excluding N2 pair atoms
    b_bond_elem_str = config.small_molecule.b_bond_element
    x_elem = Element(b_bond_elem_str)
    x_indices_all = _element_indices(reference, x_elem)
    # Exclude N2 atoms from X candidates (relevant for MeCN / MeCH=NMe where X = N)
    x_indices_all = [i for i in x_indices_all if i not in n2_indices]
    x_by_b = sorted(x_indices_all, key=lambda i: _distance(ref_coord, b_index, i))
    small_x_indices: tuple[int, ...] = tuple(x_by_b[: config.small_molecule.b_bond_multiplicity])

    # Build a core coordinate array: [B, N2_0, N2_1, small_C, small_X...]
    core_order = [b_index, n2_indices[0], n2_indices[1], small_c_index] + list(small_x_indices)
    core_coord = ref_coord[np.asarray(core_order, dtype=np.int64)] * BOHR_TO_ANGSTROM
    core_symbols = tuple(ref_symbols[i] for i in core_order)

    return ReferenceCore(
        b_index=b_index,
        n2_indices=n2_indices,
        small_c_index=small_c_index,
        small_x_indices=small_x_indices,
        coord_angstrom=core_coord,
        symbols=core_symbols,
    )


# ---------------------------------------------------------------------------
# Reaction-coordinate restraints
# ---------------------------------------------------------------------------


def reaction_coordinate_restraints(
    target: System,
    config: ReactionConfig,
    k: float,
) -> tuple[DistanceRestraint, ...]:
    """Build N-C and B-X distance restraints from the target reference."""
    ref_core = detect_reference_core(target, config)

    n_atom, c_atom, n_c_dist = _closest_pair(target, config.n2_indices, (config.small_c_index,))
    b_atom, x_atom, b_x_dist = _closest_pair(target, (config.b_index,), config.small_x_indices)

    n_label = f"N-{element_symbol(target.atom_type[c_atom]) if target.atom_type else 'C'}"
    x_label = config.b_x_label

    return (
        DistanceRestraint(int(n_atom), int(c_atom), n_c_dist, k, n_label),
        DistanceRestraint(int(b_atom), int(x_atom), b_x_dist, k, x_label),
    )


# ---------------------------------------------------------------------------
# Core RMSD between simulation frame and reference
# ---------------------------------------------------------------------------


def core_rmsd_to_reference(
    sim_coord: np.ndarray,
    config: ReactionConfig,
    ref_core: ReferenceCore,
) -> float:
    """Compute RMSD between a simulation frame's reactive core and a reference core.

    Parameters
    ----------
    sim_coord : np.ndarray (natom, 3) in Bohr
    ref_core : ReferenceCore with coord_angstrom in Angstrom
    """
    sim_ang = np.asarray(sim_coord, dtype=np.float64) * BOHR_TO_ANGSTROM
    core_indices = [
        config.b_index,
        config.n2_indices[0],
        config.n2_indices[1],
        config.small_c_index,
    ] + list(config.small_x_indices)
    sim_core = sim_ang[np.asarray(core_indices, dtype=np.int64)]
    return _rmsd(ref_core.coord_angstrom, sim_core)


# ---------------------------------------------------------------------------
# PDB / output table parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    step: int
    symbols: tuple[str, ...]
    coord_angstrom: np.ndarray


def _parse_step(line: str) -> int | None:
    match = re.search(r"Step\s*=\s*(-?\d+)", line)
    if match is None:
        return None
    return int(match.group(1))


def read_pdb_frames(filename: str | Path) -> list[Frame]:
    path = Path(filename)
    frames: list[Frame] = []
    step = 0
    symbols: list[str] = []
    coords: list[list[float]] = []

    def flush() -> None:
        nonlocal symbols, coords
        if coords:
            frames.append(
                Frame(step=step, symbols=tuple(symbols), coord_angstrom=np.asarray(coords, dtype=np.float64))
            )
        symbols = []
        coords = []

    for line in path.read_text().splitlines():
        if line.startswith("REMARK"):
            parsed = _parse_step(line)
            if parsed is not None:
                step = parsed
        elif line.startswith("ATOM"):
            symbol = line[76:78].strip() or line[12:16].strip()
            symbols.append(symbol)
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("END"):
            flush()
    flush()
    if not frames:
        raise ValueError(f"no PDB frames found in {path}")
    return frames


def read_output_rows(filename: str | Path) -> dict[int, dict[str, Any]]:
    path = Path(filename)
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty output table: {path}")
    header = lines[0].split()
    rows: dict[int, dict[str, Any]] = {}
    for line in lines[1:]:
        values = line.split()
        row: dict[str, Any] = {}
        for name, value in zip(header, values, strict=False):
            if name == "state_decision":
                row[name] = value
                continue
            try:
                parsed = float(value)
            except ValueError:
                row[name] = value
            else:
                row[name] = int(parsed) if name in {"step", "bias_used", "next_add_bias", "stopped"} else parsed
        if "step" in row:
            rows[int(row["step"])] = row
    return rows


# ---------------------------------------------------------------------------
# Frame metrics
# ---------------------------------------------------------------------------


def _min_b_x(coord: np.ndarray, config: ReactionConfig) -> float:
    return min(_distance(coord, config.b_index, xi) for xi in config.small_x_indices)


def _min_n_c(coord: np.ndarray, config: ReactionConfig) -> float:
    return min(_distance(coord, ni, config.small_c_index) for ni in config.n2_indices)


def _metric_value(row: dict[str, Any] | None, *names: str) -> float | None:
    if row is None:
        return None
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def frame_metrics(
    frame: Frame,
    config: ReactionConfig,
    output_rows: dict[int, dict[str, Any]],
    ts_ref_core: ReferenceCore | None,
    product_ref_core: ReferenceCore | None,
) -> dict[str, Any]:
    row = output_rows.get(frame.step)
    coord = frame.coord_angstrom
    b_x = _min_b_x(coord, config)
    n_c = _min_n_c(coord, config)
    metrics: dict[str, Any] = {
        "step": frame.step,
        f"min_{config.b_x_label.replace('-', '_')}_A": b_x,
        "min_N_C_A": n_c,
        "bond_score_A": b_x + n_c,
        "pot_real_Ha": _metric_value(row, "pot_real_Ha"),
        "pot_bias_Ha": _metric_value(row, "pot_rtip_Ha", "pot_idwm_Ha"),
        "pot_total_Ha": _metric_value(row, "pot_total_Ha"),
        "temp_K": _metric_value(row, "temp_K"),
        "distance_column": _metric_value(row, "rti_dist", "idw_dist"),
        "state_decision": None if row is None else row.get("state_decision"),
    }
    if ts_ref_core is not None:
        metrics["ts_core_rmsd_A"] = core_rmsd_to_reference(
            np.asarray(frame.coord_angstrom, dtype=np.float64) / BOHR_TO_ANGSTROM,
            config,
            ts_ref_core,
        )
    if product_ref_core is not None:
        metrics["product_core_rmsd_A"] = core_rmsd_to_reference(
            np.asarray(frame.coord_angstrom, dtype=np.float64) / BOHR_TO_ANGSTROM,
            config,
            product_ref_core,
        )
    return metrics


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# XYZ export
# ---------------------------------------------------------------------------


def _write_xyz_frame(filename: Path, frame: Frame, title: str) -> None:
    lines = [f"{len(frame.symbols):8d}", title]
    for symbol, coord in zip(frame.symbols, frame.coord_angstrom, strict=True):
        lines.append(f"{symbol:>4} {coord[0]:20.10f} {coord[1]:20.10f} {coord[2]:20.10f}")
    filename.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _markdown_summary(summary: dict[str, Any], config: ReactionConfig) -> str:
    rows = [
        ("final", summary["final_metrics"]),
        ("best TS-like", summary.get("best_ts_like")),
        ("best product-like", summary.get("best_product_like")),
        ("best bond-forming", summary["best_bond_forming"]),
    ]
    b_x_label = config.b_x_label.replace("-", "_")
    lines = [
        f"# RC-MD Summary — {config.reaction_name}",
        "",
        f"- PDB: `{summary['pdb']}`",
        f"- OUT: `{summary['out']}`",
        f"- Small molecule: `{config.small_molecule.value}`",
        f"- Promising bond threshold met: `{summary['promising_bond_threshold_met']}`",
        "",
        f"| frame | step | pot_real / Ha | pot_bias / Ha | min({config.b_x_label}) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in rows:
        if metrics is None:
            continue
        lines.append(
            "| "
            f"{label} | {metrics['step']} | "
            f"{_fmt(metrics.get('pot_real_Ha'))} | {_fmt(metrics.get('pot_bias_Ha'))} | "
            f"{_fmt(metrics.get(f'min_{b_x_label}_A', metrics.get('min_B_O_A')))} | "
            f"{_fmt(metrics.get('min_N_C_A'))} | "
            f"{_fmt(metrics.get('ts_core_rmsd_A'))} | {_fmt(metrics.get('product_core_rmsd_A'))} |"
        )
    lines.extend([
        "",
        "Reference distances:",
        "",
        "| reference | " + f"min({config.b_x_label}) / A | min(N-C) / A |",
        "|---|---:|---:|",
    ])
    for ref_name, ref_metrics in summary.get("reference_metrics", {}).items():
        lines.append(
            f"| {ref_name} | "
            f"{_fmt(ref_metrics.get(f'min_{b_x_label}_A', ref_metrics.get('min_B_O_A')))} | "
            f"{_fmt(ref_metrics.get('min_N_C_A'))} |"
        )
    lines.extend([
        "",
        "Frame files:",
        "",
    ])
    for label, filename in summary.get("frame_files", {}).items():
        lines.append(f"- {label}: `{filename}`")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------


def analyze_generalized_job(
    *,
    pdb_file: str | Path,
    out_file: str | Path,
    reaction_dir: str | Path,
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
) -> dict[str, Any]:
    pdb_path = Path(pdb_file)
    out_path = Path(out_file)
    reaction_path = Path(reaction_dir)
    config = detect_reaction_config(reaction_path)

    # Detect reference cores
    ts_ref_core: ReferenceCore | None = None
    product_ref_core: ReferenceCore | None = None
    ts_path = reaction_path / "ts.xyz"
    product_path = reaction_path / "product.xyz"
    if ts_path.exists():
        ts_system = System.read_xyz(ts_path)
        ts_ref_core = detect_reference_core(ts_system, config)
    if product_path.exists():
        product_system = System.read_xyz(product_path)
        product_ref_core = detect_reference_core(product_system, config)

    # Reference baseline metrics
    reference_metrics: dict[str, dict[str, float]] = {}
    if ts_ref_core is not None:
        ts_sys = System.read_xyz(ts_path)
        reference_metrics["ts"] = {
            f"min_{config.b_x_label.replace('-', '_')}_A": _min_b_x(
                _system_coord_angstrom(ts_sys), config
            ),
            "min_N_C_A": _min_n_c(_system_coord_angstrom(ts_sys), config),
        }
    if product_ref_core is not None:
        prod_sys = System.read_xyz(product_path)
        reference_metrics["product"] = {
            f"min_{config.b_x_label.replace('-', '_')}_A": _min_b_x(
                _system_coord_angstrom(prod_sys), config
            ),
            "min_N_C_A": _min_n_c(_system_coord_angstrom(prod_sys), config),
        }

    frames = read_pdb_frames(pdb_path)
    output_rows = read_output_rows(out_path)
    data_frames = [frame for frame in frames if frame.step in output_rows]
    if not data_frames:
        data_frames = frames

    all_metrics = [
        frame_metrics(frame, config, output_rows, ts_ref_core, product_ref_core)
        for frame in data_frames
    ]
    by_step = {m["step"]: m for m in all_metrics}
    final_frame = data_frames[-1]
    final_metrics = by_step[final_frame.step]

    best_bond = min(all_metrics, key=lambda m: m["bond_score_A"])

    best_ts = None
    best_product = None
    if ts_ref_core is not None:
        best_ts = min(all_metrics, key=lambda m: m.get("ts_core_rmsd_A", float("inf")))
    if product_ref_core is not None:
        best_product = min(all_metrics, key=lambda m: m.get("product_core_rmsd_A", float("inf")))

    frames_by_step = {f.step: f for f in data_frames}

    out_dir = Path(output_dir) if output_dir is not None else pdb_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or pdb_path.stem
    frame_files: dict[str, Path] = {
        "final": out_dir / f"{prefix}_final.xyz",
        "best_bond_forming": out_dir / f"{prefix}_best_bond_forming.xyz",
    }
    _write_xyz_frame(frame_files["final"], final_frame, f"final step {final_frame.step}")
    _write_xyz_frame(
        frame_files["best_bond_forming"],
        frames_by_step[best_bond["step"]],
        f"best bond-forming step {best_bond['step']}",
    )
    if best_ts is not None:
        frame_files["best_ts_like"] = out_dir / f"{prefix}_best_ts_like.xyz"
        _write_xyz_frame(
            frame_files["best_ts_like"],
            frames_by_step[best_ts["step"]],
            f"best TS-like step {best_ts['step']}",
        )
    if best_product is not None:
        frame_files["best_product_like"] = out_dir / f"{prefix}_best_product_like.xyz"
        _write_xyz_frame(
            frame_files["best_product_like"],
            frames_by_step[best_product["step"]],
            f"best product-like step {best_product['step']}",
        )

    promising = best_bond["min_N_C_A"] < 2.0 and best_bond.get(f"min_{config.b_x_label.replace('-', '_')}_A", 99) < 2.0

    summary: dict[str, Any] = {
        "reaction": config.reaction_name,
        "small_molecule": config.small_molecule.value,
        "pdb": str(pdb_path),
        "out": str(out_path),
        "input": str(input_file) if input_file else None,
        "reaction_dir": str(reaction_path),
        "config": {
            "b_index": config.b_index,
            "n2_indices": list(config.n2_indices),
            "small_c_index": config.small_c_index,
            "small_x_indices": list(config.small_x_indices),
            "b_x_label": config.b_x_label,
        },
        "final_metrics": final_metrics,
        "best_ts_like": best_ts,
        "best_product_like": best_product,
        "best_bond_forming": best_bond,
        "reference_metrics": reference_metrics,
        "promising_bond_threshold_met": promising,
        "frame_files": {name: str(path) for name, path in frame_files.items()},
    }

    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    summary_json.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n")
    summary_md.write_text(_markdown_summary(summary, config))
    return summary
