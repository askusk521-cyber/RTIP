from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rtip_jax.constants import BOHR_TO_ANGSTROM, element_symbol
from rtip_jax.system import System


B_INDEX = 5
N2_INDICES = (16, 17)
CO2_C_INDEX = 33
CO2_O_INDICES = (34, 35)
CORE_INDICES = (B_INDEX, *N2_INDICES, CO2_C_INDEX, *CO2_O_INDICES)
REFERENCE_REORDER_1TBU_CO2 = tuple(range(20)) + (
    23,
    24,
    28,
    32,
    29,
    30,
    31,
    33,
    34,
    35,
    25,
    26,
    27,
    21,
    20,
    22,
)


@dataclass(frozen=True)
class Frame:
    step: int
    symbols: tuple[str, ...]
    coord_angstrom: np.ndarray


def _system_symbols(system: System) -> tuple[str, ...]:
    if system.atom_type is None:
        return tuple("XX" for _ in range(system.natom))
    return tuple(element_symbol(element) for element in system.atom_type)


def _system_coord_angstrom(system: System) -> np.ndarray:
    return np.asarray(system.coord, dtype=np.float64) * BOHR_TO_ANGSTROM


def _reorder_reference_to_input(reference: System, input_system: System) -> Frame:
    ref_symbols = _system_symbols(reference)
    input_symbols = _system_symbols(input_system)
    ref_coord = _system_coord_angstrom(reference)
    if ref_symbols == input_symbols:
        return Frame(step=0, symbols=ref_symbols, coord_angstrom=ref_coord)

    order = REFERENCE_REORDER_1TBU_CO2
    if len(order) == reference.natom == input_system.natom:
        ordered_symbols = tuple(ref_symbols[index] for index in order)
        if ordered_symbols == input_symbols:
            return Frame(
                step=0,
                symbols=ordered_symbols,
                coord_angstrom=ref_coord[np.asarray(order, dtype=np.int64)],
            )

    raise ValueError("reference atom order cannot be matched to the input atom order")


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
            frames.append(Frame(step=step, symbols=tuple(symbols), coord_angstrom=np.asarray(coords, dtype=np.float64)))
        symbols = []
        coords = []

    for line in path.read_text().splitlines():
        if line.startswith("REMARK"):
            parsed_step = _parse_step(line)
            if parsed_step is not None:
                step = parsed_step
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


def _align_mobile_to_reference(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    ref_center = reference.mean(axis=0)
    mob_center = mobile.mean(axis=0)
    ref0 = reference - ref_center
    mob0 = mobile - mob_center
    u, _s, vt = np.linalg.svd(mob0.T @ ref0)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return mob0 @ rotation + ref_center


def _rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    aligned = _align_mobile_to_reference(reference, mobile)
    diff = aligned - reference
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _distance(coord: np.ndarray, atom_i: int, atom_j: int) -> float:
    return float(np.linalg.norm(coord[atom_i] - coord[atom_j]))


def _min_b_o(coord: np.ndarray) -> float:
    return min(_distance(coord, B_INDEX, oxygen) for oxygen in CO2_O_INDICES)


def _min_n_c(coord: np.ndarray) -> float:
    return min(_distance(coord, nitrogen, CO2_C_INDEX) for nitrogen in N2_INDICES)


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
    *,
    output_rows: dict[int, dict[str, Any]],
    ts_reference: Frame,
    product_reference: Frame,
) -> dict[str, Any]:
    row = output_rows.get(frame.step)
    coord = frame.coord_angstrom
    core = np.asarray(CORE_INDICES, dtype=np.int64)
    metrics: dict[str, Any] = {
        "step": frame.step,
        "min_B_O_A": _min_b_o(coord),
        "min_N_C_A": _min_n_c(coord),
        "ts_core_rmsd_A": _rmsd(ts_reference.coord_angstrom[core], coord[core]),
        "product_core_rmsd_A": _rmsd(product_reference.coord_angstrom[core], coord[core]),
        "bond_score_A": _min_b_o(coord) + _min_n_c(coord),
        "pot_real_Ha": _metric_value(row, "pot_real_Ha"),
        "pot_bias_Ha": _metric_value(row, "pot_rtip_Ha", "pot_idwm_Ha"),
        "pot_total_Ha": _metric_value(row, "pot_total_Ha"),
        "temp_K": _metric_value(row, "temp_K"),
        "distance_column": _metric_value(row, "rti_dist", "idw_dist"),
        "state_decision": None if row is None else row.get("state_decision"),
    }
    return metrics


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_xyz_frame(filename: Path, frame: Frame, title: str) -> None:
    lines = [f"{len(frame.symbols):8d}", title]
    for symbol, coord in zip(frame.symbols, frame.coord_angstrom, strict=True):
        lines.append(f"{symbol:>4} {coord[0]:20.10f} {coord[1]:20.10f} {coord[2]:20.10f}")
    filename.write_text("\n".join(lines) + "\n")


def _markdown_summary(summary: dict[str, Any]) -> str:
    rows = [
        ("final", summary["final_metrics"]),
        ("best TS-like", summary["best_ts_like"]),
        ("best product-like", summary["best_product_like"]),
        ("best bond-forming", summary["best_bond_forming"]),
    ]
    lines = [
        "# Bias Exploration Summary",
        "",
        f"- PDB: `{summary['pdb']}`",
        f"- OUT: `{summary['out']}`",
        f"- Promising bond threshold met: `{summary['promising_bond_threshold_met']}`",
        "",
        "| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in rows:
        lines.append(
            "| "
            f"{label} | {metrics['step']} | "
            f"{_fmt(metrics.get('pot_real_Ha'))} | {_fmt(metrics.get('pot_bias_Ha'))} | "
            f"{_fmt(metrics['min_B_O_A'])} | {_fmt(metrics['min_N_C_A'])} | "
            f"{_fmt(metrics['ts_core_rmsd_A'])} | {_fmt(metrics['product_core_rmsd_A'])} |"
        )
    lines.extend(
        [
            "",
            "Reference distances:",
            "",
            "| reference | min(B-O) / A | min(N-C) / A |",
            "|---|---:|---:|",
            f"| TS | {_fmt(summary['reference_metrics']['ts']['min_B_O_A'])} | {_fmt(summary['reference_metrics']['ts']['min_N_C_A'])} |",
            f"| product | {_fmt(summary['reference_metrics']['product']['min_B_O_A'])} | {_fmt(summary['reference_metrics']['product']['min_N_C_A'])} |",
            "",
            "Frame files:",
            "",
        ]
    )
    for label, filename in summary["frame_files"].items():
        lines.append(f"- {label}: `{filename}`")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def analyze_job(
    *,
    pdb_file: str | Path,
    out_file: str | Path,
    reaction_dir: str | Path = "examples/ic5c02384/reactions/1-tBu__CO2",
    input_file: str | Path = "examples/ic5c02384/reactions/1-tBu__CO2/IS.xyz",
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
) -> dict[str, Any]:
    pdb_path = Path(pdb_file)
    out_path = Path(out_file)
    reaction_path = Path(reaction_dir)
    input_system = System.read_xyz(input_file)
    ts_reference = _reorder_reference_to_input(System.read_xyz(reaction_path / "ts.xyz"), input_system)
    product_reference = _reorder_reference_to_input(System.read_xyz(reaction_path / "product.xyz"), input_system)

    frames = read_pdb_frames(pdb_path)
    output_rows = read_output_rows(out_path)
    data_frames = [frame for frame in frames if frame.step in output_rows]
    if not data_frames:
        data_frames = frames

    all_metrics = [
        frame_metrics(
            frame,
            output_rows=output_rows,
            ts_reference=ts_reference,
            product_reference=product_reference,
        )
        for frame in data_frames
    ]
    by_step = {metric["step"]: metric for metric in all_metrics}
    final_frame = data_frames[-1]
    final_metrics = by_step[final_frame.step]
    best_ts_metrics = min(all_metrics, key=lambda metric: metric["ts_core_rmsd_A"])
    best_product_metrics = min(all_metrics, key=lambda metric: metric["product_core_rmsd_A"])
    best_bond_metrics = min(all_metrics, key=lambda metric: metric["bond_score_A"])
    frames_by_step = {frame.step: frame for frame in data_frames}

    out_dir = Path(output_dir) if output_dir is not None else pdb_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or pdb_path.stem
    frame_files = {
        "final": out_dir / f"{prefix}_final.xyz",
        "best_ts_like": out_dir / f"{prefix}_best_ts_like.xyz",
        "best_product_like": out_dir / f"{prefix}_best_product_like.xyz",
        "best_bond_forming": out_dir / f"{prefix}_best_bond_forming.xyz",
    }
    _write_xyz_frame(frame_files["final"], final_frame, f"final step {final_frame.step}")
    _write_xyz_frame(
        frame_files["best_ts_like"],
        frames_by_step[best_ts_metrics["step"]],
        f"best TS-like step {best_ts_metrics['step']}",
    )
    _write_xyz_frame(
        frame_files["best_product_like"],
        frames_by_step[best_product_metrics["step"]],
        f"best product-like step {best_product_metrics['step']}",
    )
    _write_xyz_frame(
        frame_files["best_bond_forming"],
        frames_by_step[best_bond_metrics["step"]],
        f"best bond-forming step {best_bond_metrics['step']}",
    )

    reference_metrics = {
        "ts": {
            "min_B_O_A": _min_b_o(ts_reference.coord_angstrom),
            "min_N_C_A": _min_n_c(ts_reference.coord_angstrom),
        },
        "product": {
            "min_B_O_A": _min_b_o(product_reference.coord_angstrom),
            "min_N_C_A": _min_n_c(product_reference.coord_angstrom),
        },
    }
    summary = {
        "pdb": str(pdb_path),
        "out": str(out_path),
        "input": str(input_file),
        "reaction_dir": str(reaction_path),
        "final_metrics": final_metrics,
        "best_ts_like": best_ts_metrics,
        "best_product_like": best_product_metrics,
        "best_bond_forming": best_bond_metrics,
        "reference_metrics": reference_metrics,
        "promising_bond_threshold_met": best_bond_metrics["min_N_C_A"] < 2.0 and best_bond_metrics["min_B_O_A"] < 2.0,
        "frame_files": {name: str(path) for name, path in frame_files.items()},
    }
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    summary_json.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n")
    summary_md.write_text(_markdown_summary(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 1-tBu__CO2 RTIP bias exploration outputs.")
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reaction-dir", default="examples/ic5c02384/reactions/1-tBu__CO2")
    parser.add_argument("--input", default="examples/ic5c02384/reactions/1-tBu__CO2/IS.xyz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = analyze_job(
        pdb_file=args.pdb,
        out_file=args.out,
        reaction_dir=args.reaction_dir,
        input_file=args.input,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(_json_ready(summary["final_metrics"]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
