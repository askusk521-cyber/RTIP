"""Aggregate size_scaling A/B synthesis-MD runs into a statistical summary.

Scans the size-scaling results directory for ``synth_<reaction>_..._seed<k>_<OFF|ON>``
job folders, reads each ``summary.json`` plus the trajectory ``.out`` (for peak
temperature and final state), and emits per-seed rows plus OFF-vs-ON aggregate
statistics (mean/std of best B-X, best N-C, best bond-score, product-core RMSD).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


JOB_RE = re.compile(r"^synth_(?P<reaction>.+)_a(?P<a0>[\d.]+)_s(?P<steps>\d+)_seed(?P<seed>\d+)_(?P<scale>OFF|ON)$")


def _peak_temp_and_state(out_file: Path) -> tuple[float | None, str | None, int | None]:
    if not out_file.exists():
        return None, None, None
    header: list[str] = []
    peak = None
    last_state = None
    last_step = None
    for i, line in enumerate(out_file.read_text().splitlines()):
        if not line.strip():
            continue
        parts = line.split()
        if i == 0:
            header = parts
            continue
        row = dict(zip(header, parts))
        try:
            t = float(row.get("temp_K", "nan"))
            peak = t if peak is None else max(peak, t)
        except ValueError:
            pass
        last_state = row.get("state_decision", last_state)
        try:
            last_step = int(float(row.get("step", last_step)))
        except (ValueError, TypeError):
            pass
    return peak, last_state, last_step


def _bx_key(metrics: dict[str, Any]) -> str | None:
    for k in metrics:
        if k.startswith("min_B_") and k.endswith("_A"):
            return k
    return None


def _collect(results_dir: Path, reaction_filter: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job_dir in sorted(results_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        m = JOB_RE.match(job_dir.name)
        if m is None:
            continue
        if reaction_filter and m.group("reaction") != reaction_filter:
            continue
        summary_path = job_dir / "summary.json"
        if not summary_path.exists():
            rows.append({"job": job_dir.name, "reaction": m.group("reaction"),
                         "seed": int(m.group("seed")), "scale": m.group("scale"),
                         "status": "NO_SUMMARY"})
            continue
        summary = json.loads(summary_path.read_text())
        best = summary.get("best_bond_forming", {})
        finalm = summary.get("final_metrics", {})
        bx_key = _bx_key(best) or _bx_key(finalm)
        out_file = Path(summary.get("out", "")) if summary.get("out") else (job_dir / f"{job_dir.name}.out")
        peak_t, last_state, last_step = _peak_temp_and_state(out_file)
        rows.append({
            "job": job_dir.name,
            "reaction": m.group("reaction"),
            "seed": int(m.group("seed")),
            "scale": m.group("scale"),
            "bx_label": summary.get("config", {}).get("b_x_label"),
            "best_bx_A": best.get(bx_key) if bx_key else None,
            "best_nc_A": best.get("min_N_C_A"),
            "best_bond_score_A": best.get("bond_score_A"),
            "best_product_rmsd_A": best.get("product_core_rmsd_A"),
            "best_step": best.get("step"),
            "final_bx_A": finalm.get(bx_key) if bx_key else None,
            "final_nc_A": finalm.get("min_N_C_A"),
            "promising": summary.get("promising_bond_threshold_met"),
            "peak_temp_K": peak_t,
            "final_state": last_state,
            "last_step": last_step,
            "status": "OK",
        })
    return rows


def _agg(rows: list[dict[str, Any]], scale: str, field: str) -> tuple[float | None, float | None, int]:
    vals = [r[field] for r in rows if r.get("scale") == scale and r.get("status") == "OK"
            and isinstance(r.get(field), (int, float))]
    if not vals:
        return None, None, 0
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std, len(vals)


def _fmt(v: Any) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate size_scaling synthesis A/B runs.")
    ap.add_argument("--results-dir", default="/home/lhshen/RTIP/research/ic5c02384/results/size-scaling")
    ap.add_argument("--reaction", default=None, help="Filter to one reaction (e.g. 1-tBu__CS2).")
    ap.add_argument("--output", default=None, help="Markdown output path.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    rows = _collect(results_dir, args.reaction)
    synth_rows = [r for r in rows if r.get("status") == "OK"]

    lines: list[str] = ["# size_scaling synthesis A/B aggregate", ""]
    lines.append(f"- results_dir: `{results_dir}`")
    if args.reaction:
        lines.append(f"- reaction filter: `{args.reaction}`")
    lines.append(f"- jobs found: {len(rows)} (with summary: {len(synth_rows)})")
    lines.append("")
    lines.append("## Per-seed results")
    lines.append("")
    lines.append("| job | seed | scale | best B-X / A | best N-C / A | bond score / A | prod RMSD / A | best step | peak T / K | final state | promising |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---|:---:|")
    for r in sorted(rows, key=lambda x: (x.get("scale", ""), x.get("seed", -1))):
        if r.get("status") != "OK":
            lines.append(f"| {r['job']} | {r.get('seed')} | {r.get('scale')} | — | — | — | — | — | — | {r.get('status')} | — |")
            continue
        lines.append(
            f"| {r['job']} | {r['seed']} | {r['scale']} | {_fmt(r['best_bx_A'])} | {_fmt(r['best_nc_A'])} | "
            f"{_fmt(r['best_bond_score_A'])} | {_fmt(r['best_product_rmsd_A'])} | {_fmt(r['best_step'])} | "
            f"{_fmt(r['peak_temp_K'])} | {r['final_state']} | {r['promising']} |"
        )
    lines.append("")
    lines.append("## OFF vs ON aggregate (mean ± population std)")
    lines.append("")
    lines.append("| metric | OFF | ON |")
    lines.append("|---|---|---|")
    for field, label in [
        ("best_bx_A", "best B-X / A"),
        ("best_nc_A", "best N-C / A"),
        ("best_bond_score_A", "best bond score / A"),
        ("best_product_rmsd_A", "best product RMSD / A"),
        ("peak_temp_K", "peak T / K"),
    ]:
        off_m, off_s, off_n = _agg(synth_rows, "OFF", field)
        on_m, on_s, on_n = _agg(synth_rows, "ON", field)
        off_str = "n/a" if off_m is None else f"{off_m:.3f} ± {off_s:.3f} (n={off_n})"
        on_str = "n/a" if on_m is None else f"{on_m:.3f} ± {on_s:.3f} (n={on_n})"
        lines.append(f"| {label} | {off_str} | {on_str} |")

    n_off_promising = sum(1 for r in synth_rows if r["scale"] == "OFF" and r.get("promising"))
    n_on_promising = sum(1 for r in synth_rows if r["scale"] == "ON" and r.get("promising"))
    n_off = sum(1 for r in synth_rows if r["scale"] == "OFF")
    n_on = sum(1 for r in synth_rows if r["scale"] == "ON")
    lines.append(f"| promising bond (B-X&N-C<2A) | {n_off_promising}/{n_off} | {n_on_promising}/{n_on} |")
    lines.append("")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.output:
        Path(args.output).write_text(text)
        print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
