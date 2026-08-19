"""教科书反应验证：按 case 追踪特定原子对的成键过程。

为什么不用全局邻接事件: 原代码的 1.25 倍共价半径邻接会把乙烯分子内
1,2-C-H 近距接触(约 1.2 A)误判为键，产生假断键/假成键事件。
因此这里只追踪化学反应相关的特定原子对，采用教科书物理键长阈值:
  H-H < 0.90 A (H2 平衡 0.741)
  C-H < 1.15 A (烷基 C-H 约 1.09)
  O-H < 1.05 A (O-H 约 0.96)
  C-C < 1.65 A (C-C 单键 1.54, 双键 1.34)
  C-O < 1.50 A (C-O 单键 1.43)

原子索引按初始 xyz 顺序:
  validation_ch2o_h2:  O0 C1 H2 H3(CH2O) H4 H5(H2)
  validation_c2h4_h2:  C0 C1 H2 H3 H4 H5(乙烯) H6 H7(H2)
  validation_ch2o_ch4: O0 C1 H2 H3(CH2O) C4 H5 H6 H7 H8(CH4)

用法 (n5):
  PYTHONPATH=rtipmd/jax/src JAX/.venv/bin/python \
      formose/analyze_validation.py formose/runs/validation_<case>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


THRESHOLDS = {
    "H-H": 0.90,
    "C-H": 1.15,
    "O-H": 1.05,
    "C-C": 1.65,
    "C-O": 1.50,
}


CASES = {
    "validation_ch2o_h2": {
        "track": {
            "H-H (H2)": ((4, 5), "H-H"),
            "C-O (醛基)": ((0, 1), "C-O"),
            "O-H (H2->O)": ((0, 4), "O-H"),
            "C-H (H2->C)": ((1, 5), "C-H"),
        },
        "expect": "ch2o_h2",
    },
    "validation_c2h4_h2": {
        "track": {
            "H-H (H2)": ((6, 7), "H-H"),
            "C-C (C=C)": ((0, 1), "C-C"),
            "C-H 候选1": ((0, 6), "C-H"),
            "C-H 候选2": ((1, 7), "C-H"),
            "C-H 候选3": ((0, 7), "C-H"),
            "C-H 候选4": ((1, 6), "C-H"),
        },
        "expect": "c2h4_h2",
    },
    "validation_ch2o_ch4": {
        "track": {
            "C-C (CH2O-C + CH4-C)": ((1, 4), "C-C"),
            "C-H (CH4 内)": ((4, 5), "C-H"),
            "H-H (CH4 两 H)": ((5, 6), "H-H"),
        },
        "expect": "ch2o_ch4",
    },
}


def read_pdb_frames(path: Path) -> list[tuple[str, np.ndarray]]:
    """Return (atom_symbols, coord_angstrom) frames from a multi-frame PDB."""

    frames: list[tuple[str, np.ndarray]] = []
    symbols: list[str] = []
    coord: list[list[float]] = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            numbers = re.findall(r"-?\d+\.\d+", line)
            if len(numbers) < 4 or "nan" in line:
                continue
            symbols.append(line[76:78].strip() or line[12:16].strip().lstrip("0123456789"))
            coord.append([float(numbers[0]), float(numbers[1]), float(numbers[2])])
        elif line.startswith(("ENDMDL", "END")) and symbols:
            frames.append((tuple(symbols), np.asarray(coord, dtype=np.float64)))
            symbols, coord = [], []
    if symbols:
        frames.append((tuple(symbols), np.asarray(coord, dtype=np.float64)))
    return frames


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def analyze(run_dir: Path) -> str:
    lines: list[str] = []
    spec = None
    for key, candidate in CASES.items():
        if run_dir.name.startswith(key):
            spec = candidate
            break
    if spec is None:
        return f"# {run_dir.name}: 未知 case（未在 CASES 表中）\n"
    track = spec["track"]

    frames = read_pdb_frames(run_dir / "rtip.pdb")
    if not frames:
        return f"# {run_dir.name}: no frames\n"
    symbols0, coord0 = frames[0]
    lines.append(f"# 教科书反应验证: {run_dir.name}")
    lines.append("")
    lines.append(f"- 原子数: {len(symbols0)}; 初始分子数: 2")
    lines.append(f"- PDB 帧数: {len(frames)} (每 10 步一帧)")

    # 运行完整性
    out_rows: list[list[str]] = []
    with (run_dir / "rtip.out").open() as fh:
        next(fh)
        for line in fh:
            p = line.split()
            if len(p) >= 4:
                out_rows.append(p)
    last_step = int(out_rows[-1][0]) if out_rows else 0
    nan_count = sum(1 for p in out_rows if "nan" in p[3])
    complete = bool(out_rows) and nan_count == 0 and last_step == int(out_rows[0][0]) + len(out_rows) - 1
    lines.append(f"- rtip.out 末步: {last_step}; NaN 行: {nan_count}; 跑完: {'是' if complete else '否'}")

    # 每个追踪对记录首次 formed / broken 与末帧状态（高温振动会使键长
    # 反复穿越阈值，故只报告第一次变化，几何以末帧为准）
    first_formed: dict[str, dict] = {}
    first_broken: dict[str, dict] = {}
    bonded_now: dict[str, bool] = {}
    for fi, (symbols, coord) in enumerate(frames):
        step = fi * 10
        for name, ((i, j), kind) in track.items():
            d = dist(coord[i], coord[j])
            bonded = d < THRESHOLDS[kind]
            if name not in bonded_now:
                bonded_now[name] = bonded
                continue
            if bonded and not bonded_now[name]:
                first_formed.setdefault(name, {"step": step, "d": d})
                bonded_now[name] = True
            elif not bonded and bonded_now[name]:
                first_broken.setdefault(name, {"step": step, "d": d})
                bonded_now[name] = False

    lines.append("")
    lines.append("## 追踪键（阈值见文件头）")
    for name, ((i, j), kind) in track.items():
        d0 = dist(coord0[i], coord0[j])
        dL = dist(frames[-1][1][i], frames[-1][1][j])
        lines.append(f"- {name}: 初帧 {d0:.2f} A -> 末帧 {dL:.2f} A")

    lines.append("")
    lines.append("## 键状态事件（首次）")
    if first_formed or first_broken:
        for name in track:
            if name in first_broken:
                ev = first_broken[name]
                lines.append(f"- step {ev['step']}: {name} 断裂 ({ev['d']:.2f} A)")
            if name in first_formed:
                ev = first_formed[name]
                lines.append(f"- step {ev['step']}: {name} 成键 ({ev['d']:.2f} A)")
    else:
        lines.append("- 无")

    # 末帧分子数（简单判据: 用 1.25 倍共价半径的代码逻辑太重，这里用
    # 追踪键 + 全部原子对的阈值做一个保守估计）
    symbolsL, coordL = frames[-1]
    nmolL = estimate_nmol(symbolsL, coordL)
    lines.append("")
    lines.append(f"## 末帧估计分子数: {nmolL}")

    # 判据
    formed = set(first_formed)
    broken = set(first_broken)
    if spec["expect"] == "ch2o_h2":
        ok = (
            complete
            and "H-H (H2)" in broken
            and "O-H (H2->O)" in formed
            and "C-H (H2->C)" in formed
            and nmolL == 1
        )
        lines.append("")
        lines.append(f"- [{'x' if ok else ' '}] 正对照 CH2O+H2: H2 解离 + O-H 与 C-H 成键 + 末帧单分子 (甲醇)")
    elif spec["expect"] == "c2h4_h2":
        nch = sum(1 for k in ("C-H 候选1", "C-H 候选2", "C-H 候选3", "C-H 候选4") if k in formed)
        ok = complete and "H-H (H2)" in broken and nch >= 2 and nmolL == 1
        lines.append("")
        lines.append(f"- [{'x' if ok else ' '}] 正对照 C2H4+H2: H2 解离 + >=2 个新 C-H (实际 {nch}) + 末帧单分子 (乙烷)")
    else:  # ch2o_ch4
        ok = complete and "C-C (CH2O-C + CH4-C)" not in formed and nmolL >= 2
        lines.append("")
        lines.append(f"- [{'x' if ok else ' '}] 负对照 CH2O+CH4: 无 C-C 成键事件 + 末帧仍 >=2 分子")

    lines.append("")
    lines.append(f"结论: {'通过' if ok else '未通过'}")
    return "\n".join(lines) + "\n"


def estimate_nmol(symbols, coord) -> int:
    """分子数估计：用原代码 1.25 倍共价半径邻接（该约定对分子连通性
    计数是正确的；它只在高频键事件检测上会产生乙烯式伪影，因此键事件
    另用物理阈值，见文件头）。"""

    radii = {"H": 0.31, "C": 0.76, "O": 0.66}
    n = len(symbols)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            rsum = radii[symbols[i]] + radii[symbols[j]]
            if dist(coord[i], coord[j]) < 1.25 * rsum:
                union(i, j)
    return len({find(i) for i in range(n)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    report = analyze(args.run_dir)
    out = args.output if args.output is not None else args.run_dir / "validation_report.md"
    out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
