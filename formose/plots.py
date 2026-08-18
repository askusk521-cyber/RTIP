# -*- coding: utf-8 -*-
"""Formose RTIP-MD 结果图表生成（在 n5 的 deepmd 环境运行）。

用法:
    source /group/software/deepmd-kit-3.1.1/bin/activate
    cd /home/lhshen/RTIP
    python formose/plots.py

输出: formose/plots/*.png
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path("/home/lhshen/RTIP")
PLOT = ROOT / "formose/plots"
PLOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "rtipmd/jax/src"))
sys.path.insert(0, str(ROOT / "formose"))

# 中文字体（n5 上的 Noto CJK）
for _p in sorted(Path("/usr/share/fonts/opentype/noto").glob("NotoSerifCJK*.ttc")):
    try:
        fm.fontManager.addfont(str(_p))
    except Exception:
        pass
cjk_font = None
for _name in ("Noto Serif CJK SC", "Noto Serif CJK", "Noto Serif CJK JP"):
    try:
        fm.findfont(_name, fallback_to_default=False)
        cjk_font = _name
        break
    except Exception:
        continue
if cjk_font:
    plt.rcParams["font.family"] = [cjk_font, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PAPER_T = 1500.0
CC_BOND_A = 1.52
CC_EVENTS = {
    "seed0": (7570, 1.357),
    "seed2": (5700, 1.406),
    "r2pair": (250, 2.619),
    "r5pair": (700, 1.515),
}
RUN_NAMES = ["seed0", "seed1", "seed2", "r2pair", "r5pair"]


def read_out(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+")
    return df.replace([np.inf, -np.inf], np.nan)


def read_pdb_frames(path: Path):
    frames = []
    atom_type, coord = [], []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            numbers = re.findall(r"-?\d+\.\d+", line)
            if "nan" in line.lower() or len(numbers) < 3:
                continue
            atom_type.append(line[76:78].strip() or line[12:16].strip().lstrip("0123456789"))
            coord.append([float(numbers[0]), float(numbers[1]), float(numbers[2])])
        elif line.startswith("END") and atom_type:
            frames.append((list(atom_type), np.asarray(coord, dtype=float)))
            atom_type, coord = [], []
    if atom_type:
        frames.append((list(atom_type), np.asarray(coord, dtype=float)))
    return frames


def min_cc_series(path: Path, step: int = 10) -> np.ndarray:
    out = []
    for k, (atom_type, coord) in enumerate(read_pdb_frames(path)):
        c = [i for i, e in enumerate(atom_type) if e == "C"]
        if len(c) < 2:
            out.append((k * step, np.nan))
            continue
        d = min(np.linalg.norm(coord[c[a]] - coord[c[b]]) for a in range(len(c)) for b in range(a + 1, len(c)))
        out.append((k * step, d))
    return np.asarray(out, dtype=float)


def _cc_time(step: int) -> float:
    return step * 0.5  # 0.5 fs/step


def fig_scalars(name: str, run: Path) -> None:
    df = read_out(run / "rtip.out")
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    cols = [("temp_K", "温度 T (K)"), ("pot_real_Ha", "真实势能 E_real (Ha)"),
            ("pot_rtip_Ha", "RTIP 势能 E_rtip (Ha)"), ("rti_dist", "RTI 距离 (Bohr)")]
    for a, (col, lab) in zip(ax.ravel(), cols):
        y = df[col].to_numpy(dtype=float)
        mask = np.isfinite(y)
        a.plot(df["time_fs"].to_numpy()[mask], y[mask], lw=0.7)
        if col == "temp_K":
            a.axhline(PAPER_T, ls="--", c="gray", lw=0.8)
            a.text(0.02, 0.94, f"论文目标 1500 K", transform=a.transAxes, fontsize=8, va="top", color="gray")
        if name in CC_EVENTS:
            t = _cc_time(CC_EVENTS[name][0])
            a.axvline(t, ls=":", c="crimson", lw=1)
            a.text(t, a.get_ylim()[1], "C-C", color="crimson", fontsize=8, va="top", ha="left")
        a.set_ylabel(lab)
    ax[-1, 0].set_xlabel("时间 (fs)")
    ax[-1, 1].set_xlabel("时间 (fs)")
    fig.suptitle(f"RTIP-MD 标量时间序列 — {name}（平均温度 {df['temp_K'].mean():.0f} K）")
    fig.tight_layout()
    fig.savefig(PLOT / f"scalars_{name}.png", dpi=160)
    plt.close(fig)


def fig_mincc(name: str, run: Path) -> None:
    s = min_cc_series(run / "rtip.pdb")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s[:, 0], s[:, 1], lw=0.9, label="最近 C-C 距离")
    ax.axhline(CC_BOND_A, ls="--", c="gray", lw=0.9, label="1.52 Å 成键阈值")
    if name in CC_EVENTS:
        st, d = CC_EVENTS[name]
        ax.scatter([st], [d], c="crimson", zorder=3, s=40, label=f"C-C {d:.3f} Å @ step {st}")
    ax.set(xlabel="步数 step", ylabel="最近 C-C 距离 (Å)",
           title=f"最近 C-C 距离演化 — {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT / f"mincc_{name}.png", dpi=160)
    plt.close(fig)


def fig_cycles(name: str, run: Path) -> None:
    rows = np.atleast_2d(np.loadtxt(run / "rtip_decreasing_steps", skiprows=1))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if n <= 60:
        ax.barh(np.arange(n), rows[:, 1] - rows[:, 0], left=rows[:, 0], height=0.7,
                color="#4C72B0", label="RTIP 减小阶段")
        ax.set_ylabel("循环序号")
    else:
        max_step = rows[-1, 1] if len(rows) else 0
        state = np.zeros(int(max_step) + 2)
        for b, e in rows:
            state[int(b):int(e) + 1] = 1
        ax.step(np.arange(len(state)), state, where="post", lw=1.2, color="#4C72B0")
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["增大", "减小"])
    ax.set_xlabel("步数 step")
    ax.set_title(f"RTIP 增/减状态机 — {name}（{n} 个循环）")
    fig.tight_layout()
    fig.savefig(PLOT / f"cycles_{name}.png", dpi=160)
    plt.close(fig)


def fig_events(name: str, run: Path) -> None:
    sys.path.insert(0, str(ROOT / "formose"))
    from analyze_run import bond_events, read_pdb_frames as _read_frames
    events = bond_events(_read_frames(run / "rtip.pdb"))
    counts: dict[str, dict[str, int]] = {}
    for kind, bond, _d, _f in events:
        counts.setdefault(bond, {"formed": 0, "broken": 0})[kind] += 1
    bonds = sorted(counts)
    x = np.arange(len(bonds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, [counts[b]["formed"] for b in bonds], w, label="成键")
    ax.bar(x + w / 2, [counts[b]["broken"] for b in bonds], w, label="断键")
    ax.set_xticks(x)
    ax.set_xticklabels(bonds)
    ax.set_ylabel("次数")
    ax.set_title(f"键变化事件统计 — {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT / f"events_{name}.png", dpi=160)
    plt.close(fig)


def fig_micro() -> None:
    c = pd.read_csv(ROOT / "formose/microkinetics/runs/concentrations.csv")
    s = json.loads((ROOT / "formose/microkinetics/runs/summary.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for col, lab in [("s1", "CH2O (1)"), ("s4", "乙醇醛 (4)"), ("s7", "甘油醛 (7)"),
                     ("s19", "四碳糖 (19)"), ("s23", "四碳糖 (23)"), ("s11", "核糖 (11)")]:
        ax[0].plot(c["time_s"], c[col], lw=1, label=lab)
    ax[0].set(xlabel="时间 (s)", ylabel="浓度 (M)", title="主要物种浓度")
    ax[0].legend(fontsize=8)
    ax[1].plot(c["time_s"], c["s2"], lw=1, label="formyl anion (2)")
    ax[1].set_yscale("log")
    ax[1].set(xlabel="时间 (s)", ylabel="浓度 (M)", title="formyl anion（对数坐标）")
    ax[1].text(0.03, 0.03,
               f"c(2) ~ {s['formyl_anion_c_M']:.2e} M\n"
               f"二聚速率 ~ {s['formaldehyde_dimerization_rate_mol_L_s']:.1e} mol L⁻¹ s⁻¹\n"
               f"retroaldol 净速率转正 @ {s['retroaldol_net_rate_positive_at_s']:.2f} s",
               transform=ax[1].transAxes, fontsize=9, va="bottom")
    ax[1].legend(fontsize=8)
    fig.suptitle("微动力学模拟（对应论文 Figure 4）")
    fig.tight_layout()
    fig.savefig(PLOT / "microkinetics.png", dpi=160)
    plt.close(fig)


def fig_summary() -> None:
    rows = []
    for name in ["seed0", "seed1", "seed2"]:
        d = read_out(ROOT / f"formose/runs/{name}/rtip.out")
        report = (ROOT / f"formose/runs/{name}/acceptance_report.md").read_text()
        m = re.search(r"min C-C distance: final ([\d.]+)", report)
        rows.append({
            "seed": name,
            "T_mean": d["temp_K"].mean(),
            "T_max": d["temp_K"].max(),
            "min_cc": float(m.group(1)) if m else np.nan,
            "cc": "成键" if name in CC_EVENTS else "未成键",
        })
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(t))
    ax.bar(x - 0.18, t["T_mean"], 0.36, label="平均温度")
    ax.bar(x + 0.18, t["T_max"], 0.36, label="最高温度")
    for i, (cc, ok) in enumerate(zip(t["min_cc"], t["cc"])):
        ax.text(i, 0.03, f"末帧 min C-C {cc:.2f} Å\n{ok}", ha="center", fontsize=8)
    ax.axhline(PAPER_T, ls="--", c="gray", lw=0.8)
    ax.text(2.45, PAPER_T, "1500 K", fontsize=8, color="gray", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(t["seed"])
    ax.set_ylabel("温度 (K)")
    ax.set_title("三种子盒子 MD 统计（10000 步 / 5 ps）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT / "summary_seeds.png", dpi=160)
    plt.close(fig)


def fig_overview() -> None:
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    # 温度对比
    for i, name in enumerate(["seed0", "seed1", "seed2"]):
        d = read_out(ROOT / f"formose/runs/{name}/rtip.out")
        ax[0, 0].plot(d["time_fs"], d["temp_K"], lw=0.8, label=name)
    ax[0, 0].axhline(PAPER_T, ls="--", c="gray", lw=0.8)
    ax[0, 0].set(title="温度时间序列（三种子）", xlabel="时间 (fs)", ylabel="T (K)")
    ax[0, 0].legend(fontsize=8)
    # min C-C
    for i, name in enumerate(["seed0", "seed1", "seed2"]):
        s = min_cc_series(ROOT / f"formose/runs/{name}/rtip.pdb")
        ax[0, 1].plot(s[:, 0], s[:, 1], lw=0.9, label=name)
    ax[0, 1].axhline(CC_BOND_A, ls="--", c="gray", lw=0.8)
    ax[0, 1].set(title="最近 C-C 距离", xlabel="step", ylabel="min C-C (Å)")
    ax[0, 1].legend(fontsize=8)
    # cycles seed0
    rows = np.loadtxt(ROOT / "formose/runs/seed0/rtip_decreasing_steps", skiprows=1)
    ax[0, 2].step(rows[:, 0], np.ones(len(rows)), where="post", lw=1.2)
    ax[0, 2].step(rows[:, 1], np.zeros(len(rows)), where="post", lw=1.2, color="#C44E52")
    ax[0, 2].set(title="seed0 RTIP 循环（蓝=进入减小，红=复位）", xlabel="step")
    # 键事件 seed0
    sys.path.insert(0, str(ROOT / "formose"))
    from analyze_run import bond_events as _be, read_pdb_frames as _rf
    ev = _be(_rf(ROOT / "formose/runs/seed0/rtip.pdb"))
    counts: dict[str, int] = {}
    for kind, bond, _d, _f in ev:
        counts[bond] = counts.get(bond, 0) + 1
    ax[1, 0].bar(list(counts), list(counts.values()))
    ax[1, 0].set(title="seed0 键事件", ylabel="次数")
    # microkinetics 摘要
    c = pd.read_csv(ROOT / "formose/microkinetics/runs/concentrations.csv")
    s = json.loads((ROOT / "formose/microkinetics/runs/summary.json").read_text())
    for col, lab in [("s1", "CH2O"), ("s4", "乙醇醛"), ("s19", "四碳糖"), ("s11", "核糖")]:
        ax[1, 1].plot(c["time_s"], c[col], lw=1, label=lab)
    ax[1, 1].set(title="微动力学（Figure 4）", xlabel="时间 (s)", ylabel="浓度 (M)")
    ax[1, 1].legend(fontsize=8)
    # 汇总数值
    ax[1, 2].axis("off")
    txt = ("验收汇总\n"
           "seed0: C-C 1.357 Å @ step 7570\n"
           "seed2: C-C 1.406 Å @ step 5700\n"
           "seed1: 未成键 (min C-C 1.59 Å)\n"
           "R2 反应对: C-C 1.42 Å\n"
           "R5 反应对: C-C 1.515 Å @ step 700\n"
           f"微动力学: c(2)~{s['formyl_anion_c_M']:.1e} M, "
           f"retroaldol>0 @ {s['retroaldol_net_rate_positive_at_s']:.2f} s")
    ax[1, 2].text(0.02, 0.98, txt, va="top", fontsize=11)
    fig.suptitle("Formose RTIP-MD 复刻进展总览（JACS Au 2026, 6, 922）", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT / "overview.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    for name in RUN_NAMES:
        run = ROOT / f"formose/runs/{name}"
        if (run / "rtip.out").exists():
            fig_scalars(name, run)
            fig_mincc(name, run)
            fig_cycles(name, run)
            fig_events(name, run)
    fig_micro()
    fig_summary()
    fig_overview()
    print("已生成图表:", sorted(p.name for p in PLOT.glob("*.png")))
