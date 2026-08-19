"""教科书验证图表。

Figure 1: DeePMD 单点反应能 vs 实验/ATcT 参考 (kcal/mol)
Figure 2: 三个 MD case 的温度曲线（相位机 growing -> reducing -> off 的表现）

用法 (n5, 在 formose/ 下):
  source /group/software/deepmd-kit-3.1.1/bin/activate
  python validation_textbook/plot_validation.py \
      --runs runs/validation_ch2o_h2 runs/validation_ch2o_ch4 runs/validation_c2h4_h2 \
      --out validation_textbook/figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


REACTIONS = {
    "CH2O + H2 -> CH3OH": {"deepmd": 0.97, "ref": -21.92, "ref_name": "ATcT dH298"},
    "C2H4 + H2 -> C2H6": {"deepmd": -191.01, "ref": -32.6, "ref_name": "dH298"},
    "CH4 + 2O2 -> CO2 + 2H2O": {"deepmd": -34.4, "ref": -191.8, "ref_name": "dH298"},
}

# 每个 case 需要追踪的原子对（索引按初始 xyz 顺序）:
#   ch2o_h2:  O0 C1 H2 H3(醛氢) H4 H5(H2)
#   c2h4_h2:  C0 C1 H2 H3 H4 H5(乙烯) H6 H7(H2)
#   ch2o_ch4: O0 C1 H2 H3(CH2O) C4 H5 H6 H7 H8(CH4)
TRACKED_PAIRS = {
    "validation_ch2o_h2": {
        "H-H (H2)": (4, 5),
        "C-O (醛基)": (0, 1),
        "O-H (新生)": (0, 4),
        "C-H (新生)": (1, 5),
    },
    "validation_c2h4_h2": {
        "H-H (H2)": (6, 7),
        "C-C (C=C)": (0, 1),
        "C-H (新生1)": (0, 6),
        "C-H (新生2)": (1, 7),
    },
    "validation_ch2o_ch4": {
        "C-C (CH2O+CH4)": (1, 4),
        "C-H (CH4)": (4, 5),
    },
}


def read_temperature(path: Path) -> tuple[list[float], list[float], list[str]]:
    times, temps, states = [], [], []
    with path.open() as fh:
        next(fh)
        for line in fh:
            p = line.split()
            if len(p) >= 10 and "nan" not in p[3]:
                times.append(float(p[1]))
                temps.append(float(p[3]))
                states.append(p[9])
    return times, temps, states


def plot_energy(outdir: Path) -> None:
    labels = list(REACTIONS)
    d = [REACTIONS[k]["deepmd"] for k in labels]
    r = [REACTIONS[k]["ref"] for k in labels]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - 0.2 for i in x], d, width=0.4, label="DeePMD DPA-3.2-5M")
    ax.bar([i + 0.2 for i in x], r, width=0.4, label="实验/ATcT dH298")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("反应能 (kcal/mol)")
    ax.set_title("DeePMD 单点反应能 vs 参考值（几何均为实验键长）")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    for i in x:
        ax.text(i - 0.2, d[i], f"{d[i]:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, r[i], f"{r[i]:.1f}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig1_reaction_energy.png", dpi=150)
    plt.close(fig)
    print("saved", outdir / "fig1_reaction_energy.png")


def plot_temperatures(runs: list[Path], outdir: Path) -> None:
    fig, axes = plt.subplots(len(runs), 1, figsize=(8, 3.2 * len(runs)), sharex=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, run in zip(axes, runs):
        times, temps, states = read_temperature(run / "rtip.out")
        if not times:
            ax.set_title(run.name)
            continue
        ax.plot(times, temps, lw=0.8, color="tab:blue")
        ax.set_title(run.name)
        ax.set_ylabel("T (K)")
        # 状态区间着色: 出现 Decreasing 的阶段标黄
        in_dec = [s == "Decreasing" for s in states]
        for t0, t1, flag in zip(times[:-1], times[1:], in_dec[:-1]):
            if flag:
                ax.axvspan(t0, t1, color="gold", alpha=0.25, lw=0)
    axes[-1].set_xlabel("time (fs)")
    fig.tight_layout()
    fig.savefig(outdir / "fig2_temperature_curves.png", dpi=150)
    plt.close(fig)
    print("saved", outdir / "fig2_temperature_curves.png")


def plot_key_distances(runs: list[Path], outdir: Path) -> None:
    """每个 case 的关键原子对距离随时间演变（来自 rtip.pdb）。"""

    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, run in zip(axes, runs):
        pairs = {}
        for key, candidate in TRACKED_PAIRS.items():
            if run.name.startswith(key):
                pairs = candidate
                break
        text = (run / "rtip.pdb").read_text()
        frames = text.split("END")
        times, series = [], {k: [] for k in pairs}
        for fr in frames:
            if "ATOM" not in fr:
                continue
            coords = {}
            atom_idx = 0
            for line in fr.splitlines():
                if line.startswith("ATOM"):
                    nums = re.findall(r"-?\d+\.\d+", line)
                    if len(nums) < 4 or "nan" in line:
                        break
                    coords[atom_idx] = (float(nums[0]), float(nums[1]), float(nums[2]))
                    atom_idx += 1
            if len(coords) < 6:
                continue
            step = int(re.search(r"Step =\s*(\d+)", fr).group(1))
            times.append(step * 0.5)
            for k, (i, j) in pairs.items():
                if i in coords and j in coords:
                    dx = coords[i][0] - coords[j][0]
                    dy = coords[i][1] - coords[j][1]
                    dz = coords[i][2] - coords[j][2]
                    series[k].append((dx * dx + dy * dy + dz * dz) ** 0.5)
        for k, vals in series.items():
            if vals:
                ax.plot(times, vals, lw=1.2, label=k)
        ax.set_title(run.name)
        ax.set_ylabel("键长 (A)")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time (fs)")
    fig.tight_layout()
    fig.savefig(outdir / "fig3_key_distances.png", dpi=150)
    plt.close(fig)
    print("saved", outdir / "fig3_key_distances.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("validation_textbook/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_energy(args.out)
    plot_temperatures(args.runs, args.out)
    plot_key_distances(args.runs, args.out)


if __name__ == "__main__":
    main()
