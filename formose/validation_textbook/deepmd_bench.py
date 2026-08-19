"""DeePMD DPA-3.2-5M 基准测试（教科书验证用）。

项目 A: H2 势能曲线（检验模型对 H-H 键的描述）
项目 B: 教科书燃烧反应 CH4 + 2 O2 -> CO2 + 2 H2O
        （实验 dH298 = -802.5 kJ/mol = -191.8 kcal/mol，气相水）
项目 C: 报告各反应能的模型-实验偏差。

几何均取实验键长（见各文件注释）。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

EV_TO_HARTREE = 1.0 / 27.211386245988
KCAL_PER_HARTREE = 627.509474063


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    natom = int(lines[0])
    symbols: list[str] = []
    coord: list[list[float]] = []
    for line in lines[2 : 2 + natom]:
        parts = line.split()
        symbols.append(parts[0])
        coord.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(coord, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=str)
    ap.add_argument("--workdir", type=Path, default=Path("."))
    args = ap.parse_args()

    from deepmd.infer import DeepPot

    dp = DeepPot(args.model)
    type_map = list(dp.get_type_map())
    wd = args.workdir

    def eval_xyz(name: str) -> float:
        symbols, coord = read_xyz(wd / name)
        atype = [type_map.index(sym) for sym in symbols]
        e_ev, _, _ = dp.eval(coord.reshape(1, -1), None, atype)
        return float(np.asarray(e_ev).reshape(-1)[0])

    # A. H2 势能曲线
    print("== A. H2 potential curve ==")
    r_list = [0.50, 0.60, 0.70, 0.7414, 0.80, 0.90, 1.00, 1.20, 1.50, 2.00, 3.00, 5.00]
    e_h2 = {}
    for r in r_list:
        coord = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]], dtype=np.float64)
        e_ev, _, _ = dp.eval(coord.reshape(1, -1), None, [0, 0])
        e_h2[r] = float(np.asarray(e_ev).reshape(-1)[0])
        print(f"  r(H-H) = {r:5.2f} A   E = {e_h2[r]:10.5f} eV")
    emin = min(e_h2.values())
    rmin = min(e_h2, key=e_h2.get)
    print(f"  model minimum near r = {rmin:.2f} A (experiment 0.741 A)")
    print(f"  depth E(r=5) - E(min) = {e_h2[5.0] - emin:8.3f} eV (experiment D0 ~ 4.48 eV)")

    # B. 燃烧反应
    print("== B. CH4 + 2 O2 -> CO2 + 2 H2O ==")
    e_ch4 = eval_xyz("CH4.xyz")
    e_o2 = eval_xyz("O2.xyz")
    e_co2 = eval_xyz("CO2.xyz")
    e_h2o = eval_xyz("H2O.xyz")
    for name, e in [("CH4", e_ch4), ("O2", e_o2), ("CO2", e_co2), ("H2O", e_h2o)]:
        print(f"  {name}: {e:12.6f} eV = {e * EV_TO_HARTREE:12.6f} Ha")
    dE_ev = e_co2 + 2.0 * e_h2o - e_ch4 - 2.0 * e_o2
    dE_kcal = dE_ev * 23.060542
    print(f"  dE = {dE_ev:9.4f} eV = {dE_kcal:9.1f} kcal/mol (experiment dH298 = -191.8 kcal/mol)")

    # C. 已测反应汇总
    print("== C. reaction-energy summary (from deepmd_sp.py) ==")
    e_h2_074 = e_h2[0.7414]
    e_h2o_r = eval_xyz("H2O.xyz")  # already computed, kept for clarity
    print("  (详见 deepmd_sp.py 输出与上方结果；C2H4+H2 与 CH2O+H2 见该脚本)")


if __name__ == "__main__":
    main()
