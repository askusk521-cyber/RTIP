"""DeePMD 单点能量脚本（教科书验证用）。

对一组 xyz 文件分别调用 DPA-3.2-5M 预训练模型，输出每个结构的
总能量（eV 与 Hartree）。cell=None 表示气相（无周期性）。
单位换算：1 Hartree = 27.211386245988 eV。

用法（n5 登录节点）:
  source /group/software/deepmd-kit-3.1.1/bin/activate
  python deepmd_sp.py model.pt H2.xyz CH2O.xyz ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

EV_TO_HARTREE = 1.0 / 27.211386245988


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
    ap.add_argument("xyzs", nargs="+", type=Path)
    args = ap.parse_args()

    from deepmd.infer import DeepPot

    dp = DeepPot(args.model)
    type_map = list(dp.get_type_map())
    print(f"model: {args.model}")
    print(f"type_map: {type_map}")
    print(f"{'file':20s} {'natom':>5s} {'E_eV':>16s} {'E_Ha':>16s}")

    results: dict[str, float] = {}
    for xyz in args.xyzs:
        symbols, coord = read_xyz(xyz)
        atype = [type_map.index(sym) for sym in symbols]
        coord_flat = coord.reshape(1, -1)
        energy_ev, _, _ = dp.eval(coord_flat, None, atype)
        e_ev = float(np.asarray(energy_ev).reshape(-1)[0])
        e_ha = e_ev * EV_TO_HARTREE
        results[xyz.stem] = e_ha
        print(f"{xyz.name:20s} {len(symbols):5d} {e_ev:16.8f} {e_ha:16.8f}")

    if len(results) >= 4:
        # 常见教科书反应的反应能（单位 kcal/mol, 1 Ha = 627.5095 kcal/mol）
        def kcal(ev: float) -> float:
            return ev * 627.509474063

        if {"CH3OH", "CH2O", "H2"} <= set(results):
            drxn = results["CH3OH"] - results["CH2O"] - results["H2"]
            print(f"\nCH2O + H2 -> CH3OH  dE = {drxn:.6f} Ha = {kcal(drxn):.2f} kcal/mol")
        if {"C2H6", "C2H4", "H2"} <= set(results):
            drxn = results["C2H6"] - results["C2H4"] - results["H2"]
            print(f"C2H4 + H2 -> C2H6  dE = {drxn:.6f} Ha = {kcal(drxn):.2f} kcal/mol")
        if {"CH4", "CH2O"} <= set(results):
            print(f"CH2O + CH4: 分离分子单点见上表（负对照，预期不成键）")


if __name__ == "__main__":
    sys.exit(main())
