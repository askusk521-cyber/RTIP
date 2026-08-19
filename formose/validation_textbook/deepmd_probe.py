"""探测 DPA-3.2-5M 调用方式对能量的影响。

检查项:
  1) has_spin / spin 相关属性
  2) cell=None 与 cell=大盒子(20 A) 的能量差异
  3) 输出属性 output_def
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


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
    from deepmd.infer import DeepPot

    model, wd = sys.argv[1], Path(sys.argv[2])
    dp = DeepPot(model)
    type_map = list(dp.get_type_map())
    print("has_spin:", getattr(dp, "has_spin", "<missing>"))
    print("get_ntypes_spin:", getattr(dp, "get_ntypes_spin", lambda: "<missing>")())
    print("output_def:", dp.output_def)
    print("rcut:", dp.get_rcut())
    print("sel_type:", dp.get_sel_type())

    for name in ("CH4.xyz", "H2.xyz", "O2.xyz", "C2H6.xyz"):
        symbols, coord = read_xyz(wd / name)
        atype = [type_map.index(s) for s in symbols]
        flat = coord.reshape(1, -1)
        e_none, _, _ = dp.eval(flat, None, atype)
        cell = np.eye(3, dtype=np.float64) * 20.0
        e_box, _, _ = dp.eval(flat, cell.reshape(1, 9), atype)
        print(
            f"{name}: cell=None {float(np.asarray(e_none).reshape(-1)[0]):12.6f} eV | "
            f"cell=20A {float(np.asarray(e_box).reshape(-1)[0]):12.6f} eV | "
            f"diff {float(np.asarray(e_box).reshape(-1)[0]) - float(np.asarray(e_none).reshape(-1)[0]):10.6f} eV"
        )


if __name__ == "__main__":
    main()
