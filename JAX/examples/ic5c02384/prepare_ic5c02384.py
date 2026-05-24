"""Prepare ACS SI XYZ structures for RTIP/JAX examples.

The ACS supporting-information file ``ic5c02384_si_002.xyz`` is a multi-XYZ
file. RTIP/JAX currently reads one XYZ block per file, so this helper splits the
SI file and writes reaction-oriented inputs.
"""

from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "ic5c02384_si_002.xyz"
OUT = Path(__file__).resolve().parent
STRUCTURES = OUT / "structures"
REACTIONS = OUT / "reactions"


@dataclass(frozen=True)
class XyzBlock:
    index: int
    natom: int
    title: str
    name: str
    energy_hartree: float | None
    lines: tuple[str, ...]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=+-]+", "_", name).strip("_")


def parse_title(title: str) -> tuple[str, float | None]:
    match = re.match(r"(.+?)\s+E\s*=\s*([-+0-9.]+)\s+a\.u\.", title)
    if match:
        return match.group(1).strip(), float(match.group(2))
    return title.strip(), None


def read_multi_xyz(path: Path) -> list[XyzBlock]:
    lines = path.read_text().splitlines()
    blocks: list[XyzBlock] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        natom = int(lines[i].strip())
        title = lines[i + 1].strip()
        atom_lines = tuple(lines[i + 2 : i + 2 + natom])
        if len(atom_lines) != natom:
            raise ValueError(f"incomplete XYZ block at line {i + 1}")
        name, energy = parse_title(title)
        # The SI titles contain two harmless duplicate-name issues. Keep the
        # original title in each XYZ file, but give the generated filenames and
        # manifests unique, reaction-useful names.
        if name == "6-H-CO2.P" and natom == 29:
            name = "6-Me-CO2.P"
        if name == "1-tBu-H2CO.TS" and any(block.name == name for block in blocks):
            name = "1-tBu-H2CO.TS.duplicate"
        blocks.append(
            XyzBlock(
                index=len(blocks) + 1,
                natom=natom,
                title=title,
                name=name,
                energy_hartree=energy,
                lines=(str(natom), title, *atom_lines),
            )
        )
        i += natom + 2
    return blocks


def write_xyz(path: Path, block: XyzBlock) -> None:
    path.write_text("\n".join(block.lines) + "\n")


def copy_named(blocks_by_name: dict[str, XyzBlock], name: str, dest: Path) -> None:
    source = STRUCTURES / filename_for(blocks_by_name[name])
    shutil.copyfile(source, dest)


def filename_for(block: XyzBlock) -> str:
    return f"{block.index:03d}_{safe_name(block.name)}.xyz"


def reaction_delta(product: XyzBlock, reactants: list[XyzBlock]) -> float | None:
    if product.energy_hartree is None or any(r.energy_hartree is None for r in reactants):
        return None
    return product.energy_hartree - sum(r.energy_hartree for r in reactants if r.energy_hartree is not None)


def write_manifest(blocks: list[XyzBlock]) -> None:
    with (OUT / "manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "natom", "name", "energy_hartree", "file"])
        for block in blocks:
            writer.writerow([block.index, block.natom, block.name, block.energy_hartree, f"structures/{filename_for(block)}"])


def write_reaction_tables(blocks_by_name: dict[str, XyzBlock]) -> None:
    rows: list[dict[str, str | float | None]] = []

    # CO2 thermodynamic products for all methyleneborane-N2 families present in
    # the SI. Families 1-6 correspond to the article's substituent/NHC scans.
    for product_name in sorted(name for name in blocks_by_name if name.endswith("-CO2.P")):
        base = product_name.removesuffix("-CO2.P")
        if base not in blocks_by_name:
            continue
        product = blocks_by_name[product_name]
        reactants = [blocks_by_name[base], blocks_by_name["CO2"]]
        delta = reaction_delta(product, reactants)
        rows.append(
            {
                "reaction": f"{base} + CO2 -> {product_name}",
                "class": "CO2_product",
                "reactant_1": base,
                "reactant_2": "CO2",
                "product": product_name,
                "transition_state": "",
                "delta_e_hartree": delta,
                "delta_e_kcal_mol": None if delta is None else delta * 627.509474,
            }
        )

    # Mechanistic TS examples explicitly present in the SI.
    for ts_name in sorted(name for name in blocks_by_name if name.endswith(".TS")):
        product_name = ts_name.removesuffix(".TS") + ".P"
        base, small_molecule = infer_reactants_from_product(product_name)
        if product_name not in blocks_by_name or base not in blocks_by_name or small_molecule not in blocks_by_name:
            continue
        ts = blocks_by_name[ts_name]
        reactants = [blocks_by_name[base], blocks_by_name[small_molecule]]
        ts_delta = reaction_delta(ts, reactants)
        rows.append(
            {
                "reaction": f"{base} + {small_molecule} -> {product_name}",
                "class": "transition_state",
                "reactant_1": base,
                "reactant_2": small_molecule,
                "product": product_name,
                "transition_state": ts_name,
                "delta_e_hartree": ts_delta,
                "delta_e_kcal_mol": None if ts_delta is None else ts_delta * 627.509474,
            }
        )

    with (OUT / "reaction_manifest.csv").open("w", newline="") as handle:
        fieldnames = [
            "reaction",
            "class",
            "reactant_1",
            "reactant_2",
            "product",
            "transition_state",
            "delta_e_hartree",
            "delta_e_kcal_mol",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_reactants_from_product(product_name: str) -> tuple[str, str]:
    for small in ("MeCH=NMe", "MeCN", "H2CO", "CS2", "CO2"):
        marker = f"-{small}.P"
        if product_name.endswith(marker):
            return product_name[: -len(marker)], small
    raise ValueError(f"cannot infer reactants for {product_name}")


def write_reaction_inputs(blocks_by_name: dict[str, XyzBlock]) -> None:
    selected_pairs = [
        ("1-CN", "CO2"),
        ("1-H", "CO2"),
        ("1-Me", "CO2"),
        ("1-Ph", "CO2"),
        ("1-PMe2", "CO2"),
        ("1-SiMe3", "CO2"),
        ("1-tBu", "CO2"),
        ("1-tBu", "CS2"),
        ("1-tBu", "H2CO"),
        ("1-tBu", "MeCN"),
        ("1-tBu", "MeCH=NMe"),
    ]
    for left, right in selected_pairs:
        if left not in blocks_by_name or right not in blocks_by_name:
            continue
        product_name = f"{left}-{right}.P"
        ts_name = f"{left}-{right}.TS"
        dirname = f"{safe_name(left)}__{safe_name(right)}"
        target = REACTIONS / dirname
        target.mkdir(parents=True, exist_ok=True)
        copy_named(blocks_by_name, left, target / "1.xyz")
        copy_named(blocks_by_name, right, target / "2.xyz")
        if product_name in blocks_by_name:
            copy_named(blocks_by_name, product_name, target / "product.xyz")
        if ts_name in blocks_by_name:
            copy_named(blocks_by_name, ts_name, target / "ts.xyz")
        (target / "README_ZH.md").write_text(
            "\n".join(
                [
                    f"# {left} + {right}",
                    "",
                    "这个目录用于 RTIP/JAX 的 `synthesize --inputs` 入口。",
                    "",
                    "- `1.xyz`: 第一个反应物",
                    "- `2.xyz`: 第二个反应物",
                    "- `product.xyz`: SI 中的优化产物结构，如果存在",
                    "- `ts.xyz`: SI 中的过渡态结构，如果存在",
                    "",
                    "示例：",
                    "",
                    "```bash",
                    "cd JAX",
                    f"python -m rtip_jax.cli synthesize --inputs examples/ic5c02384/reactions/{dirname}/1.xyz examples/ic5c02384/reactions/{dirname}/2.xyz --output examples/ic5c02384/reactions/{dirname}/IS.xyz --dist 5.0 --seed 0",
                    "```",
                    "",
                    "后续如果有可用 DeePMD 模型，可以把生成的 `IS.xyz` 作为 `deepmd-pathway` 的 `--input`。",
                    "",
                ]
            )
        )


def write_elements(blocks: list[XyzBlock]) -> None:
    elements = sorted({line.split()[0] for block in blocks for line in block.lines[2:]})
    (OUT / "elements_present.txt").write_text("\n".join(elements) + "\n")


def write_readme(blocks: list[XyzBlock]) -> None:
    (OUT / "README_ZH.md").write_text(
        "\n".join(
            [
                "# IC5C02384 RTIP/JAX 输入数据",
                "",
                "来源：ACS Supporting Information `ic5c02384_si_002.xyz`。",
                "",
                "这个目录由 `prepare_ic5c02384.py` 生成，用来把 ACS multi-XYZ 文件转换成 RTIP/JAX 当前能直接读取的单结构 XYZ 文件。",
                "",
                "## 内容",
                "",
                f"- `structures/`: {len(blocks)} 个单结构 XYZ 文件。",
                "- `manifest.csv`: 每个结构的编号、原子数、名称、电子能和文件路径。",
                "- `reaction_manifest.csv`: 按反应整理的反应物、产物、过渡态和基于 SI 电子能的反应能/相对能。",
                "- `elements_present.txt`: 这些结构实际出现的元素。",
                "- `reactions/*/1.xyz` 与 `reactions/*/2.xyz`: 可直接用于 `rtip-jax synthesize --inputs` 的反应物文件。",
                "- `reactions/*/product.xyz` 和 `reactions/*/ts.xyz`: 文献 SI 给出的产物/过渡态参考结构。",
                "",
                "## 最小使用例",
                "",
                "```bash",
                "cd JAX",
                "python -m rtip_jax.cli synthesize \\",
                "  --inputs examples/ic5c02384/reactions/1-tBu__CO2/1.xyz examples/ic5c02384/reactions/1-tBu__CO2/2.xyz \\",
                "  --output examples/ic5c02384/reactions/1-tBu__CO2/IS.xyz \\",
                "  --dist 5.0 \\",
                "  --seed 0",
                "```",
                "",
                "生成的 `IS.xyz` 是 RTIP/JAX pathway 或 DeePMD pathway 的初始分离构型。`product.xyz` 和 `ts.xyz` 是文献参考结构，不是 JAX 自动生成的结果。",
                "",
                "## 验证时要注意",
                "",
                "- `reaction_manifest.csv` 里的能量差来自 SI 标题中的电子能 `E = ... a.u.`，不是论文表格里的 Gibbs 自由能 `ΔG`，所以数值不会一一等同。",
                "- RTIP/JAX 本身不是 DFT 程序；如果要复现实势能面，需要能覆盖 B/C/H/N/O/P/S/Si 的 DeePMD 模型，或接入其他真实 PES provider。",
                "- `--type-map` 的顺序必须和所用 DeePMD 模型训练时的元素顺序一致，不能只按本文件的字母顺序随便填。",
                "",
                "注意：SI 文件中的 `CS2` 是按坐标文件标题保留的名称。",
                "",
            ]
        )
    )


def main() -> None:
    blocks = read_multi_xyz(SOURCE)
    blocks_by_name = {block.name: block for block in blocks}
    if len(blocks_by_name) != len(blocks):
        duplicates = sorted({block.name for block in blocks if sum(other.name == block.name for other in blocks) > 1})
        raise ValueError(f"duplicate structure names: {duplicates}")

    STRUCTURES.mkdir(parents=True, exist_ok=True)
    REACTIONS.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        write_xyz(STRUCTURES / filename_for(block), block)

    write_manifest(blocks)
    write_reaction_tables(blocks_by_name)
    write_reaction_inputs(blocks_by_name)
    write_elements(blocks)
    write_readme(blocks)

    print(f"read {len(blocks)} structures from {SOURCE}")
    print(f"wrote {STRUCTURES}")
    print(f"wrote {REACTIONS}")


if __name__ == "__main__":
    main()
