# IC5C02384 RTIP/JAX 输入数据

来源：ACS Supporting Information `ic5c02384_si_002.xyz`。

这个目录由 `prepare_ic5c02384.py` 生成，用来把 ACS multi-XYZ 文件转换成 RTIP/JAX 当前能直接读取的单结构 XYZ 文件。

## 内容

- `structures/`: 104 个单结构 XYZ 文件。
- `manifest.csv`: 每个结构的编号、原子数、名称、电子能和文件路径。
- `reaction_manifest.csv`: 按反应整理的反应物、产物、过渡态和基于 SI 电子能的反应能/相对能。
- `elements_present.txt`: 这些结构实际出现的元素。
- `reactions/*/1.xyz` 与 `reactions/*/2.xyz`: 可直接用于 `rtip-jax synthesize --inputs` 的反应物文件。
- `reactions/*/product.xyz` 和 `reactions/*/ts.xyz`: 文献 SI 给出的产物/过渡态参考结构。

## 最小使用例

```bash
cd JAX
python -m rtip_jax.cli synthesize \
  --inputs examples/ic5c02384/reactions/1-tBu__CO2/1.xyz examples/ic5c02384/reactions/1-tBu__CO2/2.xyz \
  --output examples/ic5c02384/reactions/1-tBu__CO2/IS.xyz \
  --dist 5.0 \
  --seed 0
```

生成的 `IS.xyz` 是 RTIP/JAX pathway 或 DeePMD pathway 的初始分离构型。`product.xyz` 和 `ts.xyz` 是文献参考结构，不是 JAX 自动生成的结果。

## 验证时要注意

- `reaction_manifest.csv` 里的能量差来自 SI 标题中的电子能 `E = ... a.u.`，不是论文表格里的 Gibbs 自由能 `ΔG`，所以数值不会一一等同。
- RTIP/JAX 本身不是 DFT 程序；如果要复现实势能面，需要能覆盖 B/C/H/N/O/P/S/Si 的 DeePMD 模型，或接入其他真实 PES provider。
- `--type-map` 的顺序必须和所用 DeePMD 模型训练时的元素顺序一致，不能只按本文件的字母顺序随便填。

注意：SI 文件中的 `CS2` 是按坐标文件标题保留的名称。
