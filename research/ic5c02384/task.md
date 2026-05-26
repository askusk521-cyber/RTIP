# ic5c02384 任务摘要

英文版见 [task_EN.md](task_EN.md)。AI agent 应阅读英文版。

## 文献主题

文献题目为 **Predicting Dinitrogen Activation and Coupling with Carbon Dioxide and Other Small Molecules by Methyleneborane: A Combined DFT and Machine Learning Study**。核心目标是用 DFT 计算和机器学习分析预测：亚甲基硼烷活化后的 `methyleneborane-N2` 复合物能否继续与 `CO2` 以及其他小分子发生偶联，形成新的氮-碳键或相关极性键活化产物。

这是一项理论预测研究，不是实验合成报道。作者关注主族硼物种在 N2 偶联化学中的潜力，尝试把常见于过渡金属体系的 N2/CO2 偶联拓展到金属自由或主族体系。

## 主要化学反应

核心反应可以概括为：

```text
methyleneborane-N2 complex + CO2 -> N2-CO2 coupling product
```

在反应中，已被亚甲基硼烷活化的 N2 片段与 CO2 偶联。N2 端的氮原子向 CO2 的碳中心形成 N-C 键，同时 CO2 的氧原子向硼中心给电子，形成 B-O 相互作用。作者认为这个 N2/CO2 偶联过程是协同步骤。

文献比较了硼取代基 `R = CN, H, Me, Ph, PMe2, SiMe3, tBu` 对 `1-R + CO2` 反应的影响。论文表格中的 Gibbs 自由能显示，`1-tBu` 最有利，`1-tBu + CO2 -> 1-tBu-CO2.P` 的反应自由能约为 `-10.1 kcal/mol`。其他较有利的体系包括 `1-Me` 和 `1-PMe2`。

除 CO2 外，作者还考察了 `1-tBu` 与其他小分子的偶联/活化：

```text
1-tBu + CS2 -> 1-tBu-CS2.P
1-tBu + H2CO -> 1-tBu-H2CO.P
1-tBu + MeCN -> 1-tBu-MeCN.P
1-tBu + MeCH=NMe -> 1-tBu-MeCH=NMe.P
```

按论文 Gibbs 势垒趋势，小分子活化难度大致为：

```text
H2CO < CS2 < CO2 < MeCN < MeCH=NMe
```

注意：本目录中的 SI 坐标和清单使用 `CS2` 名称；请以 `reaction_manifest.csv` 和结构文件名为准。

## 本目录数据

本目录由 ACS Supporting Information `ic5c02384_si_002.xyz` 转换而来，已经整理成 RTIP/JAX 可读取的数据集。

- `structures/`：104 个单结构 XYZ 文件。
- `manifest.csv`：结构编号、原子数、名称、SI 标题中的电子能和文件路径。
- `reaction_manifest.csv`：按反应整理的反应物、产物、过渡态以及基于 SI 电子能计算的相对能。
- `reactions/*/1.xyz` 与 `reactions/*/2.xyz`：每条反应的两个反应物，可作为 `rtip-jax synthesize --inputs` 输入。
- `reactions/*/product.xyz`：SI 给出的产物参考结构。
- `reactions/*/ts.xyz`：SI 给出的过渡态参考结构。
- `elements_present.txt`：出现元素为 `B, C, H, N, O, P, S, Si`。

## 建议计算任务

优先从 `1-tBu__CO2` 开始，因为它是论文中最有代表性的低能 CO2 偶联体系，且目录中同时有反应物、产物和过渡态参考结构。

建议任务顺序：

1. 用 `rtip-jax synthesize` 生成 `1-tBu + CO2` 的初始分离构型 `IS.xyz`。
2. 将 `IS.xyz`、`product.xyz` 和 `ts.xyz` 作为参考，测试 RTIP/JAX 或后续 pathway 工具能否构造合理反应路径。
3. 对比 `reaction_manifest.csv` 中的电子能相对值，筛选更容易收敛或更有代表性的反应。
4. 扩展到 `1-Me__CO2`、`1-PMe2__CO2`、`1-Ph__CO2`、`1-SiMe3__CO2`，比较不同取代基对路径和能量的影响。
5. 最后比较 `1-tBu__H2CO`、`1-tBu__CS2`、`1-tBu__MeCN`、`1-tBu__MeCH=NMe`，复现小分子活化难易顺序。

## 推荐优先级

高优先级：

- `reactions/1-tBu__CO2`
- `reactions/1-Me__CO2`
- `reactions/1-PMe2__CO2`
- `reactions/1-tBu__H2CO`

中优先级：

- `reactions/1-Ph__CO2`
- `reactions/1-SiMe3__CO2`
- `reactions/1-tBu__CS2`
- `reactions/1-tBu__MeCN`
- `reactions/1-tBu__MeCH=NMe`

低优先级：

- 只有产物结构、没有 TS 参考的 `2-R` 到 `6-R` 系列 CO2 产物，可用于热力学或结构比较，但不适合作为第一批路径验证任务。

## 注意事项

- `reaction_manifest.csv` 中的能量差来自 SI 标题里的电子能 `E = ... a.u.`，不是论文正文表格的 Gibbs 自由能 `Delta G`，所以数值和论文表格不会逐项相同。
- RTIP/JAX 不是 DFT 程序；若要复现实势能面，需要可覆盖 `B/C/H/N/O/P/S/Si` 的 DeePMD 模型，或接入其他可靠 PES provider。
- 使用 DeePMD 时，`--type-map` 必须和模型训练时的元素顺序一致。
- `product.xyz` 和 `ts.xyz` 是文献 SI 参考结构，不是 JAX 自动生成的结构。

## 一个最小起步命令

```bash
cd JAX
python -m rtip_jax.cli synthesize \
  --inputs examples/ic5c02384/reactions/1-tBu__CO2/1.xyz examples/ic5c02384/reactions/1-tBu__CO2/2.xyz \
  --output examples/ic5c02384/reactions/1-tBu__CO2/IS.xyz \
  --dist 5.0 \
  --seed 0
```
