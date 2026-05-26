# ic5c02384 文献反应与验证参考

英文版见 [reaction_reference_EN.md](reaction_reference_EN.md)。AI agent 应阅读英文版。

本文档把论文 `Predicting Dinitrogen Activation and Coupling with Carbon Dioxide and Other Small Molecules by Methyleneborane: A Combined DFT and Machine Learning Study` 中涉及的主要反应、计算边界条件、能量口径和本仓库可直接验证的数据整理成 Markdown，便于继续做类似 `history/daily/20260524.md` 的 RTIP/JAX + DeePMD 验证。

## 来源

- 主文 PDF：`predicting-dinitrogen-activation-and-coupling-with-carbon-dioxide-and-other-small-molecules-by-methyleneborane-a.pdf`
- SI 坐标：`ic5c02384_si_002.xyz`
- 已整理数据：`research/ic5c02384/`
- 结构清单：`research/ic5c02384/manifest.csv`
- 反应清单：`research/ic5c02384/reaction_manifest.csv`

注意：主文摘要和正文个别位置写到 `sulfur dioxide`，但 Figure 1/7、Table 4、SI 坐标和本地数据文件均使用 `CS2`。后续验证以 SI 文件名和 `reaction_manifest.csv` 的 `CS2` 为准。

## 文献计算边界条件

- 几何优化：气相，`omegaB97X-D/cc-pVDZ`，所有原子同一水平。
- 频率计算：同一水平；minimum 为无虚频，TS 为一个虚频。
- IRC：对 TS 做 IRC，确认 TS 连接两个 minimum。
- 单点能：`omegaB97X-D/cc-pVTZ`。
- 正文 Gibbs 自由能：`cc-pVTZ` 单点能加上 `cc-pVDZ` Gibbs 校正。
- 温度：`298 K`。
- 约束：无几何约束。
- 网格：Gaussian 16，ultrafine grid。
- 其他分析：Multiwfn 做 FBO，NBO + PIO 程序做主相互作用轨道分析，EDA 使用 distortion/interaction 或 activation strain 口径。

## 反应总览

核心反应是已经被 methyleneborane 活化的 `N2` 复合物继续与小分子偶联：

```text
methyleneborane-N2 complex + X=C-R -> coupling product
```

产物共同特征：

- `N2` 末端氮与小分子碳形成新的 `N-C` 键。
- 小分子中的 `X` 原子向硼中心给电子，形成 `B-X` 相互作用或键。
- `CO2` 体系中对应的关键成键指标是 `B-O` 和 `N-C(CO2)`。
- `CS2`、`H2CO`、`MeCN`、`MeCH=NMe` 可类比监控 `B-X` 与 `N-C`，其中 `X = S/O/N`。

## 正文 Table 1：`1-R + CO2` 热力学产物

这些是正文 Gibbs 自由能和产物关键键长，不是 SI 标题电子能。

| 反应 | B-N / A | N-N / A | B-O / A | C-O / A | C-N2 / A | ΔG / kcal mol^-1 |
|------|--------:|--------:|--------:|--------:|---------:|-----------------:|
| `1-CN + CO2 -> 1-CN-CO2.P` | 1.564 | 1.238 | 1.460 | 1.341 | 1.494 | 2.0 |
| `1-H + CO2 -> 1-H-CO2.P` | 1.552 | 1.242 | 1.473 | 1.335 | 1.491 | 0.3 |
| `1-Me + CO2 -> 1-Me-CO2.P` | 1.557 | 1.241 | 1.486 | 1.332 | 1.492 | -6.5 |
| `1-Ph + CO2 -> 1-Ph-CO2.P` | 1.554 | 1.240 | 1.482 | 1.333 | 1.493 | -3.1 |
| `1-PMe2 + CO2 -> 1-PMe2-CO2.P` | 1.540 | 1.245 | 1.475 | 1.338 | 1.485 | -6.3 |
| `1-SiMe3 + CO2 -> 1-SiMe3-CO2.P` | 1.527 | 1.249 | 1.482 | 1.339 | 1.478 | -1.3 |
| `1-tBu + CO2 -> 1-tBu-CO2.P` | 1.553 | 1.242 | 1.489 | 1.331 | 1.491 | -10.1 |

结论：正文 Gibbs 自由能显示 `1-tBu` 最有利，`1-Me` 与 `1-PMe2` 也明显放热；`1-CN` 和 `1-H` 近似热中性或略吸热。

## 正文 Figure 6：`1-R + CO2` Gibbs 剖面

这些值相对于分离反应物 `1-R + CO2`，单位为 `kcal mol^-1`。`INT` 是预反应复合物，`TS` 是偶联过渡态，`P` 是产物。

| 反应 | INT ΔG | TS ΔG | P ΔG | 验证优先级 |
|------|-------:|------:|-----:|------------|
| `1-tBu + CO2` | 3.9 | 25.1 | -10.1 | 高，代表性最低势垒/最放热体系 |
| `1-Me + CO2` | 4.1 | 26.9 | -6.5 | 高，可验证取代基趋势 |
| `1-PMe2 + CO2` | -0.2 | 28.9 | -6.3 | 高，可验证取代基趋势 |
| `1-Ph + CO2` | 4.4 | 29.6 | -3.1 | 中 |
| `1-SiMe3 + CO2` | 4.2 | 32.9 | -1.3 | 中，势垒最高 |

正文 EDA 电子能口径下的 `1-R-CO2.TS` 分解如下；这些不是 Gibbs 剖面值。

| TS | ΔEdeform(1-R) | ΔEdeform(CO2) | ΔEb | ΔE |
|----|--------------:|--------------:|----:|---:|
| `1-Me-CO2.TS` | 21.4 | 33.4 | -38.5 | 16.2 |
| `1-Ph-CO2.TS` | 22.3 | 33.9 | -37.6 | 18.6 |
| `1-PMe2-CO2.TS` | 21.0 | 35.1 | -38.5 | 17.6 |
| `1-SiMe3-CO2.TS` | 29.8 | 39.4 | -47.5 | 21.8 |
| `1-tBu-CO2.TS` | 24.0 | 36.8 | -47.7 | 13.1 |

## 正文 Figure 3：NHC 骨架对 `n-tBu + CO2` 的影响

主文只在图中给出 `n = 2..6` 且 `R = tBu` 的 Gibbs 反应能：

| 反应 | ΔG / kcal mol^-1 | 备注 |
|------|-----------------:|------|
| `2-tBu + CO2 -> 2-tBu-CO2.P` | -2.7 | CAAC 型，较不利 |
| `3-tBu + CO2 -> 3-tBu-CO2.P` | -0.7 | CAAC 型，较不利 |
| `4-tBu + CO2 -> 4-tBu-CO2.P` | -6.7 | CDAC 型，较有利 |
| `5-tBu + CO2 -> 5-tBu-CO2.P` | 2.8 | CAAC 型，吸热 |
| `6-tBu + CO2 -> 6-tBu-CO2.P` | 1.5 | CAAC 型，吸热 |

正文结论：CDAC 通常比 CAAC 更有利；CAAC 同时是 sigma donor 和 pi acceptor，会与 `CO2` 的供受电子角色竞争。

## 正文 Figure 7：`1-tBu` 与其他小分子

Gibbs 剖面值相对于分离反应物，单位为 `kcal mol^-1`。

| 反应 | INT ΔG | TS ΔG | P ΔG | 关键验证指标 |
|------|-------:|------:|-----:|--------------|
| `1-tBu + H2CO -> 1-tBu-H2CO.P` | 5.2 | 11.9 | -32.7 | `B-O`、`N-C(H2CO)` |
| `1-tBu + CS2 -> 1-tBu-CS2.P` | 5.0 | 19.2 | -8.1 | `B-S`、`N-C(CS2)` |
| `1-tBu + CO2 -> 1-tBu-CO2.P` | 3.9 | 25.1 | -10.1 | `B-O`、`N-C(CO2)` |
| `1-tBu + MeCN -> 1-tBu-MeCN.P` | 0.7 | 25.7 | -15.7 | `B-N`、`N-C(MeCN)` |
| `1-tBu + MeCH=NMe -> 1-tBu-MeCH=NMe.P` | 3.4 | 27.5 | -14.0 | `B-N`、`N-C(imine)` |

势垒顺序：

```text
H2CO < CS2 < CO2 < MeCN < MeCH=NMe
```

对应 EDA 电子能分解：

| TS | ΔEdeform(1-tBu) | ΔEdeform(small molecule) | ΔEb | ΔE |
|----|----------------:|-------------------------:|----:|---:|
| `1-tBu-CO2.TS` | 24.0 | 36.8 | -47.7 | 13.1 |
| `1-tBu-CS2.TS` | 9.4 | 15.5 | -16.8 | 8.1 |
| `1-tBu-H2CO.TS` | 13.6 | 4.3 | -19.6 | -1.8 |
| `1-tBu-MeCN.TS` | 19.7 | 9.4 | -15.7 | 13.3 |
| `1-tBu-MeCH=NMe.TS` | 22.0 | 9.0 | -18.8 | 12.1 |

## 机理与机器学习要点

- PIO 分析显示，`1-tBu-CO2.P` 的主要相互作用来自 `N2` 端氮的 p 轨道向 `CO2` 的 pi* 轨道给电子，PBI 为 `0.97`，贡献约 `49.4%`。
- 第二主相互作用是 `CO2` 氧孤对电子向硼空 p 轨道 sigma donation，PBI 为 `0.65`，贡献约 `33.1%`。
- 第三相互作用是 `N2` pi 轨道向 `C-O` pi* 轨道 backdonation，PBI 为 `0.10`，贡献约 `5.3%`。
- 机器学习描述符包括 `HOMO-LUMO gap`、`NPA(B)`、`NPA(N1)`、`NPA(N2)` 和 `Delta FBO`。
- 符号回归模型的 RMSE 为 `1.9`，低于线性回归的 `2.2`；作者据此认为存在非线性关系。
- 趋势解释：增大 `HOMO-LUMO gap`、增大硼原子电荷或增大 `Delta FBO` 会降低反应能；增大氮原子电荷会升高反应能。
- `1-tBu` 的 CO2 偶联势垒最低，正文归因于 TS 中 deformed reactants 的 binding 最强；作者进一步把它联系到 fragment HOMO 能量较高。
- 小分子活化中，`H2CO` 与 `MeCH=NMe` 的势垒差异主要来自反应物 deformation energy，而不是 binding energy。

## 本地 SI 电子能反应清单

`reaction_manifest.csv` 中的数值来自 SI XYZ 标题里的电子能 `E = ... a.u.`，不是正文 Gibbs 自由能。它适合做 DeePMD 单点能趋势验证，但不要和正文 `ΔG` 直接混用。

### 选出的 RTIP/JAX 反应目录

| 目录 | 可用参考 | SI product ΔE / kcal mol^-1 | SI TS ΔE / kcal mol^-1 | 建议用途 |
|------|----------|----------------------------:|-----------------------:|----------|
| `1-tBu__CO2` | product + TS | -31.05 | 10.11 | 第一优先级；已在 `20260524` 验证 |
| `1-Me__CO2` | product + TS | -26.10 | 13.86 | 取代基趋势复核 |
| `1-PMe2__CO2` | product + TS | -26.76 | 14.21 | 取代基趋势复核 |
| `1-Ph__CO2` | product + TS | -21.99 | 16.24 | 中等难度 CO2 体系 |
| `1-SiMe3__CO2` | product + TS | -21.35 | 18.54 | 高势垒 CO2 体系 |
| `1-H__CO2` | product only | -18.83 | n/a | 热力学端点验证 |
| `1-CN__CO2` | product only | -17.20 | n/a | 热力学端点验证 |
| `1-tBu__H2CO` | product + TS | -19.60 | -4.68 | 小分子中最容易形成通道 |
| `1-tBu__CS2` | product + TS | -16.80 | 5.09 | 小分子对比 |
| `1-tBu__MeCN` | product + TS | -15.70 | 10.19 | 小分子对比 |
| `1-tBu__MeCH=NMe` | product + TS | -18.80 | 8.34 | 小分子对比 |

### 全部 CO2 产物 SI 电子能矩阵

单位为 `kcal mol^-1`，相对于 `n-R + CO2` 分离反应物。

| family | CN | H | Me | Ph | PMe2 | SiMe3 | tBu |
|--------|---:|--:|---:|---:|-----:|------:|----:|
| `1-R` | -17.20 | -18.83 | -26.10 | -21.99 | -26.76 | -21.35 | -31.05 |
| `2-R` | -10.28 | -10.65 | -16.85 | -15.61 | -14.06 | -13.97 | -22.87 |
| `3-R` | -6.66 | -7.81 | -13.40 | -13.20 | -11.81 | -11.96 | -20.00 |
| `4-R` | -16.78 | -20.26 | -24.28 | -21.01 | -20.92 | -19.50 | -28.09 |
| `5-R` | -8.83 | -9.95 | -14.46 | -13.18 | -11.34 | -11.37 | -16.41 |
| `6-R` | -9.57 | -11.28 | -14.68 | -12.68 | -13.00 | -12.35 | -19.33 |

## RTIP/JAX 验证边界条件

为了和 `history/daily/20260524.md` 的记录保持一致，建议后续每条反应都按以下边界条件写入工作日志。

### 输入边界

- 反应物：`research/ic5c02384/reactions/<reaction>/1.xyz` 和 `2.xyz`。
- 初始分离构型：用 `rtip-jax synthesize --dist 5.0 --seed 0` 生成 `IS.xyz`。
- 参考结构：优先使用 `product.xyz` 和 `ts.xyz`；无 TS 的体系只做端点能量与产物几何验证。
- 环境：气相分子簇，非周期，无溶剂，无压力/盒矢量。
- 元素范围：`B, C, H, N, O, P, S, Si`。
- DeePMD 模型：需要覆盖上述元素；`20260524` 使用 `/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt`。

### 能量口径

- `pot_real`：真实 PES 能量，判断反应趋势时优先看它。
- `pot_total`：包含 bias，不应用作真实反应能。
- 文献正文 `ΔG`：Gibbs 自由能，来自 DFT 单点 + 热校正。
- 本地 `reaction_manifest.csv`：SI 标题电子能差，适合做 DeePMD 单点能排序对比。

### 几何判据

每次验证至少记录：

- final 或 best frame 的 `pot_real` 相对初始反应物能量。
- 与 `product.xyz`、`ts.xyz` 的 all-atom RMSD 和 reactive-core RMSD。
- 形成键距离：`min(B-X)` 与 `min(N-C_small)`。
- `CO2` 特例：记录 `min(B-O)` 和 `min(N-CO2(C))`。
- 小分子特例：
  - `CS2`：记录 `min(B-S)` 和 `min(N-CS2(C))`。
  - `H2CO`：记录 `min(B-O)` 和 `min(N-H2CO(C))`。
  - `MeCN`：记录 `min(B-N_small)` 和 `min(N-MeCN(C))`。
  - `MeCH=NMe`：记录 `min(B-N_small)` 和 `min(N-imine(C))`。

注意：`product.xyz`、`ts.xyz` 和运行输出的 atom order 可能不完全一致。RMSD 应先按元素和反应片段做合理重排或使用 Kabsch 对齐后的 reactive-core RMSD；关键距离用 `min(...)` 更稳。

### 推荐验证顺序

1. `1-tBu__CO2`：基准体系，已有长记录，可继续从成功的 RC-MD 参数附近复核。
2. `1-tBu__H2CO`：正文 Gibbs 势垒最低，适合测试相同 bias 是否更容易成键。
3. `1-Me__CO2`、`1-PMe2__CO2`：与 `1-tBu__CO2` 对比取代基趋势。
4. `1-tBu__CS2`：小分子对比，监控 `B-S` 和 `N-C`。
5. `1-Ph__CO2`、`1-SiMe3__CO2`、`1-tBu__MeCN`、`1-tBu__MeCH=NMe`：作为更高势垒或更难通道。
6. `1-CN__CO2`、`1-H__CO2` 以及 `2-R` 到 `6-R` product-only 系列：先做单点能和产物几何，不作为第一批路径验证。

## 最小命令模板

```bash
rtip-jax synthesize \
  --inputs research/ic5c02384/reactions/1-tBu__CO2/1.xyz research/ic5c02384/reactions/1-tBu__CO2/2.xyz \
  --output research/ic5c02384/reactions/1-tBu__CO2/IS.xyz \
  --dist 5.0 \
  --seed 0
```

后续如果沿用 `20260524` 的 RC-MD 验证方式，建议每个反应单独建立输出目录，并把最终 `summary.md`、`summary.json`、best frame 和关键距离表同步记录到对应工作日志。

## 主要结论

- 论文是理论预测研究，不是实验合成报道。
- `1-tBu + CO2` 是最适合作为首个验证对象的主反应：正文 Gibbs 产物最稳定，势垒最低，SI 中也有 product 和 TS。
- `H2CO` 小分子通道 Gibbs 势垒最低，可能比 `CO2` 更容易在 bias/RC-MD 中形成。
- 文献认为偶联是 concerted step：`N2` 端氮向小分子碳形成 `N-C`，小分子 `X` 向硼形成 `B-X`。
- 后续验证必须区分正文 Gibbs 自由能、SI 电子能、DeePMD `pot_real` 和 bias 后的 `pot_total`。
