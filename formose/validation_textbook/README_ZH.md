# 教科书反应验证：DeePMD + RTIP-MD 正确性检验

分支: `validation/textbook`（基于 `formose/replicate`）

## 1. 目的

用高中化学教科书中的简单气相反应，检验本仓库 JAX 版 RTIP-MD
（蛙跳积分 + Berendsen 恒温器 + 相位机 Increasing→Decreasing→Falling，
即 growing→reducing→off）在 DeePMD 预训练势（DPA-3.2-5M）驱动下的
正确性，分三层:

1. 正对照: 工作流能否把两个反应物拉近并形成**正确的产物键**;
2. 负对照: 对"不应反应"的组合，工作流是否保持不反应;
3. 定量: DeePMD 单点反应能与实验/高精度参考的偏差有多大。

## 2. 体系与设计

所有几何取实验键长（见 `formose/validation_textbook/*.xyz` 注释），
初态两分子质心距 5 Å，sidecar `*.xyz.rtip.json` 的 `atom_add_pot`
覆盖全部原子（与 formose 生产相同的偏置方式）。

| case | 反应 | 类型 | 实验 ΔH°298 (kcal/mol) |
|------|------|------|------------------------|
| validation_ch2o_h2 | CH2O + H2 → CH3OH | 正对照 | −21.9 (ATcT) |
| validation_c2h4_h2 | C2H4 + H2 → C2H6 | 正对照 | −32.6 |
| validation_ch2o_ch4 | CH2O + CH4 | 负对照 | 无稳定产物 |
| validation_ch2o_ch4_gentle | 同上，弱偏置+低温 | 负对照(温和) | 同上 |

## 3. 方法

* MD 代码路径: `python -m rtip_jax.cli deepmd-evolution-md`
  （与 formose 生产完全相同，仅输入/输出目录不同）
* 积分: 蛙跳 (leapfrog), dt = 0.5 fs, 共 2000 步 (1000 fs)
* 恒温: Berendsen, tau = 10 fs, 目标温度 1500 K（gentle case 500 K）
* 相位机: a0 = 1e-5（gentle 2e-6）, size_scaling 放大 n_bias 倍,
  sigma 固定初值, decreasing_multiple = 2.0
* 真实势: DPA-3.2-5M 预训练模型（OMol25 默认域头），cell=None 气相
* 单点: 各分子实验几何 DeepPot 单点，ΔE = E(产物) − ΣE(反应物)
* 键分析: 只追踪化学反应相关原子对，教科书物理阈值
  （H–H < 0.90 Å, C–H < 1.15 Å, O–H < 1.05 Å, C–C < 1.65 Å,
  C–O < 1.50 Å）。不采用原代码 1.25 倍共价半径邻接，因为该约定会把
  乙烯分子内 1,2-C–H 近距接触(~1.2 Å)误判为键（见 analyze_validation.py
  文件头说明）。

## 4. 参考数据来源

* CH2O(g) ΔfH°298 = −109.19 ± 0.10 kJ/mol；CH3OH(g) ΔfH°298 =
  −200.92 ± 0.15 kJ/mol（ATcT 1.130）→ ΔH = −91.7 kJ/mol = −21.9 kcal/mol
* C2H4(g) ΔfH°298 = +52.5 kJ/mol；C2H6(g) = −83.8 kJ/mol（NIST）→
  ΔH = −136.3 kJ/mol = −32.6 kcal/mol（与 RSC SI 的 CCSD(T) 电子能参考一致）
* CH4 + 2O2 → CO2 + 2H2O: ΔH = −802.5 kJ/mol = −191.8 kcal/mol

## 5. 单点反应能: DeePMD vs 参考

| 反应 | DeePMD ΔE (kcal/mol) | 参考 ΔH298 (kcal/mol) | 偏差 |
|------|----------------------|----------------------|------|
| CH2O+H2→CH3OH | +0.97 | −21.9 | +22.9 |
| C2H4+H2→C2H6 | −191.0 | −32.6 | −158.4 |
| CH4+2O2→CO2+2H2O | −34.4 | −191.8 | +157.4 |

辅助检查（`deepmd_bench.py`）:
* H2 势能曲线: 极小值在 0.74 Å（实验 0.741），解离能 5.10 eV
  （实验 ~4.48 eV，偏高 0.6 eV）——键长定性正确，深度偏高。
* 燃烧反应偏差 ~157 kcal/mol；模型无自旋处理（O2 三重态按闭壳层算），
  是误差来源之一，但不是全部。

**结论: DPA-3.2-5M 对该类小分子气相反应能定量不可靠（偏差可达
100+ kcal/mol），与先前 formose SI 27 物种异构体相对能
MAE 13 kcal/mol、最大 47 kcal/mol 的发现一致。因此能量正确性验证
不通过；MD 验证只用于检验工作流（成键/断键与产物几何）。**

## 6. MD 轨迹结果

### 6.1 validation_ch2o_h2 — 正对照（CH2O + H2 → CH3OH）: 通过

* 2000 步跑完，0 行 NaN
* 事件: step 550 H–H(H2) 断裂; step 560 C–H(H2→C) 成键; step 570
  O–H(H2→O) 成键
* 末帧（step 2000）: 单分子；C–O 1.38 Å、O–H 0.98 Å、C–H 1.03 Å
  → **甲醇**（实验键长 C–O 1.43、O–H 0.96、C–H 1.09）
* 关键键长演变: H–H 0.74→2.10 Å（解离）；C–O 1.20→1.38 Å
  （双键→单键）；O–H 4.73→0.98 Å；C–H 5.47→1.03 Å

### 6.2 validation_c2h4_h2 — 正对照（C2H4 + H2 → C2H6）: 通过

* 2000 步跑完，0 行 NaN
* 事件: H–H(H2) step 420 首次拉长（0.92 Å）后恢复，step 770 最终断裂
  （0.98 Å）；step 760–770 三个 C–H 候选成键（至少 2 个有效）
* 末帧: 单分子；C–C 1.56 Å（乙烷实验 1.535）、C–H ~1.0–1.1 Å
  → **乙烷**

### 6.3 validation_ch2o_ch4 — 负对照: 未通过（过度驱动）

* 2000 步跑完，0 行 NaN
* 事件: step 680 CH2O 的 C 与 CH4 的 C 成键（1.50 Å）
* 两次相同参数运行（nondeterministic，GPU/JAX）给出不同末帧:
  - 运行 A（首次）: 末帧 3 个分子，CH2O 分解为 CO + H2，C–C 3.20 Å
    未成键（仅瞬时 C–C 1.47 Å）
  - 运行 B（重跑，结果保留在 runs/validation_ch2o_ch4）: 末帧 C–C 1.56 Å
    成键，产物为 **乙醇**（C4H3–C1H2–OH，O–H 1.04 Å）——偏置甚至把
    CH4 + CH2O "驱动"成了真实产物
* 解读: 在 a0=1e-5、浴温 1500 K 的偏置下，负对照体系被推到
  ~2200–2300 K 并发生强碰撞，出现 C–C 成键（乃至生成乙醇）。
  这不是 DeePMD 势的错误，而是该偏置参数对"不应反应"的体系过于激进。

### 6.4 validation_ch2o_ch4_gentle — 负对照（温和参数）: 通过

弱偏置（a0=2e-6）+ 低温浴（500 K），用于检验 6.3 的假阳性是否由参数
过激导致:

* 2000 步跑完，0 行 NaN
* 无 C–C 成键事件；末帧 CH2O + CH4 两分子（C–C 2.03 Å 未成键）
* 结论: 负对照"不反应"在该工作流下成立，但**前提是偏置参数足够温和**

## 7. 数值修复记录（核心代码改动）

首次 CH2O+H2 运行在第 600 步出现 NaN: 两分子质心精确重合时
（RTI 距离 = 0），`rti_pot_force` 中的 `1/d^7` 权重与
`... * distances` 除法产生 0/0（旧运行归档在
`formose/runs/_archive/validation_ch2o_h2_pre_fix_nan`，轨迹在 NaN 前
已出现 H2 解离 + O–H/C–H 成键，说明反应事件真实，只是数值奇点污染了
后续轨迹）。

修复: `rtipmd/jax/src/rtip_jax/core/rtip.py` 的 `rti_weight` /
`rti_weight_derivative` / `rti_pot_force` 将 RTI 距离平移
`_RTI_DIST_EPS = 1e-6` Bohr（5e-7 Å，远低于任何物理尺度）。
回归测试: `rtipmd/jax/tests/test_rtip.py` 新增
`test_rti_weight_and_force_finite_at_exact_coincidence`。
全套测试 91 passed。

## 8. 数据文件清单

原始输出（n5, `/home/lhshen/RTIP/`）:

```
formose/runs/validation_ch2o_h2/{rtip.out, rtip.pdb, rtip_decreasing_steps, validation_report.md}
formose/runs/validation_c2h4_h2/{rtip.out, rtip.pdb, rtip_decreasing_steps, validation_report.md}
formose/runs/validation_ch2o_ch4/{rtip.out, rtip.pdb, rtip_decreasing_steps, validation_report.md}
formose/runs/validation_ch2o_ch4_gentle/{...}
formose/runs/_archive/validation_ch2o_h2_pre_fix_nan/     (修复前 NaN 运行)
formose/runs/_archive/validation_ch2o_ch4_pre_fix_partial/ (修复前被中断的运行)
```

输入/脚本（已提交到仓库）:

```
formose/validation_textbook/*.xyz, *.rtip.json       反应物/产物几何与 sidecar
formose/validation_textbook/deepmd_sp.py             单点能量脚本
formose/validation_textbook/deepmd_bench.py          H2 曲线与燃烧基准
formose/validation_textbook/deepmd_probe.py          调用方式探测
formose/run_validation.slurm                         作业脚本
formose/para_validation_gentle.json                  温和负对照参数
formose/analyze_validation.py                        键事件与验收分析
formose/validation_textbook/plot_validation.py       图表
formose/validation_textbook/figures/                  fig1/fig2/fig3 (提交)
```

图表:

* `fig1_reaction_energy.png` — DeePMD 单点反应能 vs 参考
* `fig2_temperature_curves.png` — 四个 case 的温度曲线（相位机行为）
* `fig3_key_distances.png` — 关键键长随时间演变（成键过程）

## 9. 结论与边界

* **验证了什么**: RTIP-MD 工作流（相位机 + Berendsen + 蛙跳）在
  DeePMD 势驱动下，能把 CH2O+H2 和 C2H4+H2 正确驱动成甲醇和乙烷
  （键事件 + 产物几何层面）。这是对"代码移植正确性"的正向证据。
* **不能推广到什么**: (a) DeePMD 预训练模型的定量热化学（反应能偏差
  数十至 150+ kcal/mol）; (b) 强偏置参数下负对照的"不反应"保证
  （会过度驱动）。这些是工作流使用者必须知道的边界。
* **与 formose 主流程的关系**: 本验证独立于 formose 复现结论，只复用
  同一 `deepmd-evolution-md` 代码路径与同一 DeePMD 模型。formose 的
  微动力学等结论仍只由论文数据支持（见 GUIDE_ZH.md 的数据来源区分）。
