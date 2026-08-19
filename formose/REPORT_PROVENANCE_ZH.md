# Formose 复刻项目 — 数据来源与结论边界报告

**报告日期**：2026-08-19
**分支**：`formose/replicate`（n5: `/home/lhshen/RTIP`，GitHub: `jinzhezenggroup/RTIP`）
**性质**：对既有文档（`REPORT.md`、`ACCEPTANCE.md`、`GUIDE_ZH.md`）的**结论边界澄清**，不替代原报告；原报告一律保留不变。

---

## 1. 为什么需要这份报告

原 `REPORT.md` 把"RTIP-MD 机制复现"和"微动力学与 Figure 4 定量一致"并列呈现，容易让读者误以为两者互相支撑。实际上：

- **微动力学部分的输入 100% 来自论文**（Table S1 能垒、初始浓度、65 °C、ODE 模型），我们的贡献只是用 Python 重新实现同一模型、并用 LSODA 求解；
- 因此"与 Figure 4 一致"**只能证明我们复刻的微动力学程序是正确的**；
- 它**不能**作为"DeePMD + RTIP-MD 主体结论"的一部分，不能用来论证 DeePMD 引擎产生了正确的反应网络或正确的动力学。

本报告把所有数据按来源分类，并明确每条结论的**可主张范围**。

---

## 2. 数据来源总表

| 数据 / 结果 | 来源 | 本工作新计算？ | 能支撑的结论 |
|---|---|---|---|
| MD 轨迹 `rtip.pdb`、标量 `rtip.out`、相位机 `rtip_decreasing_steps` | DPA-3.2-5M（DeePMD）实时推理 | ✅ 是 | 机制级事件、温度/能量演化 |
| 键事件 `bond_events.csv` | 由上述轨迹按论文键监测方案（几何阈值）派生 | ✅ 是（几何判据，无新量子计算） | 反应事件的有/无、类型、时刻 |
| 2/3 种子 C–C 成键、R2/R5 反应对事件 | 上述轨迹 | ✅ 是 | **DeePMD + RTIP-MD 主体结论** |
| 温度稳定性（1500–1800 K） | 上述轨迹 | ✅ 是 | 引擎工程行为（fixed_sigma/size_scaling 配置） |
| 27 物种 + 28 TS 结构 | 论文 SI（`formose/data/species|ts/*.xyz`） | ❌ 否（解析入库） | 参照结构 / 输入 |
| `manifest.csv` 的 `energy_ha` | 论文 SI（DFT 能量） | ❌ 否 | 仅供索引，**不是本工作计算值** |
| Table S1 的 28 步正/逆能垒 | 论文 SI（`formose/data/table_S1.csv`） | ❌ 否 | 微动力学输入 |
| 微动力学初始浓度 / 温度（65 °C） | 论文 3.5 节 | ❌ 否 | 微动力学输入 |
| 微动力学 ODE 模型（质量作用 + kT/h） | 论文 Microkinetics 程序 | ❌ 否（我们复刻实现） | 程序复刻正确性 |
| 微动力学数值（Figure 4 一致） | 论文输入 + 我们的 LSODA 求解器 | ⚠️ 半（实现/求解） | **仅程序复刻验证** |
| 图表、验收报告 | 派生（我们的分析/绘图代码） | ⚠️ 派生 | 展示与记录 |

---

## 3. A 部分：DeePMD + RTIP-MD 主体（真正的新计算）

### 3.1 数据链

```text
DPA-3.2-5M (DeePMD 推理)
   └─→ evolution_md (RTIP 相位机 + 蛙跳 + Berendsen)
          └─→ rtip.out / rtip.pdb / rtip_decreasing_steps
                 └─→ analyze_trajectory.py / analyze_run.py
                        └─→ bond_events.csv / acceptance_report.md
```

### 3.2 可以主张的结论

1. **算法移植忠实**：`evolution_md` 与上游 Rust `md.rs` 逐行对应；RTIP 势/力公式三方（论文 SI / Rust / JAX）审计一致；`pytest` 90 项通过。→ 代码层结论。
2. **机制级复现（本报告的核心新计算结果）**：在 DeePMD 引擎 + 论文协议下，
   - 10 Å 盒子 5 ps（10000 步）MD，3 个独立种子中 **2/3 自发形成甲醛 C–C 缩合键**：seed0 1.357 Å @ step 7570，seed2 1.406 Å @ step 5700；
   - 反应对 R2（HCO + CH₂O）末帧持久 C–C 1.42 Å；R5（烯醇核心 + CH₂O）step 700 新 C–C 1.515 Å；
   - 轨迹中出现 C–H（烯醇化）与 H–H（H₂ 生成）事件，与论文键监测方案描述的事件类型一致。
3. **工程稳定性结论**：`fixed_sigma=true` + `size_scaling=true` 配置下，盒子 MD 温度稳定在约 1500–1800 K，无早期温度爆炸。

### 3.3 不可以主张的结论

- 不能用这些轨迹给出**能量、势垒、速率常数**：DeePMD 能量在本体系的精度未验证，且非平衡偏置下的轨迹不是热力学采样；
- 不能把事件出现频率当作**动力学概率**：1500 K 是加速条件，RTIP 偏置是非平衡驱动；
- 不能忽略 NNP 外推风险：r5pair 自 step 700 起 RTIP 侧出现 NaN（step 700 为最后一个有限帧），说明 DeePMD 在极端构型下存在失效边界。

---

## 4. B 部分：微动力学（论文数据 + 程序复刻）

### 4.1 数据链

```text
论文 SI Table S1（28 步能垒）
论文初始浓度（0.35/0.05/0.06 M、H2O 55.5 M）
论文 ODE 模型（质量作用 + kT/h 预因子，65 °C）
   └─→ simulate.py（LSODA 求解）
          └─→ concentrations.csv / summary.json
                 └─→ 与论文 Figure 4 对照
```

### 4.2 可以主张的结论

1. **我们的 Python 实现正确复刻了论文的微动力学模型/程序**：同一 ODE、同一输入，输出与 Figure 4 一致：
   - formyl anion 浓度 3.62×10⁻¹⁴ M（论文 3.6×10⁻¹⁴）；
   - CH₂O 二聚速率 1.90×10⁻⁹ mol L⁻¹ s⁻¹（论文 1.9×10⁻⁹）；
   - aldotetrose retroaldol 净速率转正 0.763 s（论文 ~0.76 s）；
   - 核糖 3.8×10⁻¹⁵ vs 线型四碳糖 0.027 M（论文：核糖微量）。
2. **论文的动力学结论被"复算"复核**：诱导期、自催化仅在低乙醇醛浓度、核糖低产率——这些是论文模型的自洽结果，我们验证了自己的实现能重现它们。

### 4.3 不可以主张的结论

- 微动力学的数字**不是由 DeePMD 计算得出**，**不能**作为 DeePMD + RTIP-MD 主体结论的一部分；
- **不能**由"微动力学一致"反推"DeePMD 轨迹/能垒正确"——两条证据链相互独立；
- 若将来把能垒替换为 DeePMD 计算值，得到的是**新模型预测**，必须另立报告，且不能再与 Figure 4 做"复现"式比较。

---

## 5. 结论边界矩阵（速查）

| 问题 | 答案 | 依据 | 边界 |
|---|---|---|---|
| DeePMD + RTIP-MD 能否机制级复现论文反应？ | 是（2/3 种子 C–C、R2/R5 事件） | A 部分轨迹 | 仅机制级，非能量级 |
| 我们的微动力学程序是否正确？ | 是 | B 部分（同模型同输入同输出） | 程序复刻验证 |
| 论文的动力学结论是否被复现？ | 是（复算） | B 部分 | 依赖论文能垒，与 DeePMD 无关 |
| DeePMD 能量/力能否用于定量动力学？ | 未验证 | — | 需单独基准测试（对比 DFT/实验） |
| 微动力学一致能否证明 DeePMD 轨迹正确？ | **不能** | 两条数据链独立 | 不能互相推导 |

---

## 6. 对验收口径的修订建议

建议将"验收"拆成两条独立轨道，本报告即按此口径书写：

- **轨道一（机制级验收，A 部分）**：2/3 种子成键 + R2/R5 事件 → ✅ 通过；
- **轨道二（程序复刻验收，B 部分）**：微动力学与 Figure 4 一致 → ✅ 通过，但**只证明程序/实现正确**；
- 原 `ACCEPTANCE.md` 把两者混合叙述，阅读时应以本报告的边界为准；原文件保留不动，后续如需可另出一版拆分后的验收文件。

---

## 7. 复现命令与原始文件位置

### A 部分（DeePMD + RTIP-MD）

```bash
cd /home/lhshen/RTIP
sbatch formose/run_formose.slurm          # seed0，10000 步
SEED=1 MAX_STEP=10000 sbatch formose/run_formose.slurm
SEED=2 MAX_STEP=10000 sbatch formose/run_formose.slurm
PYTHONPATH=rtipmd/jax/src JAX/.venv/bin/python formose/analyze_run.py formose/runs/seed0 --label seed0
```

原始数据：`/home/lhshen/RTIP/formose/runs/{seed0,seed1,seed2,r2pair,r5pair}/{rtip.out,rtip.pdb,rtip_decreasing_steps,bond_events.csv}`

### B 部分（微动力学）

```bash
source /group/software/deepmd-kit-3.1.1/bin/activate
python formose/microkinetics/simulate.py \
  --table formose/data/table_S1.csv \
  --output-dir formose/microkinetics/runs
```

原始数据：`/home/lhshen/RTIP/formose/microkinetics/runs/{concentrations.csv,summary.json}`；
输入（论文）：`/home/lhshen/RTIP/formose/data/table_S1.csv`。

---

## 8. 一句话总结

**A 部分（DeePMD + RTIP-MD）是本项目真正的新计算，结论是"机制级复现"；B 部分（微动力学）的输入全部来自论文，结论只能是"我们正确复刻了论文的微动力学程序"。两者互相独立，不能互相背书。**
