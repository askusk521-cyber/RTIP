# Formose 反应 RTIP-MD 复刻项目报告

**报告日期**：2026-08-01
**分支**：`formose/replicate`（n5: `/home/lhshen/RTIP`）
**复刻目标论文**：*Conserved-Potential-Driven Molecular Dynamics Deciphers
Formose Reaction Mechanisms*，JACS Au 2026, 6, 922（DOI:
10.1021/jacsau.5c01359，CC-BY 4.0）

---

## 1. 项目概述

本工作以课题组现有 RTIP 软件栈为基础，用 Python/JAX 复刻论文的
RTIP-MD（roto-translationally invariant potential driven molecular
dynamics）方法，并以 **formose（甲醛聚糖）反应**为对象做机制级复现与
验收。论文的核心方法是：在真实势能面上叠加一个随搜索步数线性加深的高斯型
RTIP 偏置，通过键变化检测自动控制偏置的增/减循环，驱动分子体系在无预设
反应坐标的条件下自发靠近并发生反应；随后用微动力学模拟解释实验现象。

本分支只保留与上游算法一致的部分（不使用反应坐标偏置 RCMD、不含
ic5c02384 论文工作流），真实势能由 DeePMD-kit（DPA-3.2-5M）提供，替代
论文原用的 CP2K/B97-3c。

## 2. 工作内容总览

| 项目 | 状态 | 说明 |
|---|---|---|
| 新分支 `formose/replicate` | ✅ | 基于 fork 的 `size-scaling-lite`（HEAD 72c9746），15 个提交，工作区干净 |
| 清理 | ✅ | 删除 `research/ic5c02384/`、`rtipmd/jax/examples/`、RCMD/反应坐标代码、size-scaling 之外的工程附加物（振荡、温度反馈、提前停止、多轮合成 MD、`rust_compat` 开关） |
| 算法移植 | ✅ | `RepulsivePot::rtip_nvt_md` 与 `EvolutionPot::rtip_nvt_md` 从 Rust `md.rs` 1:1 移植到 `rtipmd/jax`（蛙跳积分 + Berendsen 恒温器 + Increasing/Decreasing/Falling 状态机 + 键变化检测） |
| 数据 | ✅ | 论文 SI（Europe PMC PMC12933356）下载并解析：27 物种 + 28 过渡态结构（Å/Hartree）、Table S1（28 步反应能垒） |
| 微动力学 | ✅ | 同一 ODE 模型的 Python 实现，与论文 Figure 4 定量一致 |
| 实验 | ✅ | 完整 5 ps（10000 步）盒子 MD × 3 种子；反应对 MD（R2/R5）；参数扫描 |
| 文档 | ✅ | README、RECORD（决策与问题记录）、ACCEPTANCE（验收）、AGENTS、USAGE_ZH 全部 markdown |

## 3. 技术路线

### 3.1 算法（上游 RTIP-MD）

- **积分器**：蛙跳（leapfrog），时间步 0.5 fs，Berendsen 恒温器目标
  1500 K（`tau=10`）；
- **RTIP 偏置**：权重 `f(d)=1/d⁷` 的加权高斯组合
  `V = a·Σ(wᵢ/w)·exp(−dᵢ²/2σ²)`，向"分子质心重合"的虚拟目标构型牵引；
- **自动控制**：`judge_variation_of_bonding`（1.0/1.6 倍共价半径阈值）检测
  C–C/C–H/H–H/O–O 键变化，触发 RTIP 从 Increasing 转 Decreasing
  （`decreasing_multiple=2`，两倍速衰减），幅度回到 `decreasing_bound=0.5`
  阈值后更新键连信息重新 Increasing；
- **键监测**：忽略 C–O、H–O 及所有含 Ca 的键对（论文 3.1 节规则）。

### 3.2 引擎与环境（n5）

- DeePMD-kit 3.1.1 + DPA-3.2-5M（type_map 覆盖全部 118 元素，含 Ca）；
- GPU：RTX 5090 32 GB（node5，经 slurm 提交）；JAX 0.7.2（x64）；
- 单元换算在边界层集中处理（Bohr↔Å、eV↔Hartree、eV/Å↔Ha/Bohr），已逐项
  核对与 Rust/CP2K 边界一致。

### 3.3 初始体系

- 盒子 MD：10 Å 立方、66 原子（8 H2O + 8 CH2O + 2 Ca + 4 OH⁻），冷启动
  （零初速度 = Rust 默认）；
- 反应对 MD：formyl anion（HCO）+ CH2O（R2）、烯醇负离子核心（SI 物种 5
  原子 1–7）+ CH2O（R5），5 Å 分离；
- 生产参数：`a0=0.00001`（size-scaling 有效幅度 0.00066 ≈ Rust 默认量级）、
  `size_scaling=true`、`fixed_sigma=true`。

## 4. 验收结果（以论文为标准）

### 4.1 完整盒子 MD：甲醛自身缩合（论文 R2，限速步）

3 个独立种子 × 10000 步（5 ps，0.5 fs/步，Berendsen 1500 K）：

| 种子 | 平均温度 (K) | 最高温度 (K) | 末帧 min C–C (Å) | C–C 成键事件 |
|---|---|---|---|---|
| 0 | 1618 | 2295 | 1.51 | ✅ 1.357 Å @ step 7570 |
| 1 | 1744 | 2799 | 1.59 | ✗（接近 1.52 Å 阈值） |
| 2 | 1756 | 2781 | 1.58 | ✅ 1.406 Å @ step 5700 |

**2/3 种子在论文协议下自发形成 C–C 键**。结构确认（seed 0, step 7570）：
两个 CH2O 的碳原子（PDB 第 38/54 号原子，0-based 索引 37/53）= 1.357 Å，
各自保留 C=O 与 C–H，合并进同一
19 原子分子——即论文 R2/R3 的甲醛自身缩合（glycolaldehyde 前体）。
此外轨迹中出现 C–H（烯醇化）与 H–H（H2 生成）事件，与论文键监测方案
一致。

### 4.2 反应对 MD：R2 与 R5

- **R2（formyl anion + CH2O）**：2000 步内 C–C 形成并持续，末帧
  **1.42 Å**，产物骨架 O=C(H)–C(H)=O（umpolung 亲核进攻产物）；
- **R5（烯醇负离子 + CH2O）**：step 700 形成新 C–C **1.515 Å**，11 原子
  合并为单一偶联产物（醛醇加成，C3 糖前体）。

### 4.3 微动力学（论文 Figure 4）

用与论文 Microkinetics 程序相同的 28 步 ODE 模型（质量作用 + kT/h 预因子、
65 °C、Table S1 初始浓度）求解，结果定量吻合：

| 物理量 | 本工作 | 论文 Figure 4 |
|---|---|---|
| formyl anion 浓度 | 3.62×10⁻¹⁴ M | 3.6×10⁻¹⁴ M |
| CH2O 二聚速率 | 1.90×10⁻⁹ mol L⁻¹ s⁻¹ | 1.9×10⁻⁹ |
| aldotetrose retroaldol 净速率转正时刻 | 0.763 s | ~0.76 s |
| 核糖 vs 线型四碳糖 | 3.8×10⁻¹⁵ vs 0.027 M | 核糖微量 |

论文三大实验结论（甲醛二聚诱导期、自催化仅在低乙醇醛浓度、核糖低产率）
全部复现。

## 5. 关键技术问题与解决方案

1. **公式审计**：RTIP 势/力公式在论文、Rust、JAX 三方逐项一致（权重
   1/d⁷、加权高斯、力项 `pot/σ² + dw·Σwⱼ(uⱼ−uᵢ)/(w²d)`）；DeePMD 单位
   换算正确。
2. **size-scaling-lite**：RTI 距离随偏置原子数 ~√N 增长，per-atom 力被
   ~1/N 稀释；实现 `Para.size_scaling`（幅度 × n_bias）后恢复有效拉力。
3. **PDB 解析 bug**：分析脚本坐标错位一列（正则漏抓整数原子序号），曾导致
   "无 C–C 事件"误判；已修复并撤回旧结论。
4. **fixed_sigma（决定性修复）**：上游代码 σ 每步取当前 rti_dist（动态），
   分子接近时高斯变尖导致温度爆炸（10⁴–10⁶ K）；论文 Eq. 6 语义 σ=d_des
   为固定宽度。新增 `Para.fixed_sigma`（默认 False 保持 Rust 行为；生产
   配置开启），使盒子 MD 稳定在 **1500–1800 K** 并完成 5 ps 全程成键。

## 6. 代码与仓库

- `rtipmd/jax`：Python/JAX 包（约 2600 行），`pytest` 90 项全部通过；
- `formose/`：数据（SI 结构、Table S1、盒子/反应对输入）、构建与运行脚本
  （`build_box.py`、`run_formose.slurm`）、分析脚本
  （`analyze_trajectory.py`、`analyze_run.py`）、微动力学
  （`microkinetics/simulate.py`）、文档（README/RECORD/ACCEPTANCE）；
- 运行输出全部 gitignore，仓库 146 个跟踪文件、15 个提交、工作区干净。

### 复现命令

```bash
cd /home/lhshen/RTIP
# 盒子 MD（3 种子 × 10000 步）
sbatch --array=0-2 formose/run_formose.slurm
# 验收报告
PYTHONPATH=rtipmd/jax/src JAX/.venv/bin/python \
  formose/analyze_run.py formose/runs/seed0 --label seed0
# 微动力学
source /group/software/deepmd-kit-3.1.1/bin/activate
python formose/microkinetics/simulate.py \
  --table formose/data/table_S1.csv --output-dir formose/microkinetics/runs
```

## 7. 局限与后续计划

- **局限**：完整 28 步网络未在单条长轨迹中全部复现（R2/R5 与缩合已证）；
  DeePMD 在少数极端构型下出现 NaN；论文的 DFT/TS 重验证按约定不在本
  分支范围。
- **后续**：① 延长盒子轨迹（20000 步）观察醛醇增长后的多步网络；
  ② 补 R7（甘油醛 + 烯醇 → 五碳糖）等后续 C–C 步骤；③ 若需能量级
  复现，可在 n5 编译 CP2K 接入 B97-3c（源码已在 `/home/lhshen/cp2k`）；
  ④ 分支推送到 origin 供课题组共享。

## 8. 结论

本分支完成了 formose 论文 RTIP-MD 方法的机制级复刻：在纯上游算法 + DeePMD
引擎下，**2/3 种子在 5 ps 盒子 MD 中自发形成甲醛 C–C 缩合键（1.36/1.41 Å）**，
反应对 MD 复现 R2（1.42 Å）与 R5（1.52 Å）两类 C–C 成键，微动力学与论文
Figure 4 定量一致；公式、单位、算法语义均经审计，关键工程问题（力稀释、
σ 语义）已定位并修复，全过程有完整 markdown 记录，仓库整洁可复现。
