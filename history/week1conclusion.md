# Week 1 代码阅读结论

## 一、项目目标

用 RTIP/JAX + DeePMD (DPA-3.2-5M) 模拟 1-tBu + CO2 偶联反应，验证 ACS 文献 `ic5c02384` 的反应路径。

## 二、代码架构

```
rtip_jax/
├── constants.py    — 物理常数 + 元素周期表 + 原子质量
├── _config.py      — JAX 初始化（开双精度 x64）
├── config.py       — Para 参数包（步数、温度、力强度等）
├── system.py       — System 数据结构（坐标/元素/势能）
├── core/
│   └── rtip.py     — 核心算法：RTI 距离 + RTIP 高斯偏置势能
├── pes/
│   ├── base.py     — 势能面接口（PES / SumPES / ZeroPES）
│   └── bias.py     — 四种偏置力容器 + ReactionCoordinatePES
├── workflows/
│   ├── pathway_sampling.py  — Pathway 工作流（无温度，纯力+线搜索）
│   ├── md.py                — MD 工作流（有温度，蛙跳积分+恒温器）
│   └── synthesis.py         — 碎片拼接布局（synthesize/synthesis_target_state）
├── external/
│   └── deepmd.py   — DeePMD 接口（Angstrom/eV ↔ Bohr/Hartree 单位换算）
└── cli.py           — 命令行入口
```

## 三、核心概念

### 3.1 RTI 距离（rti_dist）

旋转平移不变距离。用四元数方法找到两个分子结构之间的最优旋转对齐，残差即为 RTI 距离。RTI 距离 = 0 表示两个结构是同构体，值越大差异越大。

### 3.2 RTIP 偏置势能（Rtip0PES）

以参考结构为中心放置高斯势：
- **a > 0**：排斥力（高斯"山峰"，推离参考点），适用解离反应
- **a < 0**：吸引力（高斯"山谷"，拉向参考点），适用结合反应
- a 的绝对值随步数增大（a = a0 × step），逐步加强

高斯宽度 sigma 用当前结构到参考结构的 RTI 距离。

### 3.3 四种偏置力类型

| 类型 | a 符号 | 参考点 | 适用场景 | Rust 来源 |
|------|--------|--------|---------|-----------|
| RepulsivePot | + (排斥) | local_min | 解离反应 AB→A+B | Rust pathway + MD |
| AttractivePot | - (吸引) | final_state (产物) | 结合反应 A+B→AB | Rust pathway 有，MD 无 |
| SynthesisPot | - (吸引) | 碎片共享质心 | 多分子合成 | Rust pathway 有，MD 无 |
| ReactionCoordinatePot | + (简谐约束) | 指定原子对距离 | 约束关键成键距离 | Rust 完全无，纯新加 |

### 3.4 两个工作流

**Pathway（pathway_sampling.py）**：无温度，纯力驱动。每一步算 force_total = force_real + force_bias，沿合力方向做一维线搜索。有状态机判定（pot_drop / pot_climb / f_epsilon 等停判）。temp_K 和 kin_Ha 显示 nan 是正常行为。

**MD（md.py）**：有温度，模拟热运动。每一步用蛙跳（Leapfrog）积分推进坐标和速度，Berendsen 恒温器控制温度。有四个 bias 分支：repulsive / attractive / synthesis / reaction-coordinate。

## 四、关键物理量判据

| 指标 | 含义 | 判断依据 |
|------|------|---------|
| pot_real | 真实势能 (DeePMD) | **核心判据**，反应走通应下降 |
| pot_bias | 偏置势能 (RTIP) | 人工加的能量，不能只看它 |
| pot_total | pot_real + pot_bias | HISTORY 反复强调不能只看 total |
| rti_dist / sigma_min | RTI 距离 | Pathway 中是 bias 宽度，MD 中仅是监控量 |
| min(B-O) | 最近硼-氧距离 | 关键成键判据，product 参考值 ~1.49 A |
| min(N-C) | 最近氮-碳距离 | 关键成键判据，product 参考值 ~1.49 A |
| RMSD | 与参考结构的偏差 | 越小越接近目标结构 |
| temp_K | 温度 | 仅 MD 有意义 |
| f_real | 真实力范数 | 越小越接近稳定点 |

**核心教训**：判断反应是否走通要看 pot_real、关键成键距离和 RMSD，不能只看 rti_dist 或 pot_total。

## 五、Rust 原始设计调查结果

对比 `/home/slh/4_28/RTIP/src/pes_exploration/` 下 Rust 源码：

### Pathway 层（pathway_sampling.rs）

三种模式 Rust 都有：
- `impl RtipPathSampling for RepulsivePot` — 第 21 行
- `impl RtipPathSampling for AttractivePot` — 第 629 行
- `impl RtipPathSampling for SynthesisPot` — 第 832 行

JAX 迁移忠实地复现了全部三种。

### MD 层（md.rs）

Rust **只有** repulsive 一种模式：
- `impl RtipNVTMD for RepulsivePot` — 第 18 行到第 307 行结束
- AttractivePot / SynthesisPot 的 RtipNVTMD **不存在**
- ReactionCoordinatePot **完全不存在于 Rust**

### 结论

| 层级 | Repulsive | Attractive | Synthesis | ReactionCoordinate |
|------|:---:|:---:|:---:|:---:|
| Rust pathway | 有 | 有 | 有 | 无 |
| Rust MD | 有 | **无** | **无** | 无 |
| JAX pathway | 有 | 有 | 有 | 无 |
| JAX MD | 有 | **这次新加的** | **这次新加的** | **这次新加的** |
| CLI 入口 | 有 | 无 | 无 | 无 |

CLI 只有 repulsive，不是遗漏，而是**原版 Rust MD 层就只有 repulsive**。Attractive / synthesis MD 以及 ReactionCoordinate 是本次研究中发现 repulsive 走不通结合反应后，在 JAX 层现场扩展的。

## 六、HISTORY 文档关键结论

1. DPA-3.2-5M 模型定性正确（TS > reactant > product 能量排序对），定量有偏差（TS 势垒偏低 ~3 kcal/mol，产物放热偏少 ~7.5 kcal/mol）
2. Stock repulsive RTIPMD 5 个种子全部失败，不适合结合反应
3. 四种 bias 探索中，ReactionCoordinate (RC_K=0.002, product target) 给出唯一 promising frame，但 pot_real 仍升高 ~24 kcal/mol，成键距离（2.5 A）仍远大于目标（1.49 A），不是 success
4. **更新（2026-05-24 晚）：RC_K 扫描（0.001-0.010, 1000 步）证实之前的失败主要是步数不足。k=0.006 product-target 成功将体系稳定在 product 盆地区域约 800 步（400 fs），最终 B-O=1.44 A, N-C=1.66 A（参考值 1.49, 1.49 A）。pot_real 仍偏高（Erel ~+43 kcal/mol），需后续无 bias 弛豫确认。详见 `HISTORY/20260524.md` RC_K 扫描章节。**
5. 下一步建议：对 k=0.006 轨迹做无 bias 弛豫；对其他取代基体系用 k=0.006 复现
