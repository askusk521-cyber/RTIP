# Rust-JAX 对齐方案

## 差异总结

### 核心算法层（公式一致，无需修改）
- rtip.py: RTI距离、四元数系统、高斯偏置势 — 公式完全一致
- idwm.py: 权重函数、IDW距离、梯度 — 公式完全一致
- optimization.py: 黄金分割线搜索 — 逻辑一致
- rotations.py: 旋转矩阵公式 — 一致

### 需要对齐的行为差异（影响数值结果）

| # | 差异 | Rust行为 | JAX行为 | 对齐方案 |
|---|------|---------|---------|---------|
| 1 | eigh UPLO方向 | Upper triangle | Lower triangle (JAX默认) | 强制对称化 S=(S+S.T)/2 后再eigh |
| 2 | 负特征值处理 | sqrt直接产生NaN | clamp到0再sqrt | 保留JAX行为（这是bug fix） |
| 3 | attractive/synthesis线搜索系统 | 传入完整系统 | 传入fragment | 改为传入完整系统（匹配Rust） |
| 4 | MD偏置振幅 | 纯线性增长 a0*step | bias_amplitude()含振荡/衰减 | 默认参数下已一致(oscillation=0) |
| 5 | MD停止条件 | 无（跑满max_step） | 10+停止条件 | 添加rust_compat模式禁用提前停止 |
| 6 | MD温度反馈 | 无 | T>1500K缩放bias | 添加开关，默认关闭以匹配Rust |

### 保持不变（DeePMD相关）
- external/deepmd.py — DeePMD PES provider
- CLI: deepmd-pathway, deepmd-md, deepmd-synthesis-md, deepmd-boundary
- eV转换常数 (EV_TO_HARTREE等)
- B/Si原子质量（研究体系需要）

### 保持不变（JAX独有扩展，不影响Rust对齐）
- AttractivePot/SynthesisPot/ReactionCoordinatePot pathway和MD
- 三阶段状态机（可通过参数禁用）
- 多轮合成MD
- SumPES/ZeroPES/HarmonicPES（测试工具）
- 结构化返回值（PathwayResult/MDResult）

## 对齐修改清单

1. core/rtip.py: 在 _quaternion_system_matrix 返回前强制对称化
2. workflows/pathway_sampling.py: attractive/synthesis线搜索改为传入完整系统
3. workflows/md.py: 添加 rust_compat 参数
   - rust_compat=True时：禁用温度反馈、禁用提前停止、纯线性振幅
4. config.py: Para添加 rust_compat 字段

## 验证实验设计

用HarmonicPES做确定性对照：
- 相同初始结构、相同seed、相同参数
- 对比Rust模式 vs JAX默认模式的轨迹
- 指标：最终坐标差异、能量差异、力差异
