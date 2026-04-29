# 途径采样辅助函数迁移说明（中文版）

Rust 源：`src/pes_exploration/pathway_sampling.rs`（共享辅助逻辑）。

Python 目标：`src/rtip_jax/workflows/pathway_sampling.py`。

## 范围

- 全系统或原子子集的均值中心随机扰动。
- 力范数帮助函数。
- 排斥/IDWM 停止条件状态更新。
- 吸引性停止条件状态更新。
- 合成停止条件状态更新。

## 保留的 Rust 语义

- 初始扰动从 `[0,1)` 采样，做均值中心化、归一化并默认缩放为 `0.1`。
- 排斥与 IDWM 停止逻辑跟踪 `pot_real_max`, `pot_real_min`, 和 `add_bias`。
- 吸引性停止逻辑在 `sigma_min < 1.0` 或 `f_bias > 1000.0` 时禁用偏置。
- 合成停止逻辑在第 1 步记录 `pot_real_initial`。

## 关键差异

- Python 辅助函数返回新的不可变状态 dataclass。
- 随机扰动使用显式 JAX PRNG keys，而不是 `ndarray-rand` 与 MPI 广播。
- 完整工作流的副作用（文件输出、PES 构造、MPI）不在本阶段迁移范围内。

## 测试

- 测试位于 `tests/test_pathway_sampling_helpers.py`，覆盖扰动可复现性与范数缩放、整系统与子集扰动、无效原子子集输入、排斥/IDWM 停止条件、吸引性停止条件，以及合成初始能量记账。
