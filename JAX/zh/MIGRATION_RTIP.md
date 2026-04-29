# RTIP 迁移说明（中文版）

Rust 源：`src/pes_exploration/rtip.rs`。

Python 目标：`src/rtip_jax/core/rtip.py`。

## Rust 语义

- 坐标通过减去几何中心来居中。
- 使用四元数方法构建 4x4 矩阵，通过对每个原子的 `A_i.T @ A_i` 求和得到。
- `rti_dist` 返回最小特征值的平方根。
- `rti_dists` 返回四个特征值的平方根。
- `rti_dist_vec` 和 `rti_dists_vecs` 将特征向量转换为旋转矩阵并返回对齐后的残差向量。
- `rti_rot_tran` 返回将 `coord1` 对齐到 `coord2` 的旋转矩阵与平移向量。
- `rti_pot` 计算以四距离加权的高斯势，其中权函数为 `f(x) = 1 / x^7`。
- `rti_pot_force` 在 `coord2` 上计算 Rust 的力表达式。
- `Rtip0PES` 评估局部极小与附近过渡态的 RTIP 贡献，可对参考或目标系统使用 `atom_add_pot` 片段。

## 输入 / 输出

- 坐标输入形状：`(natom, 3)`，内部单位为 Bohr。
- `rti_dist` 返回标量 JAX 数组。
- `rti_dists` 返回形状为 `(4,)` 的 JAX 数组。
- `rti_dist_vec` 返回 `(距离, 向量)`，向量形状为 `(natom, 3)`。
- `rti_dists_vecs` 返回 `(距离数组, 向量数组)`，向量形状为 `(4, natom, 3)`。
- `rti_rot_tran` 返回 `(rotation, translation)`，形状分别为 `(3,3)` 和 `(3,)`。
- `rti_pot_force` 返回 `(能量, 力)`，力形状为 `(natom, 3)`。
- `Rtip0PES.get_energy(system)` 返回 Python `float`；`get_energy_force` 返回 `(float, JAX array)`。

## 状态与副作用

- 核心函数为纯函数、无副作用。
- `Rtip0PES` 为冻结 dataclass。
- 片段力散布使用 JAX `.at[indices].set`。
- 不使用 IO、CP2K、MPI 或随机态。

## 数值行为

- 特征值/特征向量的行为依赖于后端求解器。
- 特征向量符号在不同实现间可能不稳定，但所得旋转应保持对齐效果。
- 在简并或近简并情形下，选取的向量可能不同。
- 保留 Rust 在加权函数 `1/x^7` 中的奇异性；精确零距离可产生 inf/NaN。
- Stage 4.2 未加入特征值夹紧或 epsilon 稳定化。

## JAX 支持

- `rti_dist`, `rti_dists`, `rti_dist_vec`, `rti_dists_vecs`, `rti_rot_tran`, `rti_pot`, `rti_pot_force` 均使用 `jax.numpy` 编写，在固定原子数下应兼容 `jit`，受限于特征分解后端。
- `Rtip0PES` 方法返回 Python floats，并在过渡态矩阵上使用 Python 循环，因此不是主要的 `jit` 目标。

## 关键差异

- 将 Rust 可变循环替换为向量化 `jax.numpy` 操作。
- `rti_dists_vecs` 从 Rust 的 `Vec<Array2<f64>>` 变为 JAX 堆叠数组 `(4, natom, 3)`。
- Rust 的隐式 shape panic 改为 Python `ValueError`。
- PES 包装器将过渡态与 sigma 值存为不可变元组。

## 测试

- 测试位于 `tests/test_rtip.py`，覆盖权重公式、四元数-旋转恒等、平移与旋转不变性、`rti_rot_tran` 重构、RTI 向量范数一致性、加权高斯势与力的数值差分验证、以及片段力散布。
