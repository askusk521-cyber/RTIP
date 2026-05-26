# IDWM 迁移说明（中文版）

Rust 源：`src/pes_exploration/idwm.rs`。

Python 目标：`src/rtip_jax/core/idwm.py`。

## Rust 语义

- `w(x)` 计算 `exp(-(x / 3)^5) + 1`。
- `dw(x)` 计算 `w(x)` 的导数。
- `wei_dist_mat(coord)` 只填充 `natom x natom` 加权距离矩阵的上三角。
- `idw_dist(w_mat0, coord)` 将参考加权矩阵与 `coord` 的加权距离比较，仅使用上三角。
- `idw_dist1(w_mat0, w_mat1)` 比较两个加权矩阵，仅使用上三角。
- `idw_dist_vec(w_mat0, coord)` 返回 IDW 距离和由 Rust 力表达式使用的标准化坐标向量。
- `idw_pot_force` 计算高斯势 `a * exp(-dist^2 / (2*sigma^2))` 及力 `vec * pot * dist / sigma^2`。
- `Idwm0PES` 存储局部极小和附近过渡态的加权矩阵，并在完整系统或 `system.atom_add_pot` 片段上评估能量和力。

## 输入 / 输出

- 坐标输入形状：`(natom, 3)`，内部单位为 Bohr。
- 加权矩阵输入形状：`(natom, natom)`，上三角有意义。
- `idw_dist` 和 `idw_dist1` 返回标量 JAX 数组。
- `idw_dist_vec` 返回 `(距离, 向量)`，向量形状为 `(natom, 3)`。
- `idw_pot_force` 返回 `(能量, 力)`，力形状为 `(natom, 3)`。
- `Idwm0PES.get_energy(system)` 返回 Python `float`。
- `Idwm0PES.get_energy_force(system)` 返回 `(float, JAX array)`。

## 状态与副作用

- 核心函数是纯函数且无副作用。
- `Idwm0PES` 为冻结的 dataclass，在 `__post_init__` 中归一化矩阵。
- 片段力散布使用 JAX 的 `.at[indices].set`。
- 本模块不涉及 IO、CP2K、MPI 或随机态。

## 数值行为

- 保留 Rust 的零距离奇异性：如果 `idw_dist_vec` 收到与参考完全相同的构型，除零归一化可能产生 NaN。
- Stage 4.1 未加入 epsilon 稳定化。
- 参考矩阵中下三角数据被忽略，匹配 Rust 的循环行为。

## JAX 支持

- `weight`, `weight_derivative`, `wei_dist_mat`, `idw_dist`, `idw_dist1`,
  `idw_dist_vec`, `idw_pot`, `idw_pot_force` 使用 `jax.numpy` 编写，应兼容在固定形状下 `jit`。
- 在外部包装器上使用 `vmap` 应可行（前提是传入固定大小坐标）。
- `Idwm0PES` 方法返回 Python floats，并在过渡态矩阵上使用 Python 循环，因此属于工作流层面的包装器而非主要的 `jit` 目标。

## 与 Rust 的关键差异

- 将 Rust 的可变循环替换为向量化的 `jax.numpy` 操作。
- 用单一的成对距离张量替代 Rust 的临时 `Array2<Array1<f64>>` 存储。
- Rust 的 shape panic 在 Python 中改为 `ValueError`。
- PES 包装器以元组形式存储，不使用可变 `Vec` 状态。

## 测试

- 测试位于 `tests/test_idwm.py`，覆盖权重与导数公式、上三角矩阵行为、
  矩阵-坐标与矩阵-矩阵距离、高斯能量公式、力方向与数值差分验证、以及片段力散布等。
