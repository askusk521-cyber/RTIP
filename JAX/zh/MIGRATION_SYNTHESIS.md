# 合成迁移说明（中文版）

Rust 源：`src/pes_exploration/synthesis.rs`。

Python 目标：`src/rtip_jax/workflows/synthesis.py`。

## Rust 语义

- 支持恰好 2、3 或 4 个分子。
- 从 `System` 提取每个分子的坐标并以其几何中心重心化。
- 对每个分子应用随机旋转矩阵。
- 将分子中心放置在固定的 2 体、三角（3 体）或四面体（4 体）偏移位置上并按 `dist` 缩放。
- 原始实现在原地修改输入 `System`。
- 使用 MPI 广播以使所有进程收到根进程的随机旋转。

## Python 语义

- `synthesize_layout(system, mol_index, dist, key=...)` 返回新的 `System`。
- 可通过 `rotations=` 提供确定性旋转以便测试。
- `key` 与 `rotations` 必须二选一提供。
- 随机旋转使用显式 JAX PRNG keys。
- 不实现 MPI 行为（此为纯辅助函数）。

## 输入 / 输出

- `system`：`System`，坐标形状 `(natom, 3)`。
- `mol_index`：2/3/4 个非空的原子索引组。
- `dist`：放置距离标量。
- 输出：返回一个新的 `System`，坐标已更新，元数据保留。

## 状态与副作用

- 纯函数，返回替换后的 `System`。
- 无 IO、CP2K、MPI 或全局随机态副作用。

## 数值行为

- 放置常数沿用 Rust 的固定值。
- 随机流与 Rust 的 `ndarray-rand` 不同，但使用显式 key 保证 Python 端可复现性。

## 测试

- 测试位于 `tests/test_synthesis.py`，覆盖 2/3/4 分子偏移、单位旋转的确定性布局、显式 key 的可复现性与输入验证。
