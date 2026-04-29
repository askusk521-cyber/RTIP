# 优化迁移说明（中文版）

Rust 源：`src/pes_exploration/optimization.rs`。

Python 目标：`src/rtip_jax/core/optimization.py`。

## Rust 语义要点

- `min_1d(fun, f0, epsilon)` 假定在零附近沿正方向函数下降。
- 初始步长 `delta_x = 0.01`。
- 若首次试探使函数值增加，则把步长反复除以 10，直到找到下降点或步长小于 `1e-15`。
- 若找不到下降点，Rust 会 panic（Python 抛出异常）。
- 在区间扩展阶段使用 `GOLDEN_RATIO2`，当函数持续下降且 `x1 < 0.1` 时扩大区间。
- 若函数在 `x1 >= 0.1` 时仍然下降，则返回受限步长。
- 否则使用黄金分割搜索直到相邻函数值差小于 `epsilon`。
- `min_1d_real_bias` 沿总力方向搜索并返回 `delta_coord`。

## 输入 / 输出

- `min_1d` 输入：可调用 `fun(x) -> scalar`、标量 `f0`、标量 `epsilon`。
- `min_1d` 输出：`(x_min, f_min)`，Python 浮点数。
- `min_1d_real_bias` 输入包括真实 PES、偏置 PES、系统坐标、总力 `force_total`、标量 `pot_total` 与 `epsilon`。
- `min_1d_real_bias` 输出：JAX 数组 `delta_coord`，形状 `(natom, 3)`。

## 状态与副作用

- `min_1d` 调用 Python 回调，无其他副作用。
- `min_1d_real_bias` 通过 `with_coord` 构造位移后的 `System`，不修改输入系统。
- 不涉及 IO、CP2K、MPI 或随机态。

## 数值行为

- 保留 Rust 的一维搜索策略，包括扩展区间时 `0.1` 的近似步长上限行为。
- 函数值转换为 Python 浮点以便控制流处理（不是 JAX 图中的原语）。
- 形状不匹配抛出 `ValueError`。
- 总力范数为零时抛出 `OptimizationError`（Rust 在该情形下可能出现除零/NaN）。

## JAX 支持与差异

- `min_1d` 和 `min_1d_real_bias` 不作为 `jit` 目标，因为使用了 Python 回调与动态控制流。
- 它们是工作流级别的辅助工具；未来可在确认行为后用 `lax` 控制流重写特定用例。

## 测试

- 测试位于 `tests/test_optimization.py`，覆盖二次函数一维最小化、初始函数增加时的拒绝、Rust 的步长上限行为、沿力方向的位移与力形状校验、以及零力方向拒绝。
