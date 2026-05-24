# MD 帮助函数迁移说明（中文版）

本模块从 Rust 的 `src/pes_exploration/md.rs` 迁移纯 MD 辅助数值逻辑。

## 范围

- 原子质量提取。
- 两倍动能：`sum_i m_i |v_i|^2`。
- 动能：`0.5 * sum_i m_i |v_i|^2`。
- 温度计算公式。
- Berendsen 恒温器缩放因子。
- 由力/质量计算加速度。
- Leapfrog 的第一与第二半步更新公式。

## 保留的 Rust 语义

- 速度与加速度以原子单位（atomic units）表示。
- 时间步 `dt` 以飞秒为输入并用 `FEMTOSECOND_TO_AU` 转换为原子单位。
- Rust 的温度使用 `sum_i m_i |v_i|^2`（不是半动能的形式）。
- Rust 在 Berendsen 缩放前对温度施加 `1.0 K` 的下限。
- 第二个 leapfrog 半步在更新速度后应用恒温器缩放。

## 与 Rust 的关键差异

- Python 辅助函数返回数组/标量，不会修改 `System`。
- `temperature` 在单原子系统上会拒绝（Rust 使用 `3 * (natom - 1)`，单原子会除以 0）。
- 完整的 MD 循环副作用被刻意省略。

## 测试

- 测试位于 `tests/test_md_helpers.py`，覆盖质量提取、动能/温度公式、
  单原子温度拒绝、Berendsen 温度下限、加速度计算以及 leapfrog 步骤。
