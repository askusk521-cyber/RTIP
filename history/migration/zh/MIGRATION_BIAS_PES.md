# 偏置 PES 容器迁移说明（中文版）

Rust 源：`src/pes_exploration/potential.rs`。

Python 目标：

- `src/rtip_jax/pes/bias.py`
- `src/rtip_jax/pes/base.py`

## Rust 语义

- `RepulsivePot` 存储：局部极小 `System`、附近的过渡态 `System` 列表、`Para`、结构输出文件名和标量输出文件名。
- `AttractivePot` 存储：初始与最终 `System`、`Para`、输出文件名。
- `SynthesisPot` 存储：初始 `System`、分子原子索引组、`Para`、输出文件名。

这些 Rust 结构体是工作流层面的配置容器，它们不直接实现 `PES` trait。

## 输入 / 输出

- `RepulsivePot(local_min, nearby_ts, para, str_output_file, output_file)`。
- `AttractivePot(initial_state, final_state, para, str_output_file, output_file)`。
- `SynthesisPot(initial_state, mol_index, para, str_output_file, output_file)`。
- `nearby_ts` 在 Python 端归一化为元组；`mol_index` 规范为嵌套整数组元组。

## 状态与副作用

- Python 容器为冻结 dataclass，只做验证/归一化。
- 不在此模块中执行 IO、CP2K、MPI、随机或数值算法。

## JAX 支持

- 这些容器属于编排层面，不作为 `jit` 目标。
- 包含的 `System.coord` 仍为 JAX 数组。

## 关键差异

- Rust 的生命周期引用被替换为普通的 Python dataclass 字段。
- 提供默认 `Para()` 与输出文件名以便更便捷构造。
- `SynthesisPot` 在构造时会验证 `mol_index` 的有效性。
- 新增 Python-only 的 `SumPES` 用于能量/力求和，Rust 中无直接对应类型。

## 测试

- 测试位于 `tests/test_bias_pes.py`，覆盖 `SumPES` 的能量/力求和、空 `SumPES` 的拒绝、`RepulsivePot` 的近邻过渡态归一化、以及 `AttractivePot` 与 `SynthesisPot` 的字段与验证。
