# Stage 5：工作流与入口迁移（中文版）

## 范围

Stage 5 将已迁移的核心模块以普通 Python 工作流与 CLI 串联起来。CP2K 仍未移植；工作流接受通用 `PES` 提供者，测试/CLI 使用确定性的 `HarmonicPES`。

## Rust -> JAX 对应关系（要点）

- 参数加载：`Para` 对应 `rtip_jax.config.load_para`。
- 输出：`output_rtip` 与 `output_cp2k` 对应 `rtip_jax.io.outputs`。
- 路径采样与 MD 运行器映射到 `run_*` 的 Python 函数。
- `src/lib*.rs` 的入口函数对应 `rtip_jax.cli` 子命令。

## 输入 / 输出

- 路径采样运行器接受 bias 配置（`RepulsivePot`/`AttractivePot`/`SynthesisPot`）和一个真实 `PES` 提供者。
- 需要显式 JAX PRNG key 以生成与 Rust 相似的初始扰动（可复现）。
- 运行器返回 `PathwayResult` 或 `MDResult`，并在 `write_outputs=True` 时写入 legacy 风格的 `rtip.pdb` 与 `rtip.out`。

## 状态与副作用

- 核心数值状态保持不可变；每次坐标更新返回新的 `System`。
- 文件输出由 `write_outputs` 控制，测试可禁用。
- MPI 根进程行为未在核心中重现；如需分布式，应在外部封装。

## 数值注记

- 工作流循环使用普通 Python 控制流，不作为 `jit` 目标。
- 当偏置被禁用时使用 `ZeroPES` 避免对零振幅求值导致的奇异算术。
- 随机扰动使用显式 JAX keys，因此可复现但不与 Rust 的随机流逐位相同。

## JIT / VMAP / Scan

- 完整的路径与 MD 运行器为 Python 协调代码，不作为 `jit` 目标。
- 单个核心函数仍适合在之后考虑 `jit`、`vmap` 或 `scan` 优化。

## 测试（已添加）

- `tests/test_stage5_config_outputs_cli.py`
- `tests/test_pathway_workflows.py`
- `tests/test_md_workflow.py`
