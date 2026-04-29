# 模块映射（中文版）

该文件将 Rust 模块映射到计划中的 Python + JAX 对应模块，并注明阶段与说明。

（摘要）

| Rust 源 | Python/JAX 目标 | 阶段 | 说明 |
| --- | --- | --- | --- |
| `src/common/constants.rs` | `src/rtip_jax/constants.py` | 3 | 物理常数与单位换算 |
| `src/common/error.rs` | `src/rtip_jax/errors.py` | 3 | 将 panic 替换为 Python 异常 |
| `src/io/input.rs` | `src/rtip_jax/config.py` | 3,5 | `Para` 为 dataclass |
| `src/io/output.rs` | `src/rtip_jax/io/outputs.py` | 5 | 输出路径帮助器 |
| `src/pes_exploration/system.rs` | `src/rtip_jax/system.py` 等 | 3 | System 数据与 IO |
| `src/matrix/mod.rs` | `src/rtip_jax/math/rotations.py` | 3 | 随机旋转需显式 PRNG key |
| `src/pes_exploration/traits.rs` | `src/rtip_jax/pes/base.py` | 2,4 | Trait -> Protocol |
| `src/pes_exploration/potential.rs` | `src/rtip_jax/pes/bias.py` | 4 | 偏置 PES 容器 |
| `src/pes_exploration/idwm.rs` | `src/rtip_jax/core/idwm.py` | 4 | IDWM 已实现 |
| `src/pes_exploration/rtip.rs` | `src/rtip_jax/core/rtip.py` | 4 | RTIP 已实现 |
| `src/pes_exploration/optimization.rs` | `src/rtip_jax/core/optimization.py` | 4 | 优化已实现 |
| `src/pes_exploration/synthesis.rs` | `src/rtip_jax/workflows/synthesis.py` | 4,5 | 合成布局 |
| `src/pes_exploration/pathway_sampling.rs` | `src/rtip_jax/workflows/pathway_sampling.py` | 4,5 | 途径采样 |
| `src/pes_exploration/md.rs` | `src/rtip_jax/workflows/md.py` | 4,5 | MD 辅助与循环 |
| `src/external/cp2k.rs` | `src/rtip_jax/external/cp2k.py` | 2 | CP2K 边界文档（未移植） |

更多细节参见原始 MODULE_MAPPING.md。
