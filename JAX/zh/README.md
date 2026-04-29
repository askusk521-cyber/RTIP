# RTIP JAX（中文版）

该目录包含从 Rust RTIP crate 至 Python + JAX 的分阶段重写内容。

迁移采取渐进式策略，目前包括包骨架、公共数据/IO 模块、核心 RTIP/IDWM/优化内核、工作流运行器与 CLI 烟雾入口。

## 开发

建议在具备依赖的远端环境中运行：

```bash
python -m pip install -e ".[dev]"
pytest
```

导入时会默认启用 JAX x64（在有 JAX 的环境下）。

示例 CLI：

```bash
rtip-jax show-default-config
rtip-jax cp2k-boundary --input cp2k.inp --output cp2k.out
rtip-jax synthesize --input input.xyz --output IS.xyz --mol-index "0,1;2,3" --dist 5.0 --seed 0
```

## 当前范围

不移植 CP2K；保留 CP2K 的输入/输出/单位契约以便将来替换真实 PES 提供者。
工作流接受通用 `PES` 提供者；包含的 `HarmonicPES` 仅用于测试与示例。
