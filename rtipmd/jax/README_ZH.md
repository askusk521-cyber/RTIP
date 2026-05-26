# RTIP JAX

本目录包含 Rust RTIP crate 的 Python + JAX 分阶段重写版本。

英文主说明见 [README.md](README.md)。AI agent 应阅读英文主说明；本文件面向中文读者。

历史迁移记录已经归档到 `../../history/migration/`。

迁移采用渐进方式。当前目录包含包骨架、公开数据/IO 模块、核心 RTIP/IDWM/optimization kernel、workflow runner，以及 CLI smoke 入口。

## 开发

```bash
python -m pip install -e ".[dev]"
pytest
```

导入 `rtip_jax` 时会默认启用 JAX x64。

常用 CLI smoke 命令：

```bash
rtip-jax show-default-config
rtip-jax cp2k-boundary --input cp2k.inp --output cp2k.out
rtip-jax deepmd-boundary --model model.pth --type-map O,H
rtip-jax synthesize --input input.xyz --output IS.xyz --mol-index "0,1;2,3" --dist 5.0 --seed 0
rtip-jax deepmd-pathway --input IS.xyz --model model.pth --type-map O,H --method rtip
rtip-jax deepmd-md --input IS.xyz --model model.pth --type-map O,H
```

## 当前范围

CP2K 不再迁移。原 CP2K 输入/输出和生命周期契约被保留为外部 PES 边界文档；真实 PES 能量和力的生产 provider 改为 DeePMD。

运行时 pathway 和 MD 函数接受通用 `PES` provider。内置 `HarmonicPES` 仅用于测试、示例和 CLI smoke 运行。

运行真实 DeePMD 模型工作流时安装 DeePMD 支持：

```bash
python -m pip install -e ".[deepmd]"
```

在 `n5` 上，系统 DeePMD 环境位于 `/group/software/deepmd-kit-3.1.1`，CUDA 位于 `/group/software/cuda-12.9.1`，DPA 模型位于 `/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt`。具体激活命令见 [USAGE_ZH.md](USAGE_ZH.md)。
