# DeePMD 集成说明

本文记录用 DeePMD-kit 推理替代旧 CP2K 真实 PES provider 的情况。

文件名沿用已请求的 `DEEPMD_INTO.md` 拼写。英文主文档见 [DEEPMD_INTO.md](DEEPMD_INTO.md)。

## 目标

DeePMD 是原 Rust workflow 中 CP2K provider 的生产替代。RTIP/JAX 核心仍然只依赖通用 `PES` 协议：

- `get_energy(system) -> float`
- `get_energy_force(system) -> tuple[float, force]`

满足该协议的外部 provider 是 `rtip_jax.external.deepmd.DeepMDPES`。

## 已替换的 Rust CP2K 契约

Rust `Cp2kPES` 提供：

- 来自 `System.coord` 的坐标。
- 可选 cell。
- 势能。
- 原子力。

JAX DeePMD provider 向 workflow 提供同等数据：

- 输入坐标：内部 Bohr 转 DeePMD Angstrom。
- 输入 cell：内部 Bohr 转 Angstrom；孤立体系传 `None`。
- 原子类型：根据模型 `type_map` 把 `System.atom_type` 符号转成 DeePMD `atype` 索引。
- 输出能量：eV 转 Hartree。
- 输出力：eV/Angstrom 转 Hartree/Bohr。
- 输出 virial：eV 转 Hartree，并保存在 `DeepMDResult`；当前 RTIP workflow 尚不消费 virial。

## Python API

```python
from rtip_jax.external import DeepMDBoundary, DeepMDPES

pes = DeepMDPES(
    DeepMDBoundary(
        model="model.pth",
        type_map=("O", "H"),
    )
)

energy, force = pes.get_energy_force(system)
```

如果省略 `type_map`，`DeepMDPES` 会尝试调用 `deep_pot.get_type_map()`。建议显式提供 `type_map`，因为这样原子索引契约更清晰且可测试。

## CLI

```bash
rtip-jax deepmd-boundary --model model.pth --type-map O,H
rtip-jax deepmd-pathway --input IS.xyz --model model.pth --type-map O,H --method rtip
rtip-jax deepmd-md --input IS.xyz --model model.pth --type-map O,H
```

旧的 `cp2k-boundary` 命令只保留为历史契约文档，不是生产 provider。

## n5 环境

当前 `n5` 安装提供：

- DeePMD-kit：`/group/software/deepmd-kit-3.1.1`
- CUDA：`/group/software/cuda-12.9.1`
- 模型：`/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt`

激活：

```bash
export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source /group/software/deepmd-kit-3.1.1/bin/activate
export PYTHONPATH=/home/lhshen/RTIP/rtipmd/jax/src${PYTHONPATH:+:${PYTHONPATH}}
```

通过模块入口运行：

```bash
python -m rtip_jax.cli deepmd-boundary \
  --model /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt
```

DPA-3.2-5M 已在 `n5` 做过 smoke 测试，并暴露完整周期表 `type_map`，因此该模型可省略 `--type-map`。登录节点没有 CUDA 设备时可能回退 CPU；生产 CUDA 运行应使用 GPU 作业或 GPU 计算节点。

## 依赖

DeePMD-kit 是可选依赖，因为测试可以用注入的 fake model 验证边界：

```bash
python -m pip install -e ".[deepmd]"
```

provider 仅在构造真实模型支持的 `DeepMDPES` 且没有注入 `deep_pot` 时导入 `deepmd.infer.DeepPot`。

## 测试

`tests/test_external_deepmd_provider.py` 覆盖：

- Bohr 到 Angstrom 的坐标转换。
- Bohr 到 Angstrom 的 cell 转换。
- 非周期 `cell=None`。
- `System.atom_type` 到 DeePMD `atype` 的 `type_map` 映射。
- eV 到 Hartree 的能量转换。
- eV/Angstrom 到 Hartree/Bohr 的力转换。
- eV 到 Hartree 的 virial 转换。
- DeePMD-backed `PES` 可进入原 CP2K 所在 workflow 位置。

`tests/test_stage5_config_outputs_cli.py` 覆盖 `deepmd-boundary` 命令和 type-map 解析。

## 假设和风险

- DeePMD 模型必须兼容 `System.atom_type` 中的化学元素。
- DeePMD 模型的能量参考可能与 CP2K 不同；完整科学验证需要标注结构或旧 CP2K 对比数据。
- 当前 workflow 不消费 virial，但 provider 会把转换后的 virial 存在 `DeepMDResult` 中，以备未来扩展。
- DeePMD 和 RTIP/JAX 可能运行在不同数组后端；边界对 DeepPot 输入使用 NumPy 数组，对返回的力使用 JAX 数组。
