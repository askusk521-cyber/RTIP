# RTIP/JAX 中文使用说明

本文说明当前 Rust -> Python + JAX 迁移版本的完成度、环境配置、测试方法和基本使用方式。

## 完成度评估

当前 JAX 版本已经完成了 Rust RTIP 项目的主要结构迁移和核心功能重写：

- Python 包结构：已完成，包名为 `rtip_jax`。
- JAX x64：已默认启用，以贴近 Rust `f64` 数值行为。
- 基础公共模块：已完成，包括常数、元素、质量、`Para` 配置、`System`、XYZ/PDB IO。
- 核心算法：已完成，包括 IDWM、RTIP、局部一维优化、RTIP/IDWM PES wrapper。
- workflow 层：已完成可运行版本，包括 repulsive RTIP/IDWM pathway、attractive RTIP pathway、synthesis pathway、RTIP NVT MD。
- DeepMD 替代 CP2K：已完成 provider 接口和测试。CP2K 只保留为历史边界文档，真实能量/力来源改为 DeePMD。
- CLI：已完成基础入口，包括配置输出、合成布局、mock 运行、DeepMD pathway、DeepMD MD。
- 测试：远端 `n5:/home/lhshen/RTIP/JAX` 已通过完整测试，结果为 `83 passed`。

仍需补齐或进一步验证的部分：

- 尚未用真实 DeePMD 模型跑科学基准。
- 尚未建立 Rust 原版 vs JAX 版的完整轨迹/标量日志对照。
- MPI rank fan-out 没有直接迁移；当前设计是普通 Python 编排层，后续如需并行调度应另包一层。
- Stage 7 的 JAX 原生性能优化尚未开始，当前优先保证行为清晰和可测试。

因此，当前完成度可以理解为：

- 工程迁移：基本完成。
- 单元/集成测试：已完成当前可测范围。
- DeepMD provider 替代 CP2K：接口和单位契约完成，真实模型验证待做。
- 科学数值验收：还需要真实模型和 Rust/JAX 对照样例。

## 目录结构

主要目录如下：

```text
RTIP/JAX/
  pyproject.toml
  README.md
  USAGE_ZH.md
  DEEPMD_INTO.md
  src/rtip_jax/
    constants.py
    config.py
    system.py
    core/
      idwm.py
      rtip.py
      optimization.py
    pes/
      base.py
      bias.py
    workflows/
      synthesis.py
      pathway_sampling.py
      md.py
    external/
      cp2k.py
      deepmd.py
    io/
      xyz.py
      pdb.py
      outputs.py
  tests/
```

## 环境配置

推荐使用 Python 3.10 或更高版本。当前远端验证使用的是 Python 3.10。

### 1. 创建虚拟环境

在 `RTIP/JAX` 目录下执行：

```bash
python -m venv .venv
source .venv/bin/activate
```

如果是在远端 `n5`，可以使用已有 conda 环境里的 Python 创建 venv，例如：

```bash
cd /home/lhshen/RTIP/JAX
/home/lhshen/miniconda3/envs/flower/bin/python -m venv .venv
source .venv/bin/activate
```

### 2. 安装开发依赖

```bash
python -m pip install -e ".[dev]"
```

这会安装：

- `jax[cpu]`
- `numpy`
- `pytest`
- `pytest-cov`

### 3. 安装 DeepMD 支持

如果要运行真实 DeePMD 模型，需要额外安装：

```bash
python -m pip install -e ".[deepmd]"
```

如果同时需要开发测试和 DeepMD：

```bash
python -m pip install -e ".[dev,deepmd]"
```

注意：单元测试不强制依赖真实 `deepmd-kit`，测试中使用 fake `DeepPot` 验证接口和单位转换。

## n5 上的 DeepMD 生产环境

当前 `n5` 已有可用的 DeePMD-kit 环境和预训练模型：

```text
DeePMD-kit: /group/software/deepmd-kit-3.1.1
CUDA:       /group/software/cuda-12.9.1
Model:      /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt
```

激活 CUDA 和 DeePMD 环境：

```bash
export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source /group/software/deepmd-kit-3.1.1/bin/activate
```

因为这个 DeepMD 环境不是通过 `pip install -e` 安装当前 JAX 项目的，所以在该环境中运行本项目时建议显式设置：

```bash
export PYTHONPATH=/home/lhshen/RTIP/JAX/src${PYTHONPATH:+:${PYTHONPATH}}
```

然后用模块方式调用 CLI：

```bash
python -m rtip_jax.cli deepmd-boundary \
  --model /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt
```

DPA-3.2-5M 模型已验证可以加载，并且模型本身暴露完整元素 `type_map`。因此使用这个模型时 `--type-map` 可以省略；如果为了可读性显式提供，也必须与模型元素顺序一致。

在登录节点上测试时可能看到类似 CUDA no device 的提示，这是因为登录节点没有可用 GPU，JAX 会回退 CPU。真正使用 CUDA 时应在有 GPU 的作业或计算节点上运行同样的环境激活命令。

已在 `n5` 做过最小真实模型 smoke 测试：

```text
energy_hartree -0.5210642014266953
force_shape (3, 3)
force_norm 0.024207633639464484
```

该 smoke 测试只说明 DeepMD provider、单位转换和模型加载链路可运行，不代表科学基准已经完成。

## 测试

运行完整测试：

```bash
cd RTIP/JAX
source .venv/bin/activate
pytest
```

远端 `n5` 已验证：

```text
83 passed
```

也可以先做语法级检查：

```bash
python -m compileall src tests
```

## 输入单位和输出单位

JAX 版本内部沿用 Rust/CP2K 风格的原子单位：

- 坐标：Bohr
- 能量：Hartree
- 力：Hartree/Bohr

文本 IO 边界：

- XYZ 输入/输出使用 Angstrom。
- 读入 XYZ 后会转换为内部 Bohr。
- 写出 XYZ/PDB 时会转换回 Angstrom。

DeepMD 边界：

- DeePMD 输入坐标：Angstrom
- DeePMD 输入 cell：Angstrom，非周期体系传 `None`
- DeePMD 输出能量：eV，转换为 Hartree
- DeePMD 输出力：eV/Angstrom，转换为 Hartree/Bohr
- DeePMD 输出 virial：eV，转换为 Hartree 并保留在 `DeepMDResult`

## 配置参数

默认参数对应 Rust `Para::new()`：

```bash
rtip-jax show-default-config
```

可以保存为 JSON，例如 `para.json`：

```json
{
  "a0": 0.002,
  "scale_ts_a0": 1.0,
  "scale_ts_sigma": 0.25,
  "max_step": 1500,
  "print_step": 1,
  "pot_climb": 0.185,
  "pot_drop": 0.02,
  "pot_epsilon": 0.00005,
  "f_epsilon": 0.001,
  "dt": 0.5,
  "tau": 500.0,
  "temp_bath": 1000.0
}
```

运行时通过 `--config para.json` 使用。

## CLI 使用

### 查看 CP2K 历史边界

CP2K 不再作为真实 provider，只保留历史输入/输出契约：

```bash
rtip-jax cp2k-boundary --input cp2k.inp --output cp2k.out
```

### 查看 DeepMD 边界

```bash
rtip-jax deepmd-boundary --model model.pth --type-map O,H
```

在 `n5` 的系统 DeePMD 环境中：

```bash
python -m rtip_jax.cli deepmd-boundary \
  --model /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt
```

`--type-map` 必须与 DeePMD 模型训练时的元素顺序一致。比如模型的 `type_map = ["O", "H"]`，则：

- O -> `atype = 0`
- H -> `atype = 1`

### 生成 synthesis 初始布局

```bash
rtip-jax synthesize \
  --input input.xyz \
  --output IS.xyz \
  --mol-index "0,1;2,3" \
  --dist 5.0 \
  --seed 0
```

`--mol-index` 用分号分隔分子，用逗号分隔原子编号。上例表示：

- 分子 1：原子 0、1
- 分子 2：原子 2、3

当前 synthesis 布局支持 2、3、4 个分子。

### 使用 mock PES 做 pathway smoke 测试

```bash
rtip-jax mock-pathway \
  --input IS.xyz \
  --output-dir run_mock_pathway \
  --method rtip \
  --max-step 5 \
  --seed 0
```

也可以测试 IDWM：

```bash
rtip-jax mock-pathway \
  --input IS.xyz \
  --output-dir run_mock_idwm \
  --method idwm \
  --max-step 5 \
  --seed 0
```

mock PES 只用于软件流程测试，不代表真实势能面。

### 使用 DeePMD 跑 pathway

```bash
rtip-jax deepmd-pathway \
  --input IS.xyz \
  --model model.pth \
  --type-map O,H \
  --output-dir run_deepmd_pathway \
  --method rtip \
  --config para.json \
  --seed 0
```

也可以把 `--method rtip` 改为 `--method idwm`。

在 `n5` 的系统 DeePMD 环境中推荐写法：

```bash
python -m rtip_jax.cli deepmd-pathway \
  --input IS.xyz \
  --model /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt \
  --output-dir run_deepmd_pathway \
  --method rtip \
  --config para.json \
  --seed 0
```

### 使用 DeePMD 跑 RTIP NVT MD

```bash
rtip-jax deepmd-md \
  --input IS.xyz \
  --model model.pth \
  --type-map O,H \
  --output-dir run_deepmd_md \
  --config para.json \
  --seed 0
```

在 `n5` 的系统 DeePMD 环境中推荐写法：

```bash
python -m rtip_jax.cli deepmd-md \
  --input IS.xyz \
  --model /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt \
  --output-dir run_deepmd_md \
  --config para.json \
  --seed 0
```

如果不希望在 MD 初始结构上加入随机扰动：

```bash
rtip-jax deepmd-md \
  --input IS.xyz \
  --model model.pth \
  --type-map O,H \
  --output-dir run_deepmd_md \
  --no-perturb
```

## n5 Slurm 调度

已在 `n5` 的项目目录写好调度脚本：

```text
/home/lhshen/RTIP/JAX/run_deepmd_rtip.slurm
```

该脚本默认设置：

- 分区：`main`
- GPU：`--gres=gpu:1`
- CPU：`-c 10`
- 时间：`24:00:00`
- DeepMD 环境：`/group/software/deepmd-kit-3.1.1`
- CUDA：`/group/software/cuda-12.9.1`
- 默认模型：`/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt`

提交 RTIP pathway：

```bash
cd /home/lhshen/RTIP/JAX
INPUT=IS.xyz OUTPUT_DIR=run_pathway MAX_STEP=100 sbatch run_deepmd_rtip.slurm
```

提交 IDWM pathway：

```bash
cd /home/lhshen/RTIP/JAX
INPUT=IS.xyz METHOD=idwm OUTPUT_DIR=run_idwm MAX_STEP=100 sbatch run_deepmd_rtip.slurm
```

提交 RTIP NVT MD：

```bash
cd /home/lhshen/RTIP/JAX
MODE=md INPUT=IS.xyz OUTPUT_DIR=run_md MAX_STEP=1000 sbatch run_deepmd_rtip.slurm
```

使用配置文件：

```bash
cd /home/lhshen/RTIP/JAX
INPUT=IS.xyz CONFIG=para.json OUTPUT_DIR=run_pathway sbatch run_deepmd_rtip.slurm
```

如果输入结构或模型不在默认位置，可以覆盖变量：

```bash
INPUT=/path/to/input.xyz \
MODEL=/path/to/model.pt \
OUTPUT_DIR=run_custom \
sbatch run_deepmd_rtip.slurm
```

## 输出文件

workflow 默认输出：

```text
rtip.pdb
rtip.out
```

如果指定 `--output-dir run1`，则输出到：

```text
run1/rtip.pdb
run1/rtip.out
```

`rtip.out` 中记录 step、RTI/IDW 距离、真实 PES 能量、bias 能量、真实 PES 力范数、bias 力范数等标量。

## Python API 示例

### 读取结构

```python
from rtip_jax.system import System

system = System.read_xyz("IS.xyz")
```

### 构造 DeepMD PES

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

### 跑 repulsive RTIP pathway

```python
from jax import random
from rtip_jax.config import Para
from rtip_jax.pes import RepulsivePot
from rtip_jax.workflows import run_rtip_repulsive_path_sampling

config = RepulsivePot(
    local_min=system,
    nearby_ts=(),
    para=Para(max_step=100),
    str_output_file="rtip.pdb",
    output_file="rtip.out",
)

result = run_rtip_repulsive_path_sampling(
    config,
    pes,
    key=random.PRNGKey(0),
)

final_system = result.system
history = result.history
```

### 跑 RTIP NVT MD

```python
from jax import random
from rtip_jax.workflows import run_rtip_nvt_md

result = run_rtip_nvt_md(
    config,
    pes,
    key=random.PRNGKey(0),
)
```

## DeepMD 替代 CP2K 的确认点

原 Rust workflow 中 CP2K 需要提供的核心数据是：

- `get_energy(system) -> f64`
- `get_energy_force(system) -> (f64, Array2<f64>)`

现在由 `DeepMDPES` 提供：

- `get_energy(system) -> float`
- `get_energy_force(system) -> tuple[float, force]`

并且测试已覆盖：

- DeepMD 输入数组 shape
- 坐标和 cell 单位转换
- 非周期体系 `cell=None`
- 元素到 `atype` 的映射
- 能量/力/virial 单位转换
- DeepMD provider 能进入原先 CP2K 所在的 workflow 调用位置

## 常见问题

### 1. `ModuleNotFoundError: No module named 'deepmd'`

说明没有安装 DeePMD 支持。执行：

```bash
python -m pip install -e ".[deepmd]"
```

### 2. 元素不在 `type_map` 里

如果结构里有 O/H/Ca，但命令只写了：

```bash
--type-map O,H
```

遇到 Ca 时会报错。需要改为：

```bash
--type-map Ca,O,H
```

顺序必须与 DeePMD 模型训练时一致。

### 3. 单原子 MD 温度报错

MD 温度公式使用 `3 * (natom - 1)` 自由度。当前 Python 版显式拒绝单原子温度计算，避免 Rust 中可能出现的除零行为。

### 4. RTIP 距离本应为 0 却出现 NaN

这个问题已经修复。JAX eigensolver 可能给半正定矩阵返回极小负特征值，当前实现会在开方前裁剪到非负。

## 后续建议

1. 提供真实 DeePMD 模型和对应 `type_map`。
2. 准备 2-3 个小体系 XYZ 作为官方 smoke examples。
3. 用 Rust 原版或已知参考数据生成标量基线。
4. 对比 JAX 版的 `rtip.out`、最终结构和能量/力统计。
5. 在行为稳定后再进入 JAX native 优化阶段。
