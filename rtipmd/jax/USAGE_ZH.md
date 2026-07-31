# RTIP-JAX 使用说明

本包是上游 Rust RTIP-MD 程序的 Python + JAX 移植，仅保留与上游算法一致
的部分（不含反应坐标偏置 RCMD、size-scaling、振荡、提前停止等工程附加
功能）。formose 复现工作流见仓库根目录 `formose/`。

## 环境（n5）

```bash
# 测试环境（Python 3.10 + jax 0.6.2）
cd /home/lhshen/RTIP/rtipmd/jax
PYTHONPATH=src /home/lhshen/RTIP/JAX/.venv/bin/python -m pytest -q

# GPU 运行环境（Python 3.12 + deepmd-kit 3.1.1 + jax 0.7.2/CUDA）
export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source /group/software/deepmd-kit-3.1.1/bin/activate
export PYTHONPATH=/home/lhshen/RTIP/rtipmd/jax/src${PYTHONPATH:+:${PYTHONPATH}}
```

登录节点无 GPU，实际运行一律通过 slurm 提交（见 `formose/run_formose.slurm`）。

## CLI

```bash
python -m rtip_jax.cli show-default-config
python -m rtip_jax.cli deepmd-evolution-md \
  --input formose/data/box_seed0.xyz \
  --model /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt \
  --config formose/para_formose.json --max-step 10000 --output-dir runs/seed0
```

内部单位为 Bohr/Hartree，XYZ 文件使用 Angstrom，DeePMD 使用 eV/Angstrom，
单位换算全部在 IO 边界完成。JAX x64 在导入时自动开启。
