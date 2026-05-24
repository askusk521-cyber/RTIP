# RTIP JAX

This directory contains the staged Python + JAX rewrite of the Rust RTIP crate.

中文使用说明见 [USAGE_ZH.md](USAGE_ZH.md)。

Historical migration notes have been archived under `../HISTORY/JAX/`.

The migration is intentionally incremental. The current tree includes the
package skeleton, public data/IO modules, core RTIP/IDWM/optimization kernels,
workflow runners, and CLI smoke entry points.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

JAX x64 is enabled when `rtip_jax` is imported.

Useful CLI smoke commands:

```bash
rtip-jax show-default-config
rtip-jax cp2k-boundary --input cp2k.inp --output cp2k.out
rtip-jax deepmd-boundary --model model.pth --type-map O,H
rtip-jax synthesize --input input.xyz --output IS.xyz --mol-index "0,1;2,3" --dist 5.0 --seed 0
rtip-jax deepmd-pathway --input IS.xyz --model model.pth --type-map O,H --method rtip
rtip-jax deepmd-md --input IS.xyz --model model.pth --type-map O,H
```

## Current Scope

CP2K is not being ported. The original CP2K input/output and lifecycle contract
is documented behind an external PES boundary, and DeePMD is the production
replacement provider for real PES energy and force data.

Runtime pathway and MD functions accept a generic `PES` provider. The included
`HarmonicPES` is only for tests, examples, and CLI smoke runs.

Install DeePMD support when running real model-backed workflows:

```bash
python -m pip install -e ".[deepmd]"
```

On `n5`, a system DeePMD environment is available at
`/group/software/deepmd-kit-3.1.1`, with CUDA at
`/group/software/cuda-12.9.1` and the DPA model at
`/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt`. See `USAGE_ZH.md` for the exact
activation commands.
