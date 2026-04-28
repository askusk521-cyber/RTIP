# RTIP JAX

This directory contains the staged Python + JAX rewrite of the Rust RTIP crate.

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
rtip-jax synthesize --input input.xyz --output IS.xyz --mol-index "0,1;2,3" --dist 5.0 --seed 0
```

## Current Scope

CP2K is not being ported. The original CP2K input/output and lifecycle contract
is documented behind an external PES boundary so another energy/force provider
can replace it later.

Runtime pathway and MD functions accept a generic `PES` provider. The included
`HarmonicPES` is only for tests, examples, and CLI smoke runs.
