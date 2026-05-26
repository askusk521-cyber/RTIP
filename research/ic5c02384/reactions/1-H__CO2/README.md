# 1-H + CO2

This directory is prepared for the RTIP/JAX `synthesize --inputs` entry point.

Chinese companion: [README_ZH.md](README_ZH.md).

- `1.xyz`: first reactant.
- `2.xyz`: second reactant.
- `product.xyz`: optimized product structure from the SI, if present.
- `ts.xyz`: transition-state structure from the SI, if present.

Example:

```bash
cd JAX
python -m rtip_jax.cli synthesize --inputs examples/ic5c02384/reactions/1-H__CO2/1.xyz examples/ic5c02384/reactions/1-H__CO2/2.xyz --output examples/ic5c02384/reactions/1-H__CO2/IS.xyz --dist 5.0 --seed 0
```

If a compatible DeePMD model is available, the generated `IS.xyz` can be used as
the `--input` for `deepmd-pathway`.
