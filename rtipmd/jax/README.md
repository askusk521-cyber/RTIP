# RTIP-JAX

Python + JAX implementation of the upstream Rust RTIP-MD program
(github.com/MillenniumDream/RTIP and .../RTIP-MD), limited to the
upstream-consistent algorithms:

* RTIP / IDWM pathway sampling (`deepmd-pathway`, `mock-pathway`);
* repulsive RTIP NVT MD (`deepmd-md`, `mock-md`);
* formose-style evolution MD with bond-variation-controlled attractive RTIP
  (`deepmd-evolution-md`);
* synthesis layout (`synthesize`);
* DeePMD-kit PES provider (replaces the legacy CP2K boundary, which is
  documented in `external/cp2k.py` but not implemented).

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

JAX x64 is enabled when `rtip_jax` is imported.

## CLI examples

```bash
rtip-jax show-default-config
rtip-jax synthesize --inputs 1.xyz 2.xyz --output IS.xyz --dist 5.0 --seed 0
rtip-jax deepmd-pathway --input IS.xyz --model model.pth --method rtip
rtip-jax deepmd-md --input IS.xyz --model model.pth
rtip-jax deepmd-evolution-md --input box.xyz --model model.pth --max-step 10000
```

See `../formose/` for the full formose-reaction workflow (box builder,
slurm runner, trajectory analysis).
