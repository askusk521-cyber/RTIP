# Optimization Migration Notes

Rust source: `src/pes_exploration/optimization.rs`.

Python target: `src/rtip_jax/core/optimization.py`.

## Rust Semantics

- `min_1d(fun, f0, epsilon)` assumes `fun` decreases along positive `x` near
  zero.
- It starts with `delta_x = 0.01`.
- If the first trial increases the function, it repeatedly divides `delta_x` by
  10 until a decreasing point is found or the step is smaller than `1e-15`.
- If no decreasing point is found, Rust panics.
- It expands the bracket using `GOLDEN_RATIO2` while the function keeps
  decreasing and `x1 < 0.1`.
- If the function still decreases after `x1 >= 0.1`, it returns the capped step.
- Otherwise it performs a golden-section search using `GOLDEN_RATIO1` until
  adjacent function values differ by less than `epsilon`.
- `min_1d_real_bias` searches along the total-force direction and returns
  `delta_coord`.

## Input and Output

- `min_1d` input: callable `fun(x) -> scalar`, scalar `f0`, scalar `epsilon`.
- `min_1d` output: `(x_min, f_min)` as Python floats.
- `min_1d_real_bias` input:
  - `real_pes` and `bias_pes` with `get_energy(system)`.
  - `system` with coordinates `(natom, 3)`.
  - `force_total` shape `(natom, 3)`.
  - scalar `pot_total` and `epsilon`.
- `min_1d_real_bias` output: JAX array `delta_coord` with shape `(natom, 3)`.

## State and Side Effects

- `min_1d` calls a Python callable and has no other side effects.
- `min_1d_real_bias` constructs displaced `System` values using `with_coord`;
  it does not mutate the input system.
- No IO, CP2K, MPI, or random state is used.

## Numerical Behavior

- The Rust one-dimensional search strategy is preserved, including the
  approximate `0.1` step cap during bracket expansion.
- Function values are converted to Python floats because this is a control-flow
  helper, not a JAX graph primitive.
- Shape mismatches raise `ValueError`.
- Zero total-force norm raises `OptimizationError`. Rust does not explicitly
  guard this and can fall into division-by-zero/NaN behavior; the Python version
  makes that invalid state explicit.

## JAX Support

- `min_1d` and `min_1d_real_bias` are not intended as `jit` targets because
  they use Python callbacks and dynamic control flow.
- They are workflow-level helpers. Later performance work may replace specific
  use cases with `lax` control flow after behavior is stable.

## Key Differences From Rust

- Rust `panic!` in `min_1d` becomes `OptimizationError`.
- Rust mutates a cloned `System`; Python creates replacement `System` objects.
- Zero total-force norm is explicitly rejected.

## Tests

Tests are in `tests/test_optimization.py` and cover:

- Quadratic one-dimensional minimization.
- Rejection when the function increases from zero.
- Rust step-cap behavior for a monotonically decreasing function.
- Force-direction displacement with harmonic mock PES.
- Force shape validation.
- Zero-force direction rejection.

