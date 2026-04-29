"""Runtime configuration for the JAX implementation."""

from __future__ import annotations


def configure_jax(*, enable_x64: bool = True) -> None:
    """Configure JAX defaults used by RTIP.

    The Rust implementation uses `f64` throughout the core numerical code. The
    Python rewrite enables JAX x64 by default to keep numerical behavior close.
    """

    from jax import config

    config.update("jax_enable_x64", enable_x64)


def is_x64_enabled() -> bool:
    """Return whether JAX x64 is currently enabled."""

    from jax import config

    try:
        return bool(config.read("jax_enable_x64"))
    except Exception:
        return bool(getattr(config, "jax_enable_x64", False))

