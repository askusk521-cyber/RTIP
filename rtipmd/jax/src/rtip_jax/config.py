"""Configuration structures and lightweight config loading.

Rust source: `src/io/input.rs`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from os import PathLike
from pathlib import Path
from typing import Any

import json

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    tomllib = None


@dataclass(frozen=True)
class Para:
    """Parameters for RTIP pathway sampling and RTIP-driven MD."""

    a0: float = 0.002
    scale_ts_a0: float = 1.0
    scale_ts_sigma: float | None = 0.25
    max_step: int = 1500
    print_step: int = 1
    pot_climb: float = 0.185
    pot_drop: float = 0.02
    pot_epsilon: float = 0.00005
    f_epsilon: float = 0.001
    dt: float = 0.5
    tau: float = 500.0
    temp_bath: float = 1000.0
    # -- paper-described RTIP-MD features (defaults preserve backward compat) --
    oscillation_period: float = 0.0
    oscillation_amplitude: float = 0.0
    reduction_rate: float = 2.0
    max_rounds: int = 1
    relax_max_steps: int = 200
    rust_compat: bool = False

    # ------------------------------------------------------------------
    # Amplitude helpers (paper basis: oscillating RTIP + 2x reduction)
    # Engineering: sinusoidal formula; linear-decrease formula.
    # ------------------------------------------------------------------

    def compute_oscillation_factor(self, step: int) -> float:
        """Return oscillation multiplier for *step*.  1.0 when disabled.

        Paper basis: oscillating (periodically modulated) RTIP well depth.
        Engineering: sinusoidal modulation ``1 + A·sin(2π·step / T)``.
        """
        if self.oscillation_period <= 0.0:
            return 1.0
        return 1.0 + self.oscillation_amplitude * math.sin(
            2.0 * math.pi * float(step) / self.oscillation_period,
        )

    def bias_amplitude(
        self,
        step: int,
        *,
        bias_phase: str = "growing",
        step_reduction_started: int | None = None,
    ) -> float:
        """Non-negative amplitude envelope for the Gaussian RTIP bias.

        * ``"growing"``  – linear growth  ``a0·step`` × oscillation factor.
        * ``"reducing"`` – monotonic linear decrease at ``reduction_rate·a0``
          per step (no oscillation), reaching zero from the peak at
          *step_reduction_started*.
        * ``"off"``      – returns 0.0.

        Paper basis: 2× reduction rate after bond detection.
        Sign (attractive / repulsive) is applied by the caller.
        """
        if bias_phase == "off":
            return 0.0
        if bias_phase == "growing":
            return self.a0 * float(step) * self.compute_oscillation_factor(step)
        if bias_phase == "reducing":
            if step_reduction_started is None:
                raise ValueError("step_reduction_started required for reducing phase")
            peak = self.a0 * float(step_reduction_started)
            decrement = self.reduction_rate * self.a0 * float(step - step_reduction_started)
            return max(0.0, peak - decrement)
        raise ValueError(f"unknown bias_phase: {bias_phase!r}")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "Para":
        """Create parameters from a mapping, rejecting unknown keys."""

        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown Para keys: {', '.join(unknown)}")
        return cls(**{key: values[key] for key in allowed if key in values})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def para_to_dict(para: Para) -> dict[str, Any]:
    return para.to_dict()


def load_para(filename: str | PathLike[str]) -> Para:
    """Load `Para` from JSON or TOML.

    TOML support uses the standard-library `tomllib` when available. On Python
    3.10, JSON remains available without adding another dependency.
    """

    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix == ".json":
        values = json.loads(path.read_text())
    elif suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML config loading requires Python 3.11+ or a caller-provided parser")
        values = tomllib.loads(path.read_text())
        values = values.get("para", values)
    else:
        raise ValueError("configuration files must use .json or .toml")
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a mapping")
    return Para.from_mapping(values)


def format_default_para() -> str:
    """Return default `Para` values as pretty JSON for CLI use."""

    return json.dumps(Para().to_dict(), indent=2, sort_keys=True) + "\n"
