"""Configuration structures and lightweight config loading.

Rust source: `src/io/input.rs`.
"""

from __future__ import annotations

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
