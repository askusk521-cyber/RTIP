"""Configuration structures and lightweight config loading.

Rust source: `src/io/input.rs` (Para / RtipPara / PathSampPara / MdPara).
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

from .constants import Element, coerce_element


@dataclass(frozen=True)
class Para:
    """Parameters for RTIP pathway sampling and RTIP-driven MD.

    Defaults match the Rust `Para::new()` in `src/io/input.rs`.
    """

    # -- RTIP (RtipPara) --
    a0: float = 0.0005
    sigma: float = 0.75
    scale_ts_a0: float = 1.0
    scale_ts_sigma: float | None = 0.25

    # -- MD (MdPara) --
    dt: float = 0.5
    tau: float = 10.0
    temp_bath: float = 1500.0
    decreasing_multiple: float = 2.0
    decreasing_bound: float = 0.5
    ignored_pair: tuple[tuple[Element, Element], ...] = (
        (Element.H, Element.O),
        (Element.C, Element.O),
        (Element.H, Element.Ca),
        (Element.C, Element.Ca),
        (Element.O, Element.Ca),
        (Element.Ca, Element.Ca),
    )
    split_step: int | None = 100
    max_step: int = 10000
    print_step: int = 1
    size_scaling: bool = False
    fixed_sigma: bool = False

    # -- Pathway sampling (PathSampPara) --
    pot_climb: float = 0.185
    pot_drop: float = 0.02
    pot_epsilon: float = 0.00005
    f_epsilon: float = 0.001

    def bias_amplitude(self, step: int, *, n_bias: int | None = None) -> float:
        """Linear RTIP amplitude ``a0 * step`` (Rust ``a_min = a0 * i``)."""

        return self.a0 * float(step)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "Para":
        """Create parameters from a mapping, rejecting unknown keys."""

        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown Para keys: {', '.join(unknown)}")
        cleaned: dict[str, Any] = {}
        for key, value in values.items():
            if key == "ignored_pair":
                cleaned[key] = tuple(
                    tuple(sorted((coerce_element(a), coerce_element(b)))) for a, b in value
                )
            else:
                cleaned[key] = value
        return cls(**cleaned)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def para_to_dict(para: Para) -> dict[str, Any]:
    return para.to_dict()


def load_para(filename: str | PathLike[str]) -> Para:
    """Load `Para` from JSON or TOML."""

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

    return json.dumps(Para().to_dict(), indent=2, sort_keys=True, default=str) + "\n"
