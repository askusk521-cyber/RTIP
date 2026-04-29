"""Output path helpers.

Rust source: `src/io/output.rs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path


@dataclass(frozen=True)
class RtipOutputPaths:
    """Structure and scalar-output paths for RTIP/IDWM workflows."""

    structure: Path
    table: Path


def _indexed_dir(base_dir: str | PathLike[str], index: int | None, *, create: bool, exist_ok: bool) -> Path:
    base = Path(base_dir)
    directory = base if index is None else base / str(index)
    if create:
        directory.mkdir(parents=True, exist_ok=exist_ok)
    return directory


def output_rtip(
    index: int | None = None,
    *,
    base_dir: str | PathLike[str] = ".",
    create: bool = True,
    exist_ok: bool = True,
) -> RtipOutputPaths:
    """Return RTIP output paths, optionally creating the indexed directory."""

    directory = _indexed_dir(base_dir, index, create=create, exist_ok=exist_ok)
    return RtipOutputPaths(structure=directory / "rtip.pdb", table=directory / "rtip.out")


def output_cp2k(
    index: int | None = None,
    *,
    base_dir: str | PathLike[str] = ".",
    create: bool = True,
    exist_ok: bool = True,
) -> Path:
    """Return the legacy CP2K output path without implementing CP2K itself."""

    directory = _indexed_dir(base_dir, index, create=create, exist_ok=exist_ok)
    return directory / "cp2k.out"


__all__ = ["RtipOutputPaths", "output_cp2k", "output_rtip"]
