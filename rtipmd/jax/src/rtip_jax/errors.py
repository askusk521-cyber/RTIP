"""Exception types and message helpers for RTIP JAX."""

from __future__ import annotations


class RTIPError(Exception):
    """Base class for RTIP JAX errors."""


class InputFormatError(RTIPError, ValueError):
    """Raised when a text input file does not match the expected format."""


class UnknownElementError(RTIPError, ValueError):
    """Raised when an element symbol is not recognized."""


class UnsupportedElementMassError(RTIPError, ValueError):
    """Raised when a known element lacks a Rust-compatible mass value."""


class OptimizationError(RTIPError, RuntimeError):
    """Raised when a local optimization precondition is not satisfied."""


def error_file(operation: str, filename: str) -> str:
    return f"ERROR: There is some problem in {operation} the file '{filename}'."


def error_read_xyz(filename: str) -> str:
    return f"ERROR: There is some problem with the input file '{filename}'. Please check it."


def error_element(symbol: str) -> str:
    return f"ERROR: Illegal chemical element type '{symbol}' has been read from the input file."


def error_unsupported_mass(symbol: str) -> str:
    return f"ERROR: No Rust-compatible atomic mass is defined for element '{symbol}'."


def error_min_1d() -> str:
    return (
        "ERROR: There is some problem with min_1d: the input function is increasing "
        "along +x direction, or the default minimum step is too large."
    )

