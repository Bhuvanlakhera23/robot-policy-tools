"""Internal numeric/schema helpers. No coercion of booleans or nonfinite data."""

import math
from collections.abc import Sequence
from numbers import Real


def finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name}: expected a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name}: cannot represent finite float") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name}: expected a finite real number")
    return result


def positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name}: expected a nonempty, unpadded string")
    return value


def names(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not values:
        raise ValueError(f"{name}: expected nonempty names")
    result = tuple(label(x, name) for x in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name}: duplicate names")
    return result


def vector(values: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if len(values) != size:
        raise ValueError(f"{name}: expected length {size}, got {len(values)}")
    return tuple(finite(v, name) for v in values)
