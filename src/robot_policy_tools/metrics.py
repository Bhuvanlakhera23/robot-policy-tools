"""Uniform-grid signal summaries; explicit units, resets and missing-data rejection.

First/second/third repeated forward differences are signal derivatives, not
physical joint acceleration/jerk unless the input is a physical joint position.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ._validation import finite, label, positive_int, vector


@dataclass(frozen=True)
class Signal:
    timestamps: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    unit: str
    sample_rate_hz: float
    segments: tuple[str, ...] | None = None
    timestamp_tolerance_s: float = 1e-9

    def __post_init__(self) -> None:
        label(self.unit, "signal unit")
        hz = finite(self.sample_rate_hz, "sample_rate_hz")
        tolerance = finite(self.timestamp_tolerance_s, "timestamp_tolerance_s")
        if hz <= 0 or tolerance < 0 or tolerance >= 0.01 / hz:
            raise ValueError(
                "frequency must be positive; timestamp tolerance must be below 1% of dt"
            )
        finite(1 / hz, "sampling interval")
        if not self.values or len(self.timestamps) != len(self.values):
            raise ValueError("signal must be nonempty with one timestamp per sample")
        size = len(self.values[0])
        positive_int(size, "signal width")
        times = vector(self.timestamps, len(self.values), "timestamps")
        rows = tuple(vector(row, size, "signal values") for row in self.values)
        segments = tuple(self.segments) if self.segments is not None else ("0",) * len(rows)
        if len(segments) != len(rows):
            raise ValueError("one segment ID required per sample")
        for segment in segments:
            label(segment, "segment ID")
        closed: set[str] = set()
        for i in range(1, len(times)):
            if times[i] <= times[i - 1]:
                raise ValueError("timestamps must be strictly increasing, including across resets")
            if segments[i] == segments[i - 1]:
                if not math.isclose(times[i] - times[i - 1], 1 / hz, rel_tol=0, abs_tol=tolerance):
                    raise ValueError("irregular/missing samples within segment; no resampling")
            else:
                closed.add(segments[i - 1])
                if segments[i] in closed:
                    raise ValueError("a closed segment ID cannot reappear")
        object.__setattr__(self, "timestamps", times)
        object.__setattr__(self, "values", rows)
        object.__setattr__(self, "segments", segments)


@dataclass(frozen=True)
class DifferenceSummary:
    order: int
    dt_normalized: bool
    unit: str
    count: int
    mean_abs_by_axis: tuple[float, ...] | None
    rms_by_axis: tuple[float, ...] | None
    max_abs_by_axis: tuple[float, ...] | None


def _average(values: Sequence[float]) -> float:
    # Normalize before summation: a representable mean must not fail merely
    # because its unnormalized sum exceeds the floating-point range.
    scale = max(abs(x) for x in values)
    if scale == 0:
        return 0.0
    return finite(scale * (math.fsum(x / scale for x in values) / len(values)), "mean")


def _mean(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(_average(tuple(row[j] for row in rows)) for j in range(len(rows[0])))


def mean_square(signal: Signal) -> tuple[float, ...]:
    """Equal-sample E[x²] per axis, in signal.unit². No time integration."""
    return _mean(
        tuple(tuple(finite(x * x, "squared signal") for x in row) for row in signal.values)
    )


def rms(signal: Signal) -> tuple[float, ...]:
    """sqrt(E[x²]) per axis, in the input unit."""
    return tuple(math.sqrt(x) for x in mean_square(signal))


def difference(signal: Signal, *, order: int = 1, divide_by_dt: bool = False) -> DifferenceSummary:
    """Repeated forward differences through order 3, excluding every reset window.

    A window of order+1 contiguous samples must share a segment ID. By default
    results are increments in the input unit. If divide_by_dt, each differencing
    stage divides by dt; order k then has input unit / s^k. Aggregation is over
    valid windows, equally, per axis. No boundary padding or zero imputation.
    """
    if type(order) is not int or order not in (1, 2, 3) or type(divide_by_dt) is not bool:
        raise ValueError("difference order must be 1, 2 or 3; divide_by_dt must be bool")
    width = len(signal.values[0])
    coefficients = tuple((-1) ** (order - j) * math.comb(order, j) for j in range(order + 1))
    try:
        factor = finite(signal.sample_rate_hz**order if divide_by_dt else 1.0, "derivative scale")
    except OverflowError as exc:
        raise ValueError("derivative scaling overflow") from exc
    rows = []
    assert signal.segments is not None
    for start in range(len(signal.values) - order):
        if len(set(signal.segments[start : start + order + 1])) != 1:
            continue
        rows.append(
            tuple(
                finite(
                    math.fsum(
                        coefficients[j] * signal.values[start + j][axis] for j in range(order + 1)
                    )
                    * factor,
                    "difference",
                )
                for axis in range(width)
            )
        )
    unit = signal.unit + (f"/s^{order}" if divide_by_dt else "")
    if not rows:
        return DifferenceSummary(order, divide_by_dt, unit, 0, None, None, None)
    return DifferenceSummary(
        order,
        divide_by_dt,
        unit,
        len(rows),
        _mean(tuple(tuple(abs(x) for x in row) for row in rows)),
        tuple(
            math.sqrt(x)
            for x in _mean(
                tuple(tuple(finite(x * x, "squared difference") for x in row) for row in rows)
            )
        ),
        tuple(max(abs(row[j]) for row in rows) for j in range(width)),
    )


def tracking_mae(command: Signal, measured: Signal) -> dict[str, tuple[float, ...]]:
    """Equal-sample signed-command-axis MAE, overall and by commanded direction.

    Each axis is partitioned by its own commanded sign; an axis near zero is
    exactly zero here (no implicit deadband). For velocity use 2 linear axes in
    m/s and yaw separately in rad/s; the caller must keep differently dimensioned axes in separate signals.
    """
    if (command.timestamps, command.sample_rate_hz, command.segments, command.unit) != (
        measured.timestamps,
        measured.sample_rate_hz,
        measured.segments,
        measured.unit,
    ):
        raise ValueError(
            "tracking signals must have identical timestamps, rate, segments and units"
        )
    if len(command.values[0]) != len(measured.values[0]):
        raise ValueError("tracking dimensions mismatch")
    errors = tuple(
        tuple(finite(abs(a - b), "tracking error") for a, b in zip(c, m, strict=False))
        for c, m in zip(command.values, measured.values, strict=False)
    )
    result = {"all": _mean(errors)}
    for axis in range(len(command.values[0])):
        for name, sign in (("positive", 1), ("negative", -1), ("zero", 0)):
            selected = [
                error[axis]
                for row, error in zip(command.values, errors, strict=False)
                if (1 if row[axis] > 0 else -1 if row[axis] < 0 else 0) == sign
            ]
            if selected:
                result[f"axis_{axis}/{name}"] = (_average(selected),)
    return result
