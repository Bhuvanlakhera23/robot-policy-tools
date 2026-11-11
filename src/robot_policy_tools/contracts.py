"""Explicit policy tensor conventions, units and freshness; never hardware safety."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ._validation import finite, label, names, positive_int, vector


@dataclass(frozen=True)
class Feature:
    name: str
    size: int
    unit: str

    def __post_init__(self) -> None:
        label(self.name, "feature name")
        positive_int(self.size, self.name + " size")
        label(self.unit, self.name + " unit")


@dataclass(frozen=True)
class ActionSpec:
    joint_names: tuple[str, ...]
    scale: tuple[float, ...]
    offset: tuple[float, ...]
    target_unit: str = "rad"
    raw_unit: str = "dimensionless"
    clip_raw: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        joints = names(self.joint_names, "action joints")
        object.__setattr__(self, "joint_names", joints)
        object.__setattr__(self, "scale", vector(self.scale, len(joints), "scale"))
        object.__setattr__(self, "offset", vector(self.offset, len(joints), "offset"))
        label(self.raw_unit, "raw action unit")
        label(self.target_unit, "target unit")
        if self.clip_raw is not None:
            limits = vector(self.clip_raw, 2, "raw clipping bounds")
            if limits[0] >= limits[1]:
                raise ValueError("raw clipping bounds must be increasing")
            object.__setattr__(self, "clip_raw", limits)


@dataclass(frozen=True)
class PolicyContract:
    features: tuple[Feature, ...]
    actions: ActionSpec
    batch_size: int = 1
    history_length: int = 1
    max_age_s: float = 0.05
    max_elements: int = 1_000_000

    def __post_init__(self) -> None:
        features = tuple(self.features)
        if any(not isinstance(x, Feature) for x in features):
            raise ValueError("features must contain Feature definitions")
        names(tuple(x.name for x in features), "features")
        if not isinstance(self.actions, ActionSpec):
            raise ValueError("actions must be an ActionSpec")
        object.__setattr__(self, "features", features)
        for name in ("batch_size", "history_length", "max_elements"):
            positive_int(getattr(self, name), name)
        age = finite(self.max_age_s, "max_age_s")
        if age < 0:
            raise ValueError("max_age_s must be nonnegative")
        if (
            self.batch_size
            * max(self.history_length * self.observation_size, len(self.actions.joint_names))
            > self.max_elements
        ):
            raise ValueError("contract exceeds max_elements resource bound")

    @property
    def observation_size(self) -> int:
        return sum(f.size for f in self.features)

    def validate_observation(
        self,
        values: Sequence[Sequence[Sequence[float]]],
        *,
        joint_names: Sequence[str],
        units: Mapping[str, str],
        timestamps: Sequence[float],
        now: float,
        previous_timestamp: float | None = None,
    ) -> tuple[tuple[tuple[float, ...], ...], ...]:
        """Validate B×H×F; shared history timestamps must be oldest to newest.

        Freshness applies to the newest sample only. All clocks must share a
        domain. Pass previous_timestamp to enforce chronology between calls.
        """
        self._joints(joint_names)
        if dict(units) != {f.name: f.unit for f in self.features}:
            raise ValueError("observation units/names do not match contract; no unit conversion")
        times = vector(timestamps, self.history_length, "history timestamps")
        current = finite(now, "now")
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("history timestamps must be strictly increasing")
        if times[-1] > current or current - times[-1] > self.max_age_s:
            raise ValueError("newest observation is from the future or stale")
        if previous_timestamp is not None and times[-1] <= finite(
            previous_timestamp, "previous_timestamp"
        ):
            raise ValueError("latest timestamp did not advance")
        if len(values) != self.batch_size:
            raise ValueError(f"observation batch: expected {self.batch_size}")
        result = []
        for history in values:
            if len(history) != self.history_length:
                raise ValueError(f"history length: expected {self.history_length}")
            result.append(
                tuple(vector(row, self.observation_size, "observation") for row in history)
            )
        return tuple(result)

    def validate_actions(
        self, values: Sequence[Sequence[float]], *, joint_names: Sequence[str], unit: str
    ) -> tuple[tuple[float, ...], ...]:
        self._joints(joint_names)
        if unit != self.actions.raw_unit:
            raise ValueError("raw action unit mismatch; no unit conversion")
        if len(values) != self.batch_size:
            raise ValueError(f"action batch: expected {self.batch_size}")
        return tuple(vector(row, len(self.actions.joint_names), "action") for row in values)

    def joint_targets(
        self, values: Sequence[Sequence[float]], *, joint_names: Sequence[str], unit: str
    ) -> tuple[tuple[float, ...], ...]:
        """Affine target = offset + scale * clipped raw action, per named joint.

        Clipping is explicit and occurs before scaling. No physical joint limits
        or actuator model are inferred. Output unit is actions.target_unit.
        """
        actions = self.validate_actions(values, joint_names=joint_names, unit=unit)
        result = []
        for row in actions:
            if self.actions.clip_raw is not None:
                low, high = self.actions.clip_raw
                row = tuple(min(high, max(low, a)) for a in row)
            result.append(
                tuple(
                    finite(o + s * a, "joint target")
                    for o, s, a in zip(self.actions.offset, self.actions.scale, row, strict=False)
                )
            )
        return tuple(result)

    def _joints(self, received: Sequence[str]) -> None:
        if names(received, "received joints") != self.actions.joint_names:
            raise ValueError("joint order/names mismatch; no implicit reordering")
