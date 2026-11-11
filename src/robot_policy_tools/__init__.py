"""Generic CPU interfaces and offline signal metrics; no safety qualification."""

from .contracts import ActionSpec, Feature, PolicyContract
from .metrics import Signal, difference, mean_square, rms, tracking_mae

__all__ = [
    "ActionSpec",
    "Feature",
    "PolicyContract",
    "Signal",
    "difference",
    "mean_square",
    "rms",
    "tracking_mae",
]
