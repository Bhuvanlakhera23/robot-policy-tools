# API and numerical definitions

## Interface contracts

`Feature(name, size, unit)` defines a contiguous feature block. Names are unique,
size is a positive integer, units are nonempty labels. Observation feature order is
the feature tuple order; the validator cannot infer or detect a permutation inside
an otherwise unlabeled numeric vector. Joint metadata must exactly equal declared
unique order. A false but shape-correct metadata declaration cannot be detected.

`ActionSpec` has J unique names and J scales/offsets. For each raw component a,
optional `[low, high]` clipping is applied first, then `offset + scale * a`.
Zero and negative scales are allowed when explicitly declared. Raw and target
units are metadata; scale is interpreted in target units per raw unit. There is
no automatic dimension conversion, joint-limit clamp or torque/PD model.

`PolicyContract` declares batch B, history H, feature width F and a resource bound
on B×max(H×F, J). `validate_observation` requires B×H×F nested sequences and H
oldest-to-newest timestamps shared across the batch. It rejects nonfinite/boolean
numbers, bad names/shapes/units, non-advancing history, future newest samples and
age above `max_age_s`. History may include older frames; freshness applies only
to the latest. Pass `previous_timestamp` to check latest-time advancement between
calls. The contract is stateless; reset/clock-session management belongs to caller.
`validate_actions` checks B×J raw actions; `joint_targets` validates then scales.
No trained-policy call, frame packing helper, encoder or physical controller exists.

## Sampled signals and resets

`Signal(timestamps, values, unit, sample_rate_hz, segments=None)` contains T×D
finite values and T finite strictly increasing timestamps in seconds. Empty and
ragged inputs fail. Sample frequency must be positive and finite. Consecutive
samples in a segment must differ by dt=1/frequency to absolute tolerance
`timestamp_tolerance_s` (default 1e-9 s; must be below 1% of dt). Missing or irregular
samples fail; there is no quiet dropping, imputation or irregular-grid derivative.

Optional string segment IDs identify reset/episode boundaries. Each ID must be
contiguous and cannot reappear after closure. Time must increase across boundaries;
a time gap at a boundary is allowed. For restarted clocks, explicitly convert to a
common monotonic clock before construction. Units and frames must already match.

## Metrics

All means are **equal sample/window** means, per component, not integrated costs
or equal-episode means. To report by fixture, keep `dict[fixture_name, Signal]` and
apply the function independently; never concatenate different fixtures or clocks.
To give episodes equal weight, calculate each episode separately, then declare
that separate aggregation explicitly. No fixture is silently pooled by this library.

| Function | Definition | Output unit | Count/limits |
|---|---|---|---|
| `tracking_mae(c, y)` | mean(abs(c-y)) per axis | Same as c and y | Identical sample times, rate, segments, width and units required |
| Direction entries | Same MAE on positive/negative/exact-zero commands for each axis | Same signal unit | Absent sign subsets are omitted, never zero-imputed; key gives axis/sign |
| `mean_square(x)` | mean(x²), per axis | Input unit² | All retained samples; use Nm input; squared-torque unit is (N·m)² |
| `rms(x)` | sqrt(mean(x²)), per axis | Input unit | E.g. joint velocity in rad/s |
| `difference(x, order=k)` | Repeated forward Δ^k x | Input unit | k=1,2,3; valid (k+1)-sample windows only |
| `difference(..., divide_by_dt=True)` | Δ^k x / dt^k | Input unit / s^k | Uniform dt; not analytic differentiation |

For order k, coefficients are `(-1)^(k-j) * binomial(k,j)` for j=0…k.
A window is omitted only when it crosses an explicitly declared reset boundary.
Absolute means, RMS and maximum absolute differences are reported per axis with
the valid-window count. If none exist (short sequence or all windows span resets),
count is zero and every numerical summary is `None`, not a fabricated zero.

Use separate signals for linear velocities (`m/s`) and yaw rates (`rad/s`). The
library cannot discover a mixed-unit vector labeled with a single false unit.
For raw actions use a declared raw unit; for applied position targets use `rad`.
Differentiating raw actions describes that signal only. These are no direct
measurements of physical joint acceleration, energy, stability, wear or safety.

Overflow/nonfinite summaries fail. No confidence interval, policy ranking, learned
skill or actuator acceptance is computed. No SciPy/NumPy derivative implementation
or private evaluation algorithm is borrowed; the finite-difference formula is an
independently implemented generic mathematical definition.
