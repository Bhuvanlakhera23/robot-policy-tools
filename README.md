# Robot policy tools

Validate declared policy interfaces and summarize offline sampled signals without
loading a policy, simulator, ROS graph or hardware transport. For robotics
researchers and engineers reviewing observation/action boundaries and evaluation
logs. It does not evaluate a learned policy or certify robot safety.

This independent project is unaffiliated with robot manufacturers.
Runtime dependencies: Python 3.10+ standard library only. Inputs are nested Python
sequences (`array.tolist()` for external array libraries).

## Installation and minimal example

From this source candidate, in a separate environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python examples/synthetic.py
python -m unittest discover -s tests -v
```

No model or robot assets are downloaded. The example is analytical synthetic data,
not a Go2 recording and not data used in any historical benchmark. Its tracking
MAE is 0.25 m/s; raw increments are 1, the dt-normalized rate is 2 raw units/s,
joint-velocity RMS is 3 rad/s and the squared-torque proxy is 4 (N·m)². These numbers
check software definitions, not locomotion capability. See `examples/synthetic.py`.

```python
from robot_policy_tools import ActionSpec, Feature, PolicyContract

contract = PolicyContract(
    features=(Feature("joint_position", 2, "rad"),),
    actions=ActionSpec(("left", "right"), (.5, .25), (0., 1.)),
)
contract.validate_observation(
    [[[0.1, -0.2]]], joint_names=("left", "right"),
    units={"joint_position": "rad"}, timestamps=(1.0,), now=1.01,
)
targets = contract.joint_targets(
    [[1., -.5]], joint_names=("left", "right"), unit="dimensionless"
)
# ((0.5, 0.875),) rad; no physical joint/torque limits have been inferred.
```

## Small public API

- `Feature`, `ActionSpec`, `PolicyContract`: immutable declarations and validation
  of B×H×F observations, B×J actions, joint order, finite values, units, chronological
  history and latest-sample freshness; explicit affine scaling/clipping.
- `Signal`: immutable rectangular sampled data with units, uniform frequency,
  monotonic timestamps and optional contiguous episode/reset segments.
- `tracking_mae`: per-axis overall and command-sign-stratified error means.
- `difference`: first/second/third discrete increments or dt-normalized differences,
  excluding windows across reset boundaries; sample count and absolute/RMS/max values.
- `rms`, `mean_square`: equal-sample per-axis signal summaries.

[API and metric definitions](docs/API.md) specify shapes, units, weighting,
missing-data behavior and short-sequence semantics. No history buffer or temporal
encoder is implemented. A signal is not a policy or physical actuator model.

## Tests and development

Build, lint and type tools are pinned in `requirements-dev.txt`; they are not runtime
requirements. Install them explicitly in a separate development environment:

```bash
python -m pip install -r requirements-dev.txt
python -m build --no-isolation
ruff check .
ruff format --check .
mypy
python -m unittest discover -s tests -v
```

CPU CI is prepared to build an sdist and its wheel, install it, run tests/example
outside the checkout, and run lint/type checks. No GitHub workflow has run for this unpublished candidate. Tests include
independent polynomial/affine oracles and deterministic randomized invariants;
all fixtures are synthetic.

## Scope and limitations

The caller must declare units, coordinate frames, joint correspondence and a
common clock truthfully. This package validates their schema, not sensor truth,
clock synchronization, calibration or physical correctness. It performs no
implicit degree/radian conversion, joint reordering, resampling, interpolation,
missing-data imputation or derivative boundary padding.

Raw action differences are not physical joint acceleration. Even discrete
joint-target derivatives do not measure actual link motion. Mean squared torque
is a proxy, not measured electrical energy, motor wear or hardware lifetime.
Sample averages are not equal-episode averages when episodes differ in length.
A contract passing does not authorize actuation or prove an actuator received it.

## License and attribution

Licensed under [MIT](LICENSE). Copyright © 2026 Bhuvan Lakhera.
No third-party source is vendored. See [provenance](PROVENANCE.md),
[license recommendation](LICENSE_RECOMMENDATION.md) and
[release boundary](RELEASE_BOUNDARY.json). `approved_for_actuation: false`.
