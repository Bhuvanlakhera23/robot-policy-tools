"""Analytical synthetic data only, not a robot or simulator recording."""

import json
from dataclasses import asdict

from robot_policy_tools import (
    ActionSpec,
    Feature,
    PolicyContract,
    Signal,
    difference,
    mean_square,
    rms,
    tracking_mae,
)


def summary() -> dict[str, object]:
    contract = PolicyContract(
        (Feature("joint_position", 2, "rad"),),
        ActionSpec(("left", "right"), (0.5, 0.25), (0.0, 1.0), clip_raw=(-1.0, 1.0)),
    )
    observed = contract.validate_observation(
        [[[0.1, -0.2]]],
        joint_names=["left", "right"],
        units={"joint_position": "rad"},
        timestamps=[1.0],
        now=1.01,
    )
    targets = contract.joint_targets(
        [[2.0, -0.5]], joint_names=["left", "right"], unit="dimensionless"
    )
    times = (0.0, 0.5, 1.0, 1.5)
    command = Signal(times, ((1.0,),) * 4, "m/s", 2.0)
    measured = Signal(times, ((0.5,), (0.75,), (1.0,), (1.25,)), "m/s", 2.0)
    action = Signal(times, ((0.0,), (1.0,), (2.0,), (3.0,)), "raw", 2.0)
    torque = Signal(times, ((2.0,), (-2.0,), (2.0,), (-2.0,)), "Nm", 2.0)
    joint_velocity = Signal(times, ((3.0,),) * 4, "rad/s", 2.0)
    return {
        "data_origin": "synthetic analytical fixture",
        "validated_observation": observed,
        "joint_targets_rad": targets,
        "linear_tracking_MAE_m_s": tracking_mae(command, measured),
        "raw_action_increment": asdict(difference(action)),
        "raw_action_rate": asdict(difference(action, divide_by_dt=True)),
        "joint_velocity_RMS_rad_s": rms(joint_velocity),
        "squared_torque_Nm2": mean_square(torque),
        "approved_for_actuation": False,
    }


if __name__ == "__main__":
    print(json.dumps(summary(), indent=2, allow_nan=False))
