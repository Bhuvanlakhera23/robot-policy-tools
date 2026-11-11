"""Synthetic policy data only; independent schema and affine-transform oracles."""

import math
import random
import unittest

from robot_policy_tools import ActionSpec, Feature, PolicyContract


class Contracts(unittest.TestCase):
    def setUp(self):
        self.joints = ("shoulder", "elbow")
        self.actions = ActionSpec(self.joints, (0.5, -2.0), (1.0, 0.25), clip_raw=(-1.0, 1.0))
        self.contract = PolicyContract(
            (Feature("position", 2, "rad"), Feature("speed", 1, "m/s")),
            self.actions,
            history_length=2,
            max_age_s=0.1,
        )

    def observation(self, **changes):
        args = dict(
            values=[[[1.0, 2.0, 0.5], [1.1, 2.1, 0.6]]],
            joint_names=self.joints,
            units={"position": "rad", "speed": "m/s"},
            timestamps=[0.9, 1.0],
            now=1.05,
        )
        args.update(changes)
        return self.contract.validate_observation(**args)

    def test_synthetic_history_order_and_shape(self):
        self.assertEqual(self.observation(), (((1.0, 2.0, 0.5), (1.1, 2.1, 0.6)),))
        for values in ([], [[1.0, 2.0, 3.0]], [[[1.0, 2.0, 3.0]]], [[[1.0, 2.0], [1.0, 2.0]]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.observation(values=values)

    def test_nonfinite_observations(self):
        for bad in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.observation(values=[[[bad, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    def test_joint_order_missing_duplicate(self):
        for received in (("elbow", "shoulder"), ("shoulder",), ("shoulder", "shoulder")):
            with self.assertRaises(ValueError):
                self.observation(joint_names=received)
        for joints in (("a", "a"), (), (" a",)):
            with self.assertRaises(ValueError):
                ActionSpec(joints, (1.0,) * len(joints), (0.0,) * len(joints))

    def test_units_are_explicit_no_implicit_conversion(self):
        for units in (
            {"position": "deg", "speed": "m/s"},
            {"position": "rad"},
            {"position": "rad", "speed": "m/s", "extra": "s"},
        ):
            with self.assertRaisesRegex(ValueError, "units"):
                self.observation(units=units)
        with self.assertRaisesRegex(ValueError, "unit"):
            self.contract.validate_actions([[0.0, 0.0]], joint_names=self.joints, unit="rad")

    def test_freshness_and_monotonicity(self):
        for changes in (
            {"timestamps": [1.0, 0.9]},
            {"timestamps": [1.0, 1.0]},
            {"now": 0.95},
            {"now": 1.11},
            {"previous_timestamp": 1.0},
            {"previous_timestamp": float("nan")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.observation(**changes)
        self.observation(previous_timestamp=0.99)

    def test_affine_clipping_and_order(self):
        # raw [2,-.5] -> clipped [1,-.5] -> [1+.5, .25+1]
        self.assertEqual(
            self.contract.joint_targets(
                [[2.0, -0.5]], joint_names=self.joints, unit="dimensionless"
            ),
            ((1.5, 1.25),),
        )
        for values in ([[0.0]], [], [[0.0, float("inf")]]):
            with self.assertRaises(ValueError):
                self.contract.joint_targets(values, joint_names=self.joints, unit="dimensionless")

    def test_invalid_declarations(self):
        for build in (
            lambda: Feature("x", 0, "rad"),
            lambda: Feature("", 1, "rad"),
            lambda: Feature("x", True, "rad"),
            lambda: ActionSpec(("a",), (), (0.0,)),
            lambda: ActionSpec(("a",), (1.0,), (0.0,), clip_raw=(1.0, -1.0)),
            lambda: PolicyContract((), self.actions),
            lambda: PolicyContract((Feature("x", 1, "raw"),), self.actions, batch_size=0),
            lambda: PolicyContract((Feature("x", 1, "raw"),), self.actions, max_age_s=-1),
            lambda: PolicyContract((Feature("x", 100, "raw"),), self.actions, max_elements=10),
        ):
            with self.assertRaises(ValueError):
                build()

    def test_target_overflow_rejected(self):
        c = PolicyContract((Feature("x", 1, "raw"),), ActionSpec(("a",), (1e308,), (1e308,)))
        with self.assertRaises(ValueError):
            c.joint_targets([[10.0]], joint_names=("a",), unit="dimensionless")

    def test_deterministic_random_affine_properties(self):
        rng = random.Random(42019)
        for _ in range(100):
            raw = [rng.uniform(-3, 3), rng.uniform(-3, 3)]
            targets = self.contract.joint_targets(
                [raw], joint_names=self.joints, unit="dimensionless"
            )[0]
            self.assertTrue(all(math.isfinite(x) for x in targets))
            self.assertAlmostEqual(targets[0], 1 + 0.5 * max(-1, min(1, raw[0])))
            self.assertAlmostEqual(targets[1], 0.25 - 2 * max(-1, min(1, raw[1])))
