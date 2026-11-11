"""Analytical synthetic sequences. No robot recordings, models or inference."""

import random
import unittest
from dataclasses import asdict

from robot_policy_tools import Signal, difference, mean_square, rms, tracking_mae


def scalar(values, dt=1.0, segments=None, unit="raw"):
    return Signal(
        tuple(i * dt for i in range(len(values))),
        tuple((x,) for x in values),
        unit,
        1 / dt,
        segments,
    )


class Metrics(unittest.TestCase):
    def test_known_tracking_error_and_direction(self):
        c = scalar([1.0, -1.0, 0.0, 1.0], unit="m/s")
        m = scalar([0.5, -0.25, 0.1, 1.25], unit="m/s")
        result = tracking_mae(c, m)
        self.assertAlmostEqual(result["all"][0], 0.4)
        self.assertEqual(result["axis_0/positive"], (0.375,))
        self.assertEqual(result["axis_0/negative"], (0.75,))
        self.assertEqual(result["axis_0/zero"], (0.1,))

    def test_yaw_unit_separate(self):
        result = tracking_mae(scalar([0.5, 0.5], unit="rad/s"), scalar([0.25, 0.75], unit="rad/s"))
        self.assertEqual(result["all"], (0.25,))
        with self.assertRaises(ValueError):
            tracking_mae(scalar([1.0, 2.0], unit="m/s"), scalar([1.0, 2.0], unit="rad/s"))

    def test_constant_zero_differences(self):
        for order in (1, 2, 3):
            result = difference(scalar([7.0] * 8, dt=0.02), order=order, divide_by_dt=True)
            self.assertEqual(result.mean_abs_by_axis, (0.0,))
            self.assertEqual(result.rms_by_axis, (0.0,))

    def test_linear_rate_and_increment_are_different(self):
        x = scalar([1.0, 2.0, 3.0, 4.0], dt=0.5)
        self.assertEqual(difference(x).mean_abs_by_axis, (1.0,))
        self.assertEqual(difference(x, divide_by_dt=True).mean_abs_by_axis, (2.0,))
        self.assertEqual(difference(x, order=2, divide_by_dt=True).rms_by_axis, (0.0,))

    def test_known_cubic_discrete_results(self):
        x = scalar([0.0, 1.0, 8.0, 27.0, 64.0])
        self.assertEqual(difference(x, order=1).mean_abs_by_axis, (16.0,))
        self.assertEqual(difference(x, order=2).mean_abs_by_axis, (12.0,))
        self.assertEqual(difference(x, order=3).rms_by_axis, (6.0,))
        y = scalar([0.0, 0.125, 1.0, 3.375, 8.0], dt=0.5)
        self.assertEqual(difference(y, order=3, divide_by_dt=True).rms_by_axis, (6.0,))

    def test_reset_transition_windows_excluded(self):
        x = scalar([0.0, 1.0, 500.0, 501.0], segments=("a", "a", "b", "b"))
        result = difference(x)
        self.assertEqual(result.count, 2)
        self.assertEqual(result.mean_abs_by_axis, (1.0,))
        result = difference(x, order=2)
        self.assertEqual(result.count, 0)
        self.assertIsNone(result.rms_by_axis)

    def test_short_sequence_is_unavailable_not_zero(self):
        for size in (1, 2, 3):
            result = difference(scalar([1.0] * size), order=3)
            self.assertEqual(result.count, 0)
            self.assertIsNone(asdict(result)["mean_abs_by_axis"])

    def test_empty_and_ragged_rejected(self):
        for args in (
            ((), (), "raw", 1.0),
            ((0.0, 1.0), ((0.0,), (1.0, 2.0)), "raw", 1.0),
            ((0.0,), ((0.0,), (1.0,)), "raw", 1.0),
        ):
            with self.assertRaises(ValueError):
                Signal(*args)

    def test_nonfinite_rejected(self):
        for x in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                scalar([0.0, x])

    def test_missing_irregular_duplicate_and_reverse_time(self):
        for times in ((0.0, 1.0, 3.0), (0.0, 0.9, 2.0), (0.0, 0.0, 1.0), (0.0, -1.0, 2.0)):
            with self.assertRaises(ValueError):
                Signal(times, ((0.0,),) * 3, "raw", 1.0)

    def test_invalid_frequency_and_tolerance(self):
        for hz in (0.0, -1.0, float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                Signal((0.0,), ((0.0,),), "raw", hz)
        with self.assertRaises(ValueError):
            Signal((0.0,), ((0.0,),), "raw", 1.0, timestamp_tolerance_s=0.1)

    def test_segment_identity_and_sampling_gaps(self):
        with self.assertRaises(ValueError):
            scalar([1.0, 2.0], segments=("a",))
        with self.assertRaises(ValueError):
            scalar([1.0, 2.0, 3.0], segments=("a", "b", "a"))
        x = Signal(
            (0.0, 1.0, 10.0, 11.0),
            ((0.0,), (1.0,), (100.0,), (101.0,)),
            "rad",
            1.0,
            ("a", "a", "b", "b"),
        )
        self.assertEqual(difference(x).count, 2)

    def test_rms_and_squared_torque_proxy(self):
        x = scalar([3.0, -3.0, 3.0], unit="rad/s")
        self.assertEqual(rms(x), (3.0,))
        torque = scalar([2.0, -2.0], unit="Nm")
        self.assertEqual(mean_square(torque), (4.0,))

    def test_tracking_alignment(self):
        a = scalar([1.0, 2.0])
        for b in (
            scalar([1.0, 2.0], dt=0.5),
            scalar([1.0, 2.0], segments=("a", "b")),
            Signal((0.0, 1.0), ((1.0, 2.0),) * 2, "raw", 1.0),
        ):
            with self.assertRaises(ValueError):
                tracking_mae(a, b)

    def test_derivative_arguments(self):
        for order in (0, 4, True):
            with self.assertRaises(ValueError):
                difference(scalar([1.0, 2.0]), order=order)

    def test_deterministic_random_translation_and_scaling_invariants(self):
        rng = random.Random(17831)
        for _ in range(50):
            values = [rng.uniform(-5, 5) for _ in range(8)]
            a = difference(scalar(values), order=2).rms_by_axis[0]
            translated = difference(scalar([x + 8 for x in values]), order=2).rms_by_axis[0]
            scaled = difference(scalar([-3 * x for x in values]), order=2).rms_by_axis[0]
            self.assertAlmostEqual(a, translated, places=12)
            self.assertAlmostEqual(3 * a, scaled, places=12)
            self.assertGreaterEqual(a, 0.0)

    def test_fixtures_keep_independent_denominators(self):
        fixtures = {"flat": scalar([0.0, 0.0]), "turn": scalar([0.0, 2.0, 4.0])}
        summary = {name: asdict(difference(x)) for name, x in fixtures.items()}
        self.assertEqual(summary["flat"]["count"], 1)
        self.assertEqual(summary["turn"]["count"], 2)
        self.assertEqual(summary["flat"]["mean_abs_by_axis"], (0.0,))
        self.assertEqual(summary["turn"]["mean_abs_by_axis"], (2.0,))
