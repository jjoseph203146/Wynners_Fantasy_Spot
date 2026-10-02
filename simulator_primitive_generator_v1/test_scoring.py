import copy
import unittest

from simulator_primitive_generator_v1.intelligence import (
    PrimitiveIntelligenceError,
    estimate_team_volume,
    estimate_team_yardage,
)
from simulator_primitive_generator_v1.scoring import (
    estimate_team_scoring,
    generate_team_primitives,
)
from simulator_recursive_state_replay_v1.core import build
from simulator_recursive_state_replay_v1.fixtures import sample
from simulator_recursive_state_replay_v1.io import contract


def state():
    c = contract()
    return build(sample(c), c)


class PrimitiveScoringTests(unittest.TestCase):

    def test_scoring_runs(self):
        result = estimate_team_scoring(state())

        self.assertEqual(result["team"], "A")
        self.assertEqual(result["opponent"], "B")

    def test_tds_are_nonnegative_integers(self):
        result = estimate_team_scoring(state())

        self.assertIsInstance(
            result["passing_tds"],
            int,
        )
        self.assertIsInstance(
            result["rushing_tds"],
            int,
        )

        self.assertGreaterEqual(
            result["passing_tds"],
            0,
        )
        self.assertGreaterEqual(
            result["rushing_tds"],
            0,
        )

    def test_points_nonnegative_integer(self):
        result = estimate_team_scoring(state())

        self.assertIsInstance(
            result["points"],
            int,
        )
        self.assertGreaterEqual(
            result["points"],
            0,
        )

    def test_points_respect_td_floor(self):
        result = estimate_team_scoring(state())

        td_floor = 6 * (
            result["passing_tds"]
            + result["rushing_tds"]
        )

        self.assertGreaterEqual(
            result["points"],
            td_floor,
        )

    def test_scoring_residual_reconciles(self):
        result = estimate_team_scoring(state())

        td_points = (
            result["diagnostics"][
                "offensive_td_points"
            ]
        )

        residual = (
            result["diagnostics"][
                "non_td_scoring_residual"
            ]
        )

        self.assertEqual(
            result["points"],
            td_points + residual,
        )

    def test_expected_synthetic_td_rates(self):
        result = estimate_team_scoring(state())
        d = result["diagnostics"]

        expected_pass = (
            (2.0 / 30.5)
            + (2.0 / 25.0)
        ) / 2.0

        expected_rush = (
            (1.0 / 20.0)
            + (1.0 / 25.0)
        ) / 2.0

        self.assertAlmostEqual(
            d["blended_pass_td_rate"],
            expected_pass,
            places=12,
        )

        self.assertAlmostEqual(
            d["blended_rush_td_rate"],
            expected_rush,
            places=12,
        )

    def test_expected_synthetic_points_baseline(self):
        result = estimate_team_scoring(state())

        self.assertEqual(
            result["diagnostics"][
                "raw_points_expectation"
            ],
            21.0,
        )

    def test_deterministic(self):
        a = estimate_team_scoring(state())
        b = estimate_team_scoring(state())

        self.assertEqual(a, b)

    def test_does_not_mutate_state(self):
        s = state()
        before = copy.deepcopy(s)

        estimate_team_scoring(s)

        self.assertEqual(s, before)

    def test_does_not_mutate_dependencies(self):
        s = state()
        volume = estimate_team_volume(s)
        yardage = estimate_team_yardage(
            s,
            volume,
        )

        volume_before = copy.deepcopy(volume)
        yardage_before = copy.deepcopy(yardage)

        estimate_team_scoring(
            s,
            volume,
            yardage,
        )

        self.assertEqual(
            volume,
            volume_before,
        )
        self.assertEqual(
            yardage,
            yardage_before,
        )

    def test_yardage_volume_mismatch_rejected(self):
        s = state()
        volume = estimate_team_volume(s)
        yardage = estimate_team_yardage(
            s,
            volume,
        )

        yardage["pass_attempts"] += 1

        with self.assertRaisesRegex(
            PrimitiveIntelligenceError,
            "YARDAGE_PASS_ATTEMPTS_MISMATCH",
        ):
            estimate_team_scoring(
                s,
                volume,
                yardage,
            )

    def test_missing_td_feature_fails_closed(self):
        s = state()

        for row in s["feature_reconstruction"]:
            if (
                row["name"]
                == "team_passing_tds_avg_3"
            ):
                row["value"] = None
                break

        with self.assertRaisesRegex(
            PrimitiveIntelligenceError,
            "REQUIRED_FEATURE_UNAVAILABLE",
        ):
            estimate_team_scoring(s)

    def test_zero_td_denominator_fails_closed(self):
        s = state()

        # Build valid upstream dependencies first so this test isolates
        # the Stage 3 scoring denominator rather than correctly failing
        # earlier inside Stage 2 yardage efficiency.
        volume = estimate_team_volume(s)
        yardage = estimate_team_yardage(
            s,
            volume,
        )

        for row in s["feature_reconstruction"]:
            if (
                row["name"]
                == "team_pass_attempts_avg_3"
            ):
                row["value"] = 0
                break

        with self.assertRaisesRegex(
            PrimitiveIntelligenceError,
            "INVALID_SCORING_DENOMINATOR:OFFENSE_PASS_TD",
        ):
            estimate_team_scoring(
                s,
                volume,
                yardage,
            )

    def test_complete_primitive_keys_exact(self):
        result = generate_team_primitives(
            state()
        )

        self.assertEqual(
            set(result),
            {
                "points",
                "pass_attempts",
                "carries",
                "passing_yards",
                "passing_tds",
                "rushing_yards",
                "rushing_tds",
            },
        )

    def test_complete_primitives_are_integers(self):
        result = generate_team_primitives(
            state()
        )

        for value in result.values():
            self.assertIsInstance(
                value,
                int,
            )

    def test_complete_primitives_nonnegative(self):
        result = generate_team_primitives(
            state()
        )

        for value in result.values():
            self.assertGreaterEqual(
                value,
                0,
            )

    def test_complete_generator_deterministic(self):
        a = generate_team_primitives(state())
        b = generate_team_primitives(state())

        self.assertEqual(a, b)

    def test_engine_version_explicit(self):
        result = estimate_team_scoring(state())

        self.assertEqual(
            result["engine_version"],
            "PRIMITIVE_INTELLIGENCE_V1_SCORING_001",
        )


if __name__ == "__main__":
    unittest.main()
