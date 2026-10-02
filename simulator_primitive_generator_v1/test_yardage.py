import copy
import unittest

from simulator_primitive_generator_v1.intelligence import (
    PrimitiveIntelligenceError,
    estimate_team_volume,
    estimate_team_yardage,
)
from simulator_recursive_state_replay_v1.core import build
from simulator_recursive_state_replay_v1.fixtures import sample
from simulator_recursive_state_replay_v1.io import contract


def state():
    c = contract()
    return build(sample(c), c)


class PrimitiveYardageTests(unittest.TestCase):

    def test_runs_from_replay_state(self):
        result = estimate_team_yardage(state())

        self.assertEqual(result["team"], "A")
        self.assertEqual(result["opponent"], "B")

    def test_generates_integer_yards(self):
        result = estimate_team_yardage(state())

        self.assertIsInstance(
            result["passing_yards"],
            int,
        )

        self.assertIsInstance(
            result["rushing_yards"],
            int,
        )

    def test_nonnegative_yards(self):
        result = estimate_team_yardage(state())

        self.assertGreaterEqual(
            result["passing_yards"],
            0,
        )

        self.assertGreaterEqual(
            result["rushing_yards"],
            0,
        )

    def test_uses_generated_volume(self):
        s = state()
        volume = estimate_team_volume(s)

        result = estimate_team_yardage(
            s,
            volume,
        )

        self.assertEqual(
            result["pass_attempts"],
            volume["pass_attempts"],
        )

        self.assertEqual(
            result["carries"],
            volume["carries"],
        )

    def test_deterministic(self):
        a = estimate_team_yardage(state())
        b = estimate_team_yardage(state())

        self.assertEqual(a, b)

    def test_does_not_mutate_state(self):
        s = state()
        before = copy.deepcopy(s)

        estimate_team_yardage(s)

        self.assertEqual(s, before)

    def test_does_not_mutate_volume(self):
        s = state()
        volume = estimate_team_volume(s)
        before = copy.deepcopy(volume)

        estimate_team_yardage(
            s,
            volume,
        )

        self.assertEqual(volume, before)

    def test_team_mismatch_rejected(self):
        s = state()
        volume = estimate_team_volume(s)
        volume["team"] = "WRONG"

        with self.assertRaisesRegex(
            PrimitiveIntelligenceError,
            "VOLUME_TEAM_MISMATCH",
        ):
            estimate_team_yardage(
                s,
                volume,
            )

    def test_missing_efficiency_feature_fails_closed(self):
        s = state()

        for row in s["feature_reconstruction"]:
            if (
                row["name"]
                == "team_passing_yards_avg_3"
            ):
                row["value"] = None
                break

        with self.assertRaisesRegex(
            PrimitiveIntelligenceError,
            "REQUIRED_FEATURE_UNAVAILABLE",
        ):
            estimate_team_yardage(s)

    def test_zero_efficiency_denominator_fails_closed(self):
        s = state()

        for row in s["feature_reconstruction"]:
            if (
                row["name"]
                == "team_pass_attempts_avg_3"
            ):
                row["value"] = 0
                break

        with self.assertRaisesRegex(
            PrimitiveIntelligenceError,
            "INVALID_EFFICIENCY_DENOMINATOR",
        ):
            estimate_team_yardage(s)

    def test_efficiency_guardrails(self):
        result = estimate_team_yardage(state())

        self.assertGreaterEqual(
            result["passing_yards_per_attempt"],
            3.0,
        )

        self.assertLessEqual(
            result["passing_yards_per_attempt"],
            12.0,
        )

        self.assertGreaterEqual(
            result["rushing_yards_per_carry"],
            2.0,
        )

        self.assertLessEqual(
            result["rushing_yards_per_carry"],
            8.0,
        )

    def test_uses_only_available_three_game_yardage_contract(self):
        result = estimate_team_yardage(state())

        self.assertEqual(
            result["diagnostics"]["efficiency_window_games"],
            3,
        )

    def test_expected_synthetic_efficiency(self):
        result = estimate_team_yardage(state())

        # Frozen Replay fixture semantics:
        #
        # A offensive passing efficiency:
        #   205 / 30.5
        #
        # team_opponent_pass_yards_allowed_avg_3 resolves from
        # the historical "other" row relative to B, which is A:
        #   205 / B's 25 pass attempts
        #
        # Do not substitute B's own 150 passing yards here.
        expected_pass = (
            (205 / 30.5) + (205 / 25)
        ) / 2.0

        # The same frozen reconstruction semantics apply to rushing:
        #   A offense = 100 / 20
        #   opponent-allowed side = A's 100 / B's 25 carries
        expected_rush = (
            (100 / 20) + (100 / 25)
        ) / 2.0

        self.assertAlmostEqual(
            result["passing_yards_per_attempt"],
            expected_pass,
            places=12,
        )

        self.assertAlmostEqual(
            result["rushing_yards_per_carry"],
            expected_rush,
            places=12,
        )

    def test_engine_version_explicit(self):
        result = estimate_team_yardage(state())

        self.assertEqual(
            result["engine_version"],
            "PRIMITIVE_INTELLIGENCE_V1_YARDAGE_002",
        )


if __name__ == "__main__":
    unittest.main()
