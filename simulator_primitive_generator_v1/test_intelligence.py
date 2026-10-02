import copy
import unittest

from simulator_primitive_generator_v1.intelligence import (
    PrimitiveIntelligenceError,
    estimate_team_volume,
)
from simulator_recursive_state_replay_v1.core import build
from simulator_recursive_state_replay_v1.fixtures import sample
from simulator_recursive_state_replay_v1.io import contract


def replay_state():
    c = contract()
    return build(sample(c), c)


class PrimitiveIntelligenceVolumeTests(unittest.TestCase):

    def test_volume_engine_runs_on_frozen_replay_state(self):
        result = estimate_team_volume(replay_state())

        self.assertEqual(result["team"], "A")
        self.assertEqual(result["opponent"], "B")

    def test_volume_conservation(self):
        result = estimate_team_volume(replay_state())

        self.assertEqual(
            result["pass_attempts"] + result["carries"],
            result["offensive_plays"],
        )

    def test_rates_sum_to_one(self):
        result = estimate_team_volume(replay_state())

        self.assertAlmostEqual(
            result["pass_rate"] + result["rush_rate"],
            1.0,
            places=12,
        )

    def test_deterministic(self):
        a = estimate_team_volume(replay_state())
        b = estimate_team_volume(replay_state())

        self.assertEqual(a, b)

    def test_does_not_mutate_replay_state(self):
        state = replay_state()
        before = copy.deepcopy(state)

        estimate_team_volume(state)

        self.assertEqual(state, before)

    def test_expected_synthetic_play_count(self):
        result = estimate_team_volume(replay_state())

        # Team A:
        # attempts avg = 30.5
        # carries avg = 20
        # plays = 50.5
        #
        # Team B:
        # attempts avg = 25
        # carries avg = 25
        # plays = 50
        #
        # midpoint = 50.25 -> Python round -> 50
        self.assertEqual(result["offensive_plays"], 50)

    def test_attempts_and_carries_are_nonnegative(self):
        result = estimate_team_volume(replay_state())

        self.assertGreaterEqual(result["pass_attempts"], 0)
        self.assertGreaterEqual(result["carries"], 0)

    def test_missing_required_feature_fails_closed(self):
        state = replay_state()

        for row in state["feature_reconstruction"]:
            if row["name"] == "team_offensive_plays_avg_3":
                row["value"] = None
                break

        with self.assertRaisesRegex(
            PrimitiveIntelligenceError,
            "REQUIRED_FEATURE_UNAVAILABLE",
        ):
            estimate_team_volume(state)

    def test_no_market_inputs_in_output(self):
        result = estimate_team_volume(replay_state())

        forbidden = {
            "spread",
            "market_spread",
            "market_total",
            "vegas_total",
            "vegas",
            "moneyline",
        }

        self.assertTrue(forbidden.isdisjoint(result.keys()))

    def test_engine_version_explicit(self):
        result = estimate_team_volume(replay_state())

        self.assertEqual(
            result["engine_version"],
            "PRIMITIVE_INTELLIGENCE_V1_VOLUME_001",
        )


if __name__ == "__main__":
    unittest.main()
