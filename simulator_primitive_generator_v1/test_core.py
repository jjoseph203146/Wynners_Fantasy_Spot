"""Focused deterministic tests for Primitive Game-State Generator V1."""

import copy
import unittest

from simulator_primitive_generator_v1.core import (
    PrimitiveContractError,
    build_primitive_result,
    content_hash,
    validate_primitive_result,
)


def sample(provenance="SIMULATED"):
    kwargs = {
        "season": 2026,
        "week": 4,
        "game_id": "2026_04_DEN_KC",
        "scenario_id": "scenario-001",
        "simulation_step": 1,
        "kickoff_utc": "2026-10-04T20:25:00Z",
        "completion_utc": "2026-10-04T23:35:00Z",
        "cutoff_utc": "2026-10-05T12:00:00Z",
        "provenance": provenance,
        "source_version": "synthetic-v1",
        "source_hash": "source-hash-001",
        "parent_state_id": "parent-state-001",
        "parent_state_hash": "parent-hash-001",
        "transition_version": "primitive-transition-v1",
        "teams": [
            {
                "team": "DEN",
                "primitives": {
                    "points": 24,
                    "pass_attempts": 34,
                    "carries": 27,
                    "passing_yards": 251.0,
                    "passing_tds": 2,
                    "rushing_yards": 118.0,
                    "rushing_tds": 1,
                },
            },
            {
                "team": "KC",
                "primitives": {
                    "points": 27,
                    "pass_attempts": 37,
                    "carries": 24,
                    "passing_yards": 286.0,
                    "passing_tds": 2,
                    "rushing_yards": 96.0,
                    "rushing_tds": 1,
                },
            },
        ],
        "coach_bindings": {
            "DEN": "coach-den",
            "KC": "coach-kc",
        },
        "schedule": {
            "game_id": "2026_04_DEN_KC",
            "home_team": "KC",
            "away_team": "DEN",
            "kickoff_utc": "2026-10-04T20:25:00Z",
        },
        "observed_at_utc": None,
        "simulated_at_utc": "2026-10-04T23:45:00Z",
    }

    if provenance == "OBSERVED":
        kwargs["observed_at_utc"] = "2026-10-04T23:45:00Z"
        kwargs["simulated_at_utc"] = None

    return build_primitive_result(**kwargs)


class PrimitiveGeneratorContractTests(unittest.TestCase):

    def test_simulated_contract_passes(self):
        result = sample()
        validate_primitive_result(result)

    def test_observed_contract_passes(self):
        result = sample("OBSERVED")
        validate_primitive_result(result)

    def test_deterministic_hash(self):
        a = sample()
        b = sample()
        self.assertEqual(a["primitive_result_hash"], b["primitive_result_hash"])

    def test_hash_changes_with_primitive(self):
        a = sample()
        kwargs = copy.deepcopy(a)
        kwargs.pop("primitive_result_hash")
        kwargs["teams"][0]["primitives"]["points"] = 25
        self.assertNotEqual(a["primitive_result_hash"], content_hash(kwargs))

    def test_unknown_primitive_is_allowed(self):
        result = sample()
        result["teams"][0]["primitives"]["passing_yards"] = None
        result.pop("primitive_result_hash")
        validate_primitive_result(result)

    def test_unknown_is_not_replaced_with_zero(self):
        result = sample()
        result["teams"][0]["primitives"]["passing_yards"] = None
        self.assertIsNone(result["teams"][0]["primitives"]["passing_yards"])

    def test_negative_primitive_rejected(self):
        result = sample()
        result["teams"][0]["primitives"]["carries"] = -1
        with self.assertRaises(PrimitiveContractError):
            validate_primitive_result(result)

    def test_fractional_integer_primitive_rejected(self):
        result = sample()
        result["teams"][0]["primitives"]["pass_attempts"] = 31.5
        with self.assertRaises(PrimitiveContractError):
            validate_primitive_result(result)

    def test_duplicate_team_rejected(self):
        result = sample()
        result["teams"][1]["team"] = "DEN"
        with self.assertRaises(PrimitiveContractError):
            validate_primitive_result(result)

    def test_schedule_team_mismatch_rejected(self):
        result = sample()
        result["schedule"]["home_team"] = "BUF"
        with self.assertRaises(PrimitiveContractError):
            validate_primitive_result(result)

    def test_schedule_game_mismatch_rejected(self):
        result = sample()
        result["schedule"]["game_id"] = "wrong-game"
        with self.assertRaises(PrimitiveContractError):
            validate_primitive_result(result)

    def test_observed_requires_observation_time(self):
        result = sample("OBSERVED")
        result["observed_at_utc"] = None
        with self.assertRaises(PrimitiveContractError):
            validate_primitive_result(result)

    def test_simulated_requires_simulation_time(self):
        result = sample()
        result["simulated_at_utc"] = None
        with self.assertRaises(PrimitiveContractError):
            validate_primitive_result(result)

    def test_observed_and_simulated_provenance_cannot_mix(self):
        result = sample()
        result["observed_at_utc"] = "2026-10-04T23:45:00Z"
        with self.assertRaises(PrimitiveContractError):
            validate_primitive_result(result)

    def test_late_observation_rejected(self):
        result = sample("OBSERVED")
        result["observed_at_utc"] = "2026-10-05T13:00:00Z"
        with self.assertRaises(PrimitiveContractError):
            validate_primitive_result(result)

    def test_completion_after_cutoff_rejected(self):
        result = sample()
        result["completion_utc"] = "2026-10-05T13:00:00Z"
        with self.assertRaises(PrimitiveContractError):
            validate_primitive_result(result)

    def test_simulated_before_completion_rejected(self):
        result = sample()
        result["simulated_at_utc"] = "2026-10-04T23:00:00Z"
        with self.assertRaisesRegex(
            PrimitiveContractError,
            "SIMULATED chronology requires completion <= simulated_at <= cutoff",
        ):
            validate_primitive_result(result)

    def test_missing_parent_rejected(self):
        result = sample()
        result["parent_state_hash"] = ""
        with self.assertRaises(PrimitiveContractError):
            validate_primitive_result(result)

    def test_missing_coach_binding_rejected(self):
        result = sample()
        del result["coach_bindings"]["DEN"]
        with self.assertRaises(PrimitiveContractError):
            validate_primitive_result(result)

    def test_null_coach_identity_is_explicitly_allowed(self):
        result = sample()
        result["coach_bindings"]["DEN"] = None
        validate_primitive_result(result)

    def test_input_is_not_mutated(self):
        teams = sample()["teams"]
        original = copy.deepcopy(teams)

        build_primitive_result(
            season=2026,
            week=4,
            game_id="2026_04_DEN_KC",
            scenario_id="scenario-002",
            simulation_step=1,
            kickoff_utc="2026-10-04T20:25:00Z",
            completion_utc="2026-10-04T23:35:00Z",
            cutoff_utc="2026-10-05T12:00:00Z",
            provenance="SIMULATED",
            teams=teams,
            coach_bindings={"DEN": "coach-den", "KC": "coach-kc"},
            schedule={
                "game_id": "2026_04_DEN_KC",
                "home_team": "KC",
                "away_team": "DEN",
                "kickoff_utc": "2026-10-04T20:25:00Z",
            },
            source_version="synthetic-v1",
            source_hash="source-hash-001",
            parent_state_id="parent-state-001",
            parent_state_hash="parent-hash-001",
            transition_version="primitive-transition-v1",
            simulated_at_utc="2026-10-04T23:45:00Z",
        )

        self.assertEqual(teams, original)


if __name__ == "__main__":
    unittest.main()
