"""Focused tests for Primitive Generator V1 -> Replay V1 adapter."""

import copy
import unittest

from simulator_primitive_generator_v1.core import build_primitive_result
from simulator_primitive_generator_v1.replay_adapter import (
    ReplayAdapterError,
    append_transition_to_replay_input,
    to_replay_transition,
)
from simulator_recursive_state_replay_v1.core import build as replay_build
from simulator_recursive_state_replay_v1.fixtures import child, sample
from simulator_recursive_state_replay_v1.io import contract


def primitive_result():
    return build_primitive_result(
        season=2025,
        week=1,
        game_id="G2",
        scenario_id="synthetic-v1",
        simulation_step=1,
        kickoff_utc="2025-01-05T17:00:00Z",
        completion_utc="2025-01-05T20:00:00Z",
        cutoff_utc="2025-01-12T16:00:00Z",
        provenance="SIMULATED",
        teams=[
            {
                "team": "A",
                "primitives": {
                    "points": 24,
                    "pass_attempts": 32,
                    "carries": 21,
                    "passing_yards": 230,
                    "passing_tds": 2,
                    "rushing_yards": 105,
                    "rushing_tds": 1,
                },
            },
            {
                "team": "B",
                "primitives": {
                    "points": 17,
                    "pass_attempts": 27,
                    "carries": 24,
                    "passing_yards": 175,
                    "passing_tds": 1,
                    "rushing_yards": 115,
                    "rushing_tds": 1,
                },
            },
        ],
        coach_bindings={"A": "CA", "B": "CB"},
        schedule={
            "game_id": "G2",
            "home_team": "A",
            "away_team": "B",
            "kickoff_utc": "2025-01-05T17:00:00Z",
        },
        source_version="snapshot-v1",
        source_hash="synthetic",
        parent_state_id="primitive-parent-state",
        parent_state_hash="primitive-parent-hash",
        transition_version="primitive-transition-v1",
        simulated_at_utc="2025-01-05T21:00:00Z",
    )



def primitive_result_for_parent(parent_version):
    base = primitive_result()

    return build_primitive_result(
        season=base["season"],
        week=base["week"],
        game_id=base["game_id"],
        scenario_id=base["scenario_id"],
        simulation_step=base["simulation_step"],
        kickoff_utc=base["kickoff_utc"],
        completion_utc=base["completion_utc"],
        cutoff_utc=base["cutoff_utc"],
        provenance=base["provenance"],
        teams=base["teams"],
        coach_bindings=base["coach_bindings"],
        schedule=base["schedule"],
        source_version=base["source_version"],
        source_hash=base["source_hash"],
        parent_state_id=parent_version,
        parent_state_hash=base["parent_state_hash"],
        transition_version=base["transition_version"],
        simulated_at_utc=base["simulated_at_utc"],
    )


class ReplayAdapterTests(unittest.TestCase):

    def test_pass_attempts_maps_to_attempts(self):
        event = to_replay_transition(primitive_result())
        self.assertEqual(event["teams"][0]["attempts"], 32)
        self.assertNotIn("pass_attempts", event["teams"][0])

    def test_all_seven_primitives_map(self):
        event = to_replay_transition(primitive_result())
        row = event["teams"][0]

        self.assertEqual(
            set(row),
            {
                "team",
                "coach_id",
                "points",
                "attempts",
                "carries",
                "passing_yards",
                "passing_tds",
                "rushing_yards",
                "rushing_tds",
            },
        )

    def test_simulated_provenance_maps(self):
        result = primitive_result()
        event = to_replay_transition(result)

        self.assertEqual(event["provenance"], "SIMULATED")
        self.assertEqual(event["scenario_version"], "synthetic-v1")
        self.assertEqual(
            event["observation_time"],
            result["simulated_at_utc"],
        )

    def test_result_hash_becomes_event_source_hash(self):
        result = primitive_result()
        event = to_replay_transition(result)

        self.assertEqual(
            event["source_hash"],
            result["primitive_result_hash"],
        )

    def test_null_primitive_remains_null(self):
        result = primitive_result()
        result["teams"][0]["primitives"]["passing_yards"] = None
        result.pop("primitive_result_hash")

        # Rebuild so the hash corresponds to the changed validated content.
        rebuilt = build_primitive_result(
            season=result["season"],
            week=result["week"],
            game_id=result["game_id"],
            scenario_id=result["scenario_id"],
            simulation_step=result["simulation_step"],
            kickoff_utc=result["kickoff_utc"],
            completion_utc=result["completion_utc"],
            cutoff_utc=result["cutoff_utc"],
            provenance=result["provenance"],
            teams=result["teams"],
            coach_bindings=result["coach_bindings"],
            schedule=result["schedule"],
            source_version=result["source_version"],
            source_hash=result["source_hash"],
            parent_state_id=result["parent_state_id"],
            parent_state_hash=result["parent_state_hash"],
            transition_version=result["transition_version"],
            simulated_at_utc=result["simulated_at_utc"],
        )

        event = to_replay_transition(rebuilt)

        self.assertIsNone(event["teams"][0]["passing_yards"])

    def test_adapter_does_not_mutate_result(self):
        result = primitive_result()
        before = copy.deepcopy(result)

        to_replay_transition(result)

        self.assertEqual(result, before)

    def test_append_does_not_mutate_replay_input(self):
        c = contract()
        root_input = sample(c)
        root_state = replay_build(root_input, c)

        target = child(root_input, root_state, simulated=True)

        # Remove fixture-generated transition so our adapter supplies it.
        target["transitions"] = []

        before = copy.deepcopy(target)

        parent_version = root_state[
            "state_manifest"
        ]["state_version"]

        adapted = append_transition_to_replay_input(
            target,
            primitive_result_for_parent(
                parent_version
            ),
        )

        self.assertEqual(target, before)
        self.assertEqual(len(adapted["transitions"]), 1)

    def test_wrong_result_parent_id_rejected(self):
        c = contract()
        root_input = sample(c)
        root_state = replay_build(
            root_input,
            c,
        )

        target = child(
            root_input,
            root_state,
            simulated=True,
        )
        target["transitions"] = []

        parent_version = root_state[
            "state_manifest"
        ]["state_version"]

        result = primitive_result_for_parent(
            parent_version
        )

        # Rebuild with a valid contract but incorrect Replay parent.
        result = build_primitive_result(
            season=result["season"],
            week=result["week"],
            game_id=result["game_id"],
            scenario_id=result["scenario_id"],
            simulation_step=result["simulation_step"],
            kickoff_utc=result["kickoff_utc"],
            completion_utc=result["completion_utc"],
            cutoff_utc=result["cutoff_utc"],
            provenance=result["provenance"],
            teams=result["teams"],
            coach_bindings=result["coach_bindings"],
            schedule=result["schedule"],
            source_version=result["source_version"],
            source_hash=result["source_hash"],
            parent_state_id="WRONG-PARENT",
            parent_state_hash=result[
                "parent_state_hash"
            ],
            transition_version=result[
                "transition_version"
            ],
            simulated_at_utc=result[
                "simulated_at_utc"
            ],
        )

        with self.assertRaisesRegex(
            ReplayAdapterError,
            "RESULT_PARENT_STATE_ID_MISMATCH",
        ):
            append_transition_to_replay_input(
                target,
                result,
            )

    def test_duplicate_event_rejected(self):
        c = contract()
        root_input = sample(c)
        root_state = replay_build(root_input, c)

        target = child(root_input, root_state, simulated=True)
        target["transitions"] = []

        parent_version = root_state[
            "state_manifest"
        ]["state_version"]

        once = append_transition_to_replay_input(
            target,
            primitive_result_for_parent(
                parent_version
            ),
        )

        with self.assertRaises(ReplayAdapterError):
            append_transition_to_replay_input(
                once,
                primitive_result_for_parent(
                    parent_version
                ),
            )

    def test_frozen_replay_accepts_adapter_transition(self):
        c = contract()

        root_input = sample(c)
        root_state = replay_build(root_input, c)

        target = child(
            root_input,
            root_state,
            simulated=True,
        )

        # child() normally creates its own synthetic G2 transition.
        # Replace only that transition with our adapter-generated G2 event.
        target["transitions"] = []

        parent_version = root_state[
            "state_manifest"
        ]["state_version"]

        adapted = append_transition_to_replay_input(
            target,
            primitive_result_for_parent(
                parent_version
            ),
        )

        output = replay_build(adapted, c)

        self.assertEqual(
            output["transition_validation"]["status"],
            "TRANSITION_COMPLETE",
        )

        self.assertIn(
            "G2",
            output["lineage"]["applied_event_ids"],
        )

        history = {
            event["game_id"]: event
            for event in output["lineage"]["history"]
        }

        self.assertEqual(
            history["G2"]["teams"][0]["attempts"],
            32,
        )

        self.assertEqual(
            history["G2"]["provenance"],
            "SIMULATED",
        )

    def test_replay_state_contains_no_unavailable_72_features(self):
        c = contract()

        root_input = sample(c)
        root_state = replay_build(root_input, c)

        target = child(root_input, root_state, simulated=True)
        target["transitions"] = []

        parent_version = root_state[
            "state_manifest"
        ]["state_version"]

        adapted = append_transition_to_replay_input(
            target,
            primitive_result_for_parent(
                parent_version
            ),
        )

        output = replay_build(adapted, c)

        self.assertEqual(
            output["transition_validation"]["unavailable_features"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
