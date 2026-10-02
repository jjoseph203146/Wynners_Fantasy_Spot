"""
End-to-end recursive closure validation.

Replay State N
    -> deterministic seven primitives
    -> Primitive Result Contract
    -> Replay Adapter
    -> Replay State N+1

SHADOW / VALIDATION ONLY.
"""

import copy
import unittest

from simulator_primitive_generator_v1.core import (
    build_primitive_result,
)
from simulator_primitive_generator_v1.replay_adapter import (
    append_transition_to_replay_input,
)
from simulator_primitive_generator_v1.scoring import (
    generate_team_primitives,
)
from simulator_recursive_state_replay_v1.core import (
    build as replay_build,
)
from simulator_recursive_state_replay_v1.fixtures import (
    child,
    sample,
)
from simulator_recursive_state_replay_v1.io import contract


def build_closure():
    c = contract()

    root_input = sample(c)
    root_state = replay_build(root_input, c)

    root_before = copy.deepcopy(root_state)
    input_before = copy.deepcopy(root_input)

    a_primitives = generate_team_primitives(
        root_state
    )

    # Replay V1 currently reconstructs the target-team perspective.
    # For closure validation, generate B deterministically by reversing
    # the team/opponent context and rebuilding the same root state.
    b_input = copy.deepcopy(root_input)
    b_input["context"]["team"] = "B"
    b_input["context"]["opponent"] = "A"

    b_state = replay_build(b_input, c)

    b_primitives = generate_team_primitives(
        b_state
    )

    parent_version = (
        root_state["state_manifest"]["state_version"]
    )

    result = build_primitive_result(
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
                "primitives": a_primitives,
            },
            {
                "team": "B",
                "primitives": b_primitives,
            },
        ],
        coach_bindings={
            "A": "CA",
            "B": "CB",
        },
        schedule={
            "game_id": "G2",
            "home_team": "A",
            "away_team": "B",
            "kickoff_utc":
                "2025-01-05T17:00:00Z",
        },
        source_version="snapshot-v1",
        source_hash="synthetic",
        parent_state_id=parent_version,
        parent_state_hash=parent_version,
        transition_version=(
            "primitive-recursive-transition-v1"
        ),
        simulated_at_utc="2025-01-05T21:00:00Z",
    )

    target = child(
        root_input,
        root_state,
        simulated=True,
    )

    # child() normally supplies its own synthetic G2.
    # Closure requires the generator to be the sole G2 source.
    target["transitions"] = []

    adapted = append_transition_to_replay_input(
        target,
        result,
    )

    child_state = replay_build(
        adapted,
        c,
    )

    return {
        "contract": c,
        "root_input": root_input,
        "root_input_before": input_before,
        "root_state": root_state,
        "root_state_before": root_before,
        "a_primitives": a_primitives,
        "b_primitives": b_primitives,
        "result": result,
        "adapted": adapted,
        "child_state": child_state,
    }


class RecursiveClosureTests(unittest.TestCase):

    def test_recursive_closure_completes(self):
        x = build_closure()

        self.assertEqual(
            x["child_state"][
                "transition_validation"
            ]["status"],
            "TRANSITION_COMPLETE",
        )

    def test_no_transition_issues(self):
        x = build_closure()

        self.assertEqual(
            x["child_state"][
                "transition_validation"
            ]["issues"],
            [],
        )

    def test_all_72_features_available(self):
        x = build_closure()

        child_state = x["child_state"]

        self.assertEqual(
            child_state["state_manifest"][
                "margin_count"
            ],
            72,
        )

        self.assertEqual(
            len(
                child_state[
                    "feature_reconstruction"
                ]
            ),
            72,
        )

        self.assertEqual(
            child_state[
                "transition_validation"
            ]["unavailable_features"],
            [],
        )

    def test_all_64_total_features_available(self):
        x = build_closure()

        self.assertEqual(
            x["child_state"]["state_manifest"][
                "total_count"
            ],
            64,
        )

    def test_generated_event_applied(self):
        x = build_closure()

        self.assertIn(
            "G2",
            x["child_state"]["lineage"][
                "applied_event_ids"
            ],
        )

    def test_generated_primitives_enter_history_exactly(self):
        x = build_closure()

        history = {
            event["game_id"]: event
            for event in x["child_state"][
                "lineage"
            ]["history"]
        }

        g2 = history["G2"]

        rows = {
            row["team"]: row
            for row in g2["teams"]
        }

        mapping = {
            "points": "points",
            "pass_attempts": "attempts",
            "carries": "carries",
            "passing_yards": "passing_yards",
            "passing_tds": "passing_tds",
            "rushing_yards": "rushing_yards",
            "rushing_tds": "rushing_tds",
        }

        for team, primitives in (
            ("A", x["a_primitives"]),
            ("B", x["b_primitives"]),
        ):
            for primitive, replay_name in (
                mapping.items()
            ):
                self.assertEqual(
                    rows[team][replay_name],
                    primitives[primitive],
                )

    def test_generated_event_is_simulated(self):
        x = build_closure()

        history = {
            event["game_id"]: event
            for event in x["child_state"][
                "lineage"
            ]["history"]
        }

        self.assertEqual(
            history["G2"]["provenance"],
            "SIMULATED",
        )

        self.assertEqual(
            history["G2"]["scenario_version"],
            "synthetic-v1",
        )

    def test_result_parent_bound_to_root_state(self):
        x = build_closure()

        version = (
            x["root_state"]["state_manifest"][
                "state_version"
            ]
        )

        self.assertEqual(
            x["result"]["parent_state_id"],
            version,
        )

        self.assertEqual(
            x["result"]["parent_state_hash"],
            version,
        )

    def test_replay_child_parent_is_root(self):
        x = build_closure()

        root_version = (
            x["root_state"]["state_manifest"][
                "state_version"
            ]
        )

        self.assertEqual(
            x["child_state"]["state_manifest"][
                "parent_state_version"
            ],
            root_version,
        )

    def test_simulated_availability_after_completion(self):
        x = build_closure()

        self.assertGreaterEqual(
            x["result"]["simulated_at_utc"],
            x["result"]["completion_utc"],
        )

    def test_result_hash_is_adapter_source_hash(self):
        x = build_closure()

        event = x["adapted"]["transitions"][0]

        self.assertEqual(
            event["source_hash"],
            x["result"]["primitive_result_hash"],
        )

    def test_root_state_not_mutated(self):
        x = build_closure()

        self.assertEqual(
            x["root_state"],
            x["root_state_before"],
        )

    def test_root_input_not_mutated(self):
        x = build_closure()

        self.assertEqual(
            x["root_input"],
            x["root_input_before"],
        )

    def test_child_advances_simulation_step(self):
        x = build_closure()

        self.assertEqual(
            x["root_state"]["state_manifest"][
                "context"
            ]["simulation_step"],
            0,
        )

        self.assertEqual(
            x["child_state"]["state_manifest"][
                "context"
            ]["simulation_step"],
            1,
        )

    def test_deterministic_child_state_version(self):
        a = build_closure()
        b = build_closure()

        self.assertEqual(
            a["result"]["primitive_result_hash"],
            b["result"]["primitive_result_hash"],
        )

        self.assertEqual(
            a["child_state"]["state_manifest"][
                "state_version"
            ],
            b["child_state"]["state_manifest"][
                "state_version"
            ],
        )

        self.assertEqual(
            a["child_state"],
            b["child_state"],
        )


if __name__ == "__main__":
    unittest.main()
