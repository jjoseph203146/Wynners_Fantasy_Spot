"""Regression tests for RotoWire-primary QB starter authority."""

import unittest
from datetime import datetime, timezone

import pandas as pd

from .core import qualified_starters


NOW = datetime(
    2030,
    10,
    6,
    16,
    0,
    tzinfo=timezone.utc,
)


def meta():
    return {
        "contract": "WFS_STARTER_VERIFICATION_CURRENT_V1",
        "source_contract": "WFS_STARTER_VERIFICATION_V1_1",
        "analysis_only": True,
        "production_influence": False,
        "season": 2030,
        "week": 3,
        "game_type": "REG",
        "identity_gate": "PASS",
        "identity_unresolved_count": 0,
        "identity_ambiguous_count": 0,
        "generated_at_utc": NOW.isoformat(),
        "role_rows": 1,
    }


def context(team):
    return {
        "season": 2030,
        "week": 3,
        "pregame_teams": [team],
    }


def row(
    team,
    *,
    depth,
    rotowire,
    verification,
    depth_gate="ALLOW",
    rw_gate="ALLOW",
    depth_available=True,
    rw_available=True,
):
    return pd.DataFrame([{
        "team": team,
        "position": "QB",
        "depth_gsis_id": depth,
        "rotowire_gsis_id": rotowire,
        "verification_status": verification,
        "comparable_to_rotowire": True,
        "depth_availability_present": depth_available,
        "rw_availability_present": rw_available,
        "depth_injury_gate": depth_gate,
        "rw_injury_gate": rw_gate,
    }])


class RotoWirePrimaryTests(unittest.TestCase):

    def test_agreement_uses_rotowire(self):
        player = "00-0039918"

        qualified, _ = qualified_starters(
            row(
                "CHI",
                depth=player,
                rotowire=player,
                verification="AGREE",
            ),
            meta(),
            context("CHI"),
            NOW,
        )

        self.assertEqual(
            qualified["CHI"],
            player,
        )

    def test_was_depth_disagreement_does_not_veto_rotowire(self):
        qualified, _ = qualified_starters(
            row(
                "WAS",
                depth="00-0039910",
                rotowire="00-0032268",
                verification="DEPTH_STARTER_BLOCKED",
                depth_gate="BLOCK",
                rw_gate="ALLOW",
            ),
            meta(),
            context("WAS"),
            NOW,
        )

        self.assertEqual(
            qualified["WAS"],
            "00-0032268",
        )

    def test_rotowire_block_is_hard_veto(self):
        with self.assertRaises(Exception) as caught:
            qualified_starters(
                row(
                    "CHI",
                    depth="00-0039918",
                    rotowire="00-0039918",
                    verification="BLOCKED_STARTER",
                    depth_gate="BLOCK",
                    rw_gate="BLOCK",
                ),
                meta(),
                context("CHI"),
                NOW,
            )

        self.assertIn(
            "ROTOWIRE_STARTER_BLOCKED",
            str(caught.exception),
        )

    def test_depth_block_does_not_block_eligible_rotowire(self):
        qualified, _ = qualified_starters(
            row(
                "WAS",
                depth="00-0039910",
                rotowire="00-0032268",
                verification="DEPTH_STARTER_BLOCKED",
                depth_gate="BLOCK",
                rw_gate="ALLOW",
                depth_available=True,
                rw_available=True,
            ),
            meta(),
            context("WAS"),
            NOW,
        )

        self.assertEqual(
            qualified["WAS"],
            "00-0032268",
        )

    def test_missing_rotowire_availability_is_healthy_by_absence(self):
        qualified, _ = qualified_starters(
            row(
                "CHI",
                depth="00-0039918",
                rotowire="00-0039918",
                verification="AGREE",
                rw_available=False,
                rw_gate="NO_CONSENSUS_ROW",
            ),
            meta(),
            context("CHI"),
            NOW,
        )

        self.assertEqual(
            qualified["CHI"],
            "00-0039918",
        )

    def test_missing_rotowire_availability_with_unexpected_gate_fails_closed(self):
        with self.assertRaises(Exception) as caught:
            qualified_starters(
                row(
                    "CHI",
                    depth="00-0039918",
                    rotowire="00-0039918",
                    verification="AGREE",
                    rw_available=False,
                    rw_gate="UNKNOWN_GATE",
                ),
                meta(),
                context("CHI"),
                NOW,
            )

        self.assertIn(
            "ROTOWIRE_STARTER_BLOCKED",
            str(caught.exception),
        )


if __name__ == "__main__":
    unittest.main()
