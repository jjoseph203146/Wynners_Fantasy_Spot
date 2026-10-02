"""V1.3 Milestone 2D RED_ZONE_ROLE claim-gate regression tests."""

import unittest

from shadow_player_analysis_v1_3.claims import build_claims


class RedZoneRoleV13Tests(unittest.TestCase):

    def event(self, text, **overrides):
        row = {
            "record_id": "record-red-zone-1",
            "event_key": "event-red-zone-1",
            "source": "SYNTHETIC_SOURCE",
            "source_event_id": "native-red-zone-1",
            "origin_id": "SYNTHETIC_ORIGIN_A",
            "content_hash": "content-red-zone-1",
            "published_at_utc": "2026-09-27T15:00:00Z",
            "retrieved_at_utc": "2026-09-27T15:01:00Z",
            "first_seen_at_utc": "2026-09-27T15:01:00Z",
            "evidence_text": text,
        }
        row.update(overrides)
        return row

    def evidence(self, **overrides):
        row = {
            "record_id": "record-red-zone-1",
            "event_key": "event-red-zone-1",
            "source_approved": True,
            "pregame_eligible": True,
            "temporal_status": "UNAMBIGUOUS_SOURCE_TIMESTAMP",
            "temporal_confidence": "RESOLVED",
            "evidence_kind": "REPORTED_EXPECTATION",
            "evidence_types": ["RED_ZONE_ROLE"],
            "relevance": "HIGH",
            "reason_codes": [],
        }
        row.update(overrides)
        return row

    def link(self, quote, **overrides):
        row = {
            "record_id": "record-red-zone-1",
            "link_id": "link-red-zone-1",
            "entity_type": "PLAYER",
            "gsis_id": "00-TEST-WR1",
            "identity_status": "RESOLVED",
            "identity_reason": "EXACT_UNIQUE_NAME_VALIDATED",
            "evidence_kind": "REPORTED_EXPECTATION",
            "evidence_types": ["RED_ZONE_ROLE"],
            "relevance": "HIGH",
            "mention": {
                "type": "PLAYER",
                "name": "Example Receiver",
                "evidence_quote": quote,
                "relationship_scope": "CLAUSE",
            },
            "reason_codes": [],
        }
        row.update(overrides)
        return row

    def build(
        self,
        text,
        quote=None,
        evidence=None,
        link=None,
        event=None,
    ):
        quote = text if quote is None else quote

        return build_claims(
            source_events=[
                event if event is not None else self.event(text)
            ],
            classified_evidence=[
                evidence if evidence is not None else self.evidence()
            ],
            entity_links=[
                link if link is not None else self.link(quote)
            ],
            cutoff_utc="2026-09-27T16:00:00Z",
        )

    # ---------------------------------------------------------------
    # Positive forward-looking RED_ZONE_ROLE propositions.
    # Expected RED until Milestone 2D grammar exists.
    # ---------------------------------------------------------------

    def test_expected_larger_red_zone_role_creates_claim(self):
        text = (
            "Example Receiver is expected to have a larger "
            "red zone role next game"
        )
        claims = self.build(text)["claims"]

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["signal_type"], "RED_ZONE_ROLE")
        self.assertEqual(claims[0]["player_name"], "Example Receiver")
        self.assertEqual(claims[0]["gsis_id"], "00-TEST-WR1")

    def test_expected_major_red_zone_role_creates_claim(self):
        text = (
            "Example Receiver is expected to have a major "
            "red zone role this week"
        )
        claims = self.build(text)["claims"]

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["signal_type"], "RED_ZONE_ROLE")

    def test_coach_plan_larger_red_zone_role_creates_claim(self):
        text = (
            "Coach plans to give Example Receiver a larger "
            "red zone role next game"
        )
        claims = self.build(text)["claims"]

        self.assertEqual(len(claims), 1)

    # ---------------------------------------------------------------
    # Historical / non-equivalent / vague propositions remain closed.
    # ---------------------------------------------------------------

    def test_historical_red_zone_role_rejected(self):
        claims = self.build(
            "Example Receiver had a major red zone role last game"
        )["claims"]

        self.assertEqual(claims, [])

    def test_last_week_red_zone_role_rejected(self):
        claims = self.build(
            "Example Receiver had a larger red zone role last week"
        )["claims"]

        self.assertEqual(claims, [])

    def test_red_zone_targets_not_equivalent_rejected(self):
        claims = self.build(
            "Example Receiver had four red zone targets last game"
        )["claims"]

        self.assertEqual(claims, [])

    def test_red_zone_touchdown_not_equivalent_rejected(self):
        claims = self.build(
            "Example Receiver scored two red zone touchdowns last game"
        )["claims"]

        self.assertEqual(claims, [])

    def test_vague_red_zone_role_rejected(self):
        claims = self.build(
            "Example Receiver should be involved in the red zone role"
        )["claims"]

        self.assertEqual(claims, [])

    def test_smaller_red_zone_role_rejected(self):
        claims = self.build(
            "Example Receiver is expected to have a smaller "
            "red zone role next game"
        )["claims"]

        self.assertEqual(claims, [])

    def test_question_rejected(self):
        claims = self.build(
            "Is Example Receiver expected to have a larger "
            "red zone role next game?"
        )["claims"]

        self.assertEqual(claims, [])

    def test_ambiguous_timestamp_rejected(self):
        text = (
            "Example Receiver is expected to have a larger "
            "red zone role next game"
        )

        evidence = self.evidence(
            pregame_eligible=False,
            temporal_status="AMBIGUOUS_SOURCE_TIMESTAMP",
            temporal_confidence="AMBIGUOUS",
        )

        claims = self.build(
            text,
            evidence=evidence,
        )["claims"]

        self.assertEqual(claims, [])


class RedZoneRoleAdversarialV13Tests(RedZoneRoleV13Tests):

    def test_no_longer_plans_rejected(self):
        claims = self.build(
            "Coach no longer plans to give Example Receiver "
            "a larger red zone role next game"
        )["claims"]

        self.assertEqual(claims, [])

    def test_withdrawn_plan_rejected(self):
        claims = self.build(
            "Coach plans to give Example Receiver a larger red zone role "
            "next game, but that plan was withdrawn"
        )["claims"]

        self.assertEqual(claims, [])

    def test_withdrawn_plans_rejected(self):
        claims = self.build(
            "Coach plans to give Example Receiver a larger red zone role "
            "next game, but those plans were withdrawn"
        )["claims"]

        self.assertEqual(claims, [])

    def test_not_expected_rejected(self):
        claims = self.build(
            "Example Receiver is not expected to have a larger "
            "red zone role next game"
        )["claims"]

        self.assertEqual(claims, [])

    def test_no_longer_expected_rejected(self):
        claims = self.build(
            "Example Receiver is no longer expected to have a larger "
            "red zone role next game"
        )["claims"]

        self.assertEqual(claims, [])

    def test_does_not_plan_rejected(self):
        claims = self.build(
            "Coach does not plan to give Example Receiver "
            "a larger red zone role next game"
        )["claims"]

        self.assertEqual(claims, [])

    def test_historical_expectation_rejected(self):
        claims = self.build(
            "Example Receiver was expected to have a larger "
            "red zone role last week"
        )["claims"]

        self.assertEqual(claims, [])

    def test_wrong_player_rejected(self):
        claims = self.build(
            "Backup Receiver is expected to have a larger "
            "red zone role next game"
        )["claims"]

        self.assertEqual(claims, [])

    def test_player_prefix_boundary_rejected(self):
        claims = self.build(
            "NotExample Receiver is expected to have a larger "
            "red zone role next game"
        )["claims"]

        self.assertEqual(claims, [])

    def test_player_suffix_boundary_rejected(self):
        claims = self.build(
            "Example ReceiverX is expected to have a larger "
            "red zone role next game"
        )["claims"]

        self.assertEqual(claims, [])

    def test_red_zone_targets_not_promoted(self):
        claims = self.build(
            "Example Receiver is expected to receive more "
            "red zone targets next game"
        )["claims"]

        self.assertEqual(claims, [])

    def test_goal_line_carries_not_promoted(self):
        claims = self.build(
            "Example Receiver is expected to receive more "
            "goal line carries next game"
        )["claims"]

        self.assertEqual(claims, [])

    def test_red_zone_touchdowns_not_promoted(self):
        claims = self.build(
            "Example Receiver is expected to score more "
            "red zone touchdowns next game"
        )["claims"]

        self.assertEqual(claims, [])


if __name__ == "__main__":
    unittest.main()
