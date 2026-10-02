"""V1.3 Milestone 2B TARGET_SHARE claim-gate regression tests."""

import unittest

from shadow_player_analysis_v1_3.claims import build_claims


class TargetShareV13Tests(unittest.TestCase):

    def event(self, text, **overrides):
        row = {
            "record_id": "record-target-1",
            "event_key": "event-target-1",
            "source": "SYNTHETIC_SOURCE",
            "source_event_id": "native-target-1",
            "origin_id": "SYNTHETIC_ORIGIN_A",
            "content_hash": "content-target-1",
            "published_at_utc": "2026-09-27T15:00:00Z",
            "retrieved_at_utc": "2026-09-27T15:01:00Z",
            "first_seen_at_utc": "2026-09-27T15:01:00Z",
            "evidence_text": text,
        }
        row.update(overrides)
        return row

    def evidence(self, **overrides):
        row = {
            "record_id": "record-target-1",
            "event_key": "event-target-1",
            "source_approved": True,
            "pregame_eligible": True,
            "temporal_status": "UNAMBIGUOUS_SOURCE_TIMESTAMP",
            "temporal_confidence": "RESOLVED",
            "evidence_kind": "REPORTED_EXPECTATION",
            "evidence_types": ["TARGET_SHARE"],
            "relevance": "HIGH",
            "reason_codes": [],
        }
        row.update(overrides)
        return row

    def link(self, quote, **overrides):
        row = {
            "record_id": "record-target-1",
            "link_id": "link-target-1",
            "entity_type": "PLAYER",
            "gsis_id": "00-TEST-WR1",
            "identity_status": "RESOLVED",
            "identity_reason": "EXACT_UNIQUE_NAME_VALIDATED",
            "evidence_kind": "REPORTED_EXPECTATION",
            "evidence_types": ["TARGET_SHARE"],
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
        quote = quote if quote is not None else text

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
    # Positive forward-looking propositions.
    # Expected RED until TARGET_SHARE grammar exists.
    # ---------------------------------------------------------------

    def test_expected_larger_target_share(self):
        text = (
            "Example Receiver is expected to have a larger target share "
            "next game"
        )
        claims = self.build(text)["claims"]

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["signal_type"], "TARGET_SHARE")
        self.assertEqual(claims[0]["player_name"], "Example Receiver")
        self.assertEqual(claims[0]["gsis_id"], "00-TEST-WR1")

    def test_expected_majority_target_share(self):
        text = (
            "Example Receiver is expected to command the majority "
            "target share this week"
        )
        claims = self.build(text)["claims"]

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["signal_type"], "TARGET_SHARE")

    def test_coach_plan_increase_target_share(self):
        text = (
            "Coach plans to give Example Receiver a larger target share "
            "next game"
        )
        claims = self.build(text)["claims"]

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["signal_type"], "TARGET_SHARE")

    # ---------------------------------------------------------------
    # Historical/descriptive evidence is not a forward claim.
    # ---------------------------------------------------------------

    def test_last_game_target_share_rejected(self):
        text = (
            "Example Receiver had a 31 percent target share last game"
        )
        self.assertEqual(self.build(text)["claims"], [])

    def test_last_week_target_share_rejected(self):
        text = (
            "Example Receiver commanded a 28 percent target share last week"
        )
        self.assertEqual(self.build(text)["claims"], [])

    # ---------------------------------------------------------------
    # Generic target counts remain insufficient.
    # ---------------------------------------------------------------

    def test_target_count_rejected(self):
        text = (
            "Example Receiver had 12 targets and eight catches last game"
        )
        self.assertEqual(self.build(text)["claims"], [])

    # ---------------------------------------------------------------
    # Vague/opposite/interrogative propositions fail closed.
    # ---------------------------------------------------------------

    def test_vague_target_share_rejected(self):
        text = (
            "Example Receiver should be involved in the target share"
        )
        self.assertEqual(self.build(text)["claims"], [])

    def test_smaller_target_share_rejected(self):
        text = (
            "Example Receiver is expected to have a smaller target share "
            "next game"
        )
        self.assertEqual(self.build(text)["claims"], [])

    def test_question_rejected(self):
        text = (
            "Is Example Receiver expected to have a larger target share "
            "next game?"
        )
        self.assertEqual(self.build(text)["claims"], [])

    # ---------------------------------------------------------------
    # Temporal ambiguity independently blocks claim creation.
    # ---------------------------------------------------------------

    def test_ambiguous_timestamp_rejected(self):
        text = (
            "Example Receiver is expected to have a larger target share "
            "next game"
        )

        evidence = self.evidence(
            pregame_eligible=False,
            temporal_status="AMBIGUOUS_SOURCE_TIMESTAMP",
            temporal_confidence="UNRESOLVED",
            reason_codes=[
                "INVALID_PUBLICATION_TIMESTAMP",
                "TEMPORAL_UNRESOLVED",
            ],
        )

        event = self.event(
            text,
            published_at_utc=None,
        )

        self.assertEqual(
            self.build(
                text,
                evidence=evidence,
                event=event,
            )["claims"],
            [],
        )


if __name__ == "__main__":
    unittest.main()


class TargetShareAdversarialV13Tests(TargetShareV13Tests):

    def test_no_longer_coach_plan_rejected(self):
        text = (
            "Coach no longer plans to give Example Receiver "
            "a larger target share next game"
        )
        self.assertEqual(self.build(text)["claims"], [])

    def test_withdrawn_coach_plan_rejected(self):
        text = (
            "Coach plans to give Example Receiver a larger target share "
            "next game, but that plan was withdrawn"
        )
        self.assertEqual(self.build(text)["claims"], [])
