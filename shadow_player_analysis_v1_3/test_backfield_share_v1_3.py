import unittest

from shadow_player_analysis_v1_3.claims import build_claims


class BackfieldShareV13Tests(unittest.TestCase):

    def event(self, text, **overrides):
        row = {
            "record_id": "record-backfield-1",
            "event_key": "event-backfield-1",
            "source": "SYNTHETIC_SOURCE",
            "source_event_id": "native-backfield-1",
            "origin_id": "SYNTHETIC_ORIGIN_A",
            "content_hash": "content-backfield-1",
            "published_at_utc": "2026-09-27T15:00:00Z",
            "retrieved_at_utc": "2026-09-27T15:01:00Z",
            "first_seen_at_utc": "2026-09-27T15:01:00Z",
            "evidence_text": text,
        }
        row.update(overrides)
        return row

    def evidence(self, **overrides):
        row = {
            "record_id": "record-backfield-1",
            "event_key": "event-backfield-1",
            "source_approved": True,
            "pregame_eligible": True,
            "temporal_status": "UNAMBIGUOUS_SOURCE_TIMESTAMP",
            "temporal_confidence": "RESOLVED",
            "evidence_kind": "REPORTED_EXPECTATION",
            "evidence_types": ["BACKFIELD_SHARE"],
            "relevance": "HIGH",
            "reason_codes": [],
        }
        row.update(overrides)
        return row

    def link(self, quote, **overrides):
        row = {
            "record_id": "record-backfield-1",
            "link_id": "link-backfield-1",
            "entity_type": "PLAYER",
            "gsis_id": "00-TEST-RB1",
            "identity_status": "RESOLVED",
            "identity_reason": "EXACT_UNIQUE_NAME_VALIDATED",
            "evidence_kind": "REPORTED_EXPECTATION",
            "evidence_types": ["BACKFIELD_SHARE"],
            "relevance": "HIGH",
            "mention": {
                "type": "PLAYER",
                "name": "Example Runner",
                "evidence_quote": quote,
                "relationship_scope": "CLAUSE",
            },
            "reason_codes": [],
        }
        row.update(overrides)
        return row

    def build(self, text, quote=None, evidence=None, link=None):
        quote = quote if quote is not None else text

        return build_claims(
            source_events=[self.event(text)],
            classified_evidence=[
                evidence if evidence is not None else self.evidence()
            ],
            entity_links=[
                link if link is not None else self.link(quote)
            ],
            cutoff_utc="2026-09-27T16:00:00Z",
        )

    def test_expected_larger_backfield_share_creates_claim(self):
        text = "Example Runner is expected to handle a larger share of the backfield next game"
        rows = self.build(text)["claims"]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["signal_type"], "BACKFIELD_SHARE")
        self.assertEqual(rows[0]["gsis_id"], "00-TEST-RB1")
        self.assertEqual(rows[0]["eligibility_status"], "ELIGIBLE")
        self.assertFalse(rows[0]["consensus_eligible"])

    def test_expected_majority_backfield_work_creates_claim(self):
        text = "Example Runner is expected to handle the majority of the backfield work this week"
        rows = self.build(text)["claims"]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["signal_type"], "BACKFIELD_SHARE")

    def test_coach_plan_for_more_backfield_work_creates_claim(self):
        text = "Coach plans to give Example Runner more backfield work next game"
        rows = self.build(text)["claims"]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["signal_type"], "BACKFIELD_SHARE")

    def test_previous_game_carries_do_not_create_claim(self):
        text = "Example Runner had 18 carries and 3 catches last game"
        self.assertEqual(self.build(text)["claims"], [])

    def test_previous_game_touch_share_does_not_create_claim(self):
        text = "Example Runner handled 70 percent of the backfield touches last week"
        self.assertEqual(self.build(text)["claims"], [])

    def test_box_score_performance_does_not_create_claim(self):
        text = "Example Runner rushed for 112 yards and two touchdowns on 20 carries"
        self.assertEqual(self.build(text)["claims"], [])

    def test_vague_backfield_comment_does_not_create_claim(self):
        text = "Example Runner remains part of the backfield"
        self.assertEqual(self.build(text)["claims"], [])

    def test_reduced_backfield_work_does_not_become_positive_share_claim(self):
        text = "Example Runner is expected to receive less backfield work next game"
        self.assertEqual(self.build(text)["claims"], [])

    def test_question_does_not_create_claim(self):
        text = "Will Example Runner handle a larger share of the backfield next game?"
        self.assertEqual(self.build(text)["claims"], [])

    def test_ambiguous_timestamp_still_blocks_backfield_claim(self):
        text = "Example Runner is expected to handle a larger share of the backfield next game"

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

        result = build_claims(
            source_events=[event],
            classified_evidence=[evidence],
            entity_links=[self.link(text)],
            cutoff_utc="2026-09-27T16:00:00Z",
        )

        self.assertEqual(result["claims"], [])


if __name__ == "__main__":
    unittest.main()


class BackfieldShareAdversarialV13Tests(BackfieldShareV13Tests):

    def assert_no_claim(self, text):
        self.assertEqual(
            self.build(text)["claims"],
            [],
        )

    def test_negated_expectation_fails_closed(self):
        self.assert_no_claim(
            "Example Runner is not expected to handle a larger share of the backfield next game"
        )

    def test_no_longer_expected_fails_closed(self):
        self.assert_no_claim(
            "Example Runner is no longer expected to handle a larger share of the backfield next game"
        )

    def test_coach_denies_plan_fails_closed(self):
        self.assert_no_claim(
            "Coach does not plan to give Example Runner more backfield work next game"
        )

    def test_coach_no_longer_plans_fails_closed(self):
        self.assert_no_claim(
            "Coach no longer plans to give Example Runner more backfield work next game"
        )

    def test_previous_expectation_fails_closed(self):
        self.assert_no_claim(
            "Example Runner was expected to handle a larger share of the backfield last week"
        )

    def test_withdrawn_plan_fails_closed(self):
        self.assert_no_claim(
            "Coach plans to give Example Runner more backfield work was the earlier expectation, but that plan was withdrawn"
        )

    def test_other_player_expectation_fails_closed(self):
        self.assert_no_claim(
            "Backup Runner is expected to handle a larger share of the backfield while Example Runner remains limited"
        )

    def test_contrasted_player_fails_closed(self):
        self.assert_no_claim(
            "Example Runner is expected to remain limited while Backup Runner handles the majority of the backfield work"
        )


class BackfieldShareNameBoundaryV13Tests(BackfieldShareV13Tests):

    def test_embedded_player_name_prefix_fails_closed(self):
        text = (
            "NotExample Runner is expected to handle a larger share "
            "of the backfield next game"
        )

        self.assertEqual(
            self.build(text)["claims"],
            [],
        )
