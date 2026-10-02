import unittest

from shadow_player_analysis_v1_3.claims import build_claims


class ClaimConstructionV13Tests(unittest.TestCase):

    def link(self, **overrides):
        row = {
            "record_id": "record-1",
            "link_id": "link-1",
            "entity_type": "PLAYER",
            "gsis_id": "00-TEST-001",
            "identity_status": "RESOLVED",
            "identity_reason": "EXACT_UNIQUE_NAME_VALIDATED",
            "evidence_kind": "REPORTED_EXPECTATION",
            "evidence_types": ["STARTER_CHANGE"],
            "relevance": "HIGH",
            "mention": {
                "type": "PLAYER",
                "name": "Example Quarterback",
                "evidence_quote": "Example Quarterback set to start",
                "relationship_scope": "CLAUSE",
            },
            "reason_codes": [],
        }
        row.update(overrides)
        return row

    def evidence(self, **overrides):
        row = {
            "record_id": "record-1",
            "event_key": "event-1",
            "source_approved": True,
            "pregame_eligible": True,
            "temporal_status": "UNAMBIGUOUS_SOURCE_TIMESTAMP",
            "temporal_confidence": "RESOLVED",
            "evidence_kind": "REPORTED_EXPECTATION",
            "evidence_types": ["STARTER_CHANGE"],
            "relevance": "HIGH",
            "reason_codes": [],
        }
        row.update(overrides)
        return row

    def event(self, **overrides):
        row = {
            "record_id": "record-1",
            "event_key": "event-1",
            "source": "SYNTHETIC_SOURCE",
            "source_event_id": "native-1",
            "origin_id": "SYNTHETIC_ORIGIN_A",
            "content_hash": "content-1",
            "published_at_utc": "2026-09-27T15:00:00Z",
            "retrieved_at_utc": "2026-09-27T15:01:00Z",
            "first_seen_at_utc": "2026-09-27T15:01:00Z",
            "evidence_text": "Example Quarterback set to start",
        }
        row.update(overrides)
        return row

    def build(self, link=None, evidence=None, event=None):
        return build_claims(
            source_events=[event or self.event()],
            classified_evidence=[evidence or self.evidence()],
            entity_links=[link or self.link()],
            cutoff_utc="2026-09-27T16:00:00Z",
        )

    def test_valid_forward_starter_change_creates_claim(self):
        result = self.build()
        self.assertEqual(len(result["claims"]), 1)
        row = result["claims"][0]
        self.assertEqual(row["gsis_id"], "00-TEST-001")
        self.assertEqual(row["signal_type"], "STARTER_CHANGE")
        self.assertEqual(row["eligibility_status"], "ELIGIBLE")
        self.assertFalse(row["consensus_eligible"])

    def test_ambiguous_timestamp_creates_no_claim(self):
        evidence = self.evidence(
            pregame_eligible=False,
            temporal_status="AMBIGUOUS_SOURCE_TIMESTAMP",
            temporal_confidence="UNRESOLVED",
            reason_codes=[
                "INVALID_PUBLICATION_TIMESTAMP",
                "TEMPORAL_UNRESOLVED",
            ],
        )
        event = self.event(published_at_utc=None)
        result = self.build(evidence=evidence, event=event)
        self.assertEqual(result["claims"], [])

    def test_unresolved_identity_creates_no_claim(self):
        link = self.link(
            gsis_id=None,
            identity_status="UNRESOLVED",
            identity_reason="NO_EXACT_NAME_MATCH",
            reason_codes=["UNRESOLVED_PLAYER_IDENTITY"],
        )
        self.assertEqual(self.build(link=link)["claims"], [])

    def test_ambiguous_identity_creates_no_claim(self):
        link = self.link(
            gsis_id=None,
            identity_status="AMBIGUOUS",
            identity_reason="AMBIGUOUS_EXACT_IDENTITY",
            reason_codes=["AMBIGUOUS_PLAYER_IDENTITY"],
        )
        self.assertEqual(self.build(link=link)["claims"], [])

    def test_low_relevance_creates_no_claim(self):
        link = self.link(
            relevance="LOW",
            evidence_types=["MARKET_CONTEXT"],
            evidence_kind="MARKET_INFORMATION",
        )
        evidence = self.evidence(
            relevance="LOW",
            evidence_types=["MARKET_CONTEXT"],
            evidence_kind="MARKET_INFORMATION",
        )
        self.assertEqual(self.build(link=link, evidence=evidence)["claims"], [])

    def test_postgame_creates_no_claim(self):
        link = self.link(
            relevance="LOW",
            evidence_types=["POST_GAME_RECAP"],
            evidence_kind="FACTUAL_OBSERVATION",
        )
        evidence = self.evidence(
            relevance="LOW",
            evidence_types=["POST_GAME_RECAP"],
            evidence_kind="FACTUAL_OBSERVATION",
            pregame_eligible=False,
            reason_codes=["POST_GAME_RECAP"],
        )
        self.assertEqual(self.build(link=link, evidence=evidence)["claims"], [])

    def test_quote_must_be_supported_by_event(self):
        link = self.link()
        link["mention"] = dict(
            link["mention"],
            evidence_quote="Text that does not occur in source event",
        )
        self.assertEqual(self.build(link=link)["claims"], [])

    def test_insufficient_signal_creates_no_claim(self):
        link = self.link(
            evidence_types=["INSUFFICIENT"],
            relevance="LOW",
        )
        evidence = self.evidence(
            evidence_types=["INSUFFICIENT"],
            relevance="LOW",
        )
        self.assertEqual(self.build(link=link, evidence=evidence)["claims"], [])

    def test_claim_id_is_deterministic(self):
        first = self.build()["claims"][0]["claim_id"]
        second = self.build()["claims"][0]["claim_id"]
        self.assertEqual(first, second)

    def test_consensus_remains_disabled(self):
        result = self.build()
        self.assertEqual(result["consensus"], [])
        self.assertFalse(result["claims"][0]["consensus_eligible"])


if __name__ == "__main__":
    unittest.main()


class ClaimConstructionV13AuditRegressionTests(ClaimConstructionV13Tests):

    def test_future_retrieval_fails_closed(self):
        event = self.event(
            retrieved_at_utc="2026-09-27T16:01:00Z",
        )
        self.assertEqual(
            self.build(event=event)["claims"],
            [],
        )

    def test_missing_origin_fails_closed(self):
        event = self.event(origin_id=None)
        self.assertEqual(
            self.build(event=event)["claims"],
            [],
        )

    def test_team_entity_fails_closed(self):
        link = self.link(
            entity_type="TEAM",
            gsis_id=None,
        )
        self.assertEqual(
            self.build(link=link)["claims"],
            [],
        )

    def test_multiple_supported_signals_fail_closed(self):
        link = self.link(
            evidence_types=["STARTER_CHANGE", "ROLE_INCREASE"],
        )
        evidence = self.evidence(
            evidence_types=["STARTER_CHANGE", "ROLE_INCREASE"],
        )
        self.assertEqual(
            self.build(link=link, evidence=evidence)["claims"],
            [],
        )

    def test_conflicting_event_join_fails_closed(self):
        from shadow_player_analysis_v1_3.claims import build_claims

        first = self.event()
        second = self.event(content_hash="different-content")

        result = build_claims(
            source_events=[first, second],
            classified_evidence=[self.evidence()],
            entity_links=[self.link()],
            cutoff_utc="2026-09-27T16:00:00Z",
        )

        self.assertEqual(result["claims"], [])

    def test_conflicting_evidence_join_fails_closed(self):
        from shadow_player_analysis_v1_3.claims import build_claims

        first = self.evidence()
        second = self.evidence(relevance="LOW")

        result = build_claims(
            source_events=[self.event()],
            classified_evidence=[first, second],
            entity_links=[self.link()],
            cutoff_utc="2026-09-27T16:00:00Z",
        )

        self.assertEqual(result["claims"], [])

    def test_deterministic_complete_claim_output(self):
        first = self.build()
        second = self.build()
        self.assertEqual(first, second)

    def test_safety_contract_remains_shadow_only(self):
        from shadow_player_analysis_v1_3.claims import (
            SAFETY,
            CONSENSUS_ENABLED,
        )

        self.assertTrue(SAFETY["ANALYSIS_ONLY"])

        self.assertFalse(
            any(
                value
                for key, value in SAFETY.items()
                if key != "ANALYSIS_ONLY"
            )
        )

        self.assertFalse(CONSENSUS_ENABLED)
        self.assertEqual(self.build()["consensus"], [])
