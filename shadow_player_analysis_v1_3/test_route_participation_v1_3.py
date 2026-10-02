import unittest

from shadow_player_analysis_v1_3.claims import build_claims


class RouteParticipationV13Tests(unittest.TestCase):

    def build(
        self,
        text,
        quote=None,
        evidence=None,
        link=None,
    ):
        quote = text if quote is None else quote

        source_events = [
            {
                "record_id": "record-route-1",
                "event_key": "event-route-1",
                "source": "SYNTHETIC_SOURCE",
                "source_event_id": "native-route-1",
                "origin_id": "SYNTHETIC_ORIGIN_A",
                "content_hash": "content-route-1",
                "published_at_utc": "2026-09-27T15:00:00Z",
                "retrieved_at_utc": "2026-09-27T15:01:00Z",
                "first_seen_at_utc": "2026-09-27T15:01:00Z",
                "evidence_text": text,
            }
        ]

        classified_evidence = [
            {
                "record_id": "record-route-1",
                "source_approved": True,
                "pregame_eligible": True,
                "temporal_status": "UNAMBIGUOUS_SOURCE_TIMESTAMP",
                "temporal_confidence": "RESOLVED",
                "evidence_kind": "REPORTED_EXPECTATION",
                "evidence_types": ["ROUTE_PARTICIPATION"],
                "relevance": "HIGH",
                "reason_codes": [],
            }
        ]

        identity_links = [
            {
                "record_id": "record-route-1",
                "link_id": "link-route-1",
                "entity_type": "PLAYER",
                "gsis_id": "00-TEST-WR1",
                "identity_status": "RESOLVED",
                "identity_reason": "EXACT_UNIQUE_NAME_VALIDATED",
                "evidence_kind": "REPORTED_EXPECTATION",
                "evidence_types": ["ROUTE_PARTICIPATION"],
                "relevance": "HIGH",
                "mention": {
                    "name": "Example Receiver",
                    "evidence_quote": quote,
                    "relationship_scope": "CLAUSE",
                },
            }
        ]

        if evidence:
            classified_evidence[0].update(evidence)

        if link:
            identity_links[0].update(link)

        return build_claims(
            source_events=source_events,
            classified_evidence=classified_evidence,
            entity_links=identity_links,
            cutoff_utc="2026-09-27T16:00:00Z",
        )

    def test_expected_larger_route_participation_creates_claim(self):
        result = self.build(
            "Example Receiver is expected to have larger route participation next game"
        )
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(
            result["claims"][0]["signal_type"],
            "ROUTE_PARTICIPATION",
        )

    def test_expected_majority_route_participation_creates_claim(self):
        result = self.build(
            "Example Receiver is expected to have majority route participation this week"
        )
        self.assertEqual(len(result["claims"]), 1)

    def test_coach_plan_route_participation_creates_claim(self):
        result = self.build(
            "Coach plans to give Example Receiver larger route participation next game"
        )
        self.assertEqual(len(result["claims"]), 1)

    def test_historical_route_participation_rejected(self):
        result = self.build(
            "Example Receiver had 82 percent route participation last game"
        )
        self.assertEqual(result["claims"], [])

    def test_last_week_route_participation_rejected(self):
        result = self.build(
            "Example Receiver had majority route participation last week"
        )
        self.assertEqual(result["claims"], [])

    def test_routes_run_not_equivalent_rejected(self):
        result = self.build(
            "Example Receiver ran 34 routes last game"
        )
        self.assertEqual(result["claims"], [])

    def test_vague_route_participation_rejected(self):
        result = self.build(
            "Example Receiver should be involved in route participation"
        )
        self.assertEqual(result["claims"], [])

    def test_smaller_route_participation_rejected(self):
        result = self.build(
            "Example Receiver is expected to have smaller route participation next game"
        )
        self.assertEqual(result["claims"], [])

    def test_question_rejected(self):
        result = self.build(
            "Is Example Receiver expected to have larger route participation next game?"
        )
        self.assertEqual(result["claims"], [])

    def test_ambiguous_timestamp_rejected(self):
        result = self.build(
            "Example Receiver is expected to have larger route participation next game",
            evidence={
                "pregame_eligible": False,
                "temporal_status": "AMBIGUOUS_SOURCE_TIMESTAMP",
                "temporal_confidence": "UNRESOLVED",
                "reason_codes": ["AMBIGUOUS_SOURCE_TIMESTAMP"],
            },
        )
        self.assertEqual(result["claims"], [])


if __name__ == "__main__":
    unittest.main()


class RouteParticipationAdversarialV13Tests(RouteParticipationV13Tests):

    def test_no_longer_plans_rejected(self):
        result = self.build(
            "Coach no longer plans to give Example Receiver "
            "larger route participation next game"
        )
        self.assertEqual(result["claims"], [])

    def test_withdrawn_plan_rejected(self):
        result = self.build(
            "Coach plans to give Example Receiver larger route participation "
            "next game, but that plan was withdrawn"
        )
        self.assertEqual(result["claims"], [])

    def test_not_expected_rejected(self):
        result = self.build(
            "Example Receiver is not expected to have larger "
            "route participation next game"
        )
        self.assertEqual(result["claims"], [])

    def test_no_longer_expected_rejected(self):
        result = self.build(
            "Example Receiver is no longer expected to have larger "
            "route participation next game"
        )
        self.assertEqual(result["claims"], [])

    def test_does_not_plan_rejected(self):
        result = self.build(
            "Coach does not plan to give Example Receiver "
            "larger route participation next game"
        )
        self.assertEqual(result["claims"], [])

    def test_historical_expectation_rejected(self):
        result = self.build(
            "Example Receiver was expected to have larger "
            "route participation last week"
        )
        self.assertEqual(result["claims"], [])

    def test_wrong_player_rejected(self):
        result = self.build(
            "Backup Receiver is expected to have larger "
            "route participation next game"
        )
        self.assertEqual(result["claims"], [])

    def test_wrong_player_majority_rejected(self):
        result = self.build(
            "Backup Receiver is expected to have majority "
            "route participation this week"
        )
        self.assertEqual(result["claims"], [])

    def test_player_prefix_boundary_rejected(self):
        result = self.build(
            "NotExample Receiver is expected to have larger "
            "route participation next game"
        )
        self.assertEqual(result["claims"], [])

    def test_player_suffix_boundary_rejected(self):
        result = self.build(
            "Example ReceiverX is expected to have larger "
            "route participation next game"
        )
        self.assertEqual(result["claims"], [])
