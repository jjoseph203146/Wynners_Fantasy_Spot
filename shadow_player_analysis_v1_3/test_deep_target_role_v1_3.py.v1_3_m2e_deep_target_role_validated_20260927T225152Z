import unittest

from shadow_player_analysis_v1_3.claims import build_claims


class DeepTargetRoleV13Tests(unittest.TestCase):

    def build(
        self,
        quote,
        *,
        temporal_status="UNAMBIGUOUS_SOURCE_TIMESTAMP",
        temporal_confidence="RESOLVED",
        pregame_eligible=True,
    ):
        source_events = [
            {
                "record_id": "record-deep-target-1",
                "event_key": "event-deep-target-1",
                "source": "SYNTHETIC_SOURCE",
                "source_event_id": "native-deep-target-1",
                "origin_id": "SYNTHETIC_ORIGIN_A",
                "content_hash": "content-deep-target-1",
                "published_at_utc": "2026-09-27T15:00:00Z",
                "retrieved_at_utc": "2026-09-27T15:01:00Z",
                "first_seen_at_utc": "2026-09-27T15:01:00Z",
                "evidence_text": quote,
            }
        ]

        classified_evidence = [
            {
                "record_id": "record-deep-target-1",
                "event_key": "event-deep-target-1",
                "source_approved": True,
                "pregame_eligible": pregame_eligible,
                "temporal_status": temporal_status,
                "temporal_confidence": temporal_confidence,
                "relevance": "HIGH",
                "evidence_kind": "REPORTED_EXPECTATION",
                "evidence_types": ["DEEP_TARGET_ROLE"],
                "reason_codes": [],
            }
        ]

        entity_links = [
            {
                "record_id": "record-deep-target-1",
                "link_id": "link-deep-target-1",
                "event_key": "event-deep-target-1",
                "entity_type": "PLAYER",
                "gsis_id": "00-TEST-WR1",
                "player_name": "Example Receiver",
                "identity_status": "RESOLVED",
                "identity_reason": "EXACT_UNIQUE_NAME_VALIDATED",
                "relevance": "HIGH",
                "evidence_kind": "REPORTED_EXPECTATION",
                "evidence_types": ["DEEP_TARGET_ROLE"],
                "reason_codes": [],
                "mention": {
                    "type": "PLAYER",
                    "name": "Example Receiver",
                    "evidence_quote": quote,
                    "relationship_scope": "CLAUSE",
                },
            }
        ]

        return build_claims(
            source_events=source_events,
            classified_evidence=classified_evidence,
            entity_links=entity_links,
            cutoff_utc="2026-09-27T16:00:00Z",
        )

    def test_expected_larger_deep_target_role_promoted(self):
        result = self.build(
            "Example Receiver is expected to have a larger "
            "deep target role next game"
        )

        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(
            result["claims"][0]["signal_type"],
            "DEEP_TARGET_ROLE",
        )

    def test_expected_major_deep_target_role_promoted(self):
        result = self.build(
            "Example Receiver is expected to have a major "
            "deep target role this week"
        )

        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(
            result["claims"][0]["signal_type"],
            "DEEP_TARGET_ROLE",
        )

    def test_coach_plan_larger_deep_target_role_promoted(self):
        result = self.build(
            "Coach plans to give Example Receiver a larger "
            "deep target role next game"
        )

        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(
            result["claims"][0]["signal_type"],
            "DEEP_TARGET_ROLE",
        )

    def test_historical_deep_target_role_rejected(self):
        result = self.build(
            "Example Receiver had a major deep target role last game"
        )

        self.assertEqual(result["claims"], [])

    def test_historical_last_week_rejected(self):
        result = self.build(
            "Example Receiver had a larger deep target role last week"
        )

        self.assertEqual(result["claims"], [])

    def test_deep_target_count_not_promoted(self):
        result = self.build(
            "Example Receiver had four deep targets last game"
        )

        self.assertEqual(result["claims"], [])

    def test_air_yards_not_promoted(self):
        result = self.build(
            "Example Receiver had 120 air yards last game"
        )

        self.assertEqual(result["claims"], [])

    def test_vague_deep_target_role_rejected(self):
        result = self.build(
            "Example Receiver should be involved in the deep target role"
        )

        self.assertEqual(result["claims"], [])

    def test_smaller_deep_target_role_rejected(self):
        result = self.build(
            "Example Receiver is expected to have a smaller "
            "deep target role next game"
        )

        self.assertEqual(result["claims"], [])

    def test_question_rejected(self):
        result = self.build(
            "Is Example Receiver expected to have a larger "
            "deep target role next game?"
        )

        self.assertEqual(result["claims"], [])

    def test_ambiguous_timestamp_rejected(self):
        result = self.build(
            "Example Receiver is expected to have a larger "
            "deep target role next game",
            temporal_status="AMBIGUOUS_SOURCE_TIMESTAMP",
            temporal_confidence="UNRESOLVED",
            pregame_eligible=False,
        )

        self.assertEqual(result["claims"], [])


if __name__ == "__main__":
    unittest.main()


class DeepTargetRoleAdversarialV13Tests(DeepTargetRoleV13Tests):

    def assert_rejected(self, text):
        result = self.build(text)
        self.assertEqual(result["claims"], [])

    def test_not_expected_rejected(self):
        self.assert_rejected(
            "Example Receiver is not expected to have a larger "
            "deep target role next game"
        )

    def test_no_longer_expected_rejected(self):
        self.assert_rejected(
            "Example Receiver is no longer expected to have a larger "
            "deep target role next game"
        )

    def test_does_not_plan_rejected(self):
        self.assert_rejected(
            "Coach does not plan to give Example Receiver a larger "
            "deep target role next game"
        )

    def test_no_longer_plans_rejected(self):
        self.assert_rejected(
            "Coach no longer plans to give Example Receiver a larger "
            "deep target role next game"
        )

    def test_plan_withdrawn_rejected(self):
        self.assert_rejected(
            "The plan was withdrawn; Coach plans to give "
            "Example Receiver a larger deep target role next game"
        )

    def test_plans_withdrawn_rejected(self):
        self.assert_rejected(
            "The plans were withdrawn; Coach plans to give "
            "Example Receiver a larger deep target role next game"
        )

    def test_historical_expectation_rejected(self):
        self.assert_rejected(
            "Example Receiver was expected to have a larger "
            "deep target role last week"
        )

    def test_wrong_player_larger_rejected(self):
        self.assert_rejected(
            "Backup Receiver is expected to have a larger "
            "deep target role next game"
        )

    def test_wrong_player_major_rejected(self):
        self.assert_rejected(
            "Backup Receiver is expected to have a major "
            "deep target role this week"
        )

    def test_prefixed_player_name_rejected(self):
        self.assert_rejected(
            "NotExample Receiver is expected to have a larger "
            "deep target role next game"
        )

    def test_suffixed_player_name_rejected(self):
        self.assert_rejected(
            "Example ReceiverJr is expected to have a larger "
            "deep target role next game"
        )

    def test_deep_targets_not_equivalent(self):
        self.assert_rejected(
            "Example Receiver is expected to see more deep targets next game"
        )

    def test_air_yards_not_equivalent(self):
        self.assert_rejected(
            "Example Receiver is expected to have more air yards next game"
        )

    def test_downfield_role_not_equivalent(self):
        self.assert_rejected(
            "Example Receiver is expected to have a larger "
            "downfield role next game"
        )

    def test_vertical_role_not_equivalent(self):
        self.assert_rejected(
            "Example Receiver is expected to have a larger "
            "vertical role next game"
        )

    def test_deep_target_percentage_not_equivalent(self):
        self.assert_rejected(
            "Example Receiver had a 35 percent deep target share last week"
        )

    def test_question_still_rejected(self):
        self.assert_rejected(
            "Is Example Receiver expected to have a major "
            "deep target role this week?"
        )

    def test_smaller_role_still_rejected(self):
        self.assert_rejected(
            "Example Receiver is expected to have a smaller "
            "deep target role next game"
        )

    def test_later_valid_occurrence_after_bad_prefix_accepted(self):
        result = self.build(
            "NotExample Receiver is expected to have a larger deep target "
            "role next game, but Example Receiver is expected to have a "
            "larger deep target role next game"
        )

        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(
            result["claims"][0]["signal_type"],
            "DEEP_TARGET_ROLE",
        )
