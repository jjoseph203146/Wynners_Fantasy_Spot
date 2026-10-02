import unittest

from shadow_player_analysis_v1_3.claims import build_claims, digest


class ClaimLifecycleV13Tests(unittest.TestCase):

    def event(self, **overrides):
        row = {
            "record_id": "record-1",
            "event_key": "event-1",
            "source": "SYNTHETIC_SOURCE",
            "source_event_id": "native-1",
            "origin_id": "SYNTHETIC_ORIGIN_A",
            "content_hash": "content-1",
            "published_at_utc": "2026-09-27T15:00:00Z",
            "updated_at_utc": "2026-09-27T15:00:30Z",
            "first_seen_at_utc": "2026-09-27T15:01:00Z",
            "retrieved_at_utc": "2026-09-27T15:01:00Z",
            "evidence_text": "Example Quarterback set to start",
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

    def player_link(self, **overrides):
        row = {
            "record_id": "record-1",
            "link_id": "player-link-1",
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

    def game_link(self, **overrides):
        row = {
            "record_id": "record-1",
            "link_id": "game-link-1",
            "entity_type": "GAME",
            "identity_status": "RESOLVED",
            "game_id": "game-1",
            "schedule_sha256": None,
            "reason_codes": [],
        }
        row.update(overrides)
        return row

    def schedule(self, **overrides):
        row = {
            "authority": "WFS_SCHEDULE_SNAPSHOT",
            "available_at_utc": "2026-09-27T15:02:00Z",
            "games": [
                {
                    "game_id": "game-1",
                    "kickoff_at_utc": "2026-09-27T17:00:00Z",
                }
            ],
        }
        row.update(overrides)
        return row

    def build(
        self,
        *,
        event=None,
        evidence=None,
        player=None,
        games=None,
        schedule=None,
        cutoff="2026-09-27T16:00:00Z",
    ):
        schedule_row = schedule or self.schedule()

        if games is None:
            game = self.game_link(
                schedule_sha256=digest(schedule_row)
            )
            game_rows = [game]
        else:
            game_rows = games

        links = [player or self.player_link()]
        links.extend(game_rows)

        return build_claims(
            source_events=[event or self.event()],
            classified_evidence=[evidence or self.evidence()],
            entity_links=links,
            cutoff_utc=cutoff,
            schedule=schedule_row,
        )

    def test_exact_resolved_game_creates_lifecycle_claim(self):
        row = self.build()["claims"][0]

        self.assertEqual(row["game_id"], "game-1")
        self.assertEqual(row["kickoff_at_utc"], "2026-09-27T17:00:00Z")
        self.assertEqual(
            row["knowledge_at_utc"],
            "2026-09-27T15:01:00+00:00",
        )
        self.assertEqual(
            row["effective_from_utc"],
            "2026-09-27T15:01:00+00:00",
        )
        self.assertEqual(
            row["valid_until_utc"],
            "2026-09-27T17:00:00Z",
        )

    def test_missing_game_link_fails_closed(self):
        self.assertEqual(self.build(games=[])["claims"], [])

    def test_multiple_game_links_fail_closed(self):
        games = [
            self.game_link(),
            self.game_link(
                link_id="game-link-2",
                game_id="game-2",
            ),
        ]
        self.assertEqual(self.build(games=games)["claims"], [])

    def test_unresolved_game_link_fails_closed(self):
        game = self.game_link(
            identity_status="UNRESOLVED",
            game_id=None,
            reason_codes=["UNRESOLVED_GAME_IDENTITY"],
        )
        self.assertEqual(self.build(games=[game])["claims"], [])

    def test_game_missing_from_schedule_fails_closed(self):
        schedule = self.schedule(
            games=[
                {
                    "game_id": "different-game",
                    "kickoff_at_utc": "2026-09-27T17:00:00Z",
                }
            ]
        )
        self.assertEqual(self.build(schedule=schedule)["claims"], [])

    def test_duplicate_game_in_schedule_fails_closed(self):
        game = {
            "game_id": "game-1",
            "kickoff_at_utc": "2026-09-27T17:00:00Z",
        }
        schedule = self.schedule(games=[dict(game), dict(game)])
        self.assertEqual(self.build(schedule=schedule)["claims"], [])

    def test_wrong_schedule_authority_fails_closed(self):
        schedule = self.schedule(authority="UNTRUSTED_SCHEDULE")
        self.assertEqual(self.build(schedule=schedule)["claims"], [])

    def test_future_schedule_snapshot_fails_closed(self):
        schedule = self.schedule(
            available_at_utc="2026-09-27T16:00:01Z"
        )
        self.assertEqual(self.build(schedule=schedule)["claims"], [])

    def test_cutoff_at_kickoff_fails_closed(self):
        self.assertEqual(
            self.build(cutoff="2026-09-27T17:00:00Z")["claims"],
            [],
        )

    def test_cutoff_after_kickoff_fails_closed(self):
        self.assertEqual(
            self.build(cutoff="2026-09-27T17:00:01Z")["claims"],
            [],
        )

    def test_updated_time_participates_in_knowledge_boundary(self):
        event = self.event(
            updated_at_utc="2026-09-27T16:00:01Z",
            first_seen_at_utc="2026-09-27T16:00:01Z",
            retrieved_at_utc="2026-09-27T16:00:01Z",
        )
        self.assertEqual(self.build(event=event)["claims"], [])

    def test_knowledge_time_uses_latest_known_timestamp(self):
        event = self.event(
            updated_at_utc="2026-09-27T15:02:00Z",
            first_seen_at_utc="2026-09-27T15:01:00Z",
            retrieved_at_utc="2026-09-27T15:03:00Z",
        )
        claims = self.build(event=event)["claims"]
        self.assertEqual(len(claims), 1)
        self.assertEqual(
            claims[0]["knowledge_at_utc"],
            "2026-09-27T15:03:00+00:00",
        )
        self.assertEqual(
            claims[0]["effective_from_utc"],
            "2026-09-27T15:03:00+00:00",
        )

    def test_kickoff_missing_fails_closed(self):
        schedule = self.schedule(
            games=[{"game_id": "game-1"}]
        )
        self.assertEqual(self.build(schedule=schedule)["claims"], [])

    def test_schedule_hash_mismatch_fails_closed(self):
        game = self.game_link(schedule_sha256="wrong-schedule-hash")
        self.assertEqual(self.build(games=[game])["claims"], [])


if __name__ == "__main__":
    unittest.main()
