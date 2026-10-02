import unittest

from shadow_player_analysis_v1_3.claims import digest
from shadow_player_analysis_v1_3.test_claim_lifecycle_v1_3 import (
    ClaimLifecycleV13Tests,
)


class ClaimLifecycleAdversarialV13Tests(ClaimLifecycleV13Tests):

    def test_cross_record_game_link_not_consumed(self):
        game = self.game_link(
            record_id="record-other",
            schedule_sha256=digest(self.schedule()),
        )
        self.assertEqual(self.build(games=[game])["claims"], [])

    def test_same_record_resolved_and_unresolved_game_links_fail_closed(self):
        schedule = self.schedule()
        good = self.game_link(
            link_id="game-good",
            schedule_sha256=digest(schedule),
        )
        bad = self.game_link(
            link_id="game-bad",
            identity_status="UNRESOLVED",
            game_id=None,
            schedule_sha256=digest(schedule),
        )
        self.assertEqual(
            self.build(schedule=schedule, games=[good, bad])["claims"],
            [],
        )

    def test_game_link_reason_code_fails_closed(self):
        schedule = self.schedule()
        game = self.game_link(
            schedule_sha256=digest(schedule),
            reason_codes=["AMBIGUOUS_GAME"],
        )
        self.assertEqual(
            self.build(schedule=schedule, games=[game])["claims"],
            [],
        )

    def test_blank_game_id_fails_closed(self):
        schedule = self.schedule()
        game = self.game_link(
            game_id="",
            schedule_sha256=digest(schedule),
        )
        self.assertEqual(
            self.build(schedule=schedule, games=[game])["claims"],
            [],
        )

    def test_schedule_game_id_blank_cannot_bind(self):
        schedule = self.schedule(
            games=[
                {
                    "game_id": "",
                    "kickoff_at_utc": "2026-09-27T17:00:00Z",
                }
            ]
        )
        game = self.game_link(
            game_id="",
            schedule_sha256=digest(schedule),
        )
        self.assertEqual(
            self.build(schedule=schedule, games=[game])["claims"],
            [],
        )

    def test_schedule_non_dict_game_cannot_bind(self):
        schedule = self.schedule(games=["game-1"])
        game = self.game_link(
            schedule_sha256=digest(schedule),
        )
        self.assertEqual(
            self.build(schedule=schedule, games=[game])["claims"],
            [],
        )

    def test_schedule_duplicate_target_game_fails_closed(self):
        schedule = self.schedule(
            games=[
                {
                    "game_id": "game-1",
                    "kickoff_at_utc": "2026-09-27T17:00:00Z",
                },
                {
                    "game_id": "game-1",
                    "kickoff_at_utc": "2026-09-27T18:00:00Z",
                },
            ]
        )
        game = self.game_link(
            schedule_sha256=digest(schedule),
        )
        self.assertEqual(
            self.build(schedule=schedule, games=[game])["claims"],
            [],
        )

    def test_schedule_available_exactly_at_cutoff_allowed(self):
        schedule = self.schedule(
            available_at_utc="2026-09-27T16:00:00Z",
        )
        game = self.game_link(
            schedule_sha256=digest(schedule),
        )
        claims = self.build(
            schedule=schedule,
            games=[game],
            cutoff="2026-09-27T16:00:00Z",
        )["claims"]
        self.assertEqual(len(claims), 1)

    def test_knowledge_exactly_at_cutoff_allowed(self):
        event = self.event(
            updated_at_utc="2026-09-27T16:00:00Z",
            first_seen_at_utc="2026-09-27T16:00:00Z",
            retrieved_at_utc="2026-09-27T16:00:00Z",
        )
        claims = self.build(event=event)["claims"]
        self.assertEqual(len(claims), 1)
        self.assertEqual(
            claims[0]["knowledge_at_utc"],
            "2026-09-27T16:00:00+00:00",
        )

    def test_publication_at_kickoff_fails_closed(self):
        event = self.event(
            published_at_utc="2026-09-27T17:00:00Z",
            updated_at_utc="2026-09-27T17:00:00Z",
            first_seen_at_utc="2026-09-27T17:00:00Z",
            retrieved_at_utc="2026-09-27T17:00:00Z",
        )
        self.assertEqual(
            self.build(
                event=event,
                cutoff="2026-09-27T17:00:00Z",
            )["claims"],
            [],
        )

    def test_updated_at_kickoff_fails_closed(self):
        event = self.event(
            updated_at_utc="2026-09-27T17:00:00Z",
            first_seen_at_utc="2026-09-27T15:01:00Z",
            retrieved_at_utc="2026-09-27T15:01:00Z",
        )
        self.assertEqual(
            self.build(event=event)["claims"],
            [],
        )

    def test_schedule_hash_provenance_is_exact(self):
        schedule = self.schedule()
        game = self.game_link(
            schedule_sha256=digest(schedule),
        )
        claims = self.build(
            schedule=schedule,
            games=[game],
        )["claims"]

        self.assertEqual(len(claims), 1)
        self.assertEqual(
            claims[0]["schedule_sha256"],
            digest(schedule),
        )

    def test_valid_until_is_exact_kickoff(self):
        claims = self.build()["claims"]
        self.assertEqual(len(claims), 1)
        self.assertEqual(
            claims[0]["valid_until_utc"],
            claims[0]["kickoff_at_utc"],
        )

    def test_effective_from_is_exact_knowledge_time(self):
        claims = self.build()["claims"]
        self.assertEqual(len(claims), 1)
        self.assertEqual(
            claims[0]["effective_from_utc"],
            claims[0]["knowledge_at_utc"],
        )


if __name__ == "__main__":
    unittest.main()
