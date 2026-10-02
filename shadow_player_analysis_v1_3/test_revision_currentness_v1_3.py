import unittest

from shadow_player_analysis_v1_3.claims import build_claims, digest
from shadow_player_analysis_v1_3.test_claim_lifecycle_v1_3 import (
    ClaimLifecycleV13Tests,
)


class RevisionCurrentnessV13Tests(ClaimLifecycleV13Tests):

    def revision_bundle(
        self,
        *,
        cutoff,
        second_retrieved,
        second_first_seen=None,
        schedule=None,
    ):
        schedule = schedule or self.schedule()

        first_event = self.event(
            record_id="record-a",
            event_key="event-family-1",
            content_hash="content-a",
            published_at_utc="2026-09-27T14:00:00Z",
            updated_at_utc="2026-09-27T14:00:00Z",
            first_seen_at_utc="2026-09-27T14:01:00Z",
            retrieved_at_utc="2026-09-27T14:01:00Z",
        )

        second_event = self.event(
            record_id="record-b",
            event_key="event-family-1",
            content_hash="content-b",
            published_at_utc="2026-09-27T14:00:00Z",
            updated_at_utc="2026-09-27T15:00:00Z",
            first_seen_at_utc=(
                second_first_seen or second_retrieved
            ),
            retrieved_at_utc=second_retrieved,
        )

        first_evidence = self.evidence(
            record_id="record-a",
            event_key="event-family-1",
        )
        second_evidence = self.evidence(
            record_id="record-b",
            event_key="event-family-1",
        )

        first_player = self.player_link(
            record_id="record-a",
            link_id="player-a",
        )
        second_player = self.player_link(
            record_id="record-b",
            link_id="player-b",
        )

        first_game = self.game_link(
            record_id="record-a",
            link_id="game-a",
            schedule_sha256=digest(schedule),
        )
        second_game = self.game_link(
            record_id="record-b",
            link_id="game-b",
            schedule_sha256=digest(schedule),
        )

        return build_claims(
            source_events=[first_event, second_event],
            classified_evidence=[
                first_evidence,
                second_evidence,
            ],
            entity_links=[
                first_player,
                second_player,
                first_game,
                second_game,
            ],
            cutoff_utc=cutoff,
            schedule=schedule,
        )

    def test_late_revision_does_not_rewrite_historical_cutoff(self):
        schedule = self.schedule(
            available_at_utc="2026-09-27T14:02:00Z"
        )

        result = self.revision_bundle(
            cutoff="2026-09-27T14:30:00Z",
            second_retrieved="2026-09-27T15:01:00Z",
            schedule=schedule,
        )

        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(
            result["claims"][0]["record_id"],
            "record-a",
        )

    def test_latest_known_revision_wins_after_it_is_known(self):
        result = self.revision_bundle(
            cutoff="2026-09-27T16:00:00Z",
            second_retrieved="2026-09-27T15:01:00Z",
        )

        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(
            result["claims"][0]["record_id"],
            "record-b",
        )

    def test_equal_knowledge_time_fails_closed(self):
        schedule = self.schedule()

        a = self.event(
            record_id="record-a",
            event_key="event-family-1",
            content_hash="content-a",
            updated_at_utc="2026-09-27T15:00:00Z",
            first_seen_at_utc="2026-09-27T15:01:00Z",
            retrieved_at_utc="2026-09-27T15:01:00Z",
        )
        b = self.event(
            record_id="record-b",
            event_key="event-family-1",
            content_hash="content-b",
            updated_at_utc="2026-09-27T15:00:00Z",
            first_seen_at_utc="2026-09-27T15:01:00Z",
            retrieved_at_utc="2026-09-27T15:01:00Z",
        )

        result = build_claims(
            source_events=[a, b],
            classified_evidence=[
                self.evidence(
                    record_id="record-a",
                    event_key="event-family-1",
                ),
                self.evidence(
                    record_id="record-b",
                    event_key="event-family-1",
                ),
            ],
            entity_links=[
                self.player_link(
                    record_id="record-a",
                    link_id="player-a",
                ),
                self.player_link(
                    record_id="record-b",
                    link_id="player-b",
                ),
                self.game_link(
                    record_id="record-a",
                    link_id="game-a",
                    schedule_sha256=digest(schedule),
                ),
                self.game_link(
                    record_id="record-b",
                    link_id="game-b",
                    schedule_sha256=digest(schedule),
                ),
            ],
            cutoff_utc="2026-09-27T16:00:00Z",
            schedule=schedule,
        )

        self.assertEqual(result["claims"], [])

    def test_input_order_cannot_choose_revision(self):
        result_a = self.revision_bundle(
            cutoff="2026-09-27T16:00:00Z",
            second_retrieved="2026-09-27T15:01:00Z",
        )

        schedule = self.schedule()

        a = self.event(
            record_id="record-a",
            event_key="event-family-1",
            content_hash="content-a",
            published_at_utc="2026-09-27T14:00:00Z",
            updated_at_utc="2026-09-27T14:00:00Z",
            first_seen_at_utc="2026-09-27T14:01:00Z",
            retrieved_at_utc="2026-09-27T14:01:00Z",
        )
        b = self.event(
            record_id="record-b",
            event_key="event-family-1",
            content_hash="content-b",
            published_at_utc="2026-09-27T14:00:00Z",
            updated_at_utc="2026-09-27T15:00:00Z",
            first_seen_at_utc="2026-09-27T15:01:00Z",
            retrieved_at_utc="2026-09-27T15:01:00Z",
        )

        result_b = build_claims(
            source_events=[b, a],
            classified_evidence=[
                self.evidence(
                    record_id="record-b",
                    event_key="event-family-1",
                ),
                self.evidence(
                    record_id="record-a",
                    event_key="event-family-1",
                ),
            ],
            entity_links=[
                self.player_link(
                    record_id="record-b",
                    link_id="player-b",
                ),
                self.player_link(
                    record_id="record-a",
                    link_id="player-a",
                ),
                self.game_link(
                    record_id="record-b",
                    link_id="game-b",
                    schedule_sha256=digest(schedule),
                ),
                self.game_link(
                    record_id="record-a",
                    link_id="game-a",
                    schedule_sha256=digest(schedule),
                ),
            ],
            cutoff_utc="2026-09-27T16:00:00Z",
            schedule=schedule,
        )

        self.assertEqual(
            result_a["claims"],
            result_b["claims"],
        )

    def test_lexical_record_id_does_not_choose_revision(self):
        result = self.revision_bundle(
            cutoff="2026-09-27T16:00:00Z",
            second_retrieved="2026-09-27T15:01:00Z",
        )

        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(
            result["claims"][0]["record_id"],
            "record-b",
        )


if __name__ == "__main__":
    unittest.main()
