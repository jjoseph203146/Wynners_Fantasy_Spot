import unittest

from shadow_player_analysis_v1_3.claims import build_claims, digest
from shadow_player_analysis_v1_3.test_claim_lifecycle_v1_3 import (
    ClaimLifecycleV13Tests,
)


class RevisionCurrentnessAdversarialV13Tests(ClaimLifecycleV13Tests):

    def build_family(self, events, cutoff="2026-09-27T16:00:00Z"):
        schedule = self.schedule()

        evidence = []
        links = []

        seen_records = set()

        for event in events:
            rid = event["record_id"]

            if rid in seen_records:
                continue

            seen_records.add(rid)

            evidence.append(
                self.evidence(
                    record_id=rid,
                    event_key=event["event_key"],
                )
            )

            links.append(
                self.player_link(
                    record_id=rid,
                    link_id=f"player-{rid}",
                )
            )

            links.append(
                self.game_link(
                    record_id=rid,
                    link_id=f"game-{rid}",
                    schedule_sha256=digest(schedule),
                )
            )

        return build_claims(
            source_events=events,
            classified_evidence=evidence,
            entity_links=links,
            cutoff_utc=cutoff,
            schedule=schedule,
        )

    def event_version(
        self,
        record_id,
        content_hash,
        *,
        updated,
        first_seen,
        retrieved,
    ):
        return self.event(
            record_id=record_id,
            event_key="revision-family",
            content_hash=content_hash,
            published_at_utc="2026-09-27T14:00:00Z",
            updated_at_utc=updated,
            first_seen_at_utc=first_seen,
            retrieved_at_utc=retrieved,
        )

    def test_three_revision_chain_latest_known_wins(self):
        events = [
            self.event_version(
                "record-a", "content-a",
                updated="2026-09-27T14:10:00Z",
                first_seen="2026-09-27T14:11:00Z",
                retrieved="2026-09-27T14:11:00Z",
            ),
            self.event_version(
                "record-b", "content-b",
                updated="2026-09-27T14:30:00Z",
                first_seen="2026-09-27T14:31:00Z",
                retrieved="2026-09-27T14:31:00Z",
            ),
            self.event_version(
                "record-c", "content-c",
                updated="2026-09-27T15:00:00Z",
                first_seen="2026-09-27T15:01:00Z",
                retrieved="2026-09-27T15:01:00Z",
            ),
        ]

        result = self.build_family(events)

        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["record_id"], "record-c")

    def test_future_third_revision_does_not_replace_second(self):
        events = [
            self.event_version(
                "record-a", "content-a",
                updated="2026-09-27T14:10:00Z",
                first_seen="2026-09-27T14:11:00Z",
                retrieved="2026-09-27T14:11:00Z",
            ),
            self.event_version(
                "record-b", "content-b",
                updated="2026-09-27T15:00:00Z",
                first_seen="2026-09-27T15:01:00Z",
                retrieved="2026-09-27T15:01:00Z",
            ),
            self.event_version(
                "record-c", "content-c",
                updated="2026-09-27T16:30:00Z",
                first_seen="2026-09-27T16:31:00Z",
                retrieved="2026-09-27T16:31:00Z",
            ),
        ]

        result = self.build_family(events)

        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["record_id"], "record-b")

    def test_latest_tie_fails_closed(self):
        events = [
            self.event_version(
                "record-a", "content-a",
                updated="2026-09-27T14:10:00Z",
                first_seen="2026-09-27T14:11:00Z",
                retrieved="2026-09-27T14:11:00Z",
            ),
            self.event_version(
                "record-b", "content-b",
                updated="2026-09-27T15:00:00Z",
                first_seen="2026-09-27T15:01:00Z",
                retrieved="2026-09-27T15:01:00Z",
            ),
            self.event_version(
                "record-c", "content-c",
                updated="2026-09-27T15:00:00Z",
                first_seen="2026-09-27T15:01:00Z",
                retrieved="2026-09-27T15:01:00Z",
            ),
        ]

        self.assertEqual(self.build_family(events)["claims"], [])

    def test_malformed_latest_revision_does_not_become_winner(self):
        events = [
            self.event_version(
                "record-a", "content-a",
                updated="2026-09-27T14:10:00Z",
                first_seen="2026-09-27T14:11:00Z",
                retrieved="2026-09-27T14:11:00Z",
            ),
            self.event_version(
                "record-b", "content-b",
                updated="NOT-A-TIMESTAMP",
                first_seen="2026-09-27T15:01:00Z",
                retrieved="2026-09-27T15:01:00Z",
            ),
        ]

        self.assertEqual(self.build_family(events)["claims"], [])

    def test_duplicate_capture_does_not_create_false_tie(self):
        a = self.event_version(
            "record-a", "content-a",
            updated="2026-09-27T14:10:00Z",
            first_seen="2026-09-27T14:11:00Z",
            retrieved="2026-09-27T14:11:00Z",
        )

        b = self.event_version(
            "record-b", "content-b",
            updated="2026-09-27T15:00:00Z",
            first_seen="2026-09-27T15:01:00Z",
            retrieved="2026-09-27T15:01:00Z",
        )

        result = self.build_family([a, b, dict(b)])

        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["record_id"], "record-b")

    def test_reversed_input_order_same_winner(self):
        a = self.event_version(
            "record-a", "content-a",
            updated="2026-09-27T14:10:00Z",
            first_seen="2026-09-27T14:11:00Z",
            retrieved="2026-09-27T14:11:00Z",
        )

        b = self.event_version(
            "record-b", "content-b",
            updated="2026-09-27T15:00:00Z",
            first_seen="2026-09-27T15:01:00Z",
            retrieved="2026-09-27T15:01:00Z",
        )

        forward = self.build_family([a, b])
        reverse = self.build_family([b, a])

        self.assertEqual(forward["claims"], reverse["claims"])
        self.assertEqual(forward["claims"][0]["record_id"], "record-b")


if __name__ == "__main__":
    unittest.main()
