import copy
import json
import builtins
import io
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from .core import SAFETY, build, consensus, events
from .fixtures import sample
from .publication import bundle, verify
from . import publication


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = sample()

    def row(self):
        return build(self.fixture)["claims"][0]

    def test_exact_identity(self):
        self.assertEqual(self.row()["gsis_id"], "SYNTHETIC-GSIS-001")
        self.assertEqual(self.row()["eligibility_status"], "ELIGIBLE")

    def test_ambiguous_identity(self):
        other = dict(self.fixture["identity"]["players"][0], gsis_id="SYNTHETIC-GSIS-002")
        self.fixture["identity"]["players"].append(other)
        self.assertEqual(self.row()["identity_reason"], "AMBIGUOUS_EXACT_IDENTITY")
        self.assertEqual(self.row()["eligibility_status"], "QUARANTINED")

    def test_unresolved_identity(self):
        self.fixture["claims"][0]["player_name"] = "Unknown Runner"
        self.assertEqual(len(build(self.fixture)["quarantine"]), 1)
        self.assertIsNone(self.row()["gsis_id"])

    def test_duplicate_event(self):
        event = self.fixture["events"][0]
        self.assertEqual(events([event]), events([event, event]))

    def test_revision(self):
        original = self.fixture["events"][0]
        revised = dict(original, evidence_text="Revised analysis", updated_at_utc="2025-09-07T11:00:00Z",
                       first_seen_at_utc="2025-09-07T11:01:00Z", retrieved_at_utc="2025-09-07T11:02:00Z")
        rows = events([original, revised])
        self.assertEqual(rows, events([revised, original]))
        self.assertEqual(rows[1]["supersedes_revision_id"], rows[0]["revision_id"])
        self.assertNotEqual(rows[0]["revision_id"], rows[1]["revision_id"])

    def test_post_kickoff(self):
        event = self.fixture["events"][0]
        for key in ("published_at_utc", "first_seen_at_utc", "retrieved_at_utc"):
            event[key] = "2025-09-07T18:00:00Z"
        self.fixture["claims"][0]["revision_id"] = events([event])[0]["revision_id"]
        self.assertIn("POST_KICKOFF_PUBLICATION", self.row()["reason_codes"])

    def test_recap(self):
        claim = self.fixture["claims"][0]
        claim.update(evidence_phase="POST_GAME_RECAP", temporal_orientation="RETROSPECTIVE",
                     forward_looking_basis=None)
        self.assertEqual(self.row()["eligibility_status"], "QUARANTINED")
        claim.update(evidence_phase="USAGE_ANALYSIS", temporal_orientation="FORWARD_LOOKING")
        self.assertIn("NO_FORWARD_PROPOSITION", self.row()["reason_codes"])

    def test_duplicate_origin(self):
        row = self.row()
        duplicate = dict(row, claim_id="copy", source="SYNTHETIC_B")
        group = consensus([row, duplicate])[0]
        self.assertEqual(group["independent_origin_count"], 1)
        self.assertEqual(group["status"], "SINGLE_ORIGIN")

    def test_conflict(self):
        row = self.row()
        contrary = dict(row, claim_id="contrary", origin_id="reporter-b", direction="DECREASE", signal_type="ROLE_DECREASE")
        self.assertEqual(consensus([row, contrary])[0]["status"], "CONFLICT")

    def test_unknown_origin_not_consensus(self):
        row = dict(self.row(), origin_id=None)
        self.assertEqual(consensus([row, dict(row, claim_id="other")])[0]["status"], "INSUFFICIENT")

    def test_deterministic_publication(self):
        self.assertEqual(bundle(self.fixture), bundle(copy.deepcopy(self.fixture)))

    def test_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, data in bundle(self.fixture).items():
                (root / name).write_bytes(data)
            self.assertEqual(verify(root)["safety"], SAFETY)
            (root / "claims.json").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "ARTIFACT_HASH"):
                verify(root)

    def test_no_database_or_network(self):
        original_open, original_io_open = builtins.open, io.open
        def read_only(opener):
            def guarded(file, mode="r", *args, **kwargs):
                self.assertFalse(any(c in mode for c in "wax+"), "unexpected file write")
                return opener(file, mode, *args, **kwargs)
            return guarded
        with patch.object(sqlite3, "connect", side_effect=AssertionError("DB access")), \
             patch.object(socket, "socket", side_effect=AssertionError("network")), \
             patch.object(builtins, "open", read_only(original_open)), \
             patch.object(io, "open", read_only(original_io_open)):
            self.assertTrue(bundle(self.fixture))
        self.assertTrue(SAFETY["ANALYSIS_ONLY"])
        self.assertFalse(any(v for k, v in SAFETY.items() if k != "ANALYSIS_ONLY"))

    def test_late_retrieval(self):
        self.fixture["events"][0]["retrieved_at_utc"] = "2025-09-07T19:00:00Z"
        self.assertIn("EVIDENCE_AFTER_CUTOFF", self.row()["reason_codes"])

    def test_future_identity(self):
        self.fixture["identity"]["available_at_utc"] = "2025-09-08T00:00:00Z"
        self.assertIn("IDENTITY_FROM_FUTURE", self.row()["reason_codes"])

    def test_missing_timezone(self):
        self.fixture["events"][0]["published_at_utc"] = "2025-09-07T10:00:00"
        with self.assertRaisesRegex(ValueError, "TIMEZONE_REQUIRED"):
            build(self.fixture)

    def test_unapproved_input(self):
        self.fixture["events"][0]["approval_id"] = "UNAPPROVED"
        with self.assertRaisesRegex(ValueError, "SOURCE_NOT_APPROVED"):
            build(self.fixture)

    def test_immutable_atomic_publisher(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(publication, "ROOT", root), patch.object(publication, "OUTPUT", root / "artifacts"), \
                 patch.object(publication, "code_hashes", return_value={"fixture": "test-only"}), \
                 patch.object(sqlite3, "connect", side_effect=AssertionError("DB access")), \
                 patch.object(socket, "socket", side_effect=AssertionError("network")):
                path = publication.publish(self.fixture)
                self.assertEqual(path, publication.publish(self.fixture))
                verify(path)
                self.assertEqual(list((root / "artifacts").iterdir()), [path])
                (path / "claims.json").write_text("corrupt")
                with self.assertRaisesRegex(ValueError, "ARTIFACT_HASH"):
                    publication.publish(self.fixture)
                self.assertEqual((path / "claims.json").read_text(), "corrupt")

    def test_superseded_claim_does_not_vote(self):
        original = self.fixture["events"][0]
        revised = dict(original, evidence_text="The original analysis is withdrawn.",
                       updated_at_utc="2025-09-07T11:00:00Z", first_seen_at_utc="2025-09-07T11:01:00Z",
                       retrieved_at_utc="2025-09-07T11:02:00Z")
        self.fixture["events"].append(revised)
        self.assertIn("SUPERSEDED_AS_OF_CUTOFF", self.row()["reason_codes"])
        self.assertEqual(build(self.fixture)["consensus"], [])

    def test_late_revision_does_not_rewrite_history(self):
        original = self.fixture["events"][0]
        self.fixture["events"].append(dict(original, evidence_text="Late update",
            updated_at_utc="2025-09-07T18:00:00Z", first_seen_at_utc="2025-09-07T18:01:00Z",
            retrieved_at_utc="2025-09-07T18:02:00Z"))
        self.assertEqual(self.row()["eligibility_status"], "ELIGIBLE")

    def test_incomplete_game_quarantines(self):
        del self.fixture["game"]["kickoff_at_utc"]
        self.assertEqual(self.row()["eligibility_status"], "QUARANTINED")

    def test_same_origin_opposition_is_not_independent(self):
        row = self.row()
        other = dict(row, claim_id="opposing-copy", direction="DECREASE", signal_type="ROLE_DECREASE")
        self.assertEqual(consensus([row, other])[0]["status"], "MIXED")

    def test_at_kickoff_is_not_pregame(self):
        self.fixture["cutoff_utc"] = self.fixture["game"]["kickoff_at_utc"]
        self.assertIn("CUTOFF_NOT_PREGAME", self.row()["reason_codes"])

    def test_box_score_words_are_not_opportunity(self):
        event = self.fixture["events"][0]
        event["evidence_text"] = "Example Runner had 18 carries, 3 catches, 100 yards and 2 touchdowns."
        claim = self.fixture["claims"][0]
        claim.update(revision_id=events([event])[0]["revision_id"], evidence_quote=event["evidence_text"],
                     signal_type="BACKFIELD_SHARE", forward_looking_basis=None)
        self.assertIn("NO_FORWARD_PROPOSITION", self.row()["reason_codes"])

    def test_failed_publication_does_not_expose_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(publication, "ROOT", root), patch.object(publication, "OUTPUT", root / "artifacts"), \
                 patch.object(publication, "code_hashes", return_value={}), \
                 patch.object(publication, "verify", side_effect=ValueError("validation failed")):
                with self.assertRaisesRegex(ValueError, "validation failed"):
                    publication.publish(self.fixture)
                self.assertEqual(list((root / "artifacts").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
