import copy
import hashlib
import json
import unittest

from . import claims
from .handoff import build_shadow_handoff, SAFETY


CUTOFF = "2026-09-27T16:00:00Z"
KICKOFF = "2026-09-27T20:00:00Z"


def schedule():
    return {"authority": "WFS_SCHEDULE_SNAPSHOT", "available_at_utc":
            "2026-09-27T12:00:00Z", "games": [{"game_id": "g1",
            "kickoff_at_utc": KICKOFF}]}


def fixture(origins=("o1",), direction_quote="role will increase"):
    sched = schedule()
    shash = claims.digest(sched)
    events, evidence, links = [], [], []
    for i, origin in enumerate(origins, 1):
        rid = "r" + str(i)
        events.append({"event_key": "e" + str(i), "record_id": rid,
                       "source": "src" + str(i), "source_event_id": "se" + str(i),
                       "content_hash": "h" + str(i), "origin_id": origin,
                       "published_at_utc": "2026-09-27T10:00:00Z",
                       "retrieved_at_utc": "2026-09-27T11:00:00Z",
                       "first_seen_at_utc": "2026-09-27T11:00:00Z",
                       "evidence_text": "Player role will increase."})
        evidence.append({"record_id": rid, "source_approved": True,
                         "pregame_eligible": True,
                         "temporal_status": "UNAMBIGUOUS_SOURCE_TIMESTAMP",
                         "temporal_confidence": "RESOLVED", "relevance": "HIGH",
                         "evidence_types": ["ROLE_INCREASE"]})
        links.extend([
            {"record_id": rid, "link_id": "p" + str(i), "entity_type": "PLAYER",
             "identity_status": "RESOLVED", "identity_reason": "EXACT_UNIQUE_NAME_VALIDATED",
             "gsis_id": "p1", "relevance": "HIGH", "evidence_types": ["ROLE_INCREASE"],
             "evidence_kind": "REPORT",
             "mention": {"name": "Player", "evidence_quote": direction_quote,
                         "relationship_scope": "PLAYER"}},
            {"record_id": rid, "link_id": "g" + str(i), "entity_type": "GAME",
             "identity_status": "RESOLVED", "reason_codes": [], "game_id": "g1",
             "schedule_sha256": shash},
        ])
    return {"source_events": events, "classified_evidence": evidence,
            "entity_links": links, "quarantine": [],
            "source_quality": [{"source": "src", "records": len(events),
                                "consensus_votes": 0}]}, sched


class HandoffTests(unittest.TestCase):
    def test_complete_handoff_and_detachment(self):
        value, sched = fixture()
        original = copy.deepcopy(value)
        packet = build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF,
                                      schedule=sched)
        self.assertEqual(len(packet["claims"]["claims"]), 1)
        self.assertEqual(len(packet["revision_provenance"]["revision_provenance"]), 1)
        self.assertEqual(packet["corroboration"]["groups"][0]["status"], "SINGLE_ORIGIN")
        self.assertEqual(value, original)
        packet["quarantine"].append({"x": []})
        self.assertEqual(value["quarantine"], [])

    def test_independent_agreement(self):
        value, sched = fixture(("o1", "o2"))
        packet = build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF,
                                      schedule=sched)
        self.assertEqual(packet["corroboration"]["groups"][0]["status"], "AGREEMENT")

    def test_revision_provenance_and_quarantine_are_preserved(self):
        value, sched = fixture()
        value["quarantine"].append({"record_id": "r1",
                                     "reason_codes": ["SOURCE_EVENT_REVISION_CONFLICT"]})
        packet = build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF,
                                      schedule=sched)
        self.assertIn("SOURCE_EVENT_REVISION_CONFLICT",
                      packet["quarantine"][0]["reason_codes"])
        self.assertEqual(len(packet["revision_provenance"]["revision_provenance"]), 1)

    def test_ambiguous_timestamp_stays_ineligible(self):
        value, sched = fixture()
        value["classified_evidence"][0]["temporal_status"] = "AMBIGUOUS_SOURCE_TIMESTAMP"
        value["classified_evidence"][0]["temporal_confidence"] = "UNRESOLVED"
        packet = build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF,
                                      schedule=sched)
        self.assertEqual(packet["claims"]["claims"], [])
        self.assertEqual(packet["corroboration"]["groups"], [])

    def test_safety_is_fixed(self):
        value, sched = fixture()
        packet = build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF,
                                      schedule=sched)
        self.assertEqual(packet["safety"], SAFETY)
        self.assertFalse(packet["corroboration"]["safety"]["CONSENSUS_ENABLED"])

    def test_deterministic_repeated_execution(self):
        value, sched = fixture(("o1", "o2"))
        a = build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF, schedule=sched)
        b = build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF, schedule=sched)
        self.assertEqual(a, b)

    def test_same_origin_is_not_agreement(self):
        value, sched = fixture(("o1", "o1"))
        value["source_events"][1]["event_key"] = "e2"
        packet = build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF,
                                      schedule=sched)
        self.assertEqual(packet["corroboration"]["groups"][0]["status"], "SINGLE_ORIGIN")


if __name__ == "__main__":
    unittest.main()
