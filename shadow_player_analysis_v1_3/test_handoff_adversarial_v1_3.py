import copy
import unittest
from unittest.mock import patch

from .handoff import build_shadow_handoff
from .test_handoff_v1_3 import CUTOFF, fixture, schedule


class HandoffAdversarialTests(unittest.TestCase):
    def test_missing_and_wrong_v12_collections_fail_closed(self):
        value, sched = fixture()
        for key in ("source_events", "classified_evidence", "entity_links",
                    "quarantine", "source_quality"):
            bad = copy.deepcopy(value)
            del bad[key]
            with self.assertRaises(ValueError):
                build_shadow_handoff(v1_2_result=bad, cutoff_utc=CUTOFF, schedule=sched)
        bad = copy.deepcopy(value)
        bad["source_events"] = {}
        with self.assertRaises(ValueError):
            build_shadow_handoff(v1_2_result=bad, cutoff_utc=CUTOFF, schedule=sched)

    def test_malformed_schedule_and_cutoff_fail_closed(self):
        value, sched = fixture()
        for bad_schedule in ({}, {"authority": "OTHER"},
                             {**sched, "games": "bad"},
                             {**sched, "available_at_utc": "bad"},
                             {**sched, "available_at_utc": "2027-01-01T00:00:00Z"}):
            with self.assertRaises(ValueError):
                build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF,
                                     schedule=bad_schedule)
        for cutoff in ("bad", "2026-09-27T16:00:00", None):
            with self.assertRaises(ValueError):
                build_shadow_handoff(v1_2_result=value, cutoff_utc=cutoff, schedule=sched)

    def test_extra_fields_and_malformed_nested_json_fail_closed(self):
        value, sched = fixture()
        value["unexpected"] = {"retained": True}
        self.assertEqual(build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF,
                                               schedule=sched)["input_audit"]["source_event_count"], 1)
        bad = copy.deepcopy(value)
        bad["source_events"] = [set(["not-json"])]
        with self.assertRaises(ValueError):
            build_shadow_handoff(v1_2_result=bad, cutoff_utc=CUTOFF, schedule=sched)

    def test_duplicate_schedule_ids_fail(self):
        value, sched = fixture()
        bad = copy.deepcopy(sched)
        bad["games"].append(copy.deepcopy(bad["games"][0]))
        with self.assertRaises(ValueError):
            build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF, schedule=bad)

    def test_future_and_ambiguous_evidence_do_not_get_repaired(self):
        value, sched = fixture()
        value["classified_evidence"][0]["pregame_eligible"] = False
        value["classified_evidence"][0]["reason_codes"] = ["SOURCE_EVENT_REVISION_CONFLICT"]
        packet = build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF, schedule=sched)
        self.assertEqual(packet["claims"]["claims"], [])
        self.assertEqual(packet["corroboration"]["groups"], [])

    def test_downstream_shape_violation_fails(self):
        value, sched = fixture()
        with patch("shadow_player_analysis_v1_3.handoff.build_claims", return_value={"claims": []}):
            with self.assertRaises(ValueError):
                build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF, schedule=sched)

    def test_input_and_output_are_detached(self):
        value, sched = fixture()
        packet = build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF, schedule=sched)
        packet["claims"]["claims"][0]["evidence_quote"] = "tampered"
        self.assertEqual(value["source_events"][0]["evidence_text"], "Player role will increase.")
        second = build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF, schedule=sched)
        self.assertEqual(second["claims"]["claims"][0]["evidence_quote"], "role will increase")

    def test_consensus_cannot_be_enabled(self):
        value, sched = fixture()
        with patch("shadow_player_analysis_v1_3.handoff.build_corroboration", return_value={
                "groups": [], "claim_states": {}, "rejected": [],
                "safety": {"CONSENSUS_ENABLED": True}}):
            with self.assertRaises(ValueError):
                build_shadow_handoff(v1_2_result=value, cutoff_utc=CUTOFF, schedule=sched)


if __name__ == "__main__":
    unittest.main()
