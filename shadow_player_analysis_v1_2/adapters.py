"""Offline bridge consuming existing ESPN normalized records, including quarantine."""
import copy
from shadow_player_analysis_v1.core import digest


def espn_records(capture):
    rows = {}
    for original in [*capture.get("events", []), *capture.get("quarantine", [])]:
        row = copy.deepcopy(original)
        row["raw_publication_timestamp"] = row.get("raw_feed_fields", {}).get("pubDate")
        # These V1.1 placeholders describe the old unclassified identity/manual
        # layer, not adapter integrity failures. All other failures are retained.
        reasons = row.pop("reason_codes", [])
        row["adapter_reason_codes"] = sorted(set(reasons) - {"IDENTITY_UNRESOLVED", "MANUAL_CLAIM_REVIEW_REQUIRED"})
        row.pop("eligibility_status", None)
        row.pop("gsis_id", None)
        rows[digest(row)] = row
    return sorted(rows.values(), key=digest)
