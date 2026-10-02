"""M4A: pure, shadow-only corroboration metadata over already-built claims.

No eligibility construction, revision selection, provenance interpretation or I/O.
Exact signal types and comparison dimensions are retained (no signal aliases).
Lifecycle presence is all-or-none; legacy claims have a separate cutoff-only
horizon. Timestamps are compared in UTC. Exact effective/knowledge boundaries
are deliberately conservative and can split otherwise similar propositions.

Only origin_id establishes independence. Neutral/unspecified observations do
not vote. An internally inconsistent origin cannot establish independent
conflict: it produces MIXED unless other, unambiguous origins prove CONFLICT.
Explicit directional labels are accepted; no direction is inferred from text.
"""

import hashlib
import json
from datetime import datetime, timezone

ANALYSIS_ONLY = True
PRODUCTION_INFLUENCE = False
SOLVER_INFLUENCE = False
PROJECTION_MUTATION = False
FORECAST_MUTATION = False
AVAILABILITY_AUTHORITY = False
INJURY_AUTHORITY = False
DEPTH_CHART_AUTHORITY = False
AI_ANALYST_INFLUENCE = False
DATABASE_MUTATION = False
CONSENSUS_ENABLED = False

SAFETY = {
    "ANALYSIS_ONLY": ANALYSIS_ONLY,
    "PRODUCTION_INFLUENCE": PRODUCTION_INFLUENCE,
    "SOLVER_INFLUENCE": SOLVER_INFLUENCE,
    "PROJECTION_MUTATION": PROJECTION_MUTATION,
    "FORECAST_MUTATION": FORECAST_MUTATION,
    "AVAILABILITY_AUTHORITY": AVAILABILITY_AUTHORITY,
    "INJURY_AUTHORITY": INJURY_AUTHORITY,
    "DEPTH_CHART_AUTHORITY": DEPTH_CHART_AUTHORITY,
    "AI_ANALYST_INFLUENCE": AI_ANALYST_INFLUENCE,
    "DATABASE_MUTATION": DATABASE_MUTATION,
    "CONSENSUS_ENABLED": CONSENSUS_ENABLED,
}
SIGNALS = frozenset(("STARTER_CHANGE", "ROLE_INCREASE", "ROLE_DECREASE", "COACH_INTENT",
                     "TARGET_SHARE", "ROUTE_PARTICIPATION", "BACKFIELD_SHARE",
                     "RED_ZONE_ROLE", "DEEP_TARGET_ROLE"))
MEANINGFUL = frozenset(("INCREASE", "DECREASE", "POSITIVE", "NEGATIVE"))
DIRECTIONS = MEANINGFUL | {"NEUTRAL", "UNSPECIFIED"}
OPPOSING = (frozenset(("INCREASE", "DECREASE")), frozenset(("POSITIVE", "NEGATIVE")))
IDENTIFIERS = ("claim_id", "record_id", "event_key", "source", "source_event_id",
               "content_hash", "gsis_id")
LIFECYCLE = ("game_id", "kickoff_at_utc", "schedule_sha256", "knowledge_at_utc",
             "effective_from_utc", "valid_until_utc")
REQUIRED = IDENTIFIERS + ("origin_id", "signal_type", "direction", "evidence_kind",
                         "evidence_quote", "published_at_utc", "cutoff_utc",
                         "eligibility_status", "claim_eligible", "consensus_eligible",
                         "consensus_gate", "corroboration_state", "safety")
COMPARISON = ("metric", "unit", "denominator", "comparison_basis", "conditions",
              "relationship_scope")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _identifier(value):
    return (isinstance(value, str) and bool(value) and value == value.strip()
            and value.isprintable() and not any(c.isspace() for c in value))


def _origin(value):
    # Explicit unknown sentinels are audit observations, never independence keys.
    return _identifier(value) and value.upper() not in {
        "UNKNOWN", "UNSPECIFIED", "UNRESOLVED", "NONE", "NULL", "N/A",
    }


def _time(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("TIMESTAMP_REQUIRED")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("TIMEZONE_REQUIRED")
    return parsed.astimezone(timezone.utc).isoformat()


def _validate(row, cutoff):
    reasons = []
    missing = set(REQUIRED) - row.keys()
    # Missing origin is retained solely as an unknown-origin audit observation.
    if missing - {"origin_id"}:
        reasons.append("MISSING_REQUIRED_FIELDS")
    if any(not _identifier(row.get(k)) for k in IDENTIFIERS):
        reasons.append("MALFORMED_IDENTIFIER")
    if row.get("signal_type") not in SIGNALS or row.get("direction") not in DIRECTIONS:
        reasons.append("INVALID_SIGNAL_OR_DIRECTION")
    if (row.get("signal_type") == "ROLE_INCREASE" and row.get("direction") != "INCREASE"
            or row.get("signal_type") == "ROLE_DECREASE" and row.get("direction") != "DECREASE"):
        reasons.append("CONTRADICTORY_SIGNAL_DIRECTION")
    if any(not isinstance(row.get(k), str) or not row[k].strip()
           for k in ("evidence_kind", "evidence_quote")):
        reasons.append("MALFORMED_EVIDENCE")
    if row.get("eligibility_status") != "ELIGIBLE" or row.get("claim_eligible") is not True:
        reasons.append("INELIGIBLE_CLAIM")
    if row.get("consensus_eligible") is not False or row.get("consensus_gate") != "NOT_ENABLED_V1_3":
        reasons.append("CONSENSUS_CONTRACT_VIOLATION")
    if row.get("corroboration_state") != "NOT_EVALUATED":
        reasons.append("NOT_AN_UNEVALUATED_CLAIM")
    safety = row.get("safety")
    if (not isinstance(safety, dict)
            or any(safety.get(k) is not v for k, v in SAFETY.items() if k != "CONSENSUS_ENABLED")
            or ("CONSENSUS_ENABLED" in safety and safety["CONSENSUS_ENABLED"] is not False)):
        reasons.append("UNSAFE_CLAIM")
    dimensions = {k: row.get(k) for k in ("gsis_id", "signal_type", "evidence_kind")}
    dimensions["comparison"] = {k: row[k] for k in COMPARISON if k in row}
    try:
        boundary = _time(row.get("cutoff_utc"))
        published = _time(row.get("published_at_utc"))
        dimensions["cutoff_utc"] = boundary
        if published > boundary:
            reasons.append("PUBLISHED_AFTER_CUTOFF")
        if cutoff is not None and boundary != cutoff:
            reasons.append("CUTOFF_MISMATCH")
        strict = any(k in row for k in LIFECYCLE)
        dimensions["lifecycle_mode"] = "STRICT" if strict else "LEGACY_CUTOFF_ONLY"
        if strict:
            if not all(k in row for k in LIFECYCLE):
                raise ValueError("PARTIAL_LIFECYCLE")
            if not _identifier(row["game_id"]) or not _identifier(row["schedule_sha256"]):
                raise ValueError("MALFORMED_LIFECYCLE_ID")
            dimensions.update({k: row[k] if k in ("game_id", "schedule_sha256") else _time(row[k])
                               for k in LIFECYCLE})
            knowledge = dimensions["knowledge_at_utc"]
            effective = dimensions["effective_from_utc"]
            until = dimensions["valid_until_utc"]
            kickoff = dimensions["kickoff_at_utc"]
            if not (published <= knowledge == effective <= boundary < until == kickoff):
                reasons.append("CONTRADICTORY_LIFECYCLE")
    except (ValueError, TypeError, OverflowError):
        reasons.append("INVALID_TIMESTAMP_OR_LIFECYCLE")
    return dimensions, reasons


def build_corroboration(claims, *, cutoff_utc=None):
    """Return groups, per-claim states, rejected observations and fixed safety.

    Invalid rows never vote. Conflicting captures of a claim ID, contradictory
    record metadata, or multiple content hashes in an event family at one
    cutoff quarantine every affected claim; no revision winner is selected.
    Exact duplicate captures are collapsed. Rejected rows without a usable
    claim ID are addressed by a semantic diagnostic hash in ``rejected``.
    An explicit cutoff must match the claims' cutoff; it never re-times them.
    Invalid cutoff/container raises ValueError rather than evaluating a batch.
    """
    cutoff = _time(cutoff_utc) if cutoff_utc is not None else None
    if not isinstance(claims, (list, tuple)):
        raise ValueError("CLAIMS_SEQUENCE_REQUIRED")
    captures = {}
    rejected = {}
    invalid_claim_ids = set()
    for row in claims:
        try:
            encoded = _canonical(row)
            # Detach all nested structures, including optional comparison data.
            copy = json.loads(encoded)
            if not isinstance(copy, dict):
                raise ValueError("CLAIM_OBJECT_REQUIRED")
        except (ValueError, TypeError, OverflowError, RecursionError):
            cid = row.get("claim_id") if isinstance(row, dict) else None
            cid = cid if _identifier(cid) else None
            if cid is not None:
                invalid_claim_ids.add(cid)
            key = _digest(["INVALID_ROW_TYPE", type(row).__name__, cid])
            rejected[key] = dict(observation_id=key, claim_id=cid,
                                 reason_codes=["NON_JSON_CLAIM_OBJECT"])
            continue
        captures[encoded] = copy

    rows = []
    for encoded, row in sorted(captures.items()):
        # Membership checks must never raise on malformed list/dict enum values.
        try:
            dimensions, reasons = _validate(row, cutoff)
        except TypeError:
            dimensions, reasons = {}, ["MALFORMED_FIELD_TYPE"]
        if _identifier(row.get("claim_id")) and row["claim_id"] in invalid_claim_ids:
            reasons.append("CONTRADICTORY_CLAIM_ID")
        rows.append(dict(row=row, dimensions=dimensions, reasons=set(reasons), encoded=encoded))

    def quarantine_collisions(keys, signature, reason):
        buckets = {}
        for entry in rows:
            row = entry["row"]
            key = keys(row)
            if key is not None:
                buckets.setdefault(key, []).append(entry)
        for entries in buckets.values():
            if len({_canonical(signature(e)) for e in entries}) > 1:
                for entry in entries:
                    entry["reasons"].add(reason)

    def at_cutoff(row, field):
        if not _identifier(row.get(field)):
            return None
        try:
            return row[field], _time(row.get("cutoff_utc"))
        except (ValueError, TypeError, OverflowError):
            # Malformed cutoff still cannot provide a vote.
            return None

    quarantine_collisions(lambda r: r.get("claim_id") if _identifier(r.get("claim_id")) else None,
                          lambda e: e["row"], "CONTRADICTORY_CLAIM_ID")
    quarantine_collisions(lambda r: at_cutoff(r, "event_key"),
                          lambda e: e["row"].get("content_hash"), "UNRESOLVED_REVISION_FAMILY")
    quarantine_collisions(lambda r: at_cutoff(r, "event_key"),
                          lambda e: {k: e["row"].get(k) for k in
                                     ("source", "source_event_id", "origin_id")},
                          "CONTRADICTORY_EVENT_IDENTITY")

    def native_key(row):
        key = at_cutoff(row, "source_event_id")
        return (row["source"], key) if key is not None and _identifier(row.get("source")) else None

    # Relabeling an event_key cannot bypass the revision-family quarantine.
    quarantine_collisions(native_key,
                          lambda e: {k: e["row"].get(k) for k in
                                     ("event_key", "origin_id", "content_hash")},
                          "CONTRADICTORY_NATIVE_EVENT")
    record_fields = ("event_key", "source", "source_event_id", "origin_id", "content_hash",
                     "published_at_utc") + LIFECYCLE
    quarantine_collisions(lambda r: at_cutoff(r, "record_id"),
                          lambda e: {k: e["row"].get(k) for k in record_fields},
                          "CONTRADICTORY_RECORD_METADATA")

    groups = {}
    states = {cid: dict(status="INSUFFICIENT", group_id=None, voting=False,
                        reason_codes=["NON_JSON_CLAIM_OBJECT"])
              for cid in invalid_claim_ids}
    for entry in rows:
        row, reasons = entry["row"], entry["reasons"]
        cid = row.get("claim_id")
        if reasons:
            diagnostic_id = _digest(["M4A_REJECTED", row])
            rejected[diagnostic_id] = dict(observation_id=diagnostic_id,
                                          claim_id=cid if _identifier(cid) else None,
                                          reason_codes=sorted(reasons))
            if _identifier(cid):
                state = states.setdefault(cid, dict(status="INSUFFICIENT", group_id=None,
                                                     voting=False, reason_codes=[]))
                state["reason_codes"] = sorted(set(state["reason_codes"]) | reasons)
            continue
        dimensions = entry["dimensions"]
        gid = _digest(["M4A_PROPOSITION_V1_3", dimensions])
        group = groups.setdefault(gid, dict(group_id=gid, **dimensions, _rows=[]))
        group["_rows"].append(row)

    for gid, group in sorted(groups.items()):
        members = group.pop("_rows")
        origins = {}
        unknown = []
        direction_ids = {}
        for row in members:
            direction_ids.setdefault(row["direction"], []).append(row["claim_id"])
            if _origin(row.get("origin_id")):
                origins.setdefault(row["origin_id"], set()).add(row["direction"])
            else:
                unknown.append(row["claim_id"])
        votes = {o: ds & MEANINGFUL for o, ds in origins.items() if ds & MEANINGFUL}
        clean = {o: next(iter(ds)) for o, ds in votes.items() if len(ds) == 1}
        conflict = any(frozenset((a, b)) in OPPOSING
                       for a in clean.values() for b in clean.values())
        mixed = any(len(ds) > 1 for ds in votes.values())
        conclusions = set().union(*votes.values()) if votes else set()
        if conflict:
            status, reasons = "CONFLICT", ["INDEPENDENT_OPPOSING_ORIGINS"]
        elif mixed:
            status, reasons = "MIXED", ["SAME_ORIGIN_INCONSISTENCY"]
        elif len(votes) == 1:
            status, reasons = "SINGLE_ORIGIN", ["ONE_VOTING_ORIGIN"]
        elif len(votes) >= 2 and len(conclusions) == 1:
            status, reasons = "AGREEMENT", ["INDEPENDENT_DIRECTIONAL_AGREEMENT"]
        else:
            status, reasons = "INSUFFICIENT", ["NO_VOTING_ORIGIN" if not votes else "NO_VALIDATED_OPPOSITION"]
        group.update(claim_ids=sorted(r["claim_id"] for r in members),
                     independent_origin_count=len(votes), known_origin_count=len(origins),
                     unknown_origin_count=len(unknown), unknown_origin_claim_ids=sorted(unknown),
                     origin_directions={o: sorted(ds) for o, ds in sorted(origins.items())},
                     direction_claim_ids={d: sorted(ids) for d, ids in sorted(direction_ids.items())},
                     status=status, reason_codes=reasons)
        for row in members:
            voting = _origin(row.get("origin_id")) and row["direction"] in MEANINGFUL
            states[row["claim_id"]] = dict(
                group_id=gid, group_status=status, status=status if voting else "INSUFFICIENT",
                voting=voting, reason_codes=reasons if voting else
                ["UNKNOWN_ORIGIN" if not _origin(row.get("origin_id")) else "NON_DIRECTIONAL"])
    return dict(groups=[groups[k] for k in sorted(groups)],
                claim_states={k: states[k] for k in sorted(states)},
                rejected=[rejected[k] for k in sorted(rejected)], safety=dict(SAFETY))
