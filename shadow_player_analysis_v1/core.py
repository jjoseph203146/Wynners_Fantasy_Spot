"""Offline contracts. Inputs are explicit snapshots; this module performs no I/O."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

CONTRACT = "PLAYER_ANALYSIS_INTELLIGENCE_V1_SHADOW"
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
SAFETY = {name: globals()[name] for name in (
    "ANALYSIS_ONLY", "PRODUCTION_INFLUENCE", "SOLVER_INFLUENCE", "PROJECTION_MUTATION",
    "FORECAST_MUTATION", "AVAILABILITY_AUTHORITY", "INJURY_AUTHORITY",
    "DEPTH_CHART_AUTHORITY", "AI_ANALYST_INFLUENCE", "DATABASE_MUTATION")}
SIGNALS = frozenset("ROLE_INCREASE ROLE_DECREASE TARGET_SHARE ROUTE_PARTICIPATION BACKFIELD_SHARE RED_ZONE_ROLE DEEP_TARGET_ROLE MATCHUP_POSITIVE MATCHUP_NEGATIVE GAME_SCRIPT OL_MATCHUP COVERAGE_MATCHUP COACH_INTENT INJURY_OPPORTUNITY DFS_VALUE DFS_LEVERAGE".split())
PHASES = frozenset("PRE_GAME_ANALYSIS POST_GAME_RECAP MARKET_PROP_CONTEXT INJURY_REPORT ROLE_ANALYSIS USAGE_ANALYSIS MATCHUP_ANALYSIS COACH_INTENT DFS_ANALYSIS".split())
KINDS = frozenset(("FACTUAL_OBSERVATION", "ANALYST_OPINION", "MARKET_INFORMATION", "WFS_CORROBORATION"))
DIRECTIONS = frozenset(("INCREASE", "DECREASE", "POSITIVE", "NEGATIVE", "NEUTRAL", "UNSPECIFIED"))
SIGNAL_FAMILIES = {"ROLE_INCREASE": "ROLE_CHANGE", "ROLE_DECREASE": "ROLE_CHANGE",
                   "MATCHUP_POSITIVE": "MATCHUP", "MATCHUP_NEGATIVE": "MATCHUP"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("TIMEZONE_REQUIRED")
    return result


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def events(captures):
    """Native IDs required. No URL fallback; revised original payloads are retained."""
    result = {}
    for capture in captures:
        row = dict(capture)
        for field in ("source", "source_event_id", "evidence_text", "published_at_utc",
                      "retrieved_at_utc", "first_seen_at_utc", "approval_id"):
            require(isinstance(row.get(field), str) and row[field].strip(), "MISSING_" + field)
        published = timestamp(row["published_at_utc"])
        seen = timestamp(row["first_seen_at_utc"])
        retrieved = timestamp(row["retrieved_at_utc"])
        updated = timestamp(row.get("updated_at_utc") or row["published_at_utc"])
        require(published <= updated <= seen <= retrieved, "EVENT_TIME_ORDER")
        row["event_key"] = digest(["EVENT_V1", row["source"], row["source_event_id"]])
        row["content_sha256"] = hashlib.sha256(row["evidence_text"].encode()).hexdigest()
        row["revision_id"] = digest(["REVISION_V1", row["event_key"], row["content_sha256"],
                                     row["published_at_utc"], row.get("updated_at_utc")])
        key = row["revision_id"]
        if key in result:
            previous = result[key]
            # Attribution disagreement is not silently resolved by capture order.
            stable_keys = (set(previous) | set(row)) - {"first_seen_at_utc", "retrieved_at_utc"}
            require(all(previous.get(k) == row.get(k) for k in stable_keys), "CAPTURE_CONFLICT")
            row["first_seen_at_utc"] = min(previous["first_seen_at_utc"], row["first_seen_at_utc"], key=timestamp)
            row["retrieved_at_utc"] = min(previous["retrieved_at_utc"], row["retrieved_at_utc"], key=timestamp)
        result[key] = row
    ordered = sorted(result.values(), key=lambda r: (r["event_key"], timestamp(r.get("updated_at_utc") or r["published_at_utc"]), r["revision_id"]))
    last = {}
    for row in ordered:
        previous = last.get(row["event_key"])
        if previous:
            require(timestamp(previous.get("updated_at_utc") or previous["published_at_utc"]) !=
                    timestamp(row.get("updated_at_utc") or row["published_at_utc"]), "UNVERSIONED_CONTENT_CHANGE")
        row["supersedes_revision_id"] = previous["revision_id"] if previous else None
        last[row["event_key"]] = row
    return ordered


def resolve(claim, snapshot, cutoff):
    """Reuse existing pure exact resolver; never call DB index builder or main."""
    from fanduel_injury_ingest import normalize_name, normalize_team, normalize_position, resolve_identity
    import pandas as pd

    require(snapshot.get("authority") == "WFS_IDENTITY_SNAPSHOT", "IDENTITY_AUTHORITY")
    require(timestamp(snapshot["available_at_utc"]) <= timestamp(cutoff), "IDENTITY_FROM_FUTURE")
    require((snapshot["season"], snapshot["week"]) == (claim["season"], claim["week"]), "IDENTITY_TARGET")
    for key in ("player_name", "team", "position"):
        require(bool(claim.get(key)), "IDENTITY_CONTEXT_MISSING")
    rows = []
    for original in snapshot["players"]:
        row = dict(original)
        require(bool(row.get("gsis_id")), "BLANK_SNAPSHOT_GSIS")
        row.update(normalized_name=normalize_name(row["player_name"]),
                   team=normalize_team(row["team"]), position=normalize_position(row["position"]))
        rows.append(row)
    require(bool(rows), "EMPTY_IDENTITY_SNAPSHOT")
    frame = pd.DataFrame(rows)
    for _, group in frame.groupby("gsis_id"):
        require(len(group[["team", "position"]].drop_duplicates()) == 1, "CONFLICTING_IDENTITY_SNAPSHOT")
    gsis, status, reason = resolve_identity(pd.Series(dict(claim, normalized_name=normalize_name(claim["player_name"]))), frame)
    return {"gsis_id": gsis or None, "identity_status": status, "identity_reason": reason,
            "identity_snapshot_sha256": digest(snapshot), "identity_method": "FANDUEL_EXACT_RESOLVER_V1"}


def evaluate(claim, event, snapshot, game, cutoff):
    """Explicit manual propositions only: no keyword-to-opportunity inference."""
    row = dict(claim)
    row.update({key: event.get(key) for key in ("source", "source_event_id", "event_key", "revision_id",
               "origin_id", "published_at_utc", "updated_at_utc", "retrieved_at_utc", "first_seen_at_utc")})
    row.update(safety=dict(SAFETY), cutoff_utc=cutoff, kickoff_at_utc=game.get("kickoff_at_utc"),
               identity_status="QUARANTINE", gsis_id=None, corroboration_state="NOT_EVALUATED")
    reasons = []
    try:
        require(row.get("signal_type") in SIGNALS, "INVALID_SIGNAL")
        require(row.get("evidence_phase") in PHASES, "INVALID_PHASE")
        require(row.get("evidence_kind") in KINDS, "INVALID_KIND")
        require(row.get("direction") in DIRECTIONS, "INVALID_DIRECTION")
        prescribed = {"ROLE_INCREASE": "INCREASE", "ROLE_DECREASE": "DECREASE",
                      "MATCHUP_POSITIVE": "POSITIVE", "MATCHUP_NEGATIVE": "NEGATIVE"}
        require(row["signal_type"] not in prescribed or row["direction"] == prescribed[row["signal_type"]], "SIGNAL_DIRECTION_CONFLICT")
        confidence = row.get("confidence")
        require(confidence is None or (type(confidence) in (int, float) and 0 <= confidence <= 1), "INVALID_CONFIDENCE")
        require(row.get("evidence_quote") and row["evidence_quote"] in event["evidence_text"], "UNSUPPORTED_QUOTE")
        require(row.get("extraction_method") == "MANUAL_OFFLINE", "UNSUPPORTED_EXTRACTOR")
        row.update(resolve(row, snapshot, cutoff))
        require(row["identity_status"] == "MATCHED", row["identity_reason"])
        for key in ("season", "week", "season_type", "game_id"):
            require(row.get(key) is not None and row[key] == game[key], "GAME_IDENTITY_MISMATCH")
        require(row["team"] in game["teams"] and row.get("opponent_team") in game["teams"]
                and row["team"] != row["opponent_team"], "GAME_TEAM_MISMATCH")
        require(timestamp(game["available_at_utc"]) <= timestamp(cutoff), "SCHEDULE_FROM_FUTURE")
        kickoff = timestamp(game["kickoff_at_utc"])
        boundary = timestamp(cutoff)
        require(boundary < kickoff, "CUTOFF_NOT_PREGAME")
        require(timestamp(event["published_at_utc"]) < kickoff, "POST_KICKOFF_PUBLICATION")
        knowledge = max(timestamp(event[k]) for k in ("published_at_utc", "first_seen_at_utc", "retrieved_at_utc"))
        if event.get("updated_at_utc"):
            knowledge = max(knowledge, timestamp(event["updated_at_utc"]))
        row["knowledge_at_utc"] = knowledge.isoformat()
        require(knowledge <= boundary, "EVIDENCE_AFTER_CUTOFF")
        require(timestamp(row["effective_from_utc"]) <= boundary < timestamp(row["valid_until_utc"]) <= kickoff, "EXPIRED_OR_INVALID_HORIZON")
        require(row.get("temporal_orientation") == "FORWARD_LOOKING", "NOT_FORWARD_LOOKING")
        require(row["evidence_phase"] != "POST_GAME_RECAP", "POST_GAME_RECAP")
        require(row.get("forward_looking_basis") and row["forward_looking_basis"] in row["evidence_quote"], "NO_FORWARD_PROPOSITION")
        # Derived claims require a separately validated dependency contract; not enabled in foundation.
        require(row["evidence_kind"] != "WFS_CORROBORATION", "CORROBORATION_DEPENDENCIES_NOT_VALIDATED")
    except (ValueError, KeyError, TypeError) as exc:
        reasons.append(str(exc))
    row["eligibility_status"] = "QUARANTINED" if reasons else "ELIGIBLE"
    row["reason_codes"] = reasons
    row["claim_id"] = digest(["CLAIM_V1", event["revision_id"], claim, row["gsis_id"]])
    return row


def consensus(claims):
    groups = {}
    for row in claims:
        if row["eligibility_status"] != "ELIGIBLE":
            continue
        fields = ("gsis_id", "game_id", "signal_type", "metric", "unit", "denominator",
                  "comparison_basis", "conditions", "effective_from_utc", "valid_until_utc", "cutoff_utc", "evidence_kind")
        dimensions = {k: row.get(k) for k in fields}
        dimensions["signal_type"] = SIGNAL_FAMILIES.get(row["signal_type"], row["signal_type"])
        key = digest(["GROUP_V1", dimensions])
        groups.setdefault(key, {})[row["claim_id"]] = row
    output = []
    for key, members in sorted(groups.items()):
        rows = list(members.values())
        origins = {}
        for row in rows:
            if row.get("origin_id"):
                origins.setdefault(row["origin_id"], set()).add(row["direction"])
        directions = set().union(*origins.values()) if origins else set()
        opposing = any(pair <= directions for pair in ({"INCREASE", "DECREASE"}, {"POSITIVE", "NEGATIVE"}))
        independent_conflict = any(a != b and any({x, y} in ({"INCREASE", "DECREASE"}, {"POSITIVE", "NEGATIVE"})
                                  for x in origins[a] for y in origins[b]) for a in origins for b in origins)
        status = ("CONFLICT" if independent_conflict else "MIXED" if opposing else
                  "AGREEMENT" if len(origins) >= 2 and len(directions) == 1 and not directions & {"NEUTRAL", "UNSPECIFIED"}
                  else "SINGLE_ORIGIN" if len(origins) == 1 else "INSUFFICIENT")
        output.append(dict(group_id=key, claim_ids=sorted(members), independent_origin_count=len(origins),
                           unknown_origin_count=sum(not r.get("origin_id") for r in rows),
                           direction_claim_ids={d: sorted(r["claim_id"] for r in rows if r["direction"] == d)
                                                for d in sorted({r["direction"] for r in rows})},
                           origin_directions={k: sorted(v) for k, v in sorted(origins.items())}, status=status))
    return output


def build(fixture):
    require(fixture.get("input_mode") == "OFFLINE_FIXTURE", "OFFLINE_ONLY")
    source_events = events(fixture["events"])
    require(all(e["approval_id"] == "SYNTHETIC_FIXTURE_ONLY" for e in source_events), "SOURCE_NOT_APPROVED")
    rows = []
    for claim in fixture["claims"]:
        candidates = [e for e in source_events if e["source"] == claim["source"] and e["source_event_id"] == claim["source_event_id"]
                      and e["revision_id"] == claim["revision_id"]]
        require(len(candidates) == 1, "CLAIM_REVISION_REFERENCE")
        rows.append(evaluate(claim, candidates[0], fixture["identity"], fixture["game"], fixture["cutoff_utc"]))
    rows = sorted({r["claim_id"]: r for r in rows}.values(), key=lambda r: r["claim_id"])
    # Retain revisions, but only the latest revision known by this cutoff may vote.
    latest = {}
    for event in source_events:
        if timestamp(event["retrieved_at_utc"]) <= timestamp(fixture["cutoff_utc"]):
            latest[event["event_key"]] = event["revision_id"]
    for row in rows:
        if row["eligibility_status"] == "ELIGIBLE" and latest.get(row["event_key"]) != row["revision_id"]:
            row["eligibility_status"] = "QUARANTINED"
            row["reason_codes"].append("SUPERSEDED_AS_OF_CUTOFF")
    groups = consensus(rows)
    states = {cid: group["status"] for group in groups for cid in group["claim_ids"]}
    for row in rows:
        row["corroboration_state"] = states.get(row["claim_id"], "NOT_ELIGIBLE")
    return {"events": source_events, "claims": rows, "consensus": groups,
            "quarantine": [r for r in rows if r["eligibility_status"] != "ELIGIBLE"]}
