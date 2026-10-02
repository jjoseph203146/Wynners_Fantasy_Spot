"""Player Analysis Intelligence V1.3 — deterministic shadow claim construction.

Consumes normalized V1.2.1 evidence products.

This module:
- does not access the network
- does not access or mutate a database
- does not mutate projections or forecasts
- does not influence the solver
- does not provide injury/availability authority
- does not perform consensus
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
}


SUPPORTED_SIGNALS = {
    "STARTER_CHANGE",
    "ROLE_INCREASE",
    "ROLE_DECREASE",
    "COACH_INTENT",
    "TARGET_SHARE",
    "ROUTE_PARTICIPATION",
    "BACKFIELD_SHARE",
    "RED_ZONE_ROLE",
    "DEEP_TARGET_ROLE",
}


DIRECTIONS = {
    "ROLE_INCREASE": "INCREASE",
    "ROLE_DECREASE": "DECREASE",
}


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def timestamp(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("TIMESTAMP_REQUIRED")

    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(text)

    if parsed.tzinfo is None:
        raise ValueError("TIMEZONE_REQUIRED")

    return parsed.astimezone(timezone.utc)


def _single_by_record(rows):
    result = {}

    for row in rows:
        record_id = row.get("record_id")

        if not isinstance(record_id, str) or not record_id:
            continue

        if record_id in result:
            # Duplicate logical rows are allowed only when identical.
            if canonical(result[record_id]) != canonical(row):
                result[record_id] = None
        else:
            result[record_id] = row

    return result


def _supported_quote(link, event):
    mention = link.get("mention")

    if not isinstance(mention, dict):
        return None

    quote = mention.get("evidence_quote")
    evidence_text = event.get("evidence_text")

    if not isinstance(quote, str) or not quote.strip():
        return None

    if not isinstance(evidence_text, str):
        return None

    if quote not in evidence_text:
        return None

    return quote


def _eligible_signal(link, evidence):
    link_types = link.get("evidence_types")
    evidence_types = evidence.get("evidence_types")

    if not isinstance(link_types, list) or not isinstance(evidence_types, list):
        return None

    shared = sorted(
        set(link_types)
        & set(evidence_types)
        & SUPPORTED_SIGNALS
    )

    # One link must express one unambiguous claim signal in V1.3.
    if len(shared) != 1:
        return None

    return shared[0]


def _forward_proposition(signal_type, quote, claim_player_name=None):
    """Conservative V1.3 proposition gate.

    Classification has already occurred upstream. This gate independently
    requires forward-looking language in the exact supported quote.
    """
    text = quote.casefold()

    if signal_type == "STARTER_CHANGE":
        phrases = (
            "expected to start",
            "set to start",
            "named the starter",
            "named starter",
            "will start at",
            "starting quarterback",
        )
        return any(phrase in text for phrase in phrases)

    if signal_type == "ROLE_INCREASE":
        phrases = (
            "role will increase",
            "workload will increase",
            "expanded role",
            "larger role",
            "bigger role",
        )
        return any(phrase in text for phrase in phrases)

    if signal_type == "ROLE_DECREASE":
        phrases = (
            "role will decrease",
            "workload will decrease",
            "reduced role",
            "smaller role",
        )
        return any(phrase in text for phrase in phrases)

    if signal_type == "COACH_INTENT":
        phrases = (
            "will have",
            "will use",
            "plans to",
            "plan to",
            "expected to",
        )
        return any(phrase in text for phrase in phrases)

    if signal_type == "ROUTE_PARTICIPATION":
        # V1.3 Milestone 2C:
        # explicit, player-anchored forward-looking route-participation
        # propositions only.
        player_name = str(claim_player_name or "").strip().casefold()

        if not player_name:
            return False

        if text.rstrip().endswith("?"):
            return False

        # Explicitly withdrawn or superseded intent is not a
        # forward-looking ROUTE_PARTICIPATION proposition.
        route_disqualifiers = (
            "no longer plans to",
            "plan was withdrawn",
            "plans were withdrawn",
        )

        if any(token in text for token in route_disqualifiers):
            return False

        player = player_name

        anchored_patterns = (
            f"{player} is expected to have larger route participation",
            f"{player} is expected to have majority route participation",
            f"plans to give {player} larger route participation",
        )

        for pattern in anchored_patterns:
            start = text.find(pattern)

            while start != -1:
                end = start + len(pattern)

                left_ok = (
                    start == 0
                    or not text[start - 1].isalnum()
                )
                right_ok = (
                    end == len(text)
                    or not text[end].isalnum()
                )

                if left_ok and right_ok:
                    return True

                start = text.find(pattern, start + 1)

        return False

    if signal_type == "TARGET_SHARE":
        # V1.3 Milestone 2B:
        # explicit, player-anchored forward-looking target-share
        # propositions only.
        player_name = str(claim_player_name or "").strip().casefold()

        if not player_name:
            return False

        if text.rstrip().endswith("?"):
            return False

        # Explicitly withdrawn or superseded intent is not a
        # forward-looking TARGET_SHARE proposition.
        target_disqualifiers = (
            "no longer plans to",
            "plan was withdrawn",
            "plans were withdrawn",
        )

        if any(token in text for token in target_disqualifiers):
            return False

        player = player_name

        anchored_patterns = (
            f"{player} is expected to have a larger target share",
            f"{player} is expected to command the majority target share",
            f"plans to give {player} a larger target share",
        )

        for pattern in anchored_patterns:
            start = text.find(pattern)

            while start != -1:
                end = start + len(pattern)

                left_ok = (
                    start == 0
                    or not text[start - 1].isalnum()
                )
                right_ok = (
                    end == len(text)
                    or not text[end].isalnum()
                )

                if left_ok and right_ok:
                    return True

                start = text.find(pattern, start + 1)

        return False

    if signal_type == "BACKFIELD_SHARE":
        # V1.3 Milestone 2A:
        # explicit, player-anchored forward-looking workload propositions only.
        #
        # The proposition must bind the resolved player's name to the usage
        # statement. Negated, historical, withdrawn, interrogative, vague,
        # or other-player propositions fail closed.
        player_name = str(claim_player_name or "").strip().casefold()

        if not player_name:
            return False

        if text.rstrip().endswith("?"):
            return False

        disqualifiers = (
            " not expected to ",
            " no longer expected to ",
            " does not plan to ",
            " do not plan to ",
            " no longer plans to ",
            " no longer plan to ",
            " was expected to ",
            " were expected to ",
            " earlier expectation",
            " plan was withdrawn",
            " plans were withdrawn",
        )

        padded = f" {text} "

        if any(phrase in padded for phrase in disqualifiers):
            return False

        player = player_name

        anchored_patterns = (
            f"{player} is expected to handle a larger share of the backfield",
            f"{player} is expected to handle the majority of the backfield work",
            f"plans to give {player} more backfield work",
            f"plan to give {player} more backfield work",
        )

        for pattern in anchored_patterns:
            start = text.find(pattern)

            while start != -1:
                end = start + len(pattern)

                left_ok = (
                    start == 0
                    or not text[start - 1].isalnum()
                )
                right_ok = (
                    end == len(text)
                    or not text[end].isalnum()
                )

                if left_ok and right_ok:
                    return True

                start = text.find(pattern, start + 1)

        return False

    if signal_type == "RED_ZONE_ROLE":
        # V1.3 Milestone 2D:
        # explicit, player-anchored forward-looking red-zone-role
        # propositions only.
        player_name = str(claim_player_name or "").strip().casefold()

        if not player_name:
            return False

        if text.rstrip().endswith("?"):
            return False

        # Explicitly withdrawn or superseded intent is not a
        # forward-looking RED_ZONE_ROLE proposition.
        red_zone_disqualifiers = (
            "no longer plans to",
            "plan was withdrawn",
            "plans were withdrawn",
        )

        if any(token in text for token in red_zone_disqualifiers):
            return False

        player = player_name

        anchored_patterns = (
            f"{player} is expected to have a larger red zone role",
            f"{player} is expected to have a major red zone role",
            f"plans to give {player} a larger red zone role",
        )

        for pattern in anchored_patterns:
            start = text.find(pattern)

            while start != -1:
                end = start + len(pattern)

                left_ok = (
                    start == 0
                    or not text[start - 1].isalnum()
                )
                right_ok = (
                    end == len(text)
                    or not text[end].isalnum()
                )

                if left_ok and right_ok:
                    return True

                start = text.find(pattern, start + 1)

        return False

    if signal_type == "DEEP_TARGET_ROLE":
        # V1.3 Milestone 2E:
        # explicit, player-anchored forward-looking deep-target-role
        # propositions only.
        player_name = str(claim_player_name or "").strip().casefold()

        if not player_name:
            return False

        if text.rstrip().endswith("?"):
            return False

        # Explicitly withdrawn or superseded intent is not a
        # forward-looking DEEP_TARGET_ROLE proposition.
        deep_target_disqualifiers = (
            "no longer plans to",
            "plan was withdrawn",
            "plans were withdrawn",
        )

        if any(token in text for token in deep_target_disqualifiers):
            return False

        player = player_name

        anchored_patterns = (
            f"{player} is expected to have a larger deep target role",
            f"{player} is expected to have a major deep target role",
            f"plans to give {player} a larger deep target role",
        )

        for pattern in anchored_patterns:
            start = text.find(pattern)

            while start != -1:
                end = start + len(pattern)

                left_ok = (
                    start == 0
                    or not text[start - 1].isalnum()
                )
                right_ok = (
                    end == len(text)
                    or not text[end].isalnum()
                )

                if left_ok and right_ok:
                    return True

                start = text.find(pattern, start + 1)

        return False

    # These signals remain fail-closed until their proposition grammar
    # receives dedicated V1.3 regression coverage.
    return False


def build_claims(
    *,
    source_events,
    classified_evidence,
    entity_links,
    cutoff_utc,
    schedule=None,
):
    """Construct deterministic shadow claims from validated V1.2.1 products."""

    cutoff = timestamp(cutoff_utc)

    events = _single_by_record(source_events)
    evidence_rows = _single_by_record(classified_evidence)

    lifecycle_enabled = schedule is not None
    schedule_hash = None
    schedule_games = {}

    if lifecycle_enabled:
        if not isinstance(schedule, dict):
            return {"claims": [], "consensus": [], "safety": dict(SAFETY)}

        if schedule.get("authority") != "WFS_SCHEDULE_SNAPSHOT":
            return {"claims": [], "consensus": [], "safety": dict(SAFETY)}

        try:
            schedule_available = timestamp(schedule.get("available_at_utc"))
        except (ValueError, TypeError):
            return {"claims": [], "consensus": [], "safety": dict(SAFETY)}

        if schedule_available > cutoff:
            return {"claims": [], "consensus": [], "safety": dict(SAFETY)}


        games = schedule.get("games")
        if not isinstance(games, list):
            return {"claims": [], "consensus": [], "safety": dict(SAFETY)}

        schedule_hash = digest(schedule)
        duplicate_game_ids = set()

        for game in games:
            if not isinstance(game, dict):
                continue

            game_id = game.get("game_id")
            if not isinstance(game_id, str) or not game_id:
                continue

            if game_id in schedule_games:
                duplicate_game_ids.add(game_id)

            schedule_games[game_id] = game

        if duplicate_game_ids:
            return {"claims": [], "consensus": [], "safety": dict(SAFETY)}

    # M3B revision currentness.
    # Only the uniquely latest content version known by this cutoff may
    # produce a claim. Future revisions cannot rewrite historical cutoffs,
    # and equal knowledge times fail closed.
    event_families = {}

    for candidate_event in source_events:
        if not isinstance(candidate_event, dict):
            continue

        event_key = candidate_event.get("event_key")
        content_hash = candidate_event.get("content_hash")

        if (
            isinstance(event_key, str)
            and event_key
            and isinstance(content_hash, str)
            and content_hash
        ):
            event_families.setdefault(
                event_key, []
            ).append(candidate_event)

    multi_version_families = {
        event_key: family
        for event_key, family in event_families.items()
        if len({
            row.get("content_hash")
            for row in family
            if isinstance(row.get("content_hash"), str)
            and row.get("content_hash")
        }) > 1
    }

    revision_current_records = set()

    for event_key, family in multi_version_families.items():
        known = []
        chronology_invalid = False

        for candidate_event in family:
            published_at_utc = candidate_event.get(
                "published_at_utc"
            )

            try:
                published = timestamp(published_at_utc)
                retrieved = timestamp(
                    candidate_event.get("retrieved_at_utc")
                )
                first_seen = timestamp(
                    candidate_event.get("first_seen_at_utc")
                )
                updated = timestamp(
                    candidate_event.get("updated_at_utc")
                    or published_at_utc
                )
            except (ValueError, TypeError):
                chronology_invalid = True
                break

            knowledge = max(
                published,
                updated,
                retrieved,
                first_seen,
            )

            if knowledge <= cutoff:
                record_id = candidate_event.get("record_id")

                if isinstance(record_id, str) and record_id:
                    known.append(
                        (knowledge, record_id)
                    )

        if chronology_invalid or not known:
            continue

        latest_knowledge = max(
            row[0] for row in known
        )

        latest_records = {
            record_id
            for knowledge, record_id in known
            if knowledge == latest_knowledge
        }

        if len(latest_records) == 1:
            revision_current_records.update(
                latest_records
            )

    claims = []

    for link in entity_links:
        record_id = link.get("record_id")

        event = events.get(record_id)
        evidence = evidence_rows.get(record_id)

        if (
            event is not None
            and event.get("event_key") in multi_version_families
            and record_id not in revision_current_records
        ):
            continue

        # Missing or conflicting upstream record joins fail closed.
        if event is None or evidence is None:
            continue

        if evidence.get("source_approved") is not True:
            continue

        if evidence.get("pregame_eligible") is not True:
            continue

        if evidence.get("temporal_status") != "UNAMBIGUOUS_SOURCE_TIMESTAMP":
            continue

        if evidence.get("temporal_confidence") != "RESOLVED":
            continue

        if evidence.get("relevance") != "HIGH":
            continue

        if evidence.get("reason_codes"):
            continue

        if link.get("entity_type") != "PLAYER":
            continue

        if link.get("identity_status") != "RESOLVED":
            continue

        if link.get("identity_reason") != "EXACT_UNIQUE_NAME_VALIDATED":
            continue

        gsis_id = link.get("gsis_id")

        if not isinstance(gsis_id, str) or not gsis_id:
            continue

        if link.get("relevance") != "HIGH":
            continue

        if link.get("reason_codes"):
            continue

        signal_type = _eligible_signal(link, evidence)

        if signal_type is None:
            continue

        quote = _supported_quote(link, event)

        if quote is None:
            continue

        mention = link.get("mention") or {}

        if not _forward_proposition(
            signal_type,
            quote,
            claim_player_name=mention.get("name"),
        ):
            continue

        published_at_utc = event.get("published_at_utc")

        try:
            published = timestamp(published_at_utc)
            retrieved = timestamp(event.get("retrieved_at_utc"))
            first_seen = timestamp(event.get("first_seen_at_utc"))
            updated = timestamp(event.get("updated_at_utc") or published_at_utc)
        except (ValueError, TypeError):
            continue

        knowledge = max(published, updated, retrieved, first_seen)

        # Never permit evidence known after the requested cutoff.
        if knowledge > cutoff:
            continue

        lifecycle = {}

        if lifecycle_enabled:
            game_links = [
                row
                for row in entity_links
                if row.get("record_id") == record_id
                and row.get("entity_type") == "GAME"
            ]

            # Exact resolved association only. Never infer a game.
            if len(game_links) != 1:
                continue

            game_link = game_links[0]

            if game_link.get("identity_status") != "RESOLVED":
                continue

            if game_link.get("reason_codes"):
                continue

            if game_link.get("schedule_sha256") != schedule_hash:
                continue

            game_id = game_link.get("game_id")

            if not isinstance(game_id, str) or not game_id:
                continue

            game = schedule_games.get(game_id)

            if game is None:
                continue

            try:
                kickoff = timestamp(game.get("kickoff_at_utc"))
            except (ValueError, TypeError):
                continue

            # Strict pregame lifecycle boundary.
            if cutoff >= kickoff:
                continue

            if published >= kickoff:
                continue

            if updated >= kickoff:
                continue

            lifecycle = {
                "game_id": game_id,
                "kickoff_at_utc": game.get("kickoff_at_utc"),
                "schedule_sha256": schedule_hash,
                "knowledge_at_utc": knowledge.isoformat(),
                "effective_from_utc": knowledge.isoformat(),
                "valid_until_utc": game.get("kickoff_at_utc"),
            }

        origin_id = event.get("origin_id")

        if not isinstance(origin_id, str) or not origin_id:
            continue

        direction = DIRECTIONS.get(signal_type, "UNSPECIFIED")

        claim_basis = {
            "record_id": record_id,
            "link_id": link.get("link_id"),
            "event_key": event.get("event_key"),
            "source": event.get("source"),
            "source_event_id": event.get("source_event_id"),
            "origin_id": origin_id,
            "content_hash": event.get("content_hash"),
            "gsis_id": gsis_id,
            "player_name": mention.get("name"),
            "signal_type": signal_type,
            "direction": direction,
            "evidence_kind": link.get("evidence_kind"),
            "evidence_quote": quote,
            "relationship_scope": mention.get("relationship_scope"),
            "published_at_utc": published_at_utc,
            "cutoff_utc": cutoff_utc,
            **lifecycle,
        }

        claim = dict(claim_basis)
        claim.update(
            claim_id=digest(["CLAIM_V1_3", claim_basis]),
            eligibility_status="ELIGIBLE",
            claim_eligible=True,
            consensus_eligible=False,
            consensus_gate="NOT_ENABLED_V1_3",
            corroboration_state="NOT_EVALUATED",
            safety=dict(SAFETY),
        )

        claims.append(claim)

    claims = sorted(
        {row["claim_id"]: row for row in claims}.values(),
        key=lambda row: row["claim_id"],
    )

    return {
        "claims": claims,
        "consensus": [],
        "safety": dict(SAFETY),
    }
