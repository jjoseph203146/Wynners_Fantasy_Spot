"""Pure source-neutral shadow orchestration. No I/O and no claim generation."""
import copy
import re

from shadow_player_analysis_v1.core import SAFETY, digest, require, timestamp
from .classification import classify
from .identity import Authority, TEAM_ALIASES, normalize_name

CONTRACT = "PLAYER_ANALYSIS_INTELLIGENCE_V1_2_SHADOW"


def extract(event, authority):
    """Full WFS names only, plus explicit adapter mentions and narrow unknown-name candidates.

    Clause-local classification avoids assigning Bagent's availability to Keenum.
    A coordinated clause whose scope is unclear is retained but not assigned a role.
    """
    text = event["evidence_text"]
    mentions = copy.deepcopy(event.get("entities", []))
    clauses = re.split(r"[,;\n]|\s+(?:and|despite|while|but)\s+", text, flags=re.I)

    # Full WFS names explicitly present anywhere in this immutable source event.
    # These may anchor a surname-only role clause, but only when the surname maps
    # to exactly one explicit full-name player in this same event.
    event_normalized = " " + normalize_name(text) + " "
    explicit_event_names = [
        name for name in authority.names
        if " " + normalize_name(name) + " " in event_normalized
    ]
    surname_anchors = {}
    for name in explicit_event_names:
        parts = normalize_name(name).split()
        if len(parts) >= 2:
            surname_anchors.setdefault(parts[-1], []).append(name)

    for clause in clauses:
        normalized = " " + normalize_name(clause) + " "
        names = [
            name for name in authority.names
            if " " + normalize_name(name) + " " in normalized
        ]

        # Narrow same-event surname bridge. It never searches authority by
        # surname alone: the full WFS name must already appear explicitly
        # elsewhere in this exact source event, and the surname must be unique
        # among those explicit event-local full names.
        surname_bridge_name = None
        if not names:
            anchored = []
            for surname, full_names in surname_anchors.items():
                if (
                    len(full_names) == 1
                    and (" " + surname + " ") in normalized
                ):
                    anchored.append(full_names[0])
            if len(set(anchored)) == 1:
                names = list(set(anchored))
                surname_bridge_name = names[0]

        # Unknown names are candidates only; they never create an authority row.
        match = re.match(r"\s*([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,2})\s+(?:will|expected|set|inactive|cleared|suffered|injured|ruled|workload|recovery)\b", clause)
        if match and not names and authority.team(match.group(1))["identity_status"] != "RESOLVED":
            names = [match.group(1)]

        for name in names:
            if not any(
                m.get("type") == "PLAYER"
                and normalize_name(m.get("name")) == normalize_name(name)
                and m.get("evidence_quote") == clause.strip()
                for m in mentions
            ):
                mention = dict(
                    type="PLAYER",
                    name=name,
                    evidence_quote=clause.strip(),
                    relationship_scope="CLAUSE" if len(names) == 1 else "UNRESOLVED",
                )
                if surname_bridge_name == name:
                    mention["relationship_scope"] = "SAME_EVENT_UNIQUE_SURNAME"
                    mention["relationship_anchor"] = name
                mentions.append(mention)
    for name in TEAM_ALIASES:
        if re.search(r"\b" + re.escape(name) + r"\b", text, re.I):
            mentions.append(dict(type="TEAM", name=name, evidence_quote=name))
    # IDs only from supplied evidence, never generated from teams, dates or text.
    if event.get("game_id"):
        mentions.append(dict(type="GAME", game_id=event["game_id"]))
    return sorted({digest(m): m for m in mentions}.values(), key=digest)


def temporal(event, game, cutoff):
    reasons = []
    if event.get("temporal_confidence") != "RESOLVED" or event.get("temporal_status") != "UNAMBIGUOUS_SOURCE_TIMESTAMP":
        reasons.append("TEMPORAL_UNRESOLVED")
    try:
        pub = timestamp(event["published_at_utc"])
        seen = timestamp(event["first_seen_at_utc"])
        retrieved = timestamp(event["retrieved_at_utc"])
        updated = timestamp(event.get("updated_at_utc") or event["published_at_utc"])
        boundary = timestamp(cutoff)
        if not pub <= updated <= seen <= retrieved:
            reasons.append("INVALID_SOURCE_TIME_ORDER")
        if max(pub, updated, seen, retrieved) > boundary:
            reasons.append("EVIDENCE_AFTER_CUTOFF")
        if game:
            kickoff = timestamp(game["kickoff_at_utc"])
            if pub >= kickoff or updated >= kickoff or boundary >= kickoff:
                reasons.append("POST_KICKOFF_FOR_PREGAME")
        else:
            reasons.append("UNRESOLVED_GAME_IDENTITY")
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        reasons.append("TEMPORAL_UNRESOLVED")
    return sorted(set(reasons))


def build(records, identity, schedule, cutoff, approvals):
    """Approvals are a caller-controlled source -> allowed approval IDs registry.

    No record may self-approve. Identity and content classification run even when
    source/time gates fail. Claims/consensus are deliberately disabled in V1.2.
    """
    authority = Authority(identity, schedule, cutoff)
    events = {}
    for original in records:
        row = copy.deepcopy(original)
        for key in ("source", "source_event_id", "origin_id", "content_hash", "evidence_text",
                    "headline", "summary", "retrieved_at_utc", "first_seen_at_utc",
                    "source_phase", "temporal_status", "temporal_confidence"):
            require(isinstance(row.get(key), str), "MISSING_" + key)
        require(bool(row["source"] and row["source_event_id"] and row["content_hash"]), "MISSING_EVENT_IDENTITY")
        require("published_at_utc" in row and "raw_publication_timestamp" in row, "MISSING_PUBLICATION_PROVENANCE")
        # Exact duplicate records collapse; differing captures/revisions are retained.
        row["event_key"] = digest([row["source"], row["source_event_id"]])
        row["record_id"] = digest(row)
        events[row["record_id"]] = row
    source_events = sorted(events.values(), key=lambda r: r["record_id"])
    evidence, links, quarantine = [], [], []
    event_versions = {}
    for event in source_events:
        event_versions.setdefault(event["event_key"], set()).add(event["content_hash"])
    for event in source_events:
        rid = event["record_id"]
        classification = classify(event["evidence_text"], event["source_phase"], event.get("evidence_kind"))
        reasons = []
        approved = event.get("approval_id") in approvals.get(event["source"], ())
        if not approved:
            reasons.append("SOURCE_NOT_APPROVED")
        if len(event_versions[event["event_key"]]) > 1:
            reasons.append("SOURCE_EVENT_REVISION_CONFLICT")
        if classification["relevance"] == "REJECT":
            reasons.append("REJECTED_CONTENT")
        if "INSUFFICIENT" in classification["evidence_types"]:
            reasons.append("INSUFFICIENT_EVIDENCE")
        if "POST_GAME_RECAP" in classification["evidence_types"]:
            reasons.append("POST_GAME_RECAP")
        game_result = authority.game(event.get("game_id"))
        time_reasons = temporal(event, game_result["game"], cutoff)
        reasons.extend(time_reasons)
        # Non-temporal adapter failures survive the handoff; they cannot be cleared.
        reasons.extend(event.get("adapter_reason_codes", []))
        event_links = []
        for mention in extract(event, authority):
            kind = mention.get("type")
            require(kind in {"PLAYER", "TEAM", "GAME"}, "UNSUPPORTED_ENTITY_TYPE")
            link_reasons = []
            local = classify(mention.get("evidence_quote", ""), supplied_kind=event.get("evidence_kind"))
            if kind == "PLAYER":
                resolved = authority.player(mention)
                if resolved["identity_status"] != "RESOLVED":
                    link_reasons.append(resolved["identity_status"] + "_PLAYER_IDENTITY")
                if not mention.get("evidence_quote") or mention["evidence_quote"] not in event["evidence_text"]:
                    local = classify("")
                    link_reasons.append("UNSUPPORTED_ENTITY_QUOTE")
                elif mention.get("name") and (" " + normalize_name(mention["name"]) + " ") not in (" " + normalize_name(mention["evidence_quote"]) + " "):
                    surname_bridge_valid = False
                    if (
                        mention.get("relationship_scope") == "SAME_EVENT_UNIQUE_SURNAME"
                        and normalize_name(mention.get("relationship_anchor", "")) == normalize_name(mention["name"])
                    ):
                        name_parts = normalize_name(mention["name"]).split()
                        if len(name_parts) >= 2:
                            surname = name_parts[-1]
                            quote_normalized = " " + normalize_name(mention["evidence_quote"]) + " "
                            surname_bridge_valid = (" " + surname + " ") in quote_normalized

                    if not surname_bridge_valid:
                        local = classify("")
                        link_reasons.append("UNSUPPORTED_ENTITY_QUOTE")
                quote_names = {normalize_name(n) for n in authority.names
                               if (" " + normalize_name(n) + " ") in (" " + normalize_name(mention.get("evidence_quote")) + " ")}
                if mention.get("relationship_scope") == "UNRESOLVED" or len(quote_names) > 1:
                    local = classify("")
                    link_reasons.append("AMBIGUOUS_ENTITY_RELATIONSHIP")
                defensive = bool(set(resolved["positions"]) & {"DE", "DT", "DL", "NT", "EDGE", "LB", "ILB", "OLB", "MLB", "CB", "DB", "S", "FS", "SS"})
                if defensive and set(local["evidence_types"]) & {"INJURY_CONTEXT", "AVAILABILITY", "STARTER_CHANGE"}:
                    local["evidence_types"] = sorted(set(local["evidence_types"]) | {"TEAM_ENVIRONMENT"})
                    classification["evidence_types"] = sorted(set(classification["evidence_types"]) | {"TEAM_ENVIRONMENT"})
            elif kind == "TEAM":
                resolved = authority.team(mention.get("name"))
                if resolved["identity_status"] != "RESOLVED":
                    link_reasons.append("UNRESOLVED_TEAM_IDENTITY")
                local = dict(relevance=classification["relevance"], evidence_types=["TEAM_ENVIRONMENT"], evidence_kind=classification["evidence_kind"])
            else:
                resolved = authority.game(mention.get("game_id"))
                resolved.pop("game")
                if resolved["identity_status"] != "RESOLVED":
                    link_reasons.append("UNRESOLVED_GAME_IDENTITY")
            # Global rejection/opinion/recap restrictions also constrain local routing.
            if classification["relevance"] == "REJECT" or classification["evidence_kind"] in {"ANALYST_OPINION", "MARKET_INFORMATION"} or "POST_GAME_RECAP" in classification["evidence_types"]:
                local = dict(classification)
            link = dict(record_id=rid, entity_type=kind, mention=mention, **resolved, **local,
                        reason_codes=sorted(set(link_reasons)), claim_eligible=False, consensus_eligible=False,
                        safety=dict(SAFETY))
            link["link_id"] = digest(link)
            event_links.append(link)
            reasons.extend(link_reasons)
        if not any(l["entity_type"] in {"PLAYER", "TEAM"} for l in event_links):
            reasons.append("NO_SUPPORTED_ENTITY")
        links.extend(event_links)
        row = dict(record_id=rid, event_key=event["event_key"], **classification,
                   source_approved=approved, temporal_status=event["temporal_status"],
                   temporal_confidence=event["temporal_confidence"], temporal_reason_codes=time_reasons,
                   pregame_eligible=not time_reasons and "POST_GAME_RECAP" not in classification["evidence_types"],
                   claim_eligible=False, consensus_eligible=False,
                   claim_gate="NOT_ENABLED_V1_2", consensus_gate="NOT_ENABLED_V1_2",
                   reason_codes=sorted(set(reasons)), safety=dict(SAFETY))
        evidence.append(row)
        if reasons:
            quarantine.append(dict(record_id=rid, reason_codes=row["reason_codes"], source_event=event,
                                   eligibility_status="QUARANTINED", safety=dict(SAFETY)))
    quality = []
    for source in sorted({e["source"] for e in source_events}):
        ids = {e["record_id"] for e in source_events if e["source"] == source}
        quality.append(dict(source=source, records=len(ids), quarantined=sum(q["record_id"] in ids for q in quarantine),
                            origins=sorted({e["origin_id"] for e in source_events if e["source"] == source}),
                            eligible_claims=0, consensus_votes=0))
    return dict(source_events=source_events, classified_evidence=evidence,
                entity_links=sorted(links, key=lambda r: r["link_id"]), quarantine=quarantine,
                source_quality=quality)
