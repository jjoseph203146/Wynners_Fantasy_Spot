"""Conservative routing labels, never automatic claims or player recommendations."""
import re

TAXONOMY = frozenset("""AVAILABILITY INJURY_CONTEXT STARTER_CHANGE ROLE_INCREASE ROLE_DECREASE
USAGE_ANALYSIS TARGET_SHARE ROUTE_PARTICIPATION BACKFIELD_SHARE RED_ZONE_ROLE DEEP_TARGET_ROLE
COACH_INTENT MATCHUP_POSITIVE MATCHUP_NEGATIVE COVERAGE_MATCHUP OL_DL_MATCHUP GAME_SCRIPT
TEAM_ENVIRONMENT MARKET_CONTEXT DFS_ANALYSIS TRANSACTION_CONTEXT POST_GAME_RECAP
NON_CURRENT_NFL NON_FOOTBALL INSUFFICIENT""".split())

RULES = (
    ("STARTER_CHANGE", r"\b(?:expected to start|set to start|named (?:the )?starter|will start at|starting quarterback)\b"),
    ("AVAILABILITY", r"\b(?:will play|inactive|active for|ruled out|cleared? (?:the )?protocol|questionable|doubtful)\b"),
    ("INJURY_CONTEXT", r"\b(?:injur(?:y|ed)|recovery|concussion|surgery|torn|sprain)\b"),
    ("ROLE_INCREASE", r"\b(?:workload (?:will increase|increase)|larger workload|expanded role|receive more carries in (?:the )?next game)\b"),
    ("ROLE_DECREASE", r"\b(?:workload (?:will decrease|decrease)|reduced workload|smaller role|reduced role)\b"),
    ("COACH_INTENT", r"\bcoach\b.{0,80}\b(?:says|said|plans|intends)\b.{0,100}\b(?:will|plan|intend|next game)\b"),
    ("TARGET_SHARE", r"\btarget share\b"),
    ("ROUTE_PARTICIPATION", r"\broute participation\b"),
    ("BACKFIELD_SHARE", r"\bbackfield share\b"),
    ("RED_ZONE_ROLE", r"\bred.zone role\b"),
    ("DEEP_TARGET_ROLE", r"\bdeep.target role\b"),
    ("USAGE_ANALYSIS", r"\busage analysis\b"),
    ("MATCHUP_POSITIVE", r"\b(?:favorable|favourable) matchup\b"),
    ("MATCHUP_NEGATIVE", r"\b(?:unfavorable|unfavourable|difficult) matchup\b"),
    ("COVERAGE_MATCHUP", r"\b(?:coverage matchup|cornerback matchup)\b"),
    ("OL_DL_MATCHUP", r"\b(?:pass rush|offensive line matchup|defensive line matchup)\b"),
    ("GAME_SCRIPT", r"\bgame script\b"),
    ("TEAM_ENVIRONMENT", r"\b(?:team environment|scoring environment)\b"),
    ("TRANSACTION_CONTEXT", r"\b(?:signed|traded|released|activated from|waived)\b"),
)


KINDS = frozenset({"FACTUAL_OBSERVATION", "REPORTED_EXPECTATION", "ANALYST_OPINION",
                   "MARKET_INFORMATION", "WFS_CORROBORATION"})


def classify(text, source_phase="", supplied_kind=None):
    if supplied_kind is not None and supplied_kind not in KINDS:
        raise ValueError("INVALID_EVIDENCE_KIND")
    def has(pattern):
        return bool(re.search(pattern, text, re.I | re.S))
    kind = "FACTUAL_OBSERVATION"
    if supplied_kind == "ANALYST_OPINION" or has(r"\b(?:analyst|opinion|i think|i like|we like|recommends?|prediction)\b"):
        kind = "ANALYST_OPINION"
    elif supplied_kind == "REPORTED_EXPECTATION" or has(r"\b(?:expected|reportedly|likely|set to|could|may)\b"):
        kind = "REPORTED_EXPECTATION"
    elif supplied_kind == "WFS_CORROBORATION":
        kind = "WFS_CORROBORATION"
    if has(r"\b(?:uniforms?|fashion|outfit|red carpet|entertainment)\b"):
        types, relevance = ["NON_FOOTBALL"], "REJECT"
    elif has(r"\b(?:college|draft prospect|historical|hall of fame|super bowl predictions|playoff projections)\b"):
        types, relevance = ["NON_CURRENT_NFL"], "REJECT"
    elif supplied_kind == "MARKET_INFORMATION" or has(r"\b(?:betting|odds|sportsbook|moneyline|prop bet|best bets)\b"):
        types, relevance, kind = ["MARKET_CONTEXT"], "LOW", "MARKET_INFORMATION"
    elif source_phase == "POST_GAME_RECAP" or has(r"\b(?:recap|in (?:a |the )?(?:win|loss)|caught \d+|had \d+|finished with|rushed for \d+)\b"):
        types, relevance = ["POST_GAME_RECAP"], "LOW"
    elif kind == "ANALYST_OPINION":
        types, relevance = ["DFS_ANALYSIS"], "LOW"
    elif has(r"\b(?:not|never|won't|denies|denied|no longer)\b") or "?" in text:
        # V1.2 cannot safely interpret negation or speculative question headlines.
        types, relevance = ["INSUFFICIENT"], "LOW"
    else:
        types = [name for name, pattern in RULES if has(pattern)]
        if not types and has(r"\bDFS analysis\b"):
            types, kind = ["DFS_ANALYSIS"], "ANALYST_OPINION"
        types = types or ["INSUFFICIENT"]
        high = {"AVAILABILITY", "STARTER_CHANGE", "INJURY_CONTEXT", "ROLE_INCREASE", "ROLE_DECREASE", "COACH_INTENT", "MATCHUP_POSITIVE", "MATCHUP_NEGATIVE"}
        relevance = "HIGH" if high.intersection(types) else "LOW" if types == ["INSUFFICIENT"] else "MEDIUM"
    return dict(relevance=relevance, evidence_types=sorted(types), evidence_kind=kind)
