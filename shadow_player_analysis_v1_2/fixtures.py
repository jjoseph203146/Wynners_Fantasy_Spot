"""Synthetic stories using illustrative names; IDs/games are explicitly fictional.

These records make no assertions about actual NFL status or current rosters.
"""
from shadow_player_analysis_v1.core import digest

STORIES = [
    "Zay Flowers will play after positive workout",
    "Case Keenum expected to start, Tyson Bagent cleared protocol for Chicago Bears",
    "Adonai Mitchell inactive",
    "Jameis Winston set to start",
    "De'Von Achane suffered knee injury during game and ruled out",
    "Myles Garrett recovery takes multiple weeks after injury",
    "Example Defender injured; defensive starter ruled out",
    "Chicago Bears reveal new uniforms",
    "Zay Flowers fashion show outfit",
    "College draft prospect Sam Future impresses scouts",
    "Betting odds: analyst likes Zay Flowers",
    "Postgame recap: Zay Flowers caught 8 passes for 100 yards in a win",
    "Zay Flowers workload will increase next game",
    "Zay Flowers workload will decrease next game",
    "Coach says Zay Flowers will have an expanded role next game",
    "Alex Common will play",
    "Unknown Player will play",
    "Zay Flowers will play after positive workout",  # duplicate source event below
    "Zay Flowers will play after positive workout",  # same origin syndication below
    "Jameis Winston set to start, Adonai Mitchell inactive",
]


def event(text=STORIES[0], index=0):
    return dict(source="SYNTHETIC_A", origin_id="SYNTHETIC_ORIGIN_A", source_event_id=f"fixture-{index}",
                approval_id="SYNTHETIC_FIXTURE_ONLY", article_url="https://example.invalid/no-fetch",
                content_hash=digest(text), evidence_text=text, headline=text, summary="",
                raw_publication_timestamp="Sun, 07 Sep 2025 10:00:00 GMT",
                published_at_utc="2025-09-07T10:00:00Z", first_seen_at_utc="2025-09-07T10:01:00Z",
                retrieved_at_utc="2025-09-07T10:02:00Z", source_phase="PRE_GAME_ANALYSIS",
                temporal_status="UNAMBIGUOUS_SOURCE_TIMESTAMP", temporal_confidence="RESOLVED",
                game_id="SYNTHETIC_GAME_001")


def sample(all_stories=False):
    names = [("Zay Flowers", "BAL", "WR"), ("Case Keenum", "CHI", "QB"), ("Tyson Bagent", "CHI", "QB"),
             ("Adonai Mitchell", "IND", "WR"), ("Jameis Winston", "NYG", "QB"), ("De'Von Achane", "MIA", "RB"),
             ("Myles Garrett", "CLE", "DE"), ("Example Defender", "CHI", "CB"),
             ("Alex Common", "BAL", "WR"), ("Alex Common", "CHI", "WR")]
    identity = dict(authority="WFS_IDENTITY_SNAPSHOT", available_at_utc="2025-09-06T00:00:00Z",
                    season=2025, week=1, players=[dict(player_name=n, team=t, position=p, gsis_id=f"SYNTHETIC_GSIS_{i:03}")
                                               for i, (n, t, p) in enumerate(names)])
    schedule = dict(authority="WFS_SCHEDULE_SNAPSHOT", available_at_utc="2025-09-01T00:00:00Z",
                    games=[dict(game_id="SYNTHETIC_GAME_001", kickoff_at_utc="2025-09-07T17:00:00Z")])
    records = [event(text, i) for i, text in enumerate(STORIES)] if all_stories else [event()]
    if all_stories:
        records[17] = event()
        records[18]["source"] = "SYNTHETIC_B"
        records[4].update(published_at_utc="2025-09-07T18:00:00Z",
                          first_seen_at_utc="2025-09-07T18:01:00Z", retrieved_at_utc="2025-09-07T18:02:00Z")
    return dict(records=records, identity=identity, schedule=schedule, cutoff="2025-09-07T16:00:00Z",
                approvals={s: ["SYNTHETIC_FIXTURE_ONLY"] for s in ("SYNTHETIC_A", "SYNTHETIC_B")})
