"""Entirely fictional historical fixtures; no captured external content."""
from .core import events


def sample():
    event = dict(source="SYNTHETIC_A", source_event_id="article-1", origin_id="reporter-a",
                 evidence_text="Example Runner is expected to receive more carries in the next game.",
                 published_at_utc="2025-09-07T10:00:00Z", updated_at_utc=None,
                 first_seen_at_utc="2025-09-07T10:01:00Z", retrieved_at_utc="2025-09-07T10:02:00Z",
                 approval_id="SYNTHETIC_FIXTURE_ONLY")
    claim = dict(source=event["source"], source_event_id=event["source_event_id"],
                 revision_id=events([event])[0]["revision_id"], player_name="Example Runner",
                 team="BUF", opponent_team="NYJ", position="RB", season=2025, week=1, season_type="REG",
                 game_id="SYNTHETIC_2025_01_NYJ_BUF", signal_type="ROLE_INCREASE",
                 evidence_phase="ROLE_ANALYSIS", evidence_kind="ANALYST_OPINION", direction="INCREASE",
                 confidence=None, evidence_quote=event["evidence_text"],
                 forward_looking_basis="is expected to receive more carries in the next game",
                 extraction_method="MANUAL_OFFLINE", temporal_orientation="FORWARD_LOOKING",
                 effective_from_utc="2025-09-07T10:00:00Z", valid_until_utc="2025-09-07T17:00:00Z",
                 metric="carries", unit="count", denominator=None, comparison_basis="prior_game", conditions=[])
    return dict(input_mode="OFFLINE_FIXTURE", events=[event], claims=[claim], cutoff_utc="2025-09-07T16:00:00Z",
                game=dict(season=2025, week=1, season_type="REG", game_id=claim["game_id"], teams=["NYJ", "BUF"],
                          kickoff_at_utc="2025-09-07T17:00:00Z", available_at_utc="2025-09-01T00:00:00Z"),
                identity=dict(authority="WFS_IDENTITY_SNAPSHOT", season=2025, week=1,
                              available_at_utc="2025-09-06T00:00:00Z",
                              players=[dict(player_name="Example Runner", team="BUF", position="RB", gsis_id="SYNTHETIC-GSIS-001")]))
