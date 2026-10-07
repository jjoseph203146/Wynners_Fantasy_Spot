"""Read-only validation of immutable pregame evidence; no production consumers."""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
import sqlite3

import pandas as pd


CONTRACT = "WFS_PLAYER_FORM_MATCHUP_PROSPECTIVE_CAPTURE_V1"
POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
IDENTITY_KEYS = ["game_id", "player_id", "team", "opponent_team", "position"]
SAFETY = {
    "analysis_only": True,
    "production_influence": False,
    "solver_influence": False,
    "projection_mutation": False,
    "eligibility_mutation": False,
    "gpp_mutation": False,
    "ui_mutation": False,
    "database_mutation": False,
}
PLAYER_METRICS = (
    "fd_avg_3", "fd_avg_5", "targets_avg_3", "carries_avg_3",
    "opportunities_avg_3", "snap_pct_avg_3", "target_share_avg_3",
)
OPPONENT_METRICS = (
    "fd_allowed_avg_3", "carries_allowed_avg_3", "targets_allowed_avg_3",
    "opportunities_allowed_avg_3", "rushing_yards_allowed_avg_3",
    "receptions_allowed_avg_3", "receiving_yards_allowed_avg_3",
)


@dataclass(frozen=True)
class PlayerIdentity:
    player_id: str
    team: str
    position: str


@dataclass(frozen=True)
class GameContext:
    season: int
    week: int
    game_id: str
    away_team: str
    home_team: str
    kickoff_utc: datetime
    completed: bool
    players: tuple[PlayerIdentity, ...]


@dataclass(frozen=True)
class PlayerEvidence:
    player_id: str
    name: str
    position: str
    team: str
    opponent: str
    metrics: tuple[tuple[str, float], ...]
    comparisons: tuple[tuple[str, str], ...]
    recent_games: int
    baseline_games: int


@dataclass(frozen=True)
class ValidatedOutlook:
    available: bool
    reason: str = ""
    season: int = 0
    week: int = 0
    away_team: str = ""
    home_team: str = ""
    captured_at_utc: datetime | None = None
    players: tuple[PlayerEvidence, ...] = ()


class InvalidOutlook(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise InvalidOutlook(reason)


def _utc(value):
    stamp = pd.Timestamp(value)
    _require(not pd.isna(stamp) and stamp.tzinfo is not None, "INVALID_TIME")
    return stamp.tz_convert("UTC").to_pydatetime()


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _file_signature(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _no_pending_journal(path):
    # immutable=1 must not silently ignore a committed WAL or pending rollback.
    for suffix in ("-wal", "-journal"):
        journal = Path(str(path) + suffix)
        _require(not journal.exists() or journal.stat().st_size == 0, "SCHEDULE_JOURNAL")


def read_game_context(*, database_path, game_id) -> GameContext:
    """Fresh exact-game/roster read. No SQLite sidecars, fallback, or cache."""
    _require(_text(game_id), "GAME_ID")
    path = Path(database_path).resolve(strict=True)
    _no_pending_journal(path)
    before = _file_signature(path)
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        games = conn.execute(
            "SELECT season, week, game_id, game_type, away_team, home_team, "
            "game_date, gametime, completed FROM games WHERE game_id = ?",
            (game_id,),
        ).fetchall()
        _require(len(games) == 1, "SCHEDULE_IDENTITY")
        game = dict(games[0])
        _require(game["game_type"] == "REG", "SCHEDULE_TYPE")
        _require(_integer(game["season"]) and _integer(game["week"]), "SCHEDULE_PERIOD")
        _require(game["completed"] in (0, 1), "SCHEDULE_COMPLETION")
        kickoff = pd.Timestamp(f"{game['game_date']} {game['gametime']}").tz_localize(
            "America/New_York", ambiguous="raise", nonexistent="raise"
        )
        roster = conn.execute(
            "SELECT gsis_id, team, position FROM weekly_rosters "
            "WHERE season = ? AND week = ? AND game_type = 'REG' "
            "AND team IN (?, ?)",
            (game["season"], game["week"], game["away_team"], game["home_team"]),
        ).fetchall()
    _no_pending_journal(path)
    _require(before == _file_signature(path), "SCHEDULE_CHANGED_DURING_READ")
    identities = tuple(sorted({tuple(row) for row in roster if row["position"] in POSITIONS}))
    context = GameContext(
        game["season"], game["week"], game_id, game["away_team"], game["home_team"],
        _utc(kickoff), bool(game["completed"]),
        tuple(PlayerIdentity(*row) for row in identities),
    )
    _validate_context(context)
    return context


def _validate_context(game):
    _require(isinstance(game, GameContext), "GAME_CONTEXT")
    _require(_integer(game.season) and _integer(game.week) and game.week > 0, "GAME_PERIOD")
    _require(_text(game.away_team) and _text(game.home_team) and game.away_team != game.home_team, "PARTICIPANTS")
    _require(game.game_id == f"{game.season}_{game.week:02d}_{game.away_team}_{game.home_team}", "GAME_IDENTITY")
    _utc(game.kickoff_utc)
    _require(type(game.completed) is bool, "GAME_COMPLETION")
    _require(bool(game.players), "ROSTER_UNAVAILABLE")
    known = {}
    for player in game.players:
        _require(isinstance(player, PlayerIdentity), "ROSTER_IDENTITY")
        _require(_text(player.player_id) and re.fullmatch(r"\d{2}-\d{7}", player.player_id), "ROSTER_IDENTITY")
        _require(player.team in {game.away_team, game.home_team} and player.position in POSITIONS, "ROSTER_IDENTITY")
        pair = (player.team, player.position)
        _require(player.player_id not in known or known[player.player_id] == pair, "AMBIGUOUS_ROSTER")
        known[player.player_id] = pair
    return known


def _history_valid(row, prefix, season, week):
    """Field-level history failures withhold claims rather than the whole game."""
    try:
        count = row[prefix + "_history_games"]
        if not _integer(count) or count <= 0:
            return False
        ids = json.loads(row[prefix + "_window_game_ids"])
        if not isinstance(ids, list) or len(ids) != min(count, 5) or len(set(ids)) != len(ids):
            return False
        periods = []
        for value in ids:
            match = re.fullmatch(r"(\d{4})_(\d{2})_[A-Z]+_[A-Z]+", value)
            if not match:
                return False
            period = tuple(map(int, match.groups()))
            if period >= (season, week):
                return False
            periods.append(period)
        latest = row[prefix + "_latest_game_id"]
        if latest not in ids:
            return False
        latest_period = tuple(map(int, latest.split("_")[:2]))
        if latest_period != max(periods) or latest_period != (row[prefix + "_latest_season"], row[prefix + "_latest_week"]):
            return False
        age = row[prefix + "_age_days_at_target"]
        return not isinstance(age, bool) and isinstance(age, Real) and math.isfinite(age) and age > 0
    except (KeyError, TypeError, ValueError):
        return False


def _metric(row, field):
    value = row.get(field)
    if row.get(field + "_status") != "AVAILABLE" or isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        return None
    if field.startswith(("snap_pct", "target_share")) and not 0 <= value <= 1:
        return None
    if not field.startswith("fd_") and value < 0:
        return None
    return float(value)


def _player_evidence(row, game):
    metrics, comparisons = {}, {}
    history = _history_valid(row, "player", game.season, game.week)
    n = row.get("player_history_games", 0) if history else 0
    samples_ok = (
        history and _integer(row.get("production_window_observations_3"))
        and _integer(row.get("production_window_observations_5"))
        and row.get("production_window_observations_3") == min(n, 3)
        and row.get("production_window_observations_5") == min(n, 5)
    )
    position = row["position"]
    allowed = set(PLAYER_METRICS)
    if position == "QB":
        allowed = {"fd_avg_3", "fd_avg_5"}
    elif position in {"WR", "TE"}:
        allowed.discard("carries_avg_3")
    if position == "TE":
        allowed.discard("target_share_avg_3")
    if row.get("opportunity_definition") != "CARRIES_PLUS_TARGETS_EXCLUDES_PASS_ATTEMPTS":
        allowed.discard("opportunities_avg_3")
    if samples_ok:
        for field in sorted(allowed):
            value = _metric(row, field)
            if value is not None:
                metrics[field] = value
        channels = [("scoring", "production", "fanduel_trend", "fd_avg_3", "fd_avg_5")]
        if position != "QB":
            channels.append(("targets", "target", "target_trend", "targets_avg_3", "targets_avg_5"))
        if position == "RB":
            channels.append(("carries", "carry", "carry_trend", "carries_avg_3", "carries_avg_5"))
        for label, dimension, trend_field, recent, baseline in channels:
            first, second = _metric(row, recent), _metric(row, baseline)
            trend = row.get(trend_field)
            if (n < 5 or row.get("production_role_windows_distinct") is not True
                    or row.get(dimension + "_direction_status") != "AVAILABLE"
                    or row.get(trend_field + "_status") != "AVAILABLE"
                    or first is None or second is None
                    or isinstance(trend, bool) or not isinstance(trend, Real) or not math.isfinite(trend)):
                continue
            change = first - second
            direction = "UP" if change > 0 else "DOWN" if change < 0 else "FLAT"
            if row.get(dimension + "_direction") == direction and math.isclose(trend, change, rel_tol=1e-9, abs_tol=1e-12):
                comparisons[label] = {"UP": "higher", "DOWN": "lower", "FLAT": "unchanged"}[direction]
    if (_history_valid(row, "dvp", game.season, game.week)
            and row.get("dvp_history_games", 0) >= 3
            and row.get("mi_dvp_minimum_history_met") is True):
        opponent_allowed = {
            "QB": {"fd_allowed_avg_3"},
            "RB": {"fd_allowed_avg_3", "carries_allowed_avg_3", "rushing_yards_allowed_avg_3", "opportunities_allowed_avg_3"},
            "WR": {"fd_allowed_avg_3", "targets_allowed_avg_3", "receptions_allowed_avg_3", "receiving_yards_allowed_avg_3"},
            "TE": {"fd_allowed_avg_3", "receptions_allowed_avg_3", "receiving_yards_allowed_avg_3"},
        }[position]
        for field in sorted(opponent_allowed):
            value = _metric(row, field)
            if value is not None:
                metrics[field] = value
    return PlayerEvidence(
        row["player_id"], row["player_name"], position, row["team"], row["opponent_team"],
        tuple(sorted(metrics.items())), tuple(sorted(comparisons.items())),
        min(n, 3) if samples_ok else 0, min(n, 5) if samples_ok else 0,
    )


def load_player_outlook(*, game_context, artifact_path, manifest_path, now_utc) -> ValidatedOutlook:
    """Never expose exceptions or provenance as public data; never fall back."""
    try:
        game = game_context
        known = _validate_context(game)
        now, kickoff = _utc(now_utc), _utc(game.kickoff_utc)
        artifact, manifest_file = Path(artifact_path), Path(manifest_path)
        _require(artifact.is_file() and manifest_file.is_file(), "MISSING_CAPTURE")
        manifest = json.loads(manifest_file.read_bytes())
        data = artifact.read_bytes()
        _require(manifest.get("contract") == CONTRACT and manifest.get("status") == "IMMUTABLE_PREKICKOFF_CAPTURED", "CAPTURE_CONTRACT")
        _require(hashlib.sha256(data).hexdigest() == manifest["output"]["sha256"], "CAPTURE_HASH")
        safety = manifest.get("safety", {})
        _require(manifest.get("analysis_only") is True and safety == SAFETY and all(safety.get(key) is value for key, value in SAFETY.items()), "SAFETY")
        _require(manifest.get("identity_keys") == IDENTITY_KEYS, "IDENTITY_CONTRACT")
        _require(_integer(manifest.get("season")) and _integer(manifest.get("week")) and manifest.get("season") == game.season and manifest.get("week") == game.week, "PERIOD_MISMATCH")
        details = manifest.get("games_detail")
        _require(manifest.get("games") == 1 and isinstance(details, list) and len(details) == 1, "GAME_COUNT")
        detail = details[0]
        _require(detail.get("game_id") == game.game_id and detail.get("away_team") == game.away_team and detail.get("home_team") == game.home_team, "GAME_MISMATCH")
        _require(_utc(detail.get("kickoff_utc")) == kickoff, "KICKOFF_MISMATCH")
        captured = _utc(manifest.get("captured_at_utc"))
        _require(captured <= now and captured < kickoff, "CAPTURE_TIME")
        chronology = manifest.get("chronology", {})
        _require(chronology.get("all_games_future_at_capture") is True and chronology.get("all_games_uncompleted_at_capture") is True and chronology.get("capture_policy") == "STRICTLY_BEFORE_TARGET_KICKOFF" and _utc(chronology.get("capture_time_utc")) == captured, "CAPTURE_CHRONOLOGY")
        frame = pd.read_parquet(io.BytesIO(data))
        required = set(IDENTITY_KEYS + [
            "season", "week", "player_name", "target_kickoff_utc", "built_at_utc",
            "captured_at_utc", "capture_mode", "prospective_contract", "evidence_cutoff",
            "chronology_status", "historical_available_at_cutoff_proven",
            "prospective_historical_knowledge_status", "source_stage1_sha256",
            "source_stage1_manifest_sha256", "source_schedule_sha256",
        ]) | set(SAFETY)
        _require(frame.columns.is_unique and required.issubset(frame.columns), "CAPTURE_SCHEMA")
        _require(len(frame) > 0 and len(frame) == manifest.get("rows") == manifest.get("players"), "ROW_COUNT")
        _require(not frame.duplicated(["game_id", "player_id"]).any(), "DUPLICATE_PLAYER")
        _require(chronology.get("stage1_chronology_status_counts") == {"PASS_PRIOR_ONLY_RECONSTRUCTED": len(frame)} and chronology.get("historical_available_at_cutoff_proven_counts") == {"false": len(frame)}, "CHRONOLOGY_COUNTS")
        source = manifest.get("source", {})
        players = []
        for row in frame.to_dict(orient="records"):
            _require(all(_text(row[field]) for field in IDENTITY_KEYS + ["player_name"]), "PLAYER_IDENTITY")
            _require(_integer(row["season"]) and _integer(row["week"]) and row["season"] == game.season and row["week"] == game.week and row["game_id"] == game.game_id, "ROW_GAME_MISMATCH")
            _require(row["position"] in POSITIONS and known.get(row["player_id"]) == (row["team"], row["position"]), "UNKNOWN_PLAYER")
            _require(row["team"] != row["opponent_team"] and {row["team"], row["opponent_team"]} == {game.away_team, game.home_team}, "ROW_PARTICIPANTS")
            _require(all(row[key] is value for key, value in SAFETY.items()), "ROW_SAFETY")
            _require(row["capture_mode"] == "IMMUTABLE_PREKICKOFF" and row["prospective_contract"] == CONTRACT, "ROW_CAPTURE")
            _require(_utc(row["target_kickoff_utc"]) == kickoff and _utc(row["captured_at_utc"]) == captured and _utc(row["built_at_utc"]) <= captured, "ROW_TIME")
            _require(row["chronology_status"] == "PASS_PRIOR_ONLY_RECONSTRUCTED" and row["historical_available_at_cutoff_proven"] is False and row["prospective_historical_knowledge_status"] == "CAPTURED_AS_KNOWN_AT_CAPTURE_TIME" and row["evidence_cutoff"] == f"{game.season}_WEEK_{game.week}_START_EXCLUSIVE", "ROW_CHRONOLOGY")
            for field, key in (("source_stage1_sha256", "stage1_sha256"), ("source_stage1_manifest_sha256", "stage1_manifest_sha256"), ("source_schedule_sha256", "schedule_sha256")):
                _require(_sha(row[field]) and row[field] == source.get(key), "SOURCE_BINDING")
            players.append(_player_evidence(row, game))
        return ValidatedOutlook(True, season=game.season, week=game.week, away_team=game.away_team, home_team=game.home_team, captured_at_utc=captured, players=tuple(players))
    except InvalidOutlook as exc:
        return ValidatedOutlook(False, reason=str(exc))
    except Exception:
        return ValidatedOutlook(False, reason="INVALID_CAPTURE")
