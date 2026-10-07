#!/usr/bin/env python3
"""Stage 1: isolated current-player evidence composition, never recommendations.

Run with venv/bin/python -B research/build_player_form_matchup_shadow_v1.py.
--self-test exercises adversarial checks in memory and writes nothing.

Only this builder's parquet and manifest under data/research are published.
No database connection, refresh, model inference, production import entrypoint,
eligibility change, UI integration or automatic consumer is introduced.

Published matrix/slate/MI values are consumed, not replaced. Pure existing
pregame/MI functions reconstruct strictly prior evidence to verify those values.
Fields absent or position-placeholder-only use the existing pregame calculation.
Unprovable or missing fields become null with an explicit reason. Existing role
flags retain their upstream meanings; this module defines no new role score.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Inspected modules: import defines helpers; no builder/main is invoked.
# config creates only its already-existing directory paths, never artifacts.
import pregame_features as pregame
from research import build_matchup_intelligence_current_shadow_v1 as mi

CONTRACT = "WFS_PLAYER_FORM_MATCHUP_SHADOW_V1"
OUTPUT = ROOT / "data/research/player_form_matchup_shadow_v1.parquet"
MANIFEST = OUTPUT.with_name("player_form_matchup_shadow_v1_manifest.json")
INPUTS = {
    "matrix": "data/parquet/nfl_current_offensive_model_matrix.parquet",
    "slate": "data/parquet/nfl_current_slate_features.parquet",
    "core_features": "data/parquet/nfl_core_projection_features.parquet",
    "usage": "data/parquet/nfl_player_weekly_usage.parquet",
    "mi": "data/research/matchup_intelligence_current_shadow_v1.parquet",
    "mi_manifest": "data/research/matchup_intelligence_current_shadow_v1_manifest.json",
    "schedule": "data/parquet/nfl_schedule.parquet",
    "stats": "data/parquet/nfl_player_game_stats.parquet",
    "situational": "data/research/situational_role_history_v1.parquet",
}
CODE_INPUTS = {
    "composer_code": "research/build_player_form_matchup_shadow_v1.py",
    "pregame_code": "pregame_features.py",
    "mi_builder_code": "research/build_matchup_intelligence_current_shadow_v1.py",
    "matrix_builder_code": "current_offensive_model_matrix.py",
    "slate_builder_code": "current_slate_features.py",
}
KEYS = ["game_id", "player_id", "team", "opponent_team", "position"]
IDENTITY = ["season", "week", *KEYS]
PRODUCTION = ["fd_last", "fd_avg_3", "fd_avg_5", "fanduel_trend", "fd_max_5", "fd_std_5"]
ROLE = [
    "opportunities_avg_3", "opportunities_avg_5", "opportunity_trend",
    "snap_pct_avg_3", "snap_pct_avg_5", "snap_trend",
    "targets_avg_3", "targets_avg_5", "target_trend",
    "carries_avg_3", "carries_avg_5", "carry_trend",
    "target_share_last", "target_share_avg_3", "air_yards_share_avg_3", "wopr_avg_3",
    "role_expansion_last", "role_decline_last", "role_expansions_3", "role_declines_3",
    "established_role_flag", "rising_role_flag", "declining_role_flag",
]
RECEIVING = {"targets_avg_3", "targets_avg_5", "target_trend", "target_share_last",
             "target_share_avg_3", "air_yards_share_avg_3", "wopr_avg_3"}
CARRY = {"carries_avg_3", "carries_avg_5", "carry_trend"}
SAFETY = {"analysis_only": True, "production_influence": False,
          "solver_influence": False, "projection_mutation": False,
          "eligibility_mutation": False, "gpp_mutation": False,
          "ui_mutation": False, "database_mutation": False}


def require(condition, reason):
    if not condition:
        raise RuntimeError(CONTRACT + ":" + reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def same_number(a, b):
    # Numerical reconstruction check only; NEVER used for direction/FLAT.
    return finite(a) and finite(b) and bool(np.isclose(a, b, rtol=1e-10, atol=1e-10))


def direction(value, comparable=True):
    if not comparable or not finite(value):
        return "UNAVAILABLE"
    return "UP" if value > 0 else "DOWN" if value < 0 else "FLAT"


def unique(frame, keys, label):
    require(set(keys).issubset(frame), label + "_SCHEMA")
    require(not frame[keys].isna().any().any(), label + "_NULL_KEY")
    require(frame[keys].astype(str).apply(lambda s: s.str.strip().ne("").all()).all(),
            label + "_BLANK_KEY")
    require(not frame.duplicated(keys).any(), label + "_DUPLICATE_KEY")


def load_inputs():
    frames, metadata = {}, {}
    for name, relative in {**INPUTS, **CODE_INPUTS}.items():
        path = ROOT / relative
        if name == "situational" and not path.exists():
            frames[name] = None
            metadata[name] = {"path": str(path), "status": "MISSING"}
            continue
        data = path.read_bytes()
        metadata[name] = {"path": str(path), "sha256": sha(data), "bytes": len(data),
                          "status": "PRESENT"}
        if name == "mi_manifest":
            frames[name] = json.loads(data)
        elif name in INPUTS:
            frames[name] = pd.read_parquet(io.BytesIO(data))
    validate_binding(frames, metadata)
    return frames, metadata


def validate_binding(frames, metadata):
    manifest = frames["mi_manifest"]
    require(manifest.get("contract") == "WFS_MATCHUP_INTELLIGENCE_CURRENT_SHADOW_V1",
            "MI_CONTRACT")
    require(manifest.get("status") == "CURRENT_SHADOW_VALIDATED", "MI_STATUS")
    for source, field in [("matrix", "matrix_sha256"), ("mi", "shadow_sha256")]:
        require(manifest.get(field) == metadata[source]["sha256"], "MI_HASH_" + source)
    require(manifest.get("matrix_rows") == len(frames["matrix"]), "MI_MATRIX_ROWS")
    require(manifest.get("rows") == len(frames["mi"]), "MI_ROWS")
    require(manifest.get("identity_keys") == KEYS, "MI_IDENTITY_CONTRACT")
    require(manifest.get("historical_policy") == "BUILDER_PRIOR_MASK_STRICTLY_BEFORE_TARGET_WEEK",
            "MI_CUTOFF_POLICY")


def schedule_index(schedule):
    unique(schedule, ["game_id"], "SCHEDULE")
    s = schedule.copy()
    require({"game_date", "gametime", "completed", "season", "week", "away_team", "home_team"}
            .issubset(s), "SCHEDULE_SCHEMA")
    dates = pd.to_datetime(s.game_date.astype(str) + " " + s.gametime.astype(str), errors="coerce")
    s["kickoff_utc"] = dates.dt.tz_localize("America/New_York", ambiguous="NaT",
                                         nonexistent="NaT").dt.tz_convert("UTC")
    return s.set_index("game_id", drop=False)


def prior_history(frame, season, week, schedule, cutoff, label):
    """Use the existing strict-week policy; prove schedule identity and timing."""
    h = frame.loc[mi.prior_mask(frame, season, week)].copy()
    unique(h, ["game_id", "player_id"], label)
    require(h.game_id.isin(schedule.index).all(), label + "_SCHEDULE_MISSING")
    games = schedule.loc[h.game_id]
    require(np.array_equal(h.season.to_numpy(), games.season.to_numpy()) and
            np.array_equal(h.week.to_numpy(), games.week.to_numpy()), label + "_SCHEDULE_WEEK")
    require(games.completed.eq(1).all(), label + "_NOT_FINAL")
    require(games.kickoff_utc.notna().all() and games.kickoff_utc.lt(cutoff).all(),
            label + "_NONPRIOR_KICKOFF")
    away = h.team.to_numpy() == games.away_team.to_numpy()
    home = h.team.to_numpy() == games.home_team.to_numpy()
    require((away | home).all(), label + "_TEAM_SCHEDULE")
    if "opponent_team" in h:
        require(np.array_equal(h.opponent_team.to_numpy(),
                               np.where(away, games.home_team.to_numpy(), games.away_team.to_numpy())),
                label + "_OPPONENT_SCHEDULE")
    h["_kickoff"] = games.kickoff_utc.to_numpy()
    h = h.sort_values(["player_id", "season", "week", "game_id"])
    preceding = h.groupby("player_id")["_kickoff"].shift()
    require((preceding.isna() | preceding.lt(h["_kickoff"])).all(), label + "_NONCHRONOLOGICAL_ORDER")
    return h


def reconstruct_pregame(current, usage):
    """Call existing shift(1) semantics on prior rows plus blank targets only."""
    h = usage.loc[usage.player_id.isin(current.player_id)].drop(columns=["_kickoff"], errors="ignore").copy()
    target = pd.DataFrame(0, index=range(len(current)), columns=h.columns)
    for col in IDENTITY:
        target[col] = current[col].to_numpy()
    combined = pd.concat([h, target], ignore_index=True)
    with redirect_stdout(io.StringIO()):
        calculated = pregame.calculate_pregame_features(pregame.normalize_numeric_columns(combined))
    result = calculated.loc[calculated.game_id.isin(current.game_id)].copy()
    unique(result, ["game_id", "player_id"], "RECONSTRUCTION")
    return result.set_index(["game_id", "player_id"])


def raw_dependencies(field):
    if field in PRODUCTION:
        return ["fanduel_points"]
    if field.startswith("opportunit"):
        return ["opportunities"]
    if field.startswith("snap"):
        return ["offense_pct"]
    if field.startswith("target_share"):
        return ["target_share"]
    if field.startswith("target"):
        return ["targets"]
    if field.startswith("carr"):
        return ["carries"]
    if field.startswith("air_yards"):
        return ["air_yards_share"]
    if field.startswith("wopr"):
        return ["wopr"]
    if field.startswith("role_expansion"):
        return ["role_expansion_flag"]
    if field.startswith("role_declin"):
        return ["role_decline_flag"]
    return ["opportunities", "offense_pct", "targets"]


def field_reason(field, history, position):
    if (field in RECEIVING and position == "QB") or (field in CARRY and position not in {"QB", "RB"}):
        return "NOT_APPLICABLE"
    if history.empty:
        return "NO_PRIOR_HISTORY"
    window = 1 if field.endswith("_last") else 3 if field.endswith("_3") else 5
    raw = raw_dependencies(field)
    if not set(raw).issubset(history):
        return "MISSING_RAW_FIELD"
    values = history.tail(window)[raw].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        return "MISSING_RAW_EVIDENCE"
    if field == "fd_std_5" and len(history) < 2:
        return "INSUFFICIENT_VARIANCE_HISTORY"
    return "AVAILABLE"


def history_metadata(history, schedule, kickoff, prefix, window=5):
    """Age measures are explicit; they do not impose a staleness threshold."""
    h = history.sort_values(["season", "week", "game_id"]).tail(window)
    ids = sorted(h.game_id.unique().tolist()) if len(h) else []
    latest = h.iloc[-1] if len(h) else None
    return {
        prefix + "_history_games": int(history.game_id.nunique()),
        prefix + "_window_game_ids": json.dumps(ids),
        prefix + "_latest_season": int(latest.season) if latest is not None else None,
        prefix + "_latest_week": int(latest.week) if latest is not None else None,
        prefix + "_latest_game_id": str(latest.game_id) if latest is not None else None,
        prefix + "_age_days_at_target": float((kickoff - schedule.loc[latest.game_id, "kickoff_utc"])
                                               .total_seconds() / 86400) if latest is not None else None,
    }


def compose(frames, metadata, built_at):
    validate_binding(frames, metadata)
    matrix, slate, shadow = (frames[k].copy() for k in ["matrix", "slate", "mi"])
    core = frames["core_features"]
    unique(core, ["position", "feature"], "CORE_FEATURES")
    core_by_position = core.groupby("position").feature.apply(set).to_dict()
    require(set(mi.POSITIONS).issubset(core_by_position), "CORE_POSITION_COVERAGE")
    core_fields = set(core.feature)
    for name, frame in [("MATRIX", matrix), ("SLATE", slate), ("MI", shadow)]:
        unique(frame, ["game_id", "player_id"], name)
        require(set(IDENTITY).issubset(frame) and not frame[IDENTITY].isna().any().any(), name + "_IDENTITY")
        require(frame.position.isin(mi.POSITIONS).all(), name + "_NONOFFENSE")
    require(len(matrix) > 0 and len(matrix[["season", "week"]].drop_duplicates()) == 1, "TARGET")
    season, week = map(int, matrix[["season", "week"]].iloc[0])
    manifest = frames["mi_manifest"]
    require((season, week) == (manifest.get("season"), manifest.get("week")), "MI_TARGET")
    published = pd.Timestamp(manifest.get("published_at_utc"))
    require(published.tzinfo is not None and published <= built_at, "MI_MANIFEST_TIMESTAMP")
    expected_keys = set(matrix[IDENTITY].itertuples(index=False, name=None))

    # Matrix is the canonical current offensive population.
    #
    # SLATE is an evidence source and may temporarily retain stale identities
    # after roster/trade/injury changes. Every canonical matrix identity must
    # still be covered, but stale SLATE extras are never allowed to expand the
    # output population because composition iterates MATRIX only.
    slate_keys = set(slate[IDENTITY].itertuples(index=False, name=None))
    require(expected_keys.issubset(slate_keys), "SLATE_IDENTITY_COVERAGE")

    # MI is downstream of the canonical matrix and must remain an exact
    # identity-bound artifact.
    mi_keys = set(shadow[IDENTITY].itertuples(index=False, name=None))
    require(mi_keys == expected_keys, "MI_IDENTITY_COVERAGE")
    schedule = schedule_index(frames["schedule"])
    require(matrix.game_id.isin(schedule.index).all(), "TARGET_SCHEDULE")
    targets = schedule.loc[matrix.game_id.unique()]
    require(targets.kickoff_utc.notna().all() and targets.kickoff_utc.gt(built_at).all()
            and targets.completed.eq(0).all(), "TARGET_NOT_PREGAME")
    require(targets.season.eq(season).all() and targets.week.eq(week).all(), "TARGET_SCHEDULE_WEEK")
    for r in matrix.itertuples():
        game = schedule.loc[r.game_id]
        require({r.team, r.opponent_team} == {game.away_team, game.home_team}, "TARGET_MATCHUP")
    cutoff = targets.kickoff_utc.min()
    usage = prior_history(frames["usage"], season, week, schedule, cutoff, "USAGE")
    stats = prior_history(frames["stats"], season, week, schedule, cutoff, "STATS")
    situational = frames["situational"]
    if situational is not None:
        situational = prior_history(situational, season, week, schedule, cutoff, "SITUATIONAL")
    rebuilt = reconstruct_pregame(matrix, usage)
    # Reuse MI component builders for provenance verification, never for replacing MI values.
    with redirect_stdout(io.StringIO()):
        dvp_check = mi.build_current_dvp(stats, season, week).set_index(["defense_team", "position"])
        high_check = mi.build_high_value(matrix, situational, season, week).set_index(["game_id", "player_id"]) \
            if situational is not None else None
    slate = slate.set_index(["game_id", "player_id"])
    shadow = shadow.set_index(["game_id", "player_id"])
    groups = {p: g for p, g in usage.groupby("player_id")}
    rows = []
    for _, player in matrix.sort_values(["game_id", "position", "player_id"]).iterrows():
        key = (player.game_id, player.player_id)
        prior = groups.get(player.player_id, usage.iloc[:0])
        n = len(prior)
        check, mi_row, slate_row = rebuilt.loc[key], shadow.loc[key], slate.loc[key]
        require(int(check.history_games) == n, "PLAYER_HISTORY_DEPTH")
        require(same_number(player.history_games_player, n), "MATRIX_HISTORY_DEPTH")
        # MI baseline counts only historical QB/RB/WR/TE rows. Matrix/form
        # history includes FB and two-way positions; verify each source's scope.
        mi_history_depth = int(prior["position"].isin(mi.POSITIONS).sum())
        require(same_number(mi_row.prior_player_games, mi_history_depth), "MI_PLAYER_HISTORY_DEPTH")
        kickoff = schedule.loc[player.game_id, "kickoff_utc"]
        row = {c: player[c] for c in IDENTITY}
        row.update(player_name=player.get("player_display_name", player.get("player_name")),
                   built_at_utc=built_at.isoformat(), target_kickoff_utc=kickoff.isoformat(),
                   evidence_cutoff=f"{season}_WEEK_{week}_START_EXCLUSIVE",
                   chronology_status="PASS_PRIOR_ONLY_RECONSTRUCTED",
                   historical_available_at_cutoff_proven=False,
                   mi_manifest_sha256=metadata["mi_manifest"]["sha256"],
                   mi_shadow_sha256=metadata["mi"]["sha256"],
                   mi_matrix_sha256=metadata["matrix"]["sha256"],
                   mi_manifest_published_at_utc=manifest.get("published_at_utc"),
                   production_role_windows_distinct=n > 3,
                   production_window_observations_3=min(n, 3),
                   production_window_observations_5=min(n, 5),
                   opportunity_definition="CARRIES_PLUS_TARGETS_EXCLUDES_PASS_ATTEMPTS",
                   **SAFETY)
        row.update(history_metadata(prior, schedule, kickoff, "player"))
        for field in PRODUCTION + ROLE:
            # current_slate_features initializes the union of core columns to
            # zero but only calculates the position's selected fields. Matrix
            # metadata can retain these placeholders. They are not observations.
            placeholder = field in core_fields and field not in core_by_position[player.position]
            source = "pregame_reconstruction" if placeholder else \
                     "slate" if field in PRODUCTION and field in slate_row.index else \
                     "pregame_reconstruction" if field in PRODUCTION else "matrix"
            evidence = slate_row if source == "slate" else check if source == "pregame_reconstruction" else player
            reason = field_reason(field, prior, player.position)
            value = evidence.get(field, np.nan)
            if reason == "AVAILABLE":
                if not finite(value):
                    reason = "SOURCE_FIELD_UNAVAILABLE"
                elif not same_number(value, check[field]):
                    reason = "SOURCE_NOT_VERIFIED_AGAINST_PRIOR_HISTORY"
            row[field] = float(value) if reason == "AVAILABLE" else None
            row[field + "_status"] = reason
            row[field + "_source"] = source
            row[field + "_source_reason"] = "POSITION_PLACEHOLDER_NOT_OBSERVED_REUSED_PREGAME_HELPER" if placeholder else \
                "FIELD_NOT_PUBLISHED_REUSED_PREGAME_HELPER" if source == "pregame_reconstruction" else "PUBLISHED_FIELD_SUBJECT_TO_STATUS"
        for dimension, field in [("production", "fanduel_trend"), ("opportunity", "opportunity_trend"),
                                 ("snap", "snap_trend"), ("target", "target_trend"), ("carry", "carry_trend")]:
            row[dimension + "_direction"] = direction(row[field], n > 3)
            row[dimension + "_direction_status"] = row[field + "_status"] if row[field] is None else \
                "AVAILABLE" if n > 3 else "INSUFFICIENT_DISTINCT_3_VS_5_WINDOWS"
        row["production_available"] = row["fanduel_trend"] is not None
        row["role_available"] = row["opportunity_trend"] is not None
        row["production_vs_opportunity"] = row["production_direction"] + "__" + row["opportunity_direction"]

        dh = stats.loc[stats.opponent_team.eq(player.opponent_team) & stats.position.eq(player.position)]
        row.update(history_metadata(dh.drop_duplicates("game_id"), schedule, kickoff, "dvp"))
        dk = (player.opponent_team, player.position)
        verified_dvp = dk in dvp_check.index and same_number(mi_row.dvp_history_games, row["dvp_history_games"])
        for field in mi.DVP_FEATURES:
            reason = "AVAILABLE" if verified_dvp and same_number(mi_row.get(field), dvp_check.loc[dk, field]) \
                else "DVP_CHRONOLOGY_OR_VALUE_UNPROVEN"
            receiving = field.startswith(("targets_", "receptions_", "receiving_"))
            rushing = field.startswith(("carries_", "rushing_"))
            if (player.position == "QB" and receiving) or (player.position in {"WR", "TE"} and rushing):
                reason = "NOT_APPLICABLE"
            row[field] = float(mi_row[field]) if reason == "AVAILABLE" else None
            row[field + "_status"] = reason
        row["matchup_available"] = row["fd_allowed_avg_3"] is not None
        row["mi_dvp_minimum_history_met"] = row["dvp_history_games"] >= 3
        row["matchup_fd_direction"] = direction(row["fd_allowed_trend"], row["dvp_history_games"] > 3)
        row["matchup_opportunity_direction"] = direction(row["opportunity_allowed_trend"], row["dvp_history_games"] > 3)
        for dimension, field in [("matchup_fd", "fd_allowed_trend"),
                                 ("matchup_opportunity", "opportunity_allowed_trend")]:
            row[dimension + "_direction_status"] = row[field + "_status"] if row[field] is None else \
                "INSUFFICIENT_DISTINCT_3_VS_5_WINDOWS" if row["dvp_history_games"] <= 3 else "AVAILABLE"
        row["matchup_direction_meaning"] = "SIGN_OF_ALLOWANCE_CHANGE_NOT_MATCHUP_STRENGTH"
        row["mi_incremental_channel"] = {"QB": "NONE", "RB": "rb_g2g_share", "WR": "opportunities_allowed_avg_3",
                                          "TE": "te_i10_share"}[player.position]
        row["mi_validated_components"] = json.dumps({
            "QB": [], "RB": ["carries", "rushing_yards"],
            "WR": ["targets", "receptions", "receiving_yards"],
            "TE": ["receptions", "receiving_yards"],
        }[player.position])
        for pos, share in [("RB", "rb_g2g_share"), ("TE", "te_i10_share")]:
            row[share] = None
            row[share + "_available"] = False
            row[share + "_status"] = "NOT_APPLICABLE"
            if player.position != pos:
                continue
            reason = "SITUATIONAL_SOURCE_MISSING"
            if high_check is not None:
                available = mi_row.get(share + "_available")
                verified = isinstance(available, (bool, np.bool_)) and available == high_check.loc[key, share + "_available"]
                reason = "NO_OBSERVED_SHARE_DENOMINATOR_OR_HISTORY"
                if not verified:
                    reason = "HIGH_VALUE_CHRONOLOGY_OR_VALUE_UNPROVEN"
                elif available:
                    reason = "AVAILABLE" if same_number(mi_row[share], high_check.loc[key, share]) else \
                        "HIGH_VALUE_CHRONOLOGY_OR_VALUE_UNPROVEN"
            row[share] = float(mi_row[share]) if reason == "AVAILABLE" else None
            row[share + "_available"] = reason == "AVAILABLE"
            row[share + "_status"] = reason
        relevant = player.position in {"RB", "TE"}
        hv = situational.loc[situational.player_id.eq(player.player_id) & situational.position.eq(player.position)] \
            if relevant and situational is not None else usage.iloc[:0]
        row.update(history_metadata(hv, schedule, kickoff, "high_value_player", window=3))
        team_ids = matrix.loc[matrix.game_id.eq(player.game_id) & matrix.team.eq(player.team)
                             & matrix.position.eq(player.position), "player_id"]
        denominator = situational.loc[situational.player_id.isin(team_ids) & situational.position.eq(player.position)] \
            if relevant and situational is not None else usage.iloc[:0]
        denominator_window = denominator.sort_values(["player_id", "season", "week", "game_id"]).groupby("player_id").tail(3)
        row.update(history_metadata(denominator_window.drop_duplicates("game_id"), schedule, kickoff,
                                    "high_value_denominator", window=len(denominator_window)))
        missing_ids = sorted(set(prior.loc[prior.position.eq(player.position), "game_id"]) - set(hv.game_id)) if relevant else []
        row["high_value_missing_observed_player_game_ids"] = json.dumps(missing_ids)
        row["high_value_missing_observed_player_games"] = len(missing_ids)
        denominator_observations = usage.loc[usage.player_id.isin(team_ids) & usage.position.eq(player.position)]
        observed_keys = set(denominator_observations[["game_id", "player_id"]].itertuples(index=False, name=None))
        covered_keys = set(denominator[["game_id", "player_id"]].itertuples(index=False, name=None))
        row["high_value_denominator_missing_player_games"] = len(observed_keys - covered_keys) if relevant else None
        row["high_value_denominator_player_ids"] = json.dumps(sorted(team_ids.tolist())) if relevant else "[]"
        row["high_value_freshness_status"] = "NOT_APPLICABLE" if not relevant else \
            "SOURCE_MISSING" if situational is None else "MISSING_OBSERVED_PRIOR_GAMES" if missing_ids else "COVERS_OBSERVED_PRIOR_GAMES"
        rows.append(row)
    output = pd.DataFrame(rows)
    unique(output, ["game_id", "player_id"], "OUTPUT")
    require(len(output) == len(matrix), "OUTPUT_COUNT")
    require(set(output[IDENTITY].itertuples(index=False, name=None)) == expected_keys, "OUTPUT_IDENTITIES")
    prohibited = {"projection", "projection_solver", "ridge_projection", "solver_eligible", "optimizer_eligible",
                  "active_flag", "injury_flag", "gpp_score_15", "gpp_score_20", "gpp_score_25"}
    require(not prohibited.intersection(output), "PRODUCTION_FIELD_EXPOSED")
    status_columns = [c for c in output if c.endswith("_status")]
    missing_counts = {c: output[c].value_counts().to_dict() for c in status_columns}
    cross = pd.crosstab(output.production_direction, output.opportunity_direction)
    report = {
        "contract": CONTRACT, "status": "SHADOW_COMPOSED", "built_at_utc": built_at.isoformat(),
        "season": season, "week": week, "rows": len(output),
        "position_counts": output.position.value_counts().to_dict(), "duplicate_key_count": 0,
        "identity_keys": KEYS, "inputs": metadata, "mi_binding": manifest,
        "evidence_cutoff": f"{season}_WEEK_{week}_START_EXCLUSIVE",
        "chronology_validation_status": "PASS_PRIOR_ONLY_RECONSTRUCTION_WITH_FIELD_LEVEL_GATES",
        "excluded_target_or_future_rows": {name: int((~mi.prior_mask(frames[name], season, week)).sum())
                                           for name in ["usage", "stats", "situational"] if frames[name] is not None},
        "historical_available_at_cutoff_proven": False,
        "chronology_note": "Completed prior-week outcomes verified against schedule; rebuilt history is not an immutable historical knowledge snapshot.",
        "direction_policy": "Exact sign only. Trend direction unavailable when <=3 observations make 3/5 windows identical; this is window identifiability, not a fitted threshold.",
        "upstream_role_flags": "Published applicable flags consumed after prior-only verification; position placeholders use the existing pregame helper. No new role definitions.",
        "position_policy": "QB opportunities exclude passing; QB receiving fields and WR/TE carry directions are not applicable. K/DST excluded; no composite role or matchup direction.",
        "production_field_sources": {f: output[f + "_source"].value_counts().to_dict() for f in PRODUCTION},
        "role_field_sources": {f: output[f + "_source"].value_counts().to_dict() for f in ROLE},
        "position_placeholder_policy": "Frozen core-feature position membership identifies zero-initialized, uncomputed slate columns. Their copies in matrix metadata are not treated as observed evidence. The existing pregame helper supplies shadow evidence instead; no input is changed.",
        "position_placeholder_counts": {f: int(output[f + "_source_reason"].eq("POSITION_PLACEHOLDER_NOT_OBSERVED_REUSED_PREGAME_HELPER").sum()) for f in PRODUCTION + ROLE},
        "published_value_mismatch_counts": {f: int(output[f + "_status"].eq("SOURCE_NOT_VERIFIED_AGAINST_PRIOR_HISTORY").sum()) for f in PRODUCTION + ROLE},
        "production_direction_counts": output.production_direction.value_counts().to_dict(),
        "opportunity_direction_counts": output.opportunity_direction.value_counts().to_dict(),
        "production_vs_opportunity": {str(p): {str(o): int(cross.loc[p, o]) for o in cross.columns} for p in cross.index},
        "requested_down_cohorts": {o: int((output.production_direction.eq("DOWN") & output.opportunity_direction.eq(o)).sum())
                                   for o in ["UP", "FLAT", "DOWN"]},
        "field_status_counts": missing_counts,
        "missing_evidence_counts": {f: int(output[f].isna().sum()) for f in PRODUCTION + ROLE + mi.DVP_FEATURES + ["rb_g2g_share", "te_i10_share"]},
        "freshness_summary": {p: {"latest_season_week_counts": output.groupby([p + "_latest_season", p + "_latest_week"], dropna=False).size()
                                      .rename_axis(["season", "week"]).reset_index(name="rows").replace({np.nan: None}).to_dict("records"),
                                    "maximum_age_days_at_target": float(output[p + "_age_days_at_target"].max())
                                      if output[p + "_age_days_at_target"].notna().any() else None}
                              for p in ["player", "dvp", "high_value_player", "high_value_denominator"]},
        "high_value_freshness_counts": output.high_value_freshness_status.value_counts().to_dict(),
        "safety": SAFETY,
    }
    return output, report


def check_unchanged(metadata):
    for name, info in metadata.items():
        path = Path(info["path"])
        require(not path.exists() if info["status"] == "MISSING" else sha(path.read_bytes()) == info["sha256"],
                "INPUT_CHANGED_DURING_BUILD:" + name)


def publish(output, report, metadata):
    """Manifest is committed last and binds exact parquet bytes; no production writes."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".player_form_matchup_", dir=OUTPUT.parent) as directory:
        parquet = Path(directory) / OUTPUT.name
        manifest = Path(directory) / MANIFEST.name
        output.to_parquet(parquet, index=False)
        pd.testing.assert_frame_equal(output, pd.read_parquet(parquet))
        report["output"] = {"path": str(OUTPUT), "sha256": sha(parquet.read_bytes()), "manifest_path": str(MANIFEST)}
        manifest.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        check_unchanged(metadata)
        os.replace(parquet, OUTPUT)
        os.replace(manifest, MANIFEST)


def self_test(frames, metadata, built_at):
    """Current-data adversarial tests; no output publication or input mutation."""
    baseline, _ = compose(frames, metadata, built_at)
    assert [direction(v) for v in [-1, 0, 1, None, np.nan]] == ["DOWN", "FLAT", "UP", "UNAVAILABLE", "UNAVAILABLE"]
    assert direction(1e-15) == "UP" and direction(-1e-15) == "DOWN"
    assert direction(0, False) == "UNAVAILABLE"
    poisoned = dict(frames)
    season, week = frames["matrix"][["season", "week"]].iloc[0]
    for name in ["usage", "stats", "situational"]:
        if frames[name] is None:
            continue
        fake = frames[name].iloc[:1].copy()
        fake["game_id"] = frames["matrix"].game_id.iloc[0]
        fake["player_id"] = frames["matrix"].player_id.iloc[0]
        fake["season"], fake["week"] = season, week
        for col in fake.select_dtypes(include="number"):
            if col not in {"season", "week"}:
                fake[col] = 999999.0
        future = fake.copy()
        future["week"] = int(week) + 1
        poisoned[name] = pd.concat([frames[name], fake, future], ignore_index=True)
    result, _ = compose(poisoned, metadata, built_at)
    pd.testing.assert_frame_equal(baseline, result)
    missing = dict(frames, situational=None)
    result, _ = compose(missing, metadata, built_at)
    assert result.rb_g2g_share.isna().all() and result.te_i10_share.isna().all()
    bad = dict(frames, matrix=pd.concat([frames["matrix"], frames["matrix"].iloc[:1]], ignore_index=True))
    try:
        compose(bad, metadata, built_at)
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate matrix accepted")
    bad_metadata = {k: dict(v) for k, v in metadata.items()}
    bad_metadata["mi"]["sha256"] = "tampered"
    try:
        validate_binding(frames, bad_metadata)
    except RuntimeError:
        pass
    else:
        raise AssertionError("tampered MI binding accepted")
    assert baseline.loc[baseline.position.eq("QB"), "target_direction"].eq("UNAVAILABLE").all()
    assert baseline.loc[baseline.position.isin(["WR", "TE"]), "carry_direction"].eq("UNAVAILABLE").all()
    # A bound identity with contaminated numeric evidence is never silently used.
    contaminated = dict(frames, matrix=frames["matrix"].copy(deep=True), slate=frames["slate"].copy(deep=True))
    tested = baseline.loc[baseline.role_available & baseline.production_available & baseline.fd_last_source.eq("slate")].iloc[0]
    mask = contaminated["matrix"].player_id.eq(tested.player_id)
    contaminated["matrix"].loc[mask, "opportunity_trend"] += 1000
    mask = contaminated["slate"].player_id.eq(tested.player_id)
    contaminated["slate"].loc[mask, "fd_last"] += 1000
    result, _ = compose(contaminated, metadata, built_at)
    tested_result = result.loc[result.player_id.eq(tested.player_id)].iloc[0]
    assert pd.isna(tested_result.opportunity_trend) and tested_result.opportunity_direction == "UNAVAILABLE"
    assert pd.isna(tested_result.fd_last) and tested_result.fd_last_status == "SOURCE_NOT_VERIFIED_AGAINST_PRIOR_HISTORY"
    # Uncomputed positional zeros are never misrepresented as observed values.
    assert baseline.loc[baseline.position.isin(["WR", "TE"]), "fd_last_source"].eq("pregame_reconstruction").all()
    null_raw = dict(frames, usage=frames["usage"].copy(deep=True))
    eligible_history = null_raw["usage"].loc[null_raw["usage"].player_id.eq(tested.player_id)
                                          & mi.prior_mask(null_raw["usage"], season, week)]
    last_index = eligible_history.sort_values(["season", "week", "game_id"]).index[-1]
    null_raw["usage"].loc[last_index, "fanduel_points"] = np.nan
    result, _ = compose(null_raw, metadata, built_at)
    tested_result = result.loc[result.player_id.eq(tested.player_id)].iloc[0]
    assert pd.isna(tested_result.fanduel_trend) and tested_result.production_direction == "UNAVAILABLE"
    assert tested_result.fanduel_trend_status == "MISSING_RAW_EVIDENCE"
    # The adapter has no authority to repair identity conflicts or time travel.
    conflict = dict(frames, slate=frames["slate"].copy(deep=True))
    conflict["slate"].loc[conflict["slate"].index[0], "team"] = "INVALID"
    for bad_frames, at in [(conflict, built_at), (frames, pd.Timestamp(baseline.target_kickoff_utc.max()))]:
        try:
            compose(bad_frames, metadata, at)
        except RuntimeError:
            pass
        else:
            raise AssertionError("identity conflict or postkickoff composition accepted")
    # Empty/short history cannot manufacture flat direction or a variance value.
    history = frames["usage"].iloc[:1]
    assert field_reason("fd_last", history.iloc[:0], "RB") == "NO_PRIOR_HISTORY"
    assert field_reason("fd_std_5", history, "RB") == "INSUFFICIENT_VARIANCE_HISTORY"
    check_unchanged(metadata)
    print("SELF_TEST=PASS: exact sign, indistinct windows, target/future poisoning, missing situational, duplicate identity, MI hash, position applicability, contaminated values, identity conflict, postkickoff rejection, short history, positional placeholders, null raw evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="In-memory adversarial validation; no files written")
    args = parser.parse_args()
    frames, metadata = load_inputs()
    built_at = pd.Timestamp(datetime.now(timezone.utc))
    if args.self_test:
        self_test(frames, metadata, built_at)
        return
    output, report = compose(frames, metadata, built_at)
    publish(output, report, metadata)
    for name in ["season", "week", "rows", "position_counts", "production_direction_counts",
                 "opportunity_direction_counts", "production_vs_opportunity", "requested_down_cohorts",
                 "missing_evidence_counts", "freshness_summary", "high_value_freshness_counts",
                 "chronology_validation_status", "safety", "output"]:
        print(name.upper() + "=" + json.dumps(report[name], sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
