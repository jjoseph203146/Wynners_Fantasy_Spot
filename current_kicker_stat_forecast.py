#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

DB = ROOT / "data/nfl.db"

F6_DIR = (
    ROOT
    / "data/model_candidates/stat_forecast"
    / "stage23f6_20260912T160612Z"
)

F6_MANIFEST = F6_DIR / "manifest.json"
F6_FEATURE_CONTRACT = F6_DIR / "feature_contract.json"

FGA_MODEL = F6_DIR / "models/kicker_fga_poisson.joblib"
XPA_MODEL = F6_DIR / "models/kicker_xpa_poisson.joblib"

FROZEN_TARGET = (
    ROOT
    / "data/audits/stat_forecast_stage23d2_r2"
    / "kicker_targets_complete_20260912T143502Z.parquet"
)

FROZEN_F5_MATRIX = (
    ROOT
    / "data/audits/stat_forecast_stage23f5"
    / "current_kicker_model_matrix_44_20260912T160059Z.parquet"
)

FROZEN_F6_FORECAST = (
    F6_DIR / "current_kicker_stat_forecasts.parquet"
)

CURRENT_TEAM_ENV = (
    ROOT / "data/parquet/nfl_current_team_environment.parquet"
)

CURRENT_MATRIX = (
    ROOT / "data/parquet/nfl_current_kicker_model_matrix.parquet"
)

OUTPUT = (
    ROOT / "data/parquet/nfl_current_kicker_stat_forecasts.parquet"
)

EXPECTED_TARGET_SHA = (
    "f287fca5c45ec60d6642193169a1259baf0f0abdde93ac20f9568d86cdbb71e6"
)

EXPECTED_FEATURE_SHA = (
    "41cf5fe71ec03fb585c5520535c939b3092975ad5466cbdd6729e66684d19838"
)

EXPECTED_FGA_SHA = (
    "b8a6d613acaa565ad67a6f7f68d457f784e71d087824fa01d4c77aca8fb6a900"
)

EXPECTED_XPA_SHA = (
    "66a8bf924a7da7a4e94ce2002af1257194e626ecd73a4f6dc743b70bcb82ea12"
)

ZERO_HISTORY_FG_RATE = 0.8519061583577713
ZERO_HISTORY_XP_RATE = 0.9580528223718281

TEAM_ALIASES = {
    "LA": "LAR",
    "WAS": "WSH",
}

KICKER_FEATURES = [
    "k_hist_games",
    "k_fga_avg3",
    "k_fga_avg5",
    "k_fgm_avg3",
    "k_fgm_avg5",
    "k_xpa_avg3",
    "k_xpa_avg5",
    "k_xpm_avg3",
    "k_xpm_avg5",
    "k_fg_distance_avg3",
    "k_fg_distance_avg5",
    "k_fg_pct_prior",
    "k_xp_pct_prior",
]

OUTPUT_COLUMNS = [
    "game_id",
    "team",
    "opponent_team",
    "gsis_id",
    "full_name",
    "k_hist_games",
    "rate_source",
    "expected_fga",
    "fg_make_rate",
    "expected_fgm",
    "expected_xpa",
    "xp_make_rate",
    "expected_xpm",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def canon_team(value) -> str:
    s = clean(value).upper()
    return TEAM_ALIASES.get(s, s)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path: Path, expected_sha: str, label: str) -> None:
    if not path.is_file():
        fail(f"MISSING_{label}:{path}")

    actual = sha256(path)

    if actual != expected_sha:
        fail(
            f"{label}_SHA_MISMATCH:"
            f"expected={expected_sha}:actual={actual}"
        )


def load_contract() -> tuple[dict, list[str], list[str]]:
    verify_file(
        FROZEN_TARGET,
        EXPECTED_TARGET_SHA,
        "FROZEN_KICKER_TARGET",
    )

    verify_file(
        F6_FEATURE_CONTRACT,
        EXPECTED_FEATURE_SHA,
        "F6_FEATURE_CONTRACT",
    )

    verify_file(FGA_MODEL, EXPECTED_FGA_SHA, "FGA_MODEL")
    verify_file(XPA_MODEL, EXPECTED_XPA_SHA, "XPA_MODEL")

    with F6_MANIFEST.open() as f:
        manifest = json.load(f)

    with F6_FEATURE_CONTRACT.open() as f:
        contract = json.load(f)

    if manifest.get("contract") != "WFS_KICKER_STAT_FORECAST_V1":
        fail("INVALID_F6_MANIFEST_CONTRACT")

    if contract.get("contract") != "WFS_KICKER_OPPORTUNITY_FEATURES_V1":
        fail("INVALID_F6_FEATURE_CONTRACT")

    features = list(contract.get("features", []))
    team_features = list(contract.get("team_features", []))

    if len(features) != 44:
        fail(f"INVALID_F6_FEATURE_COUNT:{len(features)}")

    if features[:13] != KICKER_FEATURES:
        fail("F6_KICKER_FEATURE_ORDER_DRIFT")

    if len(team_features) != 31:
        fail(
            f"INVALID_TEAM_ENV_FEATURE_COUNT:"
            f"{len(team_features)}"
        )

    return manifest, features, team_features


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    tmp = Path(tmp_name)

    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def rolling_prior_features(history: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season",
        "week",
        "game_id",
        "kicker_player_id",
        "player_display_name",
        "team",
        "fga",
        "fgm",
        "xpa",
        "xpm",
        "fg_distance_avg",
    ]

    missing = [c for c in required if c not in history.columns]
    if missing:
        fail("KICKER_HISTORY_SCHEMA_MISSING:" + ",".join(missing))

    h = history.copy()

    h["team"] = h["team"].map(canon_team)

    for c in ["season", "week", "fga", "fgm", "xpa", "xpm"]:
        h[c] = pd.to_numeric(h[c], errors="raise")

    h["fg_distance_avg"] = pd.to_numeric(
        h["fg_distance_avg"],
        errors="coerce",
    )

    h = h.sort_values(
        [
            "kicker_player_id",
            "season",
            "week",
            "game_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    if h.duplicated(
        ["season", "week", "game_id", "kicker_player_id"]
    ).any():
        fail("DUPLICATE_KICKER_HISTORY_IDENTITY")

    g = h.groupby("kicker_player_id", sort=False)

    out = h[
        [
            "season",
            "week",
            "game_id",
            "kicker_player_id",
            "player_display_name",
            "team",
        ]
    ].copy()

    out["k_hist_games"] = g.cumcount()

    for source, stem in [
        ("fga", "k_fga"),
        ("fgm", "k_fgm"),
        ("xpa", "k_xpa"),
        ("xpm", "k_xpm"),
    ]:
        shifted = g[source].shift(1)

        for window in (3, 5):
            out[f"{stem}_avg{window}"] = (
                shifted
                .groupby(h["kicker_player_id"])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    shifted_distance = g["fg_distance_avg"].shift(1)

    for window in (3, 5):
        out[f"k_fg_distance_avg{window}"] = (
            shifted_distance
            .groupby(h["kicker_player_id"])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    prior_fgm = g["fgm"].cumsum() - h["fgm"]
    prior_fga = g["fga"].cumsum() - h["fga"]

    prior_xpm = g["xpm"].cumsum() - h["xpm"]
    prior_xpa = g["xpa"].cumsum() - h["xpa"]

    out["k_fg_pct_prior"] = (
        prior_fgm / prior_fga.replace(0, np.nan)
    )

    out["k_xp_pct_prior"] = (
        prior_xpm / prior_xpa.replace(0, np.nan)
    )

    return out


def current_features_from_history(
    history: pd.DataFrame,
    current_kickers: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    h = history.copy()

    h["team"] = h["team"].map(canon_team)

    h = h.sort_values(
        ["kicker_player_id", "season", "week", "game_id"],
        kind="stable",
    )

    for row in current_kickers.itertuples(index=False):
        pid = clean(row.gsis_id)

        prior = h[
            h["kicker_player_id"].astype(str).eq(pid)
        ].copy()

        result = {
            "game_id": row.game_id,
            "team": row.team,
            "opponent_team": row.opponent_team,
            "gsis_id": pid,
            "full_name": row.full_name,
            "k_hist_games": int(len(prior)),
        }

        if prior.empty:
            for c in KICKER_FEATURES[1:]:
                result[c] = np.nan

            rows.append(result)
            continue

        for source, stem in [
            ("fga", "k_fga"),
            ("fgm", "k_fgm"),
            ("xpa", "k_xpa"),
            ("xpm", "k_xpm"),
        ]:
            values = pd.to_numeric(
                prior[source],
                errors="coerce",
            )

            result[f"{stem}_avg3"] = values.tail(3).mean()
            result[f"{stem}_avg5"] = values.tail(5).mean()

        distance = pd.to_numeric(
            prior["fg_distance_avg"],
            errors="coerce",
        )

        result["k_fg_distance_avg3"] = (
            distance.tail(3).mean()
        )

        result["k_fg_distance_avg5"] = (
            distance.tail(5).mean()
        )

        total_fga = float(
            pd.to_numeric(prior["fga"], errors="coerce").sum()
        )
        total_fgm = float(
            pd.to_numeric(prior["fgm"], errors="coerce").sum()
        )
        total_xpa = float(
            pd.to_numeric(prior["xpa"], errors="coerce").sum()
        )
        total_xpm = float(
            pd.to_numeric(prior["xpm"], errors="coerce").sum()
        )

        result["k_fg_pct_prior"] = (
            total_fgm / total_fga
            if total_fga > 0
            else np.nan
        )

        result["k_xp_pct_prior"] = (
            total_xpm / total_xpa
            if total_xpa > 0
            else np.nan
        )

        rows.append(result)

    return pd.DataFrame(rows)


def regression() -> None:
    _, features, _ = load_contract()

    frozen_matrix = pd.read_parquet(FROZEN_F5_MATRIX)
    frozen_forecast = pd.read_parquet(FROZEN_F6_FORECAST)

    if list(frozen_matrix.columns[5:]) != features:
        fail("FROZEN_F5_MATRIX_FEATURE_ORDER_DRIFT")

    print("FROZEN_MATRIX_ROWS =", len(frozen_matrix))
    print("FROZEN_FORECAST_ROWS =", len(frozen_forecast))

    # Load joblib only inside the model environment.
    import joblib

    fga_model = joblib.load(FGA_MODEL)
    xpa_model = joblib.load(XPA_MODEL)

    X = frozen_matrix[features]

    expected_fga = np.maximum(
        np.asarray(fga_model.predict(X), dtype=float),
        0.0,
    )

    expected_xpa = np.maximum(
        np.asarray(xpa_model.predict(X), dtype=float),
        0.0,
    )

    actual = frozen_matrix[
        [
            "game_id",
            "team",
            "opponent_team",
            "gsis_id",
            "full_name",
            "k_hist_games",
            "k_fg_pct_prior",
            "k_xp_pct_prior",
        ]
    ].copy()

    actual["rate_source"] = np.where(
        actual["k_hist_games"].gt(0),
        "KICKER_HISTORICAL_RATE",
        "ALL_PRIOR_HISTORY",
    )

    actual["fg_make_rate"] = np.where(
        actual["k_hist_games"].gt(0)
        & actual["k_fg_pct_prior"].notna(),
        actual["k_fg_pct_prior"],
        ZERO_HISTORY_FG_RATE,
    )

    actual["xp_make_rate"] = np.where(
        actual["k_hist_games"].gt(0)
        & actual["k_xp_pct_prior"].notna(),
        actual["k_xp_pct_prior"],
        ZERO_HISTORY_XP_RATE,
    )

    actual["expected_fga"] = expected_fga
    actual["expected_fgm"] = (
        actual["expected_fga"] * actual["fg_make_rate"]
    )

    actual["expected_xpa"] = expected_xpa
    actual["expected_xpm"] = (
        actual["expected_xpa"] * actual["xp_make_rate"]
    )

    actual = actual[OUTPUT_COLUMNS]

    frozen = frozen_forecast[OUTPUT_COLUMNS].copy()

    keys = ["game_id", "team", "gsis_id"]

    merged = frozen.merge(
        actual,
        on=keys,
        suffixes=("_frozen", "_actual"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    if not merged["_merge"].eq("both").all():
        fail("F6_REGRESSION_IDENTITY_MISMATCH")

    text_cols = [
        "opponent_team",
        "full_name",
        "rate_source",
    ]

    for c in text_cols:
        a = merged[f"{c}_frozen"].fillna("").astype(str)
        b = merged[f"{c}_actual"].fillna("").astype(str)

        if not a.eq(b).all():
            fail(f"F6_REGRESSION_TEXT_MISMATCH:{c}")

    numeric = [
        "k_hist_games",
        "expected_fga",
        "fg_make_rate",
        "expected_fgm",
        "expected_xpa",
        "xp_make_rate",
        "expected_xpm",
    ]

    max_delta = 0.0

    for c in numeric:
        a = pd.to_numeric(
            merged[f"{c}_frozen"],
            errors="coerce",
        ).to_numpy(float)

        b = pd.to_numeric(
            merged[f"{c}_actual"],
            errors="coerce",
        ).to_numpy(float)

        null_mismatch = np.logical_xor(
            np.isnan(a),
            np.isnan(b),
        )

        if null_mismatch.any():
            fail(f"F6_REGRESSION_NULL_MISMATCH:{c}")

        valid = np.isfinite(a) & np.isfinite(b)

        if valid.any():
            delta = float(
                np.max(np.abs(a[valid] - b[valid]))
            )
            max_delta = max(max_delta, delta)

            if np.any(np.abs(a[valid] - b[valid]) > 1e-8):
                fail(
                    f"F6_REGRESSION_VALUE_MISMATCH:"
                    f"{c}:max_delta={delta}"
                )

    print("FROZEN_F6_MAX_ABS_DELTA =", max_delta)
    print("FROZEN_F6_REGRESSION=PASS")


def load_active_target() -> tuple[int, int, pd.DataFrame]:
    with sqlite3.connect(DB) as conn:
        games = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                home_team,
                away_team,
                completed
            FROM games
            WHERE game_type = 'REG'
            ORDER BY season, week, game_id
            """,
            conn,
        )

    if games.empty:
        fail("NO_REGULAR_SEASON_GAMES")

    games["season"] = pd.to_numeric(
        games["season"],
        errors="raise",
    ).astype(int)

    games["week"] = pd.to_numeric(
        games["week"],
        errors="raise",
    ).astype(int)

    current_season = int(games["season"].max())

    season_games = games[
        games["season"].eq(current_season)
    ].copy()

    unfinished = season_games[
        ~season_games["completed"].fillna(0).astype(bool)
    ].copy()

    if unfinished.empty:
        fail("NO_UNFINISHED_CURRENT_SEASON_GAMES")

    target_week = int(unfinished["week"].min())

    target = unfinished[
        unfinished["week"].eq(target_week)
    ].copy()

    if target.empty:
        fail("EMPTY_ACTIVE_TARGET")

    return current_season, target_week, target


def resolve_current_kickers(
    season: int,
    week: int,
    games: pd.DataFrame,
) -> pd.DataFrame:
    with sqlite3.connect(DB) as conn:
        roster = pd.read_sql_query(
            """
            SELECT
                season,
                week,
                team,
                gsis_id,
                full_name,
                position,
                status
            FROM weekly_rosters
            WHERE season = ?
              AND week = ?
              AND UPPER(position) = 'K'
            """,
            conn,
            params=(season, week),
        )

        depth = pd.read_sql_query(
            """
            SELECT
                snapshot_dt,
                team,
                player_name,
                gsis_id,
                pos_name,
                pos_abb,
                pos_slot,
                pos_rank
            FROM depth_charts
            WHERE UPPER(COALESCE(pos_abb, '')) IN ('PK', 'K')
               OR UPPER(COALESCE(pos_name, '')) LIKE '%KICKER%'
            """,
            conn,
        )

    if roster.empty:
        # Pregame target weeks may not yet exist in weekly_rosters.
        # In that specific case, use the latest current depth-chart
        # kicker authority. Identity remains exact-GSIS and fail-closed.
        if depth.empty:
            fail(
                f"NO_KICKER_ROSTER_OR_DEPTH:{season}:{week}"
            )

        depth["team"] = depth["team"].map(canon_team)
        depth["snapshot_dt"] = (
            depth["snapshot_dt"]
            .fillna("")
            .astype(str)
        )
        depth["gsis_id"] = (
            depth["gsis_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        depth["player_name"] = (
            depth["player_name"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        depth["pos_rank"] = pd.to_numeric(
            depth["pos_rank"],
            errors="coerce",
        )

        game_rows = []
        for game in games.itertuples(index=False):
            home = canon_team(game.home_team)
            away = canon_team(game.away_team)
            game_rows.extend(
                [
                    {
                        "game_id": game.game_id,
                        "team": home,
                        "opponent_team": away,
                    },
                    {
                        "game_id": game.game_id,
                        "team": away,
                        "opponent_team": home,
                    },
                ]
            )

        target_teams = pd.DataFrame(game_rows)

        if target_teams["team"].duplicated().any():
            fail("DUPLICATE_TARGET_TEAM")

        resolved = []

        for target in target_teams.itertuples(index=False):
            team_depth = depth[
                depth["team"].eq(target.team)
            ].copy()

            if team_depth.empty:
                fail(
                    "PRIMARY_KICKER_DEPTH_AUTHORITY_MISSING:"
                    f"{target.team}"
                )

            latest_snapshot = (
                team_depth["snapshot_dt"].max()
            )
            team_depth = team_depth[
                team_depth["snapshot_dt"].eq(
                    latest_snapshot
                )
            ].copy()

            if team_depth["pos_rank"].isna().any():
                fail(
                    "PRIMARY_KICKER_DEPTH_RANK_MISSING:"
                    f"{target.team}"
                )

            best_rank = team_depth["pos_rank"].min()
            primary = team_depth[
                team_depth["pos_rank"].eq(best_rank)
            ].copy()

            if len(primary) != 1:
                fail(
                    "PRIMARY_KICKER_DEPTH_AMBIGUOUS:"
                    f"{target.team}:rank={best_rank}:"
                    f"count={len(primary)}"
                )

            selected = primary.iloc[0]
            gsis_id = str(selected["gsis_id"]).strip()
            full_name = str(
                selected["player_name"]
            ).strip()

            if not gsis_id or not full_name:
                fail(
                    "PRIMARY_KICKER_IDENTITY_MISSING:"
                    f"{target.team}"
                )

            resolved.append(
                {
                    "game_id": target.game_id,
                    "team": target.team,
                    "opponent_team":
                        target.opponent_team,
                    "gsis_id": gsis_id,
                    "full_name": full_name,
                }
            )

        current = pd.DataFrame(resolved)

        if len(current) != len(target_teams):
            fail(
                "PRIMARY_KICKER_TARGET_COUNT_MISMATCH:"
                f"{len(current)}!={len(target_teams)}"
            )

        if current["team"].duplicated().any():
            fail("DUPLICATE_PRIMARY_KICKER_TEAM")

        if current["gsis_id"].duplicated().any():
            fail("DUPLICATE_PRIMARY_KICKER_GSIS")

        print(
            "PRIMARY_KICKER_AUTHORITY="
            "CURRENT_DEPTH_EXACT_GSIS"
        )
        print(
            "PRIMARY_KICKER_DEPTH_TARGETS="
            f"{len(current)}"
        )

        return current

    roster["team"] = roster["team"].map(canon_team)
    roster["gsis_id"] = (
        roster["gsis_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    roster["status"] = (
        roster["status"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    active = roster[
        roster["status"].eq("ACT")
    ].copy()

    game_rows = []

    for game in games.itertuples(index=False):
        home = canon_team(game.home_team)
        away = canon_team(game.away_team)

        game_rows.extend(
            [
                {
                    "game_id": game.game_id,
                    "team": home,
                    "opponent_team": away,
                },
                {
                    "game_id": game.game_id,
                    "team": away,
                    "opponent_team": home,
                },
            ]
        )

    target_teams = pd.DataFrame(game_rows)

    if target_teams["team"].duplicated().any():
        fail("DUPLICATE_TARGET_TEAM")

    resolved = []

    for target in target_teams.itertuples(index=False):
        team = target.team

        candidates = active[
            active["team"].eq(team)
        ].copy()

        if len(candidates) == 0:
            fail(
                "PRIMARY_KICKER_AUTHORITY_FAILURE:"
                f"{team}=0"
            )

        if len(candidates) == 1:
            selected = candidates.iloc[0]

        else:
            if depth.empty:
                fail(
                    "PRIMARY_KICKER_DEPTH_AUTHORITY_MISSING:"
                    f"{team}"
                )

            team_depth = depth[
                depth["team"].map(canon_team).eq(team)
            ].copy()

            if team_depth.empty:
                fail(
                    "PRIMARY_KICKER_DEPTH_AUTHORITY_MISSING:"
                    f"{team}"
                )

            team_depth["snapshot_dt"] = (
                team_depth["snapshot_dt"]
                .fillna("")
                .astype(str)
            )

            latest_snapshot = (
                team_depth["snapshot_dt"].max()
            )

            team_depth = team_depth[
                team_depth["snapshot_dt"].eq(
                    latest_snapshot
                )
            ].copy()

            team_depth["gsis_id"] = (
                team_depth["gsis_id"]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            active_ids = set(
                candidates["gsis_id"].tolist()
            )

            team_depth = team_depth[
                team_depth["gsis_id"].isin(active_ids)
            ].copy()

            if team_depth.empty:
                fail(
                    "PRIMARY_KICKER_DEPTH_NO_ACTIVE_MATCH:"
                    f"{team}"
                )

            team_depth["pos_rank"] = pd.to_numeric(
                team_depth["pos_rank"],
                errors="coerce",
            )

            if team_depth["pos_rank"].isna().any():
                fail(
                    "PRIMARY_KICKER_DEPTH_RANK_MISSING:"
                    f"{team}"
                )

            best_rank = team_depth["pos_rank"].min()

            primary = team_depth[
                team_depth["pos_rank"].eq(best_rank)
            ].copy()

            if len(primary) != 1:
                fail(
                    "PRIMARY_KICKER_DEPTH_AMBIGUOUS:"
                    f"{team}:rank={best_rank}:"
                    f"count={len(primary)}"
                )

            primary_id = str(
                primary.iloc[0]["gsis_id"]
            ).strip()

            selected_rows = candidates[
                candidates["gsis_id"].eq(primary_id)
            ]

            if len(selected_rows) != 1:
                fail(
                    "PRIMARY_KICKER_EXACT_IDENTITY_FAILURE:"
                    f"{team}:{primary_id}:"
                    f"count={len(selected_rows)}"
                )

            selected = selected_rows.iloc[0]

            print(
                "PRIMARY_KICKER_DEPTH_RESOLUTION "
                f"team={team} "
                f"active_candidates={len(candidates)} "
                f"selected={selected['full_name']} "
                f"gsis_id={selected['gsis_id']} "
                f"depth_rank={int(best_rank)}"
            )

        gsis_id = str(selected["gsis_id"]).strip()
        full_name = str(selected["full_name"]).strip()

        if not gsis_id or not full_name:
            fail(
                "PRIMARY_KICKER_IDENTITY_MISSING:"
                f"{team}"
            )

        resolved.append(
            {
                "game_id": target.game_id,
                "team": team,
                "opponent_team": target.opponent_team,
                "gsis_id": gsis_id,
                "full_name": full_name,
            }
        )

    current = pd.DataFrame(resolved)

    if len(current) != len(target_teams):
        fail(
            "PRIMARY_KICKER_TARGET_COUNT_MISMATCH:"
            f"{len(current)}!={len(target_teams)}"
        )

    if current["team"].duplicated().any():
        fail("DUPLICATE_PRIMARY_KICKER_TEAM")

    if current["gsis_id"].duplicated().any():
        fail("DUPLICATE_PRIMARY_KICKER_GSIS")

    return current


def build_incremental_2026_targets(
    season: int,
    target_week: int,
) -> pd.DataFrame:
    import nflreadpy as nfl

    with sqlite3.connect(DB) as conn:
        completed = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                home_team,
                away_team,
                completed
            FROM games
            WHERE season = ?
              AND game_type = 'REG'
              AND week < ?
              AND completed = 1
            ORDER BY week, game_id
            """,
            conn,
            params=(season, target_week),
        )

        roster = pd.read_sql_query(
            """
            SELECT
                season,
                week,
                team,
                gsis_id,
                full_name,
                position,
                status
            FROM weekly_rosters
            WHERE season = ?
              AND week < ?
              AND UPPER(position) = 'K'
              AND UPPER(status) = 'ACT'
            """,
            conn,
            params=(season, target_week),
        )

    if completed.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "game_id",
                "kicker_player_id",
                "player_display_name",
                "team",
                "fga",
                "fgm",
                "xpa",
                "xpm",
                "fg_distance_avg",
                "fg_distance_min",
                "fg_distance_max",
                "kicking_points_made",
            ]
        )

    roster["team"] = roster["team"].map(canon_team)

    # Build authoritative kicker-game population before touching PBP.
    population_rows = []

    for game in completed.itertuples(index=False):
        game_week = int(game.week)

        for raw_team in [game.home_team, game.away_team]:
            team = canon_team(raw_team)

            candidates = roster[
                roster["week"].eq(game_week)
                & roster["team"].eq(team)
            ].copy()

            if len(candidates) != 1:
                fail(
                    "HISTORICAL_PRIMARY_KICKER_AUTHORITY_FAILURE:"
                    f"game={game.game_id}:team={team}:"
                    f"active={len(candidates)}"
                )

            k = candidates.iloc[0]

            population_rows.append(
                {
                    "season": int(game.season),
                    "week": game_week,
                    "game_id": str(game.game_id),
                    "kicker_player_id": clean(k["gsis_id"]),
                    "player_display_name": clean(k["full_name"]),
                    "team": team,
                }
            )

    population = pd.DataFrame(population_rows)

    if population.duplicated(
        ["game_id", "team"]
    ).any():
        fail("CURRENT_KICKER_POPULATION_DUPLICATE_TEAM_GAME")

    if population["kicker_player_id"].eq("").any():
        fail("CURRENT_KICKER_POPULATION_BLANK_GSIS")

    raw = nfl.load_pbp(season)

    if hasattr(raw, "to_pandas"):
        raw = raw.to_pandas()

    required = [
        "season",
        "week",
        "game_id",
        "posteam",
        "kicker_player_id",
        "kicker_player_name",
        "field_goal_attempt",
        "field_goal_result",
        "extra_point_attempt",
        "extra_point_result",
        "kick_distance",
    ]

    missing = [
        c for c in required
        if c not in raw.columns
    ]

    if missing:
        fail(
            "CURRENT_PBP_SCHEMA_MISSING:"
            + ",".join(missing)
        )

    raw["season_num"] = pd.to_numeric(
        raw["season"],
        errors="coerce",
    )

    raw["week_num"] = pd.to_numeric(
        raw["week"],
        errors="coerce",
    )

    expected_games = set(
        completed["game_id"].astype(str)
    )

    prior = raw[
        raw["season_num"].eq(season)
        & raw["week_num"].lt(target_week)
        & raw["game_id"].astype(str).isin(expected_games)
    ].copy()

    actual_games = set(
        prior["game_id"].astype(str).unique()
    )

    missing_games = sorted(
        expected_games - actual_games
    )

    if missing_games:
        fail(
            "CURRENT_PBP_MISSING_COMPLETED_GAMES:"
            + ",".join(missing_games)
        )

    def flag(series):
        return (
            series.fillna(0)
            .astype(str)
            .isin(["1", "1.0", "True", "true"])
        )

    fg_mask = flag(prior["field_goal_attempt"])
    xp_mask = flag(prior["extra_point_attempt"])

    kicks = prior[fg_mask | xp_mask].copy()

    kicks["team"] = kicks["posteam"].map(canon_team)

    kicks["kicker_player_id"] = (
        kicks["kicker_player_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if kicks["kicker_player_id"].eq("").any():
        fail("CURRENT_PBP_KICK_EVENT_MISSING_GSIS")

    kicks["fg"] = flag(
        kicks["field_goal_attempt"]
    ).astype(int)

    kicks["xp"] = flag(
        kicks["extra_point_attempt"]
    ).astype(int)

    kicks["field_goal_result_clean"] = (
        kicks["field_goal_result"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    kicks["extra_point_result_clean"] = (
        kicks["extra_point_result"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    kicks["fgm"] = (
        kicks["fg"].eq(1)
        & kicks["field_goal_result_clean"].eq("made")
    ).astype(int)

    kicks["xpm"] = (
        kicks["xp"].eq(1)
        & kicks["extra_point_result_clean"].isin(
            ["good", "made"]
        )
    ).astype(int)

    kicks["kick_distance_num"] = pd.to_numeric(
        kicks["kick_distance"],
        errors="coerce",
    )

    # Reject kicking events attributed to a different kicker than the
    # authoritative ACT roster identity for that team-game.
    authority = population[
        [
            "game_id",
            "team",
            "kicker_player_id",
        ]
    ].rename(
        columns={
            "kicker_player_id":
                "authority_kicker_player_id",
        }
    )

    kicks = kicks.merge(
        authority,
        on=["game_id", "team"],
        how="left",
        validate="many_to_one",
    )

    if kicks[
        "authority_kicker_player_id"
    ].isna().any():
        fail("PBP_KICK_EVENT_WITHOUT_TEAM_GAME_AUTHORITY")

    identity_bad = kicks[
        kicks["kicker_player_id"].ne(
            kicks["authority_kicker_player_id"]
        )
    ]

    if not identity_bad.empty:
        detail = identity_bad[
            [
                "game_id",
                "team",
                "kicker_player_id",
                "authority_kicker_player_id",
            ]
        ].drop_duplicates()

        fail(
            "PBP_KICKER_IDENTITY_MISMATCH:"
            + detail.to_json(
                orient="records"
            )
        )

    # Distance aggregation must use FG attempts only.
    kicks["fg_distance"] = np.where(
        kicks["fg"].eq(1),
        kicks["kick_distance_num"],
        np.nan,
    )

    if kicks.empty:
        event_agg = pd.DataFrame(
            columns=[
                "game_id",
                "team",
                "kicker_player_id",
                "fga",
                "fgm",
                "xpa",
                "xpm",
                "fg_distance_avg",
                "fg_distance_min",
                "fg_distance_max",
            ]
        )
    else:
        event_agg = (
            kicks.groupby(
                [
                    "game_id",
                    "team",
                    "kicker_player_id",
                ],
                as_index=False,
            )
            .agg(
                fga=("fg", "sum"),
                fgm=("fgm", "sum"),
                xpa=("xp", "sum"),
                xpm=("xpm", "sum"),
                fg_distance_avg=(
                    "fg_distance",
                    "mean",
                ),
                fg_distance_min=(
                    "fg_distance",
                    "min",
                ),
                fg_distance_max=(
                    "fg_distance",
                    "max",
                ),
            )
        )

    # LEFT JOIN is intentional:
    # zero-event kicker-games remain in the population.
    out = population.merge(
        event_agg,
        on=[
            "game_id",
            "team",
            "kicker_player_id",
        ],
        how="left",
        validate="one_to_one",
    )

    for c in ["fga", "fgm", "xpa", "xpm"]:
        out[c] = (
            pd.to_numeric(
                out[c],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    out["kicking_points_made"] = (
        3 * out["fgm"]
        + out["xpm"]
    )

    expected_rows = len(completed) * 2

    if len(out) != expected_rows:
        fail(
            "CURRENT_KICKER_HISTORY_COVERAGE_FAILURE:"
            f"expected={expected_rows}:actual={len(out)}"
        )

    if out.duplicated(
        [
            "season",
            "week",
            "game_id",
            "kicker_player_id",
        ]
    ).any():
        fail("CURRENT_KICKER_HISTORY_DUPLICATE_IDENTITY")

    zero_events = (
        out["fga"].eq(0)
        & out["xpa"].eq(0)
    ).sum()

    print(
        "INCREMENTAL_COMPLETED_GAMES =",
        len(completed),
    )
    print(
        "INCREMENTAL_KICKER_ROWS =",
        len(out),
    )
    print(
        "INCREMENTAL_ZERO_EVENT_KICKERS =",
        int(zero_events),
    )

    return out


def build_current_matrix() -> None:
    _, features, team_features = load_contract()

    season, week, games = load_active_target()

    print(
        f"ACTIVE_TARGET season={season} "
        f"week={week} games={len(games)}"
    )

    current_kickers = resolve_current_kickers(
        season,
        week,
        games,
    )

    print(
        "PRIMARY_KICKERS =",
        len(current_kickers),
    )

    baseline = pd.read_parquet(FROZEN_TARGET)

    incremental = build_incremental_2026_targets(
        season,
        week,
    )

    history = pd.concat(
        [baseline, incremental],
        ignore_index=True,
        sort=False,
    )

    if history.duplicated(
        [
            "season",
            "week",
            "game_id",
            "kicker_player_id",
        ]
    ).any():
        fail("COMBINED_KICKER_HISTORY_DUPLICATE")

    current13 = current_features_from_history(
        history,
        current_kickers,
    )

    if not CURRENT_TEAM_ENV.is_file():
        fail(
            "MISSING_CURRENT_TEAM_ENV:"
            + str(CURRENT_TEAM_ENV)
        )

    env = pd.read_parquet(
        CURRENT_TEAM_ENV
    ).copy()

    env["team"] = env["team"].map(canon_team)

    missing_env_features = [
        c for c in team_features
        if c not in env.columns
    ]

    if missing_env_features:
        fail(
            "CURRENT_TEAM_ENV_FEATURE_MISSING:"
            + ",".join(missing_env_features)
        )

    if env.duplicated(
        ["game_id", "team"]
    ).any():
        fail("CURRENT_TEAM_ENV_DUPLICATE_IDENTITY")

    matrix = current13.merge(
        env[
            ["game_id", "team"]
            + team_features
        ],
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )

    if len(matrix) != len(current_kickers):
        fail("CURRENT_KICKER_MATRIX_ROW_LOSS")

    for c in team_features:
        values = pd.to_numeric(
            matrix[c],
            errors="coerce",
        )

        if values.isna().any():
            fail(
                f"CURRENT_KICKER_TEAM_FEATURE_NULL:{c}"
            )

        if not np.isfinite(
            values.to_numpy(float)
        ).all():
            fail(
                "CURRENT_KICKER_TEAM_FEATURE_NONFINITE:"
                + c
            )

    matrix_columns = [
        "game_id",
        "team",
        "opponent_team",
        "gsis_id",
        "full_name",
    ] + features

    matrix = matrix[matrix_columns].copy()

    if list(matrix.columns[5:]) != features:
        fail("CURRENT_KICKER_FEATURE_ORDER_DRIFT")

    if matrix.duplicated(
        ["game_id", "team"]
    ).any():
        fail("CURRENT_KICKER_MATRIX_DUPLICATE_TEAM")

    expected_rows = len(games) * 2

    if len(matrix) != expected_rows:
        fail(
            "CURRENT_KICKER_MATRIX_COVERAGE_FAILURE:"
            f"expected={expected_rows}:actual={len(matrix)}"
        )

    atomic_parquet(
        matrix,
        CURRENT_MATRIX,
    )

    print("MATRIX_ROWS =", len(matrix))
    print(
        "MATRIX_GAMES =",
        matrix["game_id"].nunique(),
    )
    print(
        "MATRIX_TEAMS =",
        matrix["team"].nunique(),
    )
    print(
        "MATRIX_ZERO_HISTORY_KICKERS =",
        int(
            pd.to_numeric(
                matrix["k_hist_games"],
                errors="coerce",
            ).eq(0).sum()
        ),
    )
    print("MATRIX_OUTPUT =", CURRENT_MATRIX)
    print(
        "MATRIX_SHA256 =",
        sha256(CURRENT_MATRIX),
    )
    print("CURRENT_KICKER_MATRIX=PASS")


def score_current_matrix() -> None:
    _, features, _ = load_contract()

    if not CURRENT_MATRIX.is_file():
        fail(
            "MISSING_CURRENT_KICKER_MATRIX:"
            + str(CURRENT_MATRIX)
        )

    matrix = pd.read_parquet(
        CURRENT_MATRIX
    ).copy()

    expected_columns = [
        "game_id",
        "team",
        "opponent_team",
        "gsis_id",
        "full_name",
    ] + features

    if list(matrix.columns) != expected_columns:
        fail("CURRENT_KICKER_MATRIX_SCHEMA_DRIFT")

    if matrix.empty:
        fail("EMPTY_CURRENT_KICKER_MATRIX")

    if matrix.duplicated(
        ["game_id", "team"]
    ).any():
        fail("CURRENT_KICKER_MATRIX_DUPLICATE_TEAM")

    for c in features:
        values = pd.to_numeric(
            matrix[c],
            errors="coerce",
        )

        # Model contract uses SimpleImputer for nullable
        # kicker-history features. Infinity is never allowed.
        finite_or_null = (
            values.isna()
            | np.isfinite(
                values.fillna(0).to_numpy(float)
            )
        )

        if not finite_or_null.all():
            fail(
                "CURRENT_KICKER_MATRIX_NONFINITE:"
                + c
            )

    try:
        import joblib
    except ImportError:
        fail(
            "MODEL_ENVIRONMENT_REQUIRED:"
            "joblib/sklearn unavailable"
        )

    fga_model = joblib.load(FGA_MODEL)
    xpa_model = joblib.load(XPA_MODEL)

    X = matrix[features]

    expected_fga = np.maximum(
        np.asarray(
            fga_model.predict(X),
            dtype=float,
        ),
        0.0,
    )

    expected_xpa = np.maximum(
        np.asarray(
            xpa_model.predict(X),
            dtype=float,
        ),
        0.0,
    )

    out = matrix[
        [
            "game_id",
            "team",
            "opponent_team",
            "gsis_id",
            "full_name",
            "k_hist_games",
            "k_fg_pct_prior",
            "k_xp_pct_prior",
        ]
    ].copy()

    fg_established = (
        out["k_hist_games"].gt(0)
        & out["k_fg_pct_prior"].notna()
    )

    xp_established = (
        out["k_hist_games"].gt(0)
        & out["k_xp_pct_prior"].notna()
    )

    # Preserve frozen F6 label semantics.
    out["rate_source"] = np.where(
        out["k_hist_games"].gt(0),
        "KICKER_HISTORICAL_RATE",
        "ALL_PRIOR_HISTORY",
    )

    out["fg_make_rate"] = np.where(
        fg_established,
        out["k_fg_pct_prior"],
        ZERO_HISTORY_FG_RATE,
    )

    out["xp_make_rate"] = np.where(
        xp_established,
        out["k_xp_pct_prior"],
        ZERO_HISTORY_XP_RATE,
    )

    out["expected_fga"] = expected_fga
    out["expected_fgm"] = (
        out["expected_fga"]
        * out["fg_make_rate"]
    )

    out["expected_xpa"] = expected_xpa
    out["expected_xpm"] = (
        out["expected_xpa"]
        * out["xp_make_rate"]
    )

    out = out[OUTPUT_COLUMNS].copy()

    numeric = [
        "k_hist_games",
        "expected_fga",
        "fg_make_rate",
        "expected_fgm",
        "expected_xpa",
        "xp_make_rate",
        "expected_xpm",
    ]

    for c in numeric:
        values = pd.to_numeric(
            out[c],
            errors="coerce",
        )

        if values.isna().any():
            fail(
                f"CURRENT_KICKER_OUTPUT_NULL:{c}"
            )

        if not np.isfinite(
            values.to_numpy(float)
        ).all():
            fail(
                f"CURRENT_KICKER_OUTPUT_NONFINITE:{c}"
            )

        if (values < 0).any():
            fail(
                f"CURRENT_KICKER_OUTPUT_NEGATIVE:{c}"
            )

    if out.duplicated(
        ["game_id", "team"]
    ).any():
        fail("CURRENT_KICKER_OUTPUT_DUPLICATE_TEAM")

    atomic_parquet(
        out,
        OUTPUT,
    )

    print("ROWS =", len(out))
    print(
        "GAMES =",
        out["game_id"].nunique(),
    )
    print(
        "TEAMS =",
        out["team"].nunique(),
    )
    print(
        "ZERO_HISTORY_KICKERS =",
        int(out["k_hist_games"].eq(0).sum()),
    )
    print("OUTPUT =", OUTPUT)
    print("SHA256 =", sha256(OUTPUT))
    print(
        "CURRENT_KICKER_STAT_FORECAST=PASS"
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--regression-week1",
        action="store_true",
    )

    group.add_argument(
        "--build-current-matrix",
        action="store_true",
    )

    group.add_argument(
        "--score-current-matrix",
        action="store_true",
    )

    args = parser.parse_args()

    if args.regression_week1:
        regression()
    elif args.build_current_matrix:
        build_current_matrix()
    elif args.score_current_matrix:
        score_current_matrix()
    else:
        fail("NO_EXECUTION_MODE")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"FAIL_CLOSED:{type(exc).__name__}:{exc}",
            file=sys.stderr,
        )
        raise
