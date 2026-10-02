#!/usr/bin/env python3

"""
current_dst_stat_forecast.py

Current-season inference bridge for the frozen Stage23G4 D/ST
stat-forecast candidate.

Architecture
------------
- Schedule authority: data/nfl.db -> games
- Feature authority: nfl_current_dst_features.parquet
- Model authority: frozen Stage23G4 serialized models
- Fumble recovery policy: STRICT_PRIOR_3_GAME_BASELINE
- No model training
- No model parameter changes
- No post-prediction clipping
- No fuzzy identity matching
- No historical table writes

The active target is the earliest unfinished 2026 REG week.
Only unfinished games from that week are scored.

Output contract
---------------
game_id
season
week
team
is_home
expected_sacks
expected_interceptions
expected_points_allowed
expected_defensive_tds
expected_fumble_recoveries
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import os
import sqlite3
import tempfile

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

DATABASE_PATH = ROOT / "data" / "nfl.db"

FEATURE_PATH = (
    ROOT
    / "data"
    / "parquet"
    / "nfl_current_dst_features.parquet"
)

MODEL_ROOT = (
    ROOT
    / "data"
    / "model_candidates"
    / "stat_forecast"
    / "stage23g4_20260912T161643Z"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "parquet"
    / "nfl_current_dst_stat_forecasts.parquet"
)

CURRENT_SEASON = 2026

FEATURES = [
    "def_fum_rec_avg_3",
    "def_ints_avg_5",
    "def_pa_avg_5",
    "def_takeaways_avg_3",
    "def_takeaways_avg_5",
    "is_home",
    "opp_pass_attempts_avg_5",
    "opp_passing_tds_avg_3",
    "opp_passing_tds_avg_5",
    "opp_rush_attempts_avg_5",
    "opp_rush_attempts_last",
    "opp_rushing_tds_avg_5",
    "opp_rushing_tds_last",
    "opp_rushing_yards_avg_5",
    "opp_rushing_yards_last",
    "opp_sacks_allowed_avg_5",
    "opp_sacks_allowed_last",
]

MODEL_FILES = {
    "expected_sacks":
        "dst_sacks_model.joblib",

    "expected_interceptions":
        "dst_interceptions_model.joblib",

    "expected_points_allowed":
        "dst_opponent_score_model.joblib",

    "expected_defensive_tds":
        "dst_defensive_tds_model.joblib",
}

OUTPUT_COLUMNS = [
    "game_id",
    "season",
    "week",
    "team",
    "is_home",
    "expected_sacks",
    "expected_interceptions",
    "expected_points_allowed",
    "expected_defensive_tds",
    "expected_fumble_recoveries",
]


def section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_active_schedule() -> tuple[int, pd.DataFrame]:
    with sqlite3.connect(DATABASE_PATH) as conn:
        games = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                game_type,
                home_team,
                away_team,
                completed
            FROM games
            WHERE season = ?
              AND game_type = 'REG'
            ORDER BY week, game_id
            """,
            conn,
            params=(CURRENT_SEASON,),
        )

    if games.empty:
        raise RuntimeError(
            "NO_CURRENT_SEASON_REGULAR_GAMES"
        )

    games["week"] = pd.to_numeric(
        games["week"],
        errors="coerce",
    )

    games["completed_num"] = pd.to_numeric(
        games["completed"],
        errors="coerce",
    ).fillna(0)

    unfinished = games[
        games["completed_num"].ne(1)
    ].copy()

    if unfinished.empty:
        raise RuntimeError(
            "NO_UNFINISHED_CURRENT_SEASON_REGULAR_GAMES"
        )

    active_week = int(
        unfinished["week"].min()
    )

    target = unfinished[
        unfinished["week"].eq(active_week)
    ].copy()

    if target.empty:
        raise RuntimeError(
            "ACTIVE_WEEK_HAS_ZERO_UNFINISHED_GAMES"
        )

    if target["game_id"].duplicated().any():
        raise RuntimeError(
            "ACTIVE_SCHEDULE_DUPLICATE_GAME_ID"
        )

    return active_week, target


def expected_team_population(
    schedule: pd.DataFrame,
) -> pd.DataFrame:

    home = schedule[
        ["game_id", "season", "week", "home_team"]
    ].rename(
        columns={"home_team": "team"}
    )

    home["schedule_is_home"] = 1

    away = schedule[
        ["game_id", "season", "week", "away_team"]
    ].rename(
        columns={"away_team": "team"}
    )

    away["schedule_is_home"] = 0

    expected = pd.concat(
        [home, away],
        ignore_index=True,
    )

    expected["team"] = (
        expected["team"]
        .astype(str)
        .str.strip()
    )

    if expected[
        ["game_id", "team"]
    ].duplicated().any():
        raise RuntimeError(
            "EXPECTED_TEAM_POPULATION_DUPLICATES"
        )

    return expected


def load_current_features(
    active_week: int,
    expected: pd.DataFrame,
) -> pd.DataFrame:

    if not FEATURE_PATH.is_file():
        raise FileNotFoundError(
            f"MISSING_CURRENT_DST_FEATURES:{FEATURE_PATH}"
        )

    df = pd.read_parquet(FEATURE_PATH)

    required = {
        "game_id",
        "season",
        "week",
        "team",
        "is_home",
        *FEATURES,
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "CURRENT_DST_FEATURE_SCHEMA_MISSING:"
            + ",".join(missing)
        )

    season_num = pd.to_numeric(
        df["season"],
        errors="coerce",
    )

    week_num = pd.to_numeric(
        df["week"],
        errors="coerce",
    )

    df = df[
        season_num.eq(CURRENT_SEASON)
        & week_num.eq(active_week)
    ].copy()

    if df.empty:
        raise RuntimeError(
            "ACTIVE_WEEK_DST_FEATURES_EMPTY"
        )

    df["team"] = (
        df["team"]
        .astype(str)
        .str.strip()
    )

    if df[
        ["game_id", "team"]
    ].duplicated().any():
        raise RuntimeError(
            "ACTIVE_DST_FEATURE_IDENTITY_DUPLICATES"
        )

    feature_keys = set(
        map(
            tuple,
            df[
                ["game_id", "team"]
            ].astype(str).to_numpy(),
        )
    )

    expected_keys = set(
        map(
            tuple,
            expected[
                ["game_id", "team"]
            ].astype(str).to_numpy(),
        )
    )

    missing_keys = sorted(
        expected_keys - feature_keys
    )

    extra_keys = sorted(
        feature_keys - expected_keys
    )

    if missing_keys:
        raise RuntimeError(
            "ACTIVE_DST_FEATURE_MISSING_SCHEDULE_TEAMS:"
            f"{missing_keys}"
        )

    if extra_keys:
        raise RuntimeError(
            "ACTIVE_DST_FEATURE_EXTRA_TEAMS:"
            f"{extra_keys}"
        )

    schedule_home = expected[
        ["game_id", "team", "schedule_is_home"]
    ]

    df = df.merge(
        schedule_home,
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )

    if df["schedule_is_home"].isna().any():
        raise RuntimeError(
            "DST_FEATURE_SCHEDULE_HOME_IDENTITY_FAILURE"
        )

    feature_home = pd.to_numeric(
        df["is_home"],
        errors="coerce",
    )

    schedule_home_num = pd.to_numeric(
        df["schedule_is_home"],
        errors="coerce",
    )

    if feature_home.isna().any():
        raise RuntimeError(
            "DST_FEATURE_IS_HOME_NONNUMERIC"
        )

    if not feature_home.eq(
        schedule_home_num
    ).all():
        raise RuntimeError(
            "DST_FEATURE_IS_HOME_SCHEDULE_MISMATCH"
        )

    x = df[FEATURES].copy()

    for col in FEATURES:
        x[col] = pd.to_numeric(
            x[col],
            errors="coerce",
        )

    if x.isna().any().any():
        bad = (
            x.columns[
                x.isna().any()
            ].tolist()
        )

        raise RuntimeError(
            "DST_FEATURE_NULL_VALUES:"
            + ",".join(bad)
        )

    if not np.isfinite(
        x.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "DST_FEATURE_NONFINITE_VALUES"
        )

    return (
        df
        .sort_values(
            ["game_id", "team"]
        )
        .reset_index(drop=True)
    )


def load_models() -> dict[str, object]:
    models = {}

    for target, filename in MODEL_FILES.items():
        path = MODEL_ROOT / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"MISSING_FROZEN_G4_MODEL:{path}"
            )

        model = joblib.load(path)

        if not hasattr(model, "predict"):
            raise RuntimeError(
                f"G4_MODEL_HAS_NO_PREDICT:{filename}"
            )

        names = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if names is None:
            raise RuntimeError(
                f"G4_MODEL_MISSING_FEATURE_NAMES:{filename}"
            )

        names = list(names)

        if names != FEATURES:
            raise RuntimeError(
                f"G4_MODEL_FEATURE_CONTRACT_DRIFT:{filename}"
            )

        n_features = getattr(
            model,
            "n_features_in_",
            None,
        )

        if int(n_features) != len(FEATURES):
            raise RuntimeError(
                f"G4_MODEL_FEATURE_COUNT_DRIFT:{filename}"
            )

        models[target] = model

    return models


def score(
    features: pd.DataFrame,
    models: dict[str, object],
) -> pd.DataFrame:

    x = features[FEATURES].copy()

    for col in FEATURES:
        x[col] = pd.to_numeric(
            x[col],
            errors="raise",
        )

    out = features[
        [
            "game_id",
            "season",
            "week",
            "team",
            "is_home",
        ]
    ].copy()

    for target, model in models.items():
        pred = np.asarray(
            model.predict(x),
            dtype=float,
        )

        if len(pred) != len(out):
            raise RuntimeError(
                f"G4_PREDICTION_LENGTH_FAILURE:{target}"
            )

        if not np.isfinite(pred).all():
            raise RuntimeError(
                f"G4_NONFINITE_PREDICTION:{target}"
            )

        if (pred < 0).any():
            raise RuntimeError(
                f"G4_NEGATIVE_PREDICTION:{target}"
            )

        out[target] = pred

    # Frozen G4 regression proved this is exactly equivalent
    # to STRICT_PRIOR_3_GAME_BASELINE.
    out["expected_fumble_recoveries"] = (
        pd.to_numeric(
            features["def_fum_rec_avg_3"],
            errors="raise",
        ).to_numpy(dtype=float)
    )

    if not np.isfinite(
        out[
            [
                "expected_sacks",
                "expected_interceptions",
                "expected_points_allowed",
                "expected_defensive_tds",
                "expected_fumble_recoveries",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "G4_OUTPUT_NONFINITE"
        )

    if (
        out[
            [
                "expected_sacks",
                "expected_interceptions",
                "expected_points_allowed",
                "expected_defensive_tds",
                "expected_fumble_recoveries",
            ]
        ] < 0
    ).any().any():
        raise RuntimeError(
            "G4_OUTPUT_NEGATIVE"
        )

    return out[OUTPUT_COLUMNS]


def atomic_write_parquet(
    df: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    os.close(fd)

    tmp = Path(tmp_name)

    try:
        df.to_parquet(
            tmp,
            index=False,
        )

        os.replace(
            tmp,
            path,
        )

    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> None:
    section(
        "CURRENT STAGE23G4 D/ST STAT FORECAST"
    )

    print(
        "Model authority:",
        MODEL_ROOT,
    )
    print(
        "Feature authority:",
        FEATURE_PATH,
    )
    print(
        "Training: DISABLED"
    )
    print(
        "Post-prediction clipping: DISABLED"
    )

    active_week, schedule = (
        load_active_schedule()
    )

    expected = expected_team_population(
        schedule
    )

    section("SCHEDULE TARGET")

    print("Season:", CURRENT_SEASON)
    print("Active week:", active_week)
    print(
        "Unfinished games:",
        schedule["game_id"].nunique(),
    )
    print(
        "Expected team rows:",
        len(expected),
    )

    features = load_current_features(
        active_week,
        expected,
    )

    section("FEATURE CONTRACT")

    print("Rows:", len(features))
    print(
        "Games:",
        features["game_id"].nunique(),
    )
    print(
        "Teams:",
        features["team"].nunique(),
    )
    print(
        "Features:",
        len(FEATURES),
    )

    models = load_models()

    output = score(
        features,
        models,
    )

    if len(output) != len(expected):
        raise RuntimeError(
            "G4_OUTPUT_ROW_COUNT_FAILURE"
        )

    if output[
        ["game_id", "team"]
    ].duplicated().any():
        raise RuntimeError(
            "G4_OUTPUT_IDENTITY_DUPLICATES"
        )

    if (
        output["game_id"].nunique()
        != schedule["game_id"].nunique()
    ):
        raise RuntimeError(
            "G4_OUTPUT_GAME_COVERAGE_FAILURE"
        )

    if (
        output["team"].nunique()
        != expected["team"].nunique()
    ):
        raise RuntimeError(
            "G4_OUTPUT_TEAM_COVERAGE_FAILURE"
        )

    output = (
        output
        .sort_values(
            ["game_id", "team"]
        )
        .reset_index(drop=True)
    )

    atomic_write_parquet(
        output,
        OUTPUT_PATH,
    )

    section("CURRENT G4 OUTPUT")

    print("Rows:", len(output))
    print(
        "Games:",
        output["game_id"].nunique(),
    )
    print(
        "Teams:",
        output["team"].nunique(),
    )

    print()
    print(
        output[
            [
                "team",
                "expected_sacks",
                "expected_interceptions",
                "expected_points_allowed",
                "expected_defensive_tds",
                "expected_fumble_recoveries",
            ]
        ]
        .sort_values("team")
        .to_string(index=False)
    )

    print()
    print(
        "Parquet:",
        OUTPUT_PATH,
    )
    print(
        "SHA256:",
        sha256_file(OUTPUT_PATH),
    )

    section(
        "CURRENT STAGE23G4 D/ST STAT FORECAST PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 88)
        print(
            "CURRENT STAGE23G4 D/ST STAT FORECAST FAILED"
        )
        print("=" * 88)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        raise
