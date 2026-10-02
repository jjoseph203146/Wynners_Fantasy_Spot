#!/usr/bin/env python3

"""
WFS NFL Season & Playoff Simulator
Simulator V1 — Live Deployment Contract Audit

READ ONLY with respect to production.

Purpose
-------
Determine whether the validated EXP003 historical model feature contract
can actually be constructed for all remaining 2026 regular-season games.

This script:
1. Loads the frozen EXP003 audit.
2. Reconstructs the 92-feature margin contract.
3. Loads the frozen 64-feature totals contract.
4. Finds current project tabular artifacts containing 2026 game data.
5. Identifies candidate artifacts containing model-feature overlap.
6. Audits feature availability.
7. Separates:
   - directly available features
   - missing features
   - likely recursive/state features
   - near-term contextual features
8. Audits remaining 2026 schedule coverage where a schedule source can
   be identified.
9. Writes ONLY isolated audit artifacts under processed/simulator_v1/.

No model fitting.
No production mutation.
"""

from pathlib import Path
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")
PROCESSED = ROOT / "processed"
DATA = ROOT / "data"
OUTDIR = PROCESSED / "simulator_v1"

TRAINING_SOURCE = (
    PROCESSED
    / "forecast_v1_team_game_training.csv"
)

EXP003_AUDIT = (
    OUTDIR
    / "simulator_v1_experiment_003_audit.json"
)

REPORT_JSON = (
    OUTDIR
    / "simulator_v1_live_deployment_contract_audit.json"
)

CANDIDATES_CSV = (
    OUTDIR
    / "simulator_v1_live_feature_source_candidates.csv"
)

FEATURE_COVERAGE_CSV = (
    OUTDIR
    / "simulator_v1_live_feature_coverage.csv"
)

SCHEDULE_CANDIDATES_CSV = (
    OUTDIR
    / "simulator_v1_schedule_source_candidates.csv"
)

SEASON = 2026

MARKET_COLUMNS = {
    "market_home_spread_raw",
    "market_total",
    "team_moneyline",
    "opponent_moneyline",
}

TARGET_COLUMNS = {
    "target_team_points",
    "target_opponent_points",
    "target_margin",
    "target_total_points",
    "target_win",
    "target_result",
}

IDENTITY_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "game_date",
    "weekday",
    "gametime",
    "team",
    "opponent_team",
}

DIRECT_IDENTITY_EXCLUSIONS = {
    "coach",
    "opponent_coach",
    "team_qb_name",
    "opponent_qb_name",
}

DEFERRED_COLUMNS = {
    "temp",
    "wind",
}

TABULAR_SUFFIXES = {
    ".csv",
    ".parquet",
    ".pq",
}

SKIP_DIR_PARTS = {
    "venv",
    ".venv",
    ".git",
    "__pycache__",
    "backups",
    "backup",
    "archive",
    "archives",
}


def fail(message):
    print(f"FAIL | {message}")
    raise SystemExit(1)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def safe_columns(path):
    try:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return list(
                pd.read_csv(
                    path,
                    nrows=5,
                ).columns
            )

        if suffix in {
            ".parquet",
            ".pq",
        }:
            return list(
                pd.read_parquet(
                    path,
                ).columns
            )

    except Exception:
        return []

    return []


def safe_read(path):
    try:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path)

        if suffix in {
            ".parquet",
            ".pq",
        }:
            return pd.read_parquet(path)

    except Exception:
        return None

    return None


def classify_feature(name):
    low = name.lower()

    if (
        "injury" in low
        or "questionable" in low
        or "doubtful" in low
        or "_out_" in low
        or low.endswith("_out_count")
        or "qb_questionable" in low
    ):
        return "NEAR_TERM_INJURY_CONTEXT"

    if "coach_prior" in low:
        return "RECURSIVE_PRIOR_STATE"

    if (
        "avg_3" in low
        or "avg_5" in low
        or "_last" in low
        or "_trend" in low
        or "history_games" in low
    ):
        return "RECURSIVE_PRIOR_STATE"

    if "rest" in low:
        return "SCHEDULE_DERIVABLE"

    if "player_rows" in low:
        return "ROSTER_CONTEXT"

    return "OTHER_PREGAME_STATE"


def likely_schedule_columns(cols):
    cols = set(cols)

    signals = {
        "game_id",
        "season",
        "week",
    }

    team_pairs = [
        {"home_team", "away_team"},
        {"team", "opponent_team"},
    ]

    if not signals.issubset(cols):
        return False

    return any(
        pair.issubset(cols)
        for pair in team_pairs
    )


def season_mask(df):
    if "season" not in df.columns:
        return None

    season = pd.to_numeric(
        df["season"],
        errors="coerce",
    )

    return season == SEASON


def game_status_column(df):
    candidates = [
        "game_status",
        "status",
        "game_state",
        "state",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def final_mask(df):
    status_col = game_status_column(df)

    if status_col is None:
        return None

    status = (
        df[status_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return status.str.contains(
        r"FINAL|COMPLETE|COMPLETED",
        regex=True,
        na=False,
    )


def feature_nonnull_rate(df, feature):
    if feature not in df.columns:
        return None

    if len(df) == 0:
        return None

    return float(
        df[feature].notna().mean()
    )


print(
    "===== WFS SIMULATOR V1 "
    "LIVE DEPLOYMENT CONTRACT AUDIT ====="
)

if not TRAINING_SOURCE.exists():
    fail(
        f"training source missing: {TRAINING_SOURCE}"
    )

if not EXP003_AUDIT.exists():
    fail(
        f"EXP003 audit missing: {EXP003_AUDIT}"
    )

training_sha_before = sha256(
    TRAINING_SOURCE
)

exp003_sha_before = sha256(
    EXP003_AUDIT
)

with EXP003_AUDIT.open() as f:
    exp003 = json.load(f)

if (
    exp003.get("experiment")
    != "SIMULATOR_V1_EXPERIMENT_003"
):
    fail(
        "unexpected EXP003 audit identity"
    )

# ------------------------------------------------------------
# Reconstruct frozen margin contract.
# ------------------------------------------------------------

hist = pd.read_csv(
    TRAINING_SOURCE
)

home = hist[
    pd.to_numeric(
        hist["is_home"],
        errors="coerce",
    ) == 1
].copy()

dev = home[
    home["season"].isin(
        [2023, 2024]
    )
].copy()

excluded = (
    MARKET_COLUMNS
    | TARGET_COLUMNS
    | IDENTITY_COLUMNS
    | DIRECT_IDENTITY_EXCLUSIONS
    | DEFERRED_COLUMNS
)

margin_features = []

for col in home.columns:
    if col in excluded:
        continue

    if not pd.api.types.is_numeric_dtype(
        home[col]
    ):
        continue

    x = pd.to_numeric(
        dev[col],
        errors="coerce",
    )

    if x.notna().sum() == 0:
        continue

    if x.dropna().nunique() <= 1:
        continue

    margin_features.append(col)

margin_features = sorted(
    margin_features
)

if len(margin_features) != 92:
    fail(
        "EXP003 margin contract drift | "
        f"expected=92 actual={len(margin_features)}"
    )

total_features = (
    exp003
    .get("contract", {})
    .get(
        "total_target_specific_features",
        [],
    )
)

total_features = sorted(
    set(total_features)
)

if len(total_features) != 64:
    fail(
        "EXP003 totals contract drift | "
        f"expected=64 actual={len(total_features)}"
    )

all_required_features = sorted(
    set(margin_features)
    | set(total_features)
)

print(
    f"MARGIN_FEATURES={len(margin_features)}"
)
print(
    f"TOTAL_FEATURES={len(total_features)}"
)
print(
    f"UNION_FEATURES={len(all_required_features)}"
)

# ------------------------------------------------------------
# Search project tabular artifacts.
# ------------------------------------------------------------

roots = [
    PROCESSED,
    DATA,
]

paths = []

for root in roots:
    if not root.exists():
        continue

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in TABULAR_SUFFIXES:
            continue

        rel_parts = {
            p.lower()
            for p in path.relative_to(ROOT).parts
        }

        if rel_parts & SKIP_DIR_PARTS:
            continue

        if path == TRAINING_SOURCE:
            continue

        paths.append(path)

paths = sorted(
    set(paths)
)

print(
    f"TABULAR_ARTIFACTS_SCANNED={len(paths)}"
)

candidate_rows = []
schedule_rows = []

best_path = None
best_score = -1
best_columns = []

for path in paths:
    cols = safe_columns(path)

    if not cols:
        continue

    colset = set(cols)

    margin_overlap = sorted(
        set(margin_features) & colset
    )

    total_overlap = sorted(
        set(total_features) & colset
    )

    union_overlap = sorted(
        set(all_required_features)
        & colset
    )

    score = len(union_overlap)

    if score > 0:
        candidate_rows.append(
            {
                "path": str(
                    path.relative_to(ROOT)
                ),
                "column_count": len(cols),
                "margin_features_present":
                    len(margin_overlap),
                "margin_features_required":
                    len(margin_features),
                "total_features_present":
                    len(total_overlap),
                "total_features_required":
                    len(total_features),
                "union_features_present":
                    len(union_overlap),
                "union_features_required":
                    len(all_required_features),
                "has_game_id":
                    "game_id" in colset,
                "has_season":
                    "season" in colset,
                "has_week":
                    "week" in colset,
            }
        )

        if score > best_score:
            best_score = score
            best_path = path
            best_columns = cols

    if likely_schedule_columns(cols):
        schedule_rows.append(
            {
                "path": str(
                    path.relative_to(ROOT)
                ),
                "column_count": len(cols),
                "has_status":
                    any(
                        c in colset
                        for c in [
                            "game_status",
                            "status",
                            "game_state",
                            "state",
                        ]
                    ),
                "has_game_date":
                    "game_date" in colset,
                "has_kickoff":
                    any(
                        c in colset
                        for c in [
                            "kickoff",
                            "kickoff_time",
                            "gametime",
                            "game_time",
                        ]
                    ),
                "feature_overlap":
                    len(
                        set(
                            all_required_features
                        ) & colset
                    ),
            }
        )

candidates_df = pd.DataFrame(
    candidate_rows
)

if len(candidates_df):
    candidates_df = (
        candidates_df
        .sort_values(
            [
                "union_features_present",
                "margin_features_present",
                "total_features_present",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

schedule_df = pd.DataFrame(
    schedule_rows
)

if len(schedule_df):
    schedule_df = (
        schedule_df
        .sort_values(
            [
                "feature_overlap",
                "has_status",
                "has_game_date",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

candidates_df.to_csv(
    CANDIDATES_CSV,
    index=False,
)

schedule_df.to_csv(
    SCHEDULE_CANDIDATES_CSV,
    index=False,
)

print(
    f"FEATURE_SOURCE_CANDIDATES={len(candidates_df)}"
)
print(
    f"SCHEDULE_SOURCE_CANDIDATES={len(schedule_df)}"
)

print(
    "\n===== TOP FEATURE SOURCE CANDIDATES ====="
)

if len(candidates_df):
    print(
        candidates_df.head(15).to_string(
            index=False
        )
    )
else:
    print(
        "NO_FEATURE_SOURCE_CANDIDATES"
    )

print(
    "\n===== TOP SCHEDULE SOURCE CANDIDATES ====="
)

if len(schedule_df):
    print(
        schedule_df.head(15).to_string(
            index=False
        )
    )
else:
    print(
        "NO_SCHEDULE_SOURCE_CANDIDATES"
    )

# ------------------------------------------------------------
# Audit best live feature source.
# ------------------------------------------------------------

feature_rows = []

live_2026_rows = 0
best_rel = None

if best_path is not None:
    best_rel = str(
        best_path.relative_to(ROOT)
    )

    live = safe_read(
        best_path
    )

    if live is not None:
        smask = season_mask(live)

        if smask is not None:
            live_scope = live[
                smask
            ].copy()
        else:
            live_scope = live.copy()

        live_2026_rows = len(
            live_scope
        )

        for feature in all_required_features:
            present = (
                feature
                in live_scope.columns
            )

            rate = (
                feature_nonnull_rate(
                    live_scope,
                    feature,
                )
                if present
                else None
            )

            feature_rows.append(
                {
                    "feature":
                        feature,
                    "required_for_margin":
                        feature
                        in margin_features,
                    "required_for_total":
                        feature
                        in total_features,
                    "category":
                        classify_feature(
                            feature
                        ),
                    "present_in_best_source":
                        present,
                    "nonnull_rate":
                        rate,
                    "best_source":
                        best_rel,
                }
            )
else:
    for feature in all_required_features:
        feature_rows.append(
            {
                "feature":
                    feature,
                "required_for_margin":
                    feature
                    in margin_features,
                "required_for_total":
                    feature
                    in total_features,
                "category":
                    classify_feature(
                        feature
                    ),
                "present_in_best_source":
                    False,
                "nonnull_rate":
                    None,
                "best_source":
                    None,
            }
        )

feature_df = pd.DataFrame(
    feature_rows
)

feature_df.to_csv(
    FEATURE_COVERAGE_CSV,
    index=False,
)

present_count = int(
    feature_df[
        "present_in_best_source"
    ].sum()
)

missing_count = (
    len(feature_df)
    - present_count
)

margin_present = int(
    feature_df[
        feature_df[
            "required_for_margin"
        ]
    ][
        "present_in_best_source"
    ].sum()
)

total_present = int(
    feature_df[
        feature_df[
            "required_for_total"
        ]
    ][
        "present_in_best_source"
    ].sum()
)

print(
    "\n===== BEST LIVE FEATURE SOURCE ====="
)
print(
    f"BEST_SOURCE={best_rel}"
)
print(
    f"BEST_SOURCE_2026_ROWS={live_2026_rows}"
)
print(
    f"MARGIN_FEATURES_PRESENT={margin_present}/{len(margin_features)}"
)
print(
    f"TOTAL_FEATURES_PRESENT={total_present}/{len(total_features)}"
)
print(
    f"UNION_FEATURES_PRESENT={present_count}/{len(all_required_features)}"
)

# ------------------------------------------------------------
# Feature-category deployment analysis.
# ------------------------------------------------------------

print(
    "\n===== FEATURE CATEGORY AUDIT ====="
)

category_summary = []

for category, g in feature_df.groupby(
    "category",
    sort=True,
):
    required = len(g)
    present = int(
        g[
            "present_in_best_source"
        ].sum()
    )

    category_summary.append(
        {
            "category":
                category,
            "features":
                required,
            "present":
                present,
            "missing":
                required - present,
        }
    )

category_df = pd.DataFrame(
    category_summary
)

if len(category_df):
    print(
        category_df.to_string(
            index=False
        )
    )

print(
    "\n===== MISSING FEATURES ====="
)

missing_df = feature_df[
    ~feature_df[
        "present_in_best_source"
    ]
].copy()

if len(missing_df):
    print(
        missing_df[
            [
                "feature",
                "required_for_margin",
                "required_for_total",
                "category",
            ]
        ].to_string(
            index=False
        )
    )
else:
    print(
        "NONE"
    )

# ------------------------------------------------------------
# Schedule audit.
# ------------------------------------------------------------

schedule_best = None
schedule_scope = None
remaining_games = None
completed_games = None

if len(schedule_df):
    # Prefer schedule-like artifact with status, then date.
    ranked = schedule_df.copy()

    ranked["_score"] = (
        ranked["has_status"].astype(int) * 100
        + ranked["has_game_date"].astype(int) * 20
        + ranked["has_kickoff"].astype(int) * 10
        + ranked["feature_overlap"]
    )

    ranked = ranked.sort_values(
        "_score",
        ascending=False,
    )

    for _, row in ranked.iterrows():
        candidate = (
            ROOT
            / row["path"]
        )

        sdf = safe_read(
            candidate
        )

        if sdf is None:
            continue

        smask = season_mask(
            sdf
        )

        if smask is None:
            continue

        scope = sdf[
            smask
        ].copy()

        if len(scope) == 0:
            continue

        schedule_best = candidate
        schedule_scope = scope
        break

if schedule_scope is not None:
    fmask = final_mask(
        schedule_scope
    )

    if fmask is not None:
        completed_games = int(
            schedule_scope.loc[
                fmask,
                "game_id",
            ].nunique()
        )

        remaining_games = int(
            schedule_scope.loc[
                ~fmask,
                "game_id",
            ].nunique()
        )
    else:
        completed_games = None
        remaining_games = None

print(
    "\n===== 2026 SCHEDULE AUDIT ====="
)

print(
    "SCHEDULE_SOURCE="
    + (
        str(
            schedule_best.relative_to(
                ROOT
            )
        )
        if schedule_best is not None
        else "NONE"
    )
)

print(
    "COMPLETED_2026_GAMES="
    + (
        str(completed_games)
        if completed_games is not None
        else "UNKNOWN"
    )
)

print(
    "REMAINING_2026_GAMES="
    + (
        str(remaining_games)
        if remaining_games is not None
        else "UNKNOWN"
    )
)

# ------------------------------------------------------------
# Deployment decision.
# ------------------------------------------------------------

recursive_features = sorted(
    feature_df.loc[
        feature_df["category"]
        == "RECURSIVE_PRIOR_STATE",
        "feature",
    ].tolist()
)

injury_features = sorted(
    feature_df.loc[
        feature_df["category"]
        == "NEAR_TERM_INJURY_CONTEXT",
        "feature",
    ].tolist()
)

schedule_features = sorted(
    feature_df.loc[
        feature_df["category"]
        == "SCHEDULE_DERIVABLE",
        "feature",
    ].tolist()
)

# Historical model validity != schedule-wide deployment validity.
#
# PASS_FULL requires every feature directly available in a current
# source. Otherwise the result intentionally reports that a long-range
# state-generation contract is still required.
if (
    margin_present
    == len(margin_features)
    and total_present
    == len(total_features)
):
    deployment_status = (
        "PASS_FULL_DIRECT_AVAILABILITY"
    )
else:
    deployment_status = (
        "NEEDS_LONG_RANGE_STATE_CONTRACT"
    )

report = {
    "status":
        "PASS_READ_ONLY_AUDIT",
    "audit":
        "SIMULATOR_V1_LIVE_DEPLOYMENT_CONTRACT",
    "season":
        SEASON,
    "protected_inputs": {
        "training_source": {
            "path":
                str(TRAINING_SOURCE),
            "sha256":
                training_sha_before,
        },
        "exp003_audit": {
            "path":
                str(EXP003_AUDIT),
            "sha256":
                exp003_sha_before,
        },
    },
    "frozen_model_contract": {
        "margin_features":
            margin_features,
        "margin_feature_count":
            len(margin_features),
        "total_features":
            total_features,
        "total_feature_count":
            len(total_features),
        "margin_alpha":
            exp003.get(
                "margin",
                {}
            ).get(
                "selected_alpha"
            ),
        "total_alpha":
            exp003.get(
                "totals",
                {}
            ).get(
                "selected_alpha"
            ),
        "probability_architecture":
            exp003.get(
                "margin",
                {}
            ).get(
                "probability_architecture"
            ),
    },
    "live_feature_source": {
        "best_source":
            best_rel,
        "rows_in_2026_scope":
            live_2026_rows,
        "margin_features_present":
            margin_present,
        "margin_features_required":
            len(margin_features),
        "total_features_present":
            total_present,
        "total_features_required":
            len(total_features),
        "union_features_present":
            present_count,
        "union_features_required":
            len(all_required_features),
    },
    "feature_categories": {
        "recursive_prior_state":
            recursive_features,
        "near_term_injury_context":
            injury_features,
        "schedule_derivable":
            schedule_features,
    },
    "schedule": {
        "source":
            (
                str(
                    schedule_best.relative_to(
                        ROOT
                    )
                )
                if schedule_best
                is not None
                else None
            ),
        "completed_games":
            completed_games,
        "remaining_games":
            remaining_games,
    },
    "deployment_status":
        deployment_status,
    "production": {
        "model_training_performed":
            False,
        "production_forecast_modified":
            False,
        "optimizer_modified":
            False,
        "database_written":
            False,
    },
    "outputs": {
        "feature_source_candidates":
            str(CANDIDATES_CSV),
        "schedule_source_candidates":
            str(
                SCHEDULE_CANDIDATES_CSV
            ),
        "feature_coverage":
            str(FEATURE_COVERAGE_CSV),
    },
}

REPORT_JSON.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

# ------------------------------------------------------------
# Protected-input immutability.
# ------------------------------------------------------------

if (
    sha256(TRAINING_SOURCE)
    != training_sha_before
):
    fail(
        "training source changed during audit"
    )

if (
    sha256(EXP003_AUDIT)
    != exp003_sha_before
):
    fail(
        "EXP003 audit changed during audit"
    )

print(
    "\n===== DEPLOYMENT CONTRACT RESULT ====="
)
print(
    f"DEPLOYMENT_STATUS={deployment_status}"
)
print(
    f"RECURSIVE_PRIOR_STATE_FEATURES={len(recursive_features)}"
)
print(
    f"NEAR_TERM_INJURY_FEATURES={len(injury_features)}"
)
print(
    f"SCHEDULE_DERIVABLE_FEATURES={len(schedule_features)}"
)

print(
    "\n===== OUTPUT ====="
)
print(
    f"REPORT={REPORT_JSON}"
)
print(
    f"FEATURE_CANDIDATES={CANDIDATES_CSV}"
)
print(
    f"SCHEDULE_CANDIDATES={SCHEDULE_CANDIDATES_CSV}"
)
print(
    f"FEATURE_COVERAGE={FEATURE_COVERAGE_CSV}"
)
print(
    f"REPORT_SHA256={sha256(REPORT_JSON)}"
)
print(
    f"FEATURE_CANDIDATES_SHA256={sha256(CANDIDATES_CSV)}"
)
print(
    f"SCHEDULE_CANDIDATES_SHA256={sha256(SCHEDULE_CANDIDATES_CSV)}"
)
print(
    f"FEATURE_COVERAGE_SHA256={sha256(FEATURE_COVERAGE_CSV)}"
)

print()
print(
    "MODEL_TRAINING_PERFORMED=FALSE"
)
print(
    "PRODUCTION_FORECAST_MODIFIED=FALSE"
)
print(
    "OPTIMIZER_MODIFIED=FALSE"
)
print(
    "PRODUCTION_DATABASE_WRITTEN=FALSE"
)
print(
    "PROTECTED_INPUTS_UNCHANGED=TRUE"
)
print(
    "SIMULATOR_V1_LIVE_DEPLOYMENT_AUDIT=PASS"
)
