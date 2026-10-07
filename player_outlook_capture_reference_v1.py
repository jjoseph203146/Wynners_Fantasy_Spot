#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

STAGE1 = ROOT / "data/research/player_form_matchup_shadow_v1.parquet"
STAGE1_MANIFEST = ROOT / "data/research/player_form_matchup_shadow_v1_manifest.json"
SCHEDULE = ROOT / "data/parquet/nfl_schedule.parquet"

CAPTURE_DIR = ROOT / "data/research/prospective/player_form_matchup_v1"

CONTRACT = "WFS_PLAYER_FORM_MATCHUP_PROSPECTIVE_CAPTURE_V1"
SOURCE_CONTRACT = "WFS_PLAYER_FORM_MATCHUP_SHADOW_V1"

IDENTITY_KEYS = [
    "game_id",
    "player_id",
    "team",
    "opponent_team",
    "position",
]

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def canonical_team(value) -> str:
    x = str(value).strip().upper()

    aliases = {
        "LA": "LAR",
        "WAS": "WSH",
        "JAC": "JAX",
    }

    return aliases.get(x, x)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_name, path)

    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp.parquet",
        dir=str(path.parent),
    )
    os.close(fd)

    try:
        df.to_parquet(tmp_name, index=False)

        # Round-trip validation before publication.
        check = pd.read_parquet(tmp_name)

        if len(check) != len(df):
            raise RuntimeError(
                f"Parquet round-trip row mismatch: {len(df)} != {len(check)}"
            )

        os.replace(tmp_name, path)

    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def require_files() -> None:
    for p in [STAGE1, STAGE1_MANIFEST, SCHEDULE]:
        if not p.exists():
            raise RuntimeError(f"Required input missing: {p}")


def load_stage1_manifest() -> dict:
    with STAGE1_MANIFEST.open("r", encoding="utf-8") as f:
        m = json.load(f)

    if m.get("contract") != SOURCE_CONTRACT:
        raise RuntimeError(
            "Unexpected Stage-1 contract: "
            f"{m.get('contract')!r}"
        )

    safety = m.get("safety", {})

    expected_false = [
        "production_influence",
        "solver_influence",
        "projection_mutation",
        "eligibility_mutation",
        "gpp_mutation",
        "ui_mutation",
        "database_mutation",
    ]

    if safety.get("analysis_only") is not True:
        raise RuntimeError("Stage-1 manifest is not analysis-only.")

    for key in expected_false:
        if safety.get(key) is not False:
            raise RuntimeError(
                f"Stage-1 safety contract violation: {key}={safety.get(key)!r}"
            )

    return m


def validate_stage1_hash(stage1_manifest: dict) -> str:
    actual = sha256_file(STAGE1)

    expected = (
        stage1_manifest
        .get("output", {})
        .get("sha256")
    )

    if not expected:
        raise RuntimeError("Stage-1 manifest lacks output SHA-256.")

    if actual != expected:
        raise RuntimeError(
            "Stage-1 parquet hash does not match Stage-1 manifest.\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )

    return actual


def build_schedule_kickoff(schedule: pd.DataFrame) -> pd.DataFrame:
    required = [
        "game_id",
        "season",
        "week",
        "game_date",
        "gametime",
        "away_team",
        "home_team",
        "completed",
    ]

    missing = [c for c in required if c not in schedule.columns]
    if missing:
        raise RuntimeError(
            f"Schedule missing required columns: {missing}"
        )

    s = schedule[required].copy()

    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s["week"] = pd.to_numeric(s["week"], errors="coerce")

    if s["game_id"].duplicated().any():
        dupes = s.loc[
            s["game_id"].duplicated(keep=False),
            "game_id"
        ].tolist()

        raise RuntimeError(
            f"Schedule has duplicate game_id values: {dupes[:10]}"
        )

    # nflverse schedule gametime is Eastern local time.
    local_text = (
        s["game_date"].astype(str).str.strip()
        + " "
        + s["gametime"].astype(str).str.strip()
    )

    local = pd.to_datetime(
        local_text,
        errors="coerce",
    )

    if local.isna().any():
        bad = s.loc[
            local.isna(),
            ["game_id", "game_date", "gametime"]
        ]

        raise RuntimeError(
            "Unable to parse schedule kickoff values:\n"
            + bad.to_string(index=False)
        )

    s["schedule_kickoff_utc"] = (
        local
        .dt.tz_localize(
            "America/New_York",
            ambiguous="raise",
            nonexistent="raise",
        )
        .dt.tz_convert("UTC")
    )

    s["away_team_canonical"] = s["away_team"].map(canonical_team)
    s["home_team_canonical"] = s["home_team"].map(canonical_team)

    return s


def validate_identity_and_schedule(
    stage1: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:

    required = [
        "season",
        "week",
        "game_id",
        "player_id",
        "team",
        "opponent_team",
        "position",
        "player_name",
        "target_kickoff_utc",
        "evidence_cutoff",
        "chronology_status",
        "historical_available_at_cutoff_proven",
    ]

    missing = [c for c in required if c not in stage1.columns]

    if missing:
        raise RuntimeError(
            f"Stage-1 artifact missing required columns: {missing}"
        )

    if stage1.duplicated(IDENTITY_KEYS).any():
        raise RuntimeError(
            "Stage-1 contains duplicate exact identity rows."
        )

    x = stage1.copy()

    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)

    x["team_canonical"] = x["team"].map(canonical_team)
    x["opponent_canonical"] = x["opponent_team"].map(canonical_team)

    x["target_kickoff_utc"] = pd.to_datetime(
        x["target_kickoff_utc"],
        utc=True,
        errors="raise",
    )

    s = build_schedule_kickoff(schedule)

    merge_cols = [
        "game_id",
        "season",
        "week",
        "away_team",
        "home_team",
        "completed",
        "schedule_kickoff_utc",
        "away_team_canonical",
        "home_team_canonical",
    ]

    x = x.merge(
        s[merge_cols],
        on=["game_id", "season", "week"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not x["_merge"].eq("both").all():
        bad = x.loc[
            ~x["_merge"].eq("both"),
            ["season", "week", "game_id"]
        ].drop_duplicates()

        raise RuntimeError(
            "Stage-1 game missing from authoritative schedule:\n"
            + bad.to_string(index=False)
        )

    x = x.drop(columns="_merge")

    # Validate team/opponent against actual scheduled participants.
    def valid_pair(row) -> bool:
        participants = {
            row["away_team_canonical"],
            row["home_team_canonical"],
        }

        return (
            row["team_canonical"] in participants
            and row["opponent_canonical"] in participants
            and row["team_canonical"] != row["opponent_canonical"]
            and {
                row["team_canonical"],
                row["opponent_canonical"],
            } == participants
        )

    pair_ok = x.apply(valid_pair, axis=1)

    if not pair_ok.all():
        bad = x.loc[
            ~pair_ok,
            [
                "game_id",
                "team",
                "opponent_team",
                "away_team",
                "home_team",
            ]
        ]

        raise RuntimeError(
            "Stage-1 team/opponent identity conflicts with schedule:\n"
            + bad.to_string(index=False)
        )

    # Stage-1 kickoff and independently reconstructed schedule kickoff
    # must agree exactly.
    kickoff_match = (
        x["target_kickoff_utc"]
        == x["schedule_kickoff_utc"]
    )

    if not kickoff_match.all():
        bad = x.loc[
            ~kickoff_match,
            [
                "game_id",
                "target_kickoff_utc",
                "schedule_kickoff_utc",
            ]
        ].drop_duplicates()

        raise RuntimeError(
            "Kickoff mismatch between Stage-1 and schedule:\n"
            + bad.to_string(index=False)
        )

    return x


def capture() -> None:
    require_files()

    now = utc_now()

    stage1_manifest = load_stage1_manifest()
    stage1_sha = validate_stage1_hash(stage1_manifest)

    stage1 = pd.read_parquet(STAGE1)
    schedule = pd.read_parquet(SCHEDULE)

    bound = validate_identity_and_schedule(stage1, schedule)

    if bound.empty:
        raise RuntimeError("No Stage-1 rows available for capture.")

    seasons = sorted(bound["season"].unique().tolist())
    weeks = sorted(bound["week"].unique().tolist())

    if len(seasons) != 1 or len(weeks) != 1:
        raise RuntimeError(
            "Prospective capture must contain exactly one season/week. "
            f"Found seasons={seasons}, weeks={weeks}"
        )

    season = int(seasons[0])
    week = int(weeks[0])

    # ------------------------------------------------------------
    # STRICT PRE-KICKOFF GATE
    # ------------------------------------------------------------
    late = bound["target_kickoff_utc"] <= now

    if late.any():
        bad = (
            bound.loc[
                late,
                ["game_id", "target_kickoff_utc"]
            ]
            .drop_duplicates()
            .sort_values("target_kickoff_utc")
        )

        raise RuntimeError(
            "REFUSING CAPTURE: at least one target game has already "
            "reached/passed kickoff.\n"
            + bad.to_string(index=False)
        )

    # Completed games must never be prospectively captured.
    completed_numeric = pd.to_numeric(
        bound["completed"],
        errors="coerce",
    ).fillna(0)

    if completed_numeric.ne(0).any():
        bad = bound.loc[
            completed_numeric.ne(0),
            ["game_id", "completed"]
        ].drop_duplicates()

        raise RuntimeError(
            "REFUSING CAPTURE: schedule marks target game completed:\n"
            + bad.to_string(index=False)
        )

    # ------------------------------------------------------------
    # Capture exactly what Stage 1 knew.
    # Add capture metadata; do not recompute Stage-1 evidence.
    # ------------------------------------------------------------
    capture_df = stage1.copy()

    capture_df.insert(
        0,
        "prospective_contract",
        CONTRACT,
    )

    capture_df.insert(
        1,
        "captured_at_utc",
        now.isoformat(),
    )

    capture_df.insert(
        2,
        "capture_mode",
        "IMMUTABLE_PREKICKOFF",
    )

    capture_df.insert(
        3,
        "source_stage1_sha256",
        stage1_sha,
    )

    capture_df.insert(
        4,
        "source_stage1_manifest_sha256",
        sha256_file(STAGE1_MANIFEST),
    )

    capture_df.insert(
        5,
        "source_schedule_sha256",
        sha256_file(SCHEDULE),
    )

    capture_df.insert(
        6,
        "prospective_historical_knowledge_status",
        "CAPTURED_AS_KNOWN_AT_CAPTURE_TIME",
    )

    # Stage-2 research interpretation only.
    # These labels DO NOT influence production.
    capture_df["stage2_position_research_status"] = (
        capture_df["position"].map({
            "RB": "SUPPORTED_PROSPECTIVE_HYPOTHESIS",
            "WR": "MODEST_RESEARCH_SIGNAL",
            "TE": "DESCRIPTIVE_ONLY",
            "QB": "QB_ROLE_DEFINITION_REQUIRED",
        }).fillna("NOT_APPLICABLE")
    )

    capture_df["stage2_rb_matchup_channels"] = ""
    rb = capture_df["position"].eq("RB")

    capture_df.loc[
        rb,
        "stage2_rb_matchup_channels"
    ] = (
        "OPPORTUNITIES_ALLOWED|CARRIES_ALLOWED|"
        "RUSHING_YARDS_ALLOWED"
    )

    capture_df["stage2_threshold_policy"] = (
        "NO_PRODUCTION_THRESHOLD_AUTHORIZED"
    )

    for key, value in SAFETY.items():
        # Stage-1 already has several of these columns.
        # Reassert exact frozen safety values.
        capture_df[key] = value

    capture_df = capture_df.sort_values(
        IDENTITY_KEYS
    ).reset_index(drop=True)

    capture_key = f"{season}_week_{week:02d}"

    parquet_path = (
        CAPTURE_DIR
        / f"{capture_key}_pregame.parquet"
    )

    manifest_path = (
        CAPTURE_DIR
        / f"{capture_key}_pregame_manifest.json"
    )

    # ------------------------------------------------------------
    # IMMUTABILITY GATE
    # ------------------------------------------------------------
    if parquet_path.exists() or manifest_path.exists():

        if not parquet_path.exists() or not manifest_path.exists():
            raise RuntimeError(
                "Partial prior capture exists. Refusing overwrite.\n"
                f"parquet_exists={parquet_path.exists()}\n"
                f"manifest_exists={manifest_path.exists()}"
            )

        existing_sha = sha256_file(parquet_path)

        with manifest_path.open("r", encoding="utf-8") as f:
            existing_manifest = json.load(f)

        manifest_sha = (
            existing_manifest
            .get("output", {})
            .get("sha256")
        )

        if existing_sha != manifest_sha:
            raise RuntimeError(
                "Existing immutable capture failed hash validation. "
                "REFUSING overwrite."
            )

        print("\nIMMUTABLE CAPTURE ALREADY EXISTS")
        print("=" * 80)
        print("Parquet:", parquet_path)
        print("Manifest:", manifest_path)
        print("SHA256:", existing_sha)
        print("Action: NONE")
        return

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    # Publish parquet first.
    atomic_write_parquet(parquet_path, capture_df)

    published_sha = sha256_file(parquet_path)

    # Verify published file itself.
    verify = pd.read_parquet(parquet_path)

    if len(verify) != len(capture_df):
        raise RuntimeError(
            "Published capture row-count verification failed."
        )

    if verify.duplicated(IDENTITY_KEYS).any():
        raise RuntimeError(
            "Published capture contains duplicate identity rows."
        )

    # Manifest is committed last.
    game_summary = (
        bound[
            [
                "game_id",
                "target_kickoff_utc",
                "away_team",
                "home_team",
            ]
        ]
        .drop_duplicates()
        .sort_values(["target_kickoff_utc", "game_id"])
    )

    manifest = {
        "contract": CONTRACT,
        "status": "IMMUTABLE_PREKICKOFF_CAPTURED",
        "analysis_only": True,

        "season": season,
        "week": week,

        "captured_at_utc": now.isoformat(),
        "rows": int(len(capture_df)),
        "games": int(capture_df["game_id"].nunique()),
        "players": int(capture_df["player_id"].nunique()),

        "identity_keys": IDENTITY_KEYS,

        "source": {
            "stage1_contract": SOURCE_CONTRACT,
            "stage1_path": str(STAGE1),
            "stage1_sha256": stage1_sha,
            "stage1_manifest_path": str(STAGE1_MANIFEST),
            "stage1_manifest_sha256": sha256_file(STAGE1_MANIFEST),
            "schedule_path": str(SCHEDULE),
            "schedule_sha256": sha256_file(SCHEDULE),
        },

        "chronology": {
            "capture_policy": "STRICTLY_BEFORE_TARGET_KICKOFF",
            "capture_time_utc": now.isoformat(),
            "all_games_future_at_capture": True,
            "all_games_uncompleted_at_capture": True,
            "stage1_chronology_status_counts": (
                capture_df["chronology_status"]
                .value_counts(dropna=False)
                .to_dict()
            ),
            "historical_available_at_cutoff_proven_counts": (
                capture_df[
                    "historical_available_at_cutoff_proven"
                ]
                .value_counts(dropna=False)
                .to_dict()
            ),
        },

        "games_detail": [
            {
                "game_id": row.game_id,
                "kickoff_utc": row.target_kickoff_utc.isoformat(),
                "away_team": row.away_team,
                "home_team": row.home_team,
            }
            for row in game_summary.itertuples(index=False)
        ],

        "stage2_research_policy": {
            "universal_down_up_rebound": "NOT_VALIDATED",
            "universal_dvp_ceiling_signal": "NOT_VALIDATED",

            "rb": {
                "status": "SUPPORTED_PROSPECTIVE_HYPOTHESIS",
                "channels": [
                    "opportunities_allowed_avg_3",
                    "carries_allowed_avg_3",
                    "rushing_yards_allowed_avg_3",
                ],
                "production_influence": False,
            },

            "wr": {
                "status": "MODEST_RESEARCH_SIGNAL",
                "production_influence": False,
            },

            "te": {
                "status": "DESCRIPTIVE_ONLY",
                "production_influence": False,
            },

            "qb": {
                "status": "QB_ROLE_DEFINITION_REQUIRED",
                "reason": (
                    "Current opportunity definition excludes "
                    "passing attempts."
                ),
                "production_influence": False,
            },

            "threshold_policy": (
                "NO_HOT_COLD_ELITE_OR_PRODUCTION_THRESHOLD_AUTHORIZED"
            ),
        },

        "safety": SAFETY,

        "output": {
            "path": str(parquet_path),
            "sha256": published_sha,
        },
    }

    atomic_write_json(manifest_path, manifest)

    # Final manifest/output binding check.
    with manifest_path.open("r", encoding="utf-8") as f:
        final_manifest = json.load(f)

    if (
        final_manifest.get("output", {}).get("sha256")
        != sha256_file(parquet_path)
    ):
        raise RuntimeError(
            "Final manifest/output hash binding failed."
        )

    print("\n" + "=" * 80)
    print("STAGE 3A PROSPECTIVE CAPTURE")
    print("=" * 80)

    print("Contract:", CONTRACT)
    print("Status: IMMUTABLE_PREKICKOFF_CAPTURED")
    print("Season:", season)
    print("Week:", week)
    print("Rows:", len(capture_df))
    print("Games:", capture_df["game_id"].nunique())
    print("Players:", capture_df["player_id"].nunique())
    print("Captured at:", now.isoformat())

    print("\nTARGET GAMES:")
    print(
        game_summary.to_string(index=False)
    )

    print("\nPOSITION COUNTS:")
    print(
        capture_df["position"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nRESEARCH STATUS:")
    print(
        capture_df[
            [
                "position",
                "stage2_position_research_status",
            ]
        ]
        .drop_duplicates()
        .sort_values("position")
        .to_string(index=False)
    )

    print("\nOUTPUT:")
    print(parquet_path)

    print("\nMANIFEST:")
    print(manifest_path)

    print("\nSHA256:")
    print(published_sha)

    print("\nSAFETY:")
    for key, value in SAFETY.items():
        print(f"{key}: {value}")

    print("\nProduction changes: NONE")
    print("Solver changes: NONE")
    print("Eligibility changes: NONE")
    print("Projection changes: NONE")
    print("GPP changes: NONE")
    print("Public UI changes: NONE")
    print("Services touched: NONE")
    print("NBA APP touched: NONE")


if __name__ == "__main__":
    try:
        capture()
    except Exception as exc:
        print("\nSTAGE 3A CAPTURE FAILED CLOSED", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        sys.exit(1)
