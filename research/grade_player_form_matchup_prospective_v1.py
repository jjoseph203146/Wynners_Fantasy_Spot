#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CAPTURE_DIR = ROOT / "data/research/prospective/player_form_matchup_v1"
SCHEDULE = ROOT / "data/parquet/nfl_schedule.parquet"
STATS = ROOT / "data/parquet/nfl_player_game_stats.parquet"

SEASON = None
WEEK = None
CAPTURE = None
CAPTURE_MANIFEST = None
GRADE = None
GRADE_MANIFEST = None

CAPTURE_CONTRACT = "WFS_PLAYER_FORM_MATCHUP_PROSPECTIVE_CAPTURE_V1"
GRADE_CONTRACT = "WFS_PLAYER_FORM_MATCHUP_PROSPECTIVE_GRADE_V1"

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


CAPTURE_NAME_RE = re.compile(
    r"^(?P<season>\d{4})_week_(?P<week>\d{2})_pregame\.parquet$"
)


def discover_captures():
    found = []

    if not CAPTURE_DIR.exists():
        return found

    for path in CAPTURE_DIR.glob("*_pregame.parquet"):
        m = CAPTURE_NAME_RE.match(path.name)

        if not m:
            continue

        season = int(m.group("season"))
        week = int(m.group("week"))

        manifest = (
            CAPTURE_DIR
            / f"{season}_week_{week:02d}_pregame_manifest.json"
        )

        found.append(
            {
                "season": season,
                "week": week,
                "capture": path,
                "manifest": manifest,
            }
        )

    return sorted(
        found,
        key=lambda x: (x["season"], x["week"]),
    )


def resolve_target(args):
    global SEASON
    global WEEK
    global CAPTURE
    global CAPTURE_MANIFEST
    global GRADE
    global GRADE_MANIFEST

    captures = discover_captures()

    if not captures:
        raise RuntimeError(
            "No immutable prospective captures were discovered."
        )

    if args.latest:
        target = captures[-1]

    else:
        matches = [
            x for x in captures
            if x["season"] == args.season
            and x["week"] == args.week
        ]

        if len(matches) != 1:
            available = [
                f"{x['season']}-W{x['week']:02d}"
                for x in captures
            ]

            raise RuntimeError(
                "Requested immutable capture was not found. "
                f"Requested={args.season}-W{args.week:02d}; "
                f"available={available}"
            )

        target = matches[0]

    if not target["manifest"].exists():
        raise RuntimeError(
            "Capture parquet exists without its immutable manifest: "
            f"{target['capture']}"
        )

    SEASON = target["season"]
    WEEK = target["week"]
    CAPTURE = target["capture"]
    CAPTURE_MANIFEST = target["manifest"]

    GRADE = (
        CAPTURE_DIR
        / f"{SEASON}_week_{WEEK:02d}_postgame_grade.parquet"
    )

    GRADE_MANIFEST = (
        CAPTURE_DIR
        / f"{SEASON}_week_{WEEK:02d}_postgame_grade_manifest.json"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                indent=2,
                sort_keys=True,
                default=str,
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp.parquet",
        dir=str(path.parent),
    )
    os.close(fd)

    try:
        df.to_parquet(tmp, index=False)

        check = pd.read_parquet(tmp)

        if len(check) != len(df):
            raise RuntimeError(
                f"Parquet round-trip row mismatch: "
                f"{len(df)} != {len(check)}"
            )

        os.replace(tmp, path)

    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def require_files() -> None:
    for p in [
        CAPTURE,
        CAPTURE_MANIFEST,
        SCHEDULE,
        STATS,
    ]:
        if not p.exists():
            raise RuntimeError(f"Required input missing: {p}")


def load_capture_manifest() -> dict:
    with CAPTURE_MANIFEST.open("r", encoding="utf-8") as f:
        m = json.load(f)

    if m.get("contract") != CAPTURE_CONTRACT:
        raise RuntimeError(
            "Unexpected capture contract: "
            f"{m.get('contract')!r}"
        )

    if m.get("status") != "IMMUTABLE_PREKICKOFF_CAPTURED":
        raise RuntimeError(
            "Capture manifest is not an immutable pre-kickoff capture."
        )

    if int(m.get("season")) != SEASON or int(m.get("week")) != WEEK:
        raise RuntimeError(
            "Capture season/week does not match requested grading target."
        )

    safety = m.get("safety", {})

    if safety.get("analysis_only") is not True:
        raise RuntimeError("Capture is not analysis-only.")

    for key, expected in SAFETY.items():
        if safety.get(key) != expected:
            raise RuntimeError(
                f"Capture safety mismatch: "
                f"{key}={safety.get(key)!r}"
            )

    return m


def validate_capture_hash(manifest: dict) -> str:
    expected = manifest.get("output", {}).get("sha256")

    if not expected:
        raise RuntimeError(
            "Capture manifest lacks output SHA-256."
        )

    actual = sha256_file(CAPTURE)

    if actual != expected:
        raise RuntimeError(
            "IMMUTABLE CAPTURE HASH FAILURE.\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )

    return actual


def validate_completed_games(
    capture: pd.DataFrame,
    schedule: pd.DataFrame,
) -> list[str]:

    required = [
        "game_id",
        "season",
        "week",
        "completed",
        "away_score",
        "home_score",
    ]

    missing = [c for c in required if c not in schedule.columns]
    if missing:
        raise RuntimeError(
            f"Schedule missing required columns: {missing}"
        )

    game_ids = sorted(
        capture["game_id"].astype(str).unique().tolist()
    )

    s = schedule[
        schedule["game_id"].astype(str).isin(game_ids)
    ].copy()

    if len(s) != len(game_ids):
        found = set(s["game_id"].astype(str))
        missing_games = sorted(set(game_ids) - found)

        raise RuntimeError(
            f"Capture games missing from schedule: {missing_games}"
        )

    if s["game_id"].duplicated().any():
        raise RuntimeError(
            "Schedule contains duplicate target game_id."
        )

    completed = pd.to_numeric(
        s["completed"],
        errors="coerce",
    ).fillna(0)

    scores_present = (
        pd.to_numeric(s["away_score"], errors="coerce").notna()
        & pd.to_numeric(s["home_score"], errors="coerce").notna()
    )

    ready = completed.eq(1) & scores_present

    if not ready.all():
        pending = s.loc[
            ~ready,
            [
                "game_id",
                "completed",
                "away_score",
                "home_score",
            ]
        ]

        raise RuntimeError(
            "POSTGAME OUTCOMES NOT READY. "
            "REFUSING TO GRADE.\n"
            "Every captured game must be completed with final scores.\n\n"
            + pending.to_string(index=False)
        )

    return game_ids


def prepare_stats(
    stats: pd.DataFrame,
    game_ids: list[str],
) -> pd.DataFrame:

    required = [
        "game_id",
        "player_id",
        "position",
        "team",
        "opponent_team",
        "fanduel_points",
        "carries",
        "targets",
        "receptions",
        "rushing_yards",
        "receiving_yards",
        "rushing_tds",
        "receiving_tds",
    ]

    missing = [c for c in required if c not in stats.columns]
    if missing:
        raise RuntimeError(
            f"Stats authority missing required columns: {missing}"
        )

    s = stats[
        stats["game_id"].astype(str).isin(game_ids)
    ].copy()

    if s.empty:
        raise RuntimeError(
            "Completed target game has no player-game stats."
        )

    s["team"] = s["team"].map(canonical_team)
    s["opponent_team"] = s["opponent_team"].map(canonical_team)
    s["position"] = s["position"].astype(str).str.upper()

    # Restrict grading to fantasy offensive positions in capture.
    s = s[
        s["position"].isin(["QB", "RB", "WR", "TE"])
    ].copy()

    if s.duplicated(
        ["game_id", "player_id", "team", "opponent_team", "position"]
    ).any():
        dupes = s.loc[
            s.duplicated(
                [
                    "game_id",
                    "player_id",
                    "team",
                    "opponent_team",
                    "position",
                ],
                keep=False,
            ),
            [
                "game_id",
                "player_id",
                "team",
                "opponent_team",
                "position",
            ],
        ]

        raise RuntimeError(
            "Stats authority contains duplicate exact identities:\n"
            + dupes.to_string(index=False)
        )

    numeric = [
        "fanduel_points",
        "carries",
        "targets",
        "receptions",
        "rushing_yards",
        "receiving_yards",
        "rushing_tds",
        "receiving_tds",
    ]

    for c in numeric:
        s[c] = pd.to_numeric(s[c], errors="coerce")

    return s


def grade() -> None:
    require_files()

    capture_manifest = load_capture_manifest()
    capture_sha = validate_capture_hash(capture_manifest)

    capture = pd.read_parquet(CAPTURE)
    schedule = pd.read_parquet(SCHEDULE)

    # ------------------------------------------------------------
    # FAIL CLOSED UNTIL ALL CAPTURED GAMES ARE FINAL
    # ------------------------------------------------------------
    game_ids = validate_completed_games(capture, schedule)

    stats = prepare_stats(
        pd.read_parquet(STATS),
        game_ids,
    )

    c = capture.copy()

    c["team"] = c["team"].map(canonical_team)
    c["opponent_team"] = c["opponent_team"].map(canonical_team)
    c["position"] = c["position"].astype(str).str.upper()

    if c.duplicated(IDENTITY_KEYS).any():
        raise RuntimeError(
            "Immutable capture contains duplicate identities."
        )

    outcome_cols = [
        "game_id",
        "player_id",
        "team",
        "opponent_team",
        "position",
        "fanduel_points",
        "carries",
        "targets",
        "receptions",
        "rushing_yards",
        "receiving_yards",
        "rushing_tds",
        "receiving_tds",
    ]

    out = stats[outcome_cols].rename(
        columns={
            "fanduel_points": "actual_fd",
            "carries": "actual_carries",
            "targets": "actual_targets",
            "receptions": "actual_receptions",
            "rushing_yards": "actual_rushing_yards",
            "receiving_yards": "actual_receiving_yards",
            "rushing_tds": "actual_rushing_tds",
            "receiving_tds": "actual_receiving_tds",
        }
    )

    g = c.merge(
        out,
        on=IDENTITY_KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    # A captured player may legitimately record no box-score stats.
    # We do NOT silently convert an unmatched identity into zero.
    # Missing exact outcome identity remains explicit.
    # ------------------------------------------------------------
    g["outcome_identity_status"] = np.where(
        g["_merge"].eq("both"),
        "EXACT_MATCH",
        "NO_EXACT_OUTCOME_ROW",
    )

    g = g.drop(columns="_merge")

    actual_numeric = [
        "actual_fd",
        "actual_carries",
        "actual_targets",
        "actual_receptions",
        "actual_rushing_yards",
        "actual_receiving_yards",
        "actual_rushing_tds",
        "actual_receiving_tds",
    ]

    exact = g["outcome_identity_status"].eq("EXACT_MATCH")

    # Exact matched outcome rows must have authoritative FD.
    if g.loc[exact, "actual_fd"].isna().any():
        bad = g.loc[
            exact & g["actual_fd"].isna(),
            IDENTITY_KEYS,
        ]

        raise RuntimeError(
            "Exact outcome rows missing authoritative FanDuel points:\n"
            + bad.to_string(index=False)
        )

    # ------------------------------------------------------------
    # Prospective outcome grading.
    # Only exact outcome identities receive grades.
    # ------------------------------------------------------------
    g["actual_opportunities"] = (
        g["actual_carries"] + g["actual_targets"]
    )

    g["fd_change_vs_captured_avg3"] = np.where(
        exact,
        g["actual_fd"] - pd.to_numeric(
            g["fd_avg_3"],
            errors="coerce",
        ),
        np.nan,
    )

    g["beat_captured_avg3"] = np.where(
        exact & g["fd_avg_3"].notna(),
        g["actual_fd"] > g["fd_avg_3"],
        pd.NA,
    )

    g["beat_captured_max5"] = np.where(
        exact & g["fd_max_5"].notna(),
        g["actual_fd"] > g["fd_max_5"],
        pd.NA,
    )

    g["plus5_rebound"] = np.where(
        exact & g["fd_avg_3"].notna(),
        g["actual_fd"] >= (g["fd_avg_3"] + 5.0),
        pd.NA,
    )

    g["one_sigma_game"] = np.where(
        exact
        & g["fd_avg_5"].notna()
        & g["fd_std_5"].notna(),
        g["actual_fd"]
        >= (
            g["fd_avg_5"]
            + g["fd_std_5"]
        ),
        pd.NA,
    )

    # Role outcome is descriptive only.
    g["opportunity_change_vs_captured_avg3"] = np.where(
        exact
        & g["opportunities_avg_3"].notna()
        & ~g["position"].eq("QB"),
        g["actual_opportunities"]
        - g["opportunities_avg_3"],
        np.nan,
    )

    # Explicitly prohibit interpreting carries+targets as QB role.
    g["qb_role_grade_status"] = np.where(
        g["position"].eq("QB"),
        "NOT_GRADED_PASS_ATTEMPTS_EXCLUDED",
        "NOT_APPLICABLE",
    )

    g.insert(
        0,
        "grade_contract",
        GRADE_CONTRACT,
    )

    g.insert(
        1,
        "graded_at_utc",
        utc_now(),
    )

    g.insert(
        2,
        "source_capture_sha256",
        capture_sha,
    )

    for key, value in SAFETY.items():
        g[key] = value

    g = g.sort_values(IDENTITY_KEYS).reset_index(drop=True)

    # ------------------------------------------------------------
    # Do not silently replace an existing grade.
    # ------------------------------------------------------------
    if GRADE.exists() or GRADE_MANIFEST.exists():

        if not GRADE.exists() or not GRADE_MANIFEST.exists():
            raise RuntimeError(
                "Partial existing grade artifact detected. "
                "REFUSING overwrite."
            )

        existing_sha = sha256_file(GRADE)

        with GRADE_MANIFEST.open("r", encoding="utf-8") as f:
            existing_manifest = json.load(f)

        expected = (
            existing_manifest
            .get("output", {})
            .get("sha256")
        )

        if existing_sha != expected:
            raise RuntimeError(
                "Existing grade artifact fails manifest hash binding. "
                "REFUSING overwrite."
            )

        print("\nPOSTGAME GRADE ALREADY EXISTS")
        print("=" * 80)
        print("Grade:", GRADE)
        print("Manifest:", GRADE_MANIFEST)
        print("SHA256:", existing_sha)
        print("Action: NONE")
        return

    atomic_write_parquet(GRADE, g)

    grade_sha = sha256_file(GRADE)

    exact_count = int(
        g["outcome_identity_status"].eq("EXACT_MATCH").sum()
    )

    missing_count = int(
        g["outcome_identity_status"].eq(
            "NO_EXACT_OUTCOME_ROW"
        ).sum()
    )

    position_summary = {}

    for pos, p in g.groupby("position"):
        valid = p[p["outcome_identity_status"].eq("EXACT_MATCH")]

        position_summary[str(pos)] = {
            "captured_rows": int(len(p)),
            "exact_outcome_rows": int(len(valid)),
            "avg_actual_fd": (
                float(valid["actual_fd"].mean())
                if len(valid)
                else None
            ),
            "beat_avg3_rate": (
                float(
                    pd.to_numeric(
                        valid["beat_captured_avg3"],
                        errors="coerce",
                    ).mean()
                )
                if len(valid)
                else None
            ),
            "plus5_rebound_rate": (
                float(
                    pd.to_numeric(
                        valid["plus5_rebound"],
                        errors="coerce",
                    ).mean()
                )
                if len(valid)
                else None
            ),
            "one_sigma_rate": (
                float(
                    pd.to_numeric(
                        valid["one_sigma_game"],
                        errors="coerce",
                    ).mean()
                )
                if len(valid)
                else None
            ),
        }

    rb = g[
        g["position"].eq("RB")
        & g["outcome_identity_status"].eq("EXACT_MATCH")
    ].copy()

    rb_summary = {
        "rows": int(len(rb)),
        "research_status": "PROSPECTIVE_OBSERVATION_ONLY",
        "hypothesis_channels": [
            "opportunities_allowed_avg_3",
            "carries_allowed_avg_3",
            "rushing_yards_allowed_avg_3",
        ],
        "production_authorized": False,
    }

    manifest = {
        "contract": GRADE_CONTRACT,
        "status": "POSTGAME_GRADED",
        "analysis_only": True,

        "season": SEASON,
        "week": WEEK,

        "graded_at_utc": utc_now(),

        "source_capture": {
            "path": str(CAPTURE),
            "sha256": capture_sha,
            "manifest_path": str(CAPTURE_MANIFEST),
            "manifest_sha256": sha256_file(CAPTURE_MANIFEST),
        },

        "outcome_authority": {
            "schedule_path": str(SCHEDULE),
            "schedule_sha256": sha256_file(SCHEDULE),
            "stats_path": str(STATS),
            "stats_sha256": sha256_file(STATS),
            "games": game_ids,
            "completed_gate": "PASS",
        },

        "identity": {
            "keys": IDENTITY_KEYS,
            "captured_rows": int(len(g)),
            "exact_outcome_rows": exact_count,
            "no_exact_outcome_rows": missing_count,
            "missing_policy": (
                "NO_SILENT_ZERO_FILL_FOR_UNMATCHED_OUTCOME_IDENTITIES"
            ),
        },

        "grading_policy": {
            "central": [
                "actual_fd",
                "fd_change_vs_captured_avg3",
                "beat_captured_avg3",
            ],
            "ceiling": [
                "beat_captured_max5",
                "plus5_rebound",
                "one_sigma_game",
            ],
            "role": [
                "actual_carries",
                "actual_targets",
                "actual_opportunities",
                "opportunity_change_vs_captured_avg3",
            ],
            "qb_role_policy": (
                "NOT_GRADED_FROM_CARRIES_PLUS_TARGETS"
            ),
            "classification_policy": (
                "NO_HOT_COLD_ELITE_OR_PRODUCTION_THRESHOLD_AUTHORIZED"
            ),
        },

        "position_summary": position_summary,
        "rb_prospective_hypothesis": rb_summary,

        "safety": SAFETY,

        "output": {
            "path": str(GRADE),
            "sha256": grade_sha,
        },
    }

    atomic_write_json(GRADE_MANIFEST, manifest)

    # Final binding.
    with GRADE_MANIFEST.open("r", encoding="utf-8") as f:
        verify_manifest = json.load(f)

    if (
        verify_manifest.get("output", {}).get("sha256")
        != sha256_file(GRADE)
    ):
        raise RuntimeError(
            "Final grade manifest/output hash binding failed."
        )

    print("\n" + "=" * 80)
    print("STAGE 3B POSTGAME GRADING")
    print("=" * 80)

    print("Contract:", GRADE_CONTRACT)
    print("Status: POSTGAME_GRADED")
    print("Season:", SEASON)
    print("Week:", WEEK)
    print("Captured rows:", len(g))
    print("Exact outcomes:", exact_count)
    print("No exact outcome row:", missing_count)

    print("\nPOSITION SUMMARY:")
    for pos in sorted(position_summary):
        print(pos, position_summary[pos])

    print("\nRB PROSPECTIVE HYPOTHESIS:")
    print(rb_summary)

    print("\nOUTPUT:")
    print(GRADE)

    print("\nMANIFEST:")
    print(GRADE_MANIFEST)

    print("\nSHA256:")
    print(grade_sha)

    print("\nProduction changes: NONE")
    print("Solver changes: NONE")
    print("Eligibility changes: NONE")
    print("Projection changes: NONE")
    print("GPP changes: NONE")
    print("Public UI changes: NONE")
    print("Services touched: NONE")
    print("NBA APP touched: NONE")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Grade an immutable prospective player-form/matchup "
            "capture after all captured games are complete."
        )
    )

    target = parser.add_mutually_exclusive_group(required=True)

    target.add_argument(
        "--latest",
        action="store_true",
        help="Grade the latest immutable capture discovered on disk.",
    )

    target.add_argument(
        "--season",
        type=int,
        help="Season of the immutable capture to grade.",
    )

    parser.add_argument(
        "--week",
        type=int,
        help="Week of the immutable capture to grade.",
    )

    args = parser.parse_args()

    if args.latest:
        if args.week is not None:
            parser.error("--week cannot be used with --latest")
    else:
        if args.season is None or args.week is None:
            parser.error("--season requires --week")

        if args.week < 1 or args.week > 22:
            parser.error("--week must be between 1 and 22")

    try:
        resolve_target(args)

        print(
            f"Resolved immutable capture: "
            f"{SEASON}-W{WEEK:02d}"
        )

        grade()

    except Exception as exc:
        print(
            "\nSTAGE 3B GRADER FAILED CLOSED",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
