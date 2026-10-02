#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "processed"
    / "coaching_intelligence"
    / "postgame_coaching_evidence_v1.csv"
)

OUT_DIR = (
    ROOT
    / "processed"
    / "coaching_intelligence"
)

OUT_CSV = (
    OUT_DIR
    / "team_coaching_evidence_profile_v1.csv"
)

OUT_JSON = (
    OUT_DIR
    / "team_coaching_evidence_profile_v1.json"
)

VERSION = "WFS_TEAM_COACHING_EVIDENCE_PROFILE_V1"


RATE_COLUMNS = [
    "final_pass_rate",
    "early_down_pass_rate",
    "red_zone_pass_rate",
    "third_down_pass_rate",
    "short_yardage_pass_rate",
    "shotgun_rate",
    "no_huddle_rate",
]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
):
    v = pd.to_numeric(
        values,
        errors="coerce",
    )

    w = pd.to_numeric(
        weights,
        errors="coerce",
    )

    mask = (
        v.notna()
        & w.notna()
        & (w > 0)
    )

    if not mask.any():
        return np.nan

    return float(
        np.average(
            v[mask],
            weights=w[mask],
        )
    )


def confidence_from_evidence(
    games: int,
    plays: int,
) -> str:
    # Descriptive only.
    # No production influence.
    if games >= 6 and plays >= 300:
        return "HIGH"

    if games >= 3 and plays >= 140:
        return "MEDIUM"

    return "LOW"


def build_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for team, g in df.groupby(
        "team",
        sort=True,
    ):
        g = g.copy()

        plays = pd.to_numeric(
            g["scrimmage_plays"],
            errors="coerce",
        ).fillna(0)

        total_plays = int(
            plays.sum()
        )

        games = int(
            len(g)
        )

        baseline_mask = (
            pd.to_numeric(
                g["expected_pass_rate"],
                errors="coerce",
            ).notna()
            &
            pd.to_numeric(
                g["pass_rate_delta"],
                errors="coerce",
            ).notna()
        )

        baseline_games = int(
            baseline_mask.sum()
        )

        row = {
            "version": VERSION,
            "team": team,
            "games": games,
            "scrimmage_plays":
                total_plays,

            "baseline_games":
                baseline_games,

            "baseline_coverage":
                (
                    baseline_games / games
                    if games
                    else np.nan
                ),

            "evidence_confidence":
                confidence_from_evidence(
                    games,
                    total_plays,
                ),
        }

        for col in RATE_COLUMNS:
            row[
                f"weighted_{col}"
            ] = weighted_mean(
                g[col],
                plays,
            )

        baseline_weights = plays[
            baseline_mask
        ]

        row[
            "weighted_expected_pass_rate"
        ] = weighted_mean(
            g.loc[
                baseline_mask,
                "expected_pass_rate",
            ],
            baseline_weights,
        )

        row[
            "weighted_pass_rate_delta"
        ] = weighted_mean(
            g.loc[
                baseline_mask,
                "pass_rate_delta",
            ],
            baseline_weights,
        )

        row[
            "total_pass_calls"
        ] = int(
            pd.to_numeric(
                g["pass_calls"],
                errors="coerce",
            ).fillna(0).sum()
        )

        row[
            "total_rush_calls"
        ] = int(
            pd.to_numeric(
                g["rush_calls"],
                errors="coerce",
            ).fillna(0).sum()
        )

        row[
            "total_fourth_down_scrimmage"
        ] = int(
            pd.to_numeric(
                g["fourth_down_scrimmage"],
                errors="coerce",
            ).fillna(0).sum()
        )

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("=" * 72)
    print(VERSION)
    print("=" * 72)

    if not INPUT.exists():
        raise SystemExit(
            f"FAIL: missing ledger: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = {
        "event_id",
        "game_id",
        "team",
        "scrimmage_plays",
        "expected_pass_rate",
        "final_pass_rate",
        "pass_rate_delta",
        "pass_calls",
        "rush_calls",
        "fourth_down_scrimmage",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise SystemExit(
            "FAIL: missing columns: "
            + ", ".join(missing)
        )

    if df.empty:
        raise SystemExit(
            "FAIL: postgame ledger empty"
        )

    profile = build_profile(df)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile.to_csv(
        OUT_CSV,
        index=False,
    )

    OUT_JSON.write_text(
        json.dumps(
            {
                "version": VERSION,
                "generated_at_utc":
                    utc_now(),
                "policy": {
                    "production_influence":
                        False,
                    "solver_influence":
                        False,
                    "forecast_mutation":
                        False,
                    "persistent_coach_prior_mutation":
                        False,
                    "weighting":
                        "scrimmage_play_count",
                    "identity_scope":
                        "team_evidence_only",
                },
                "rows":
                    profile.replace(
                        {np.nan: None}
                    ).to_dict(
                        orient="records"
                    ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        profile[
            [
                "team",
                "games",
                "scrimmage_plays",
                "baseline_games",
                "weighted_final_pass_rate",
                "weighted_expected_pass_rate",
                "weighted_pass_rate_delta",
                "evidence_confidence",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("CSV :", OUT_CSV)
    print("JSON:", OUT_JSON)
    print("PASS")


if __name__ == "__main__":
    main()
