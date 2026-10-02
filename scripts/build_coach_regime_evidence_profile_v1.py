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
    / "postgame_coach_attached_evidence_v1.csv"
)

OUT_DIR = (
    ROOT
    / "processed"
    / "coaching_intelligence"
)

OUT_CSV = (
    OUT_DIR
    / "coach_regime_evidence_profile_v1.csv"
)

OUT_JSON = (
    OUT_DIR
    / "coach_regime_evidence_profile_v1.json"
)

VERSION = "WFS_COACH_REGIME_EVIDENCE_PROFILE_V1"


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


def evidence_confidence(
    games: int,
    plays: int,
) -> str:
    # Descriptive only.
    # No production or solver influence.
    if games >= 6 and plays >= 300:
        return "HIGH"

    if games >= 3 and plays >= 140:
        return "MEDIUM"

    return "LOW"


def build_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    group_cols = [
        "coach_identity",
        "identity_lookup_team",
    ]

    for (
        coach_identity,
        identity_team,
    ), g in df.groupby(
        group_cols,
        sort=True,
        dropna=False,
    ):

        g = g.copy()

        plays = pd.to_numeric(
            g["scrimmage_plays"],
            errors="coerce",
        ).fillna(0)

        games = int(
            len(g)
        )

        total_plays = int(
            plays.sum()
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

        regime_start_rows = int(
            pd.to_numeric(
                g[
                    "coach_team_era_start_flag"
                ],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .sum()
        )

        coach_change_rows = int(
            pd.to_numeric(
                g["coach_change_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .sum()
        )

        prior_era_games = pd.to_numeric(
            g[
                "coach_team_era_games_prior"
            ],
            errors="coerce",
        )

        min_prior_era_games = (
            int(prior_era_games.min())
            if prior_era_games.notna().any()
            else None
        )

        max_prior_era_games = (
            int(prior_era_games.max())
            if prior_era_games.notna().any()
            else None
        )

        row = {
            "version":
                VERSION,

            "coach_identity":
                str(coach_identity),

            "identity_lookup_team":
                str(identity_team),

            "coach_team_regime_key":
                (
                    f"{coach_identity}|"
                    f"{identity_team}"
                ),

            "games":
                games,

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

            "regime_start_rows":
                regime_start_rows,

            "coach_change_rows":
                coach_change_rows,

            "observed_new_regime_flag":
                int(
                    (
                        regime_start_rows > 0
                    )
                    or
                    (
                        coach_change_rows > 0
                    )
                ),

            "min_coach_team_era_games_prior":
                min_prior_era_games,

            "max_coach_team_era_games_prior":
                max_prior_era_games,

            "evidence_confidence":
                evidence_confidence(
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
            )
            .fillna(0)
            .sum()
        )

        row[
            "total_rush_calls"
        ] = int(
            pd.to_numeric(
                g["rush_calls"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        row[
            "total_fourth_down_scrimmage"
        ] = int(
            pd.to_numeric(
                g[
                    "fourth_down_scrimmage"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        if (
            row["total_pass_calls"]
            + row["total_rush_calls"]
        ) != total_plays:
            raise RuntimeError(
                "FAIL: pass+rush call total "
                "does not equal scrimmage plays "
                f"for {coach_identity}|"
                f"{identity_team}"
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    print("=" * 72)
    print(VERSION)
    print("=" * 72)

    if not INPUT.exists():
        raise SystemExit(
            f"FAIL: missing input: {INPUT}"
        )

    df = pd.read_csv(
        INPUT
    )

    if df.empty:
        raise SystemExit(
            "FAIL: coach-attached evidence is empty"
        )

    required = {
        "event_id",
        "game_id",
        "team",
        "identity_lookup_team",
        "coach_identity",
        "coach_change_flag",
        "coach_team_era_start_flag",
        "coach_team_era_games_prior",
        "expected_pass_rate",
        "final_pass_rate",
        "pass_rate_delta",
        "scrimmage_plays",
        "pass_calls",
        "rush_calls",
        "fourth_down_scrimmage",
    }

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise SystemExit(
            "FAIL: missing required columns: "
            + ", ".join(missing)
        )

    unresolved = df[
        df["coach_identity"].isna()
        |
        df[
            "identity_lookup_team"
        ].isna()
    ]

    if not unresolved.empty:
        print(
            "FAIL: unresolved coach/regime rows"
        )

        print(
            unresolved[
                [
                    "event_id",
                    "game_id",
                    "team",
                    "identity_lookup_team",
                    "coach_identity",
                ]
            ].to_string(
                index=False
            )
        )

        raise SystemExit(1)

    expected_key = (
        df["coach_identity"]
        .astype(str)
        + "|"
        + df[
            "identity_lookup_team"
        ].astype(str)
    )

    if (
        "coach_team_regime_key"
        in df.columns
    ):
        supplied_key = (
            df[
                "coach_team_regime_key"
            ].astype(str)
        )

        bad_key = df[
            supplied_key
            != expected_key
        ]

        if not bad_key.empty:
            print(
                "FAIL: regime key mismatch"
            )

            print(
                bad_key[
                    [
                        "event_id",
                        "game_id",
                        "team",
                        "coach_identity",
                        "identity_lookup_team",
                        "coach_team_regime_key",
                    ]
                ].to_string(
                    index=False
                )
            )

            raise SystemExit(1)

    profile = build_profile(
        df
    )

    if profile.empty:
        raise SystemExit(
            "FAIL: no regime profile rows created"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile = profile.sort_values(
        [
            "identity_lookup_team",
            "coach_identity",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    profile.to_csv(
        OUT_CSV,
        index=False,
    )

    records = (
        profile
        .replace(
            {np.nan: None}
        )
        .to_dict(
            orient="records"
        )
    )

    OUT_JSON.write_text(
        json.dumps(
            {
                "version":
                    VERSION,

                "generated_at_utc":
                    utc_now(),

                "policy": {
                    "scope":
                        "coach_team_regime",

                    "regime_key":
                        "coach_identity|identity_lookup_team",

                    "weighting":
                        "scrimmage_play_count",

                    "cross_regime_blending":
                        False,

                    "fuzzy_matching":
                        False,

                    "production_influence":
                        False,

                    "forecast_mutation":
                        False,

                    "solver_influence":
                        False,

                    "persistent_coach_prior_mutation":
                        False,
                },

                "rows":
                    records,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )

    display = profile[
        [
            "identity_lookup_team",
            "coach_identity",
            "games",
            "scrimmage_plays",
            "baseline_games",
            "weighted_final_pass_rate",
            "weighted_expected_pass_rate",
            "weighted_pass_rate_delta",
            "observed_new_regime_flag",
            "evidence_confidence",
        ]
    ]

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "REGIMES:",
        len(profile),
    )

    print(
        "NEW REGIME FLAGS:",
        int(
            profile[
                "observed_new_regime_flag"
            ].sum()
        ),
    )

    print()
    print(
        "CSV :",
        OUT_CSV,
    )

    print(
        "JSON:",
        OUT_JSON,
    )

    print("PASS")


if __name__ == "__main__":
    main()
