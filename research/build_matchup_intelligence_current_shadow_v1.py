from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SLATE = ROOT / "data/parquet/nfl_current_slate_features.parquet"
TEAM_ENV = ROOT / "data/parquet/nfl_current_team_environment.parquet"
PLAYER_USAGE = ROOT / "data/parquet/nfl_player_weekly_usage.parquet"
PLAYER_STATS = ROOT / "data/parquet/nfl_player_game_stats.parquet"
SITUATIONAL = ROOT / "data/research/situational_role_history_v1.parquet"

OUTPUT = ROOT / "data/research/matchup_intelligence_current_shadow_v1.parquet"

POSITIONS = ["QB", "RB", "WR", "TE"]

DVP_SUM_COLS = [
    "fanduel_points",
    "targets",
    "carries",
    "receptions",
    "receiving_yards",
    "rushing_yards",
    "receiving_tds",
    "rushing_tds",
    "opportunities",
]

DVP_FEATURES = [
    "fd_allowed_last",
    "fd_allowed_avg_3",
    "fd_allowed_avg_5",
    "targets_allowed_avg_3",
    "carries_allowed_avg_3",
    "receptions_allowed_avg_3",
    "receiving_yards_allowed_avg_3",
    "rushing_yards_allowed_avg_3",
    "receiving_tds_allowed_avg_3",
    "rushing_tds_allowed_avg_3",
    "opportunities_allowed_avg_3",
    "fd_allowed_trend",
    "opportunity_allowed_trend",
]


def prior_mask(df, season, week):
    return (
        (df["season"] < season)
        |
        (
            df["season"].eq(season)
            & df["week"].lt(week)
        )
    )


def build_player_baseline(current, usage, season, week):
    prior = usage[
        prior_mask(usage, season, week)
        & usage["position"].isin(POSITIONS)
    ].copy()

    raw = [
        "carries",
        "targets",
        "offense_pct",
        "fanduel_points",
    ]

    for col in raw:
        prior[col] = pd.to_numeric(
            prior[col],
            errors="coerce",
        )

    if prior[raw].isna().any().any():
        raise RuntimeError("PLAYER_BASELINE_RAW_NULL")

    prior["usage_value"] = (
        prior["carries"]
        + prior["targets"]
    )

    prior = prior.sort_values(
        ["player_id", "season", "week", "game_id"]
    )

    rows = []

    for _, player in current.iterrows():
        ph = prior[
            prior["player_id"].eq(player["player_id"])
        ].tail(3)

        n_all = int(
            (
                prior["player_id"]
                == player["player_id"]
            ).sum()
        )

        row = {
            "game_id": player["game_id"],
            "player_id": player["player_id"],
            "prior_player_games": n_all,
            "player_baseline_available": n_all > 0,
        }

        if n_all > 0:
            row.update(
                {
                    "usage_3g":
                        float(ph["usage_value"].mean()),
                    "snap_pct_3g":
                        float(ph["offense_pct"].mean()),
                    "target_3g":
                        float(ph["targets"].mean()),
                    "carry_3g":
                        float(ph["carries"].mean()),
                    "fanduel_3g":
                        float(ph["fanduel_points"].mean()),
                }
            )
        else:
            row.update(
                {
                    "usage_3g": np.nan,
                    "snap_pct_3g": np.nan,
                    "target_3g": np.nan,
                    "carry_3g": np.nan,
                    "fanduel_3g": np.nan,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_current_dvp(stats, season, week):
    hist = stats[
        prior_mask(stats, season, week)
        & stats["position"].isin(POSITIONS)
    ].copy()

    numeric = [
        "fanduel_points",
        "targets",
        "carries",
        "receptions",
        "receiving_yards",
        "rushing_yards",
        "receiving_tds",
        "rushing_tds",
    ]

    for col in numeric:
        hist[col] = pd.to_numeric(
            hist[col],
            errors="coerce",
        )

    if hist[numeric].isna().any().any():
        raise RuntimeError("DVP_RAW_NULL")

    hist["opportunities"] = (
        hist["carries"]
        + hist["targets"]
    )

    game_dvp = (
        hist.groupby(
            [
                "game_id",
                "season",
                "week",
                "opponent_team",
                "position",
            ],
            as_index=False,
        )[DVP_SUM_COLS]
        .sum()
        .rename(
            columns={
                "opponent_team": "defense_team"
            }
        )
        .sort_values(
            [
                "defense_team",
                "position",
                "season",
                "week",
                "game_id",
            ]
        )
        .reset_index(drop=True)
    )

    rows = []

    for (defense, position), g in game_dvp.groupby(
        ["defense_team", "position"],
        sort=True,
    ):
        g = g.sort_values(
            ["season", "week", "game_id"]
        )

        row = {
            "defense_team": defense,
            "position": position,
            "dvp_history_games": int(len(g)),
            "fd_allowed_last":
                float(g["fanduel_points"].iloc[-1]),
            "fd_allowed_avg_3":
                float(g["fanduel_points"].tail(3).mean()),
            "fd_allowed_avg_5":
                float(g["fanduel_points"].tail(5).mean()),
            "targets_allowed_avg_3":
                float(g["targets"].tail(3).mean()),
            "carries_allowed_avg_3":
                float(g["carries"].tail(3).mean()),
            "receptions_allowed_avg_3":
                float(g["receptions"].tail(3).mean()),
            "receiving_yards_allowed_avg_3":
                float(g["receiving_yards"].tail(3).mean()),
            "rushing_yards_allowed_avg_3":
                float(g["rushing_yards"].tail(3).mean()),
            "receiving_tds_allowed_avg_3":
                float(g["receiving_tds"].tail(3).mean()),
            "rushing_tds_allowed_avg_3":
                float(g["rushing_tds"].tail(3).mean()),
            "opportunities_allowed_avg_3":
                float(g["opportunities"].tail(3).mean()),
        }

        row["fd_allowed_trend"] = (
            row["fd_allowed_avg_3"]
            - float(
                g["fanduel_points"]
                .tail(5)
                .mean()
            )
        )

        row["opportunity_allowed_trend"] = (
            row["opportunities_allowed_avg_3"]
            - float(
                g["opportunities"]
                .tail(5)
                .mean()
            )
        )

        rows.append(row)

    return pd.DataFrame(rows)


def build_high_value(current, situational, season, week):
    prior = situational[
        prior_mask(
            situational,
            season,
            week,
        )
    ].copy()

    configs = {
        "RB": ("g2g_carries", "rb_g2g_share"),
        "TE": ("i10_targets", "te_i10_share"),
    }

    rows = []

    for pos, (raw_col, share_col) in configs.items():
        hp = prior[
            prior["position"].eq(pos)
        ].copy()

        hp[raw_col] = pd.to_numeric(
            hp[raw_col],
            errors="coerce",
        )

        if hp[raw_col].isna().any():
            raise RuntimeError(
                f"{pos}_HIGH_VALUE_RAW_NULL"
            )

        cur = current[
            current["position"].eq(pos)
        ]

        pos_rows = []

        for _, player in cur.iterrows():
            ph = hp[
                hp["player_id"].eq(
                    player["player_id"]
                )
            ].sort_values(
                ["season", "week", "game_id"]
            )

            has_history = not ph.empty

            value = (
                float(
                    ph[raw_col]
                    .tail(3)
                    .mean()
                )
                if has_history
                else np.nan
            )

            pos_rows.append(
                {
                    "game_id": player["game_id"],
                    "player_id": player["player_id"],
                    "team": player["team"],
                    "position": pos,
                    "high_value_3g": value,
                    "high_value_history_available":
                        bool(has_history),
                }
            )

        x = pd.DataFrame(pos_rows)

        available = x[
            x["high_value_history_available"]
        ]

        denom = (
            available.groupby(
                ["game_id", "team"],
                as_index=False,
            )
            .agg(
                high_value_team_3g=(
                    "high_value_3g",
                    "sum",
                )
            )
        )

        x = x.merge(
            denom,
            on=["game_id", "team"],
            how="left",
            validate="many_to_one",
        )

        num = pd.to_numeric(
            x["high_value_3g"],
            errors="coerce",
        ).to_numpy(dtype=float)

        den = pd.to_numeric(
            x["high_value_team_3g"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = (
            x["high_value_history_available"]
            .to_numpy()
            & np.isfinite(num)
            & np.isfinite(den)
            & (den > 0)
        )

        share = np.full(
            len(x),
            np.nan,
            dtype=float,
        )

        np.divide(
            num,
            den,
            out=share,
            where=valid,
        )

        x[share_col] = share
        x[
            f"{share_col}_available"
        ] = valid

        rows.append(
            x[
                [
                    "game_id",
                    "player_id",
                    share_col,
                    f"{share_col}_available",
                ]
            ]
        )

    rb = rows[0]
    te = rows[1]

    out = current[
        ["game_id", "player_id"]
    ].copy()

    out = out.merge(
        rb,
        on=["game_id", "player_id"],
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        te,
        on=["game_id", "player_id"],
        how="left",
        validate="one_to_one",
    )

    for col in [
        "rb_g2g_share_available",
        "te_i10_share_available",
    ]:
        out[col] = (
            out[col]
            .fillna(False)
            .astype(bool)
        )

    return out


def main():
    print(
        "=== BUILD MATCHUP INTELLIGENCE "
        "CURRENT SHADOW V1 ==="
    )
    print("MODE=ANALYSIS_ONLY")

    slate = pd.read_parquet(SLATE)
    team_env = pd.read_parquet(TEAM_ENV)
    usage = pd.read_parquet(PLAYER_USAGE)
    stats = pd.read_parquet(PLAYER_STATS)
    situational = pd.read_parquet(SITUATIONAL)

    if (
        slate["season"].nunique() != 1
        or slate["week"].nunique() != 1
    ):
        raise RuntimeError(
            "CURRENT_TARGET_NOT_UNIQUE"
        )

    season = int(slate["season"].iloc[0])
    week = int(slate["week"].iloc[0])

    current = (
        slate[
            slate["position"].isin(POSITIONS)
        ][
            [
                "game_id",
                "season",
                "week",
                "player_id",
                "player_display_name",
                "position",
                "team",
                "opponent_team",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    if current.duplicated(
        ["game_id", "player_id"]
    ).any():
        raise RuntimeError(
            "CURRENT_PLAYER_DUPLICATION"
        )

    baseline = build_player_baseline(
        current,
        usage,
        season,
        week,
    )

    dvp = build_current_dvp(
        stats,
        season,
        week,
    )

    high_value = build_high_value(
        current,
        situational,
        season,
        week,
    )

    team = team_env.copy()

    team = team.rename(
        columns={
            "history_games":
                "team_history_games"
        }
    )

    shadow = current.merge(
        baseline,
        on=["game_id", "player_id"],
        how="left",
        validate="one_to_one",
    )

    shadow = shadow.merge(
        team,
        on=[
            "game_id",
            "season",
            "week",
            "team",
            "opponent_team",
        ],
        how="left",
        validate="many_to_one",
    )

    shadow = shadow.merge(
        dvp,
        left_on=[
            "opponent_team",
            "position",
        ],
        right_on=[
            "defense_team",
            "position",
        ],
        how="left",
        validate="many_to_one",
    )

    shadow = shadow.merge(
        high_value,
        on=["game_id", "player_id"],
        how="left",
        validate="one_to_one",
    )

    shadow["minimum_history_3"] = (
        shadow["prior_player_games"]
        .ge(3)
        & shadow["team_history_games"]
        .ge(3)
        & shadow["dvp_history_games"]
        .ge(3)
    )

    shadow["matchup_shadow_available"] = (
        shadow["minimum_history_3"]
    )

    # --------------------------------------------------------
    # Deterministic validation.
    # --------------------------------------------------------

    if len(shadow) != len(current):
        raise RuntimeError(
            "ROW_COUNT_CHANGED"
        )

    if shadow.duplicated(
        ["game_id", "player_id"]
    ).any():
        raise RuntimeError(
            "OUTPUT_DUPLICATION"
        )

    if shadow[
        "team_history_games"
    ].isna().any():
        raise RuntimeError(
            "TEAM_ENVIRONMENT_BIND_FAILURE"
        )

    if shadow[
        "dvp_history_games"
    ].isna().any():
        raise RuntimeError(
            "DVP_BIND_FAILURE"
        )

    if shadow[
        DVP_FEATURES
    ].isna().any().any():
        raise RuntimeError(
            "DVP_FEATURE_NULL"
        )

    cold = shadow[
        "prior_player_games"
    ].eq(0)

    baseline_cols = [
        "usage_3g",
        "snap_pct_3g",
        "target_3g",
        "carry_3g",
        "fanduel_3g",
    ]

    if shadow.loc[
        cold,
        baseline_cols,
    ].notna().any().any():
        raise RuntimeError(
            "COLD_START_BASELINE_NOT_EXPLICIT"
        )

    if shadow.loc[
        ~cold,
        baseline_cols,
    ].isna().any().any():
        raise RuntimeError(
            "HISTORICAL_PLAYER_BASELINE_MISSING"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shadow.to_parquet(
        OUTPUT,
        index=False,
    )

    print(f"TARGET={season}_WEEK_{week}")
    print(f"ROWS={len(shadow)}")
    print(
        f"GAMES={shadow['game_id'].nunique()}"
    )
    print(
        f"TEAMS={shadow['team'].nunique()}"
    )
    print(
        f"PLAYERS={shadow['player_id'].nunique()}"
    )
    print(
        "COLD_START_PLAYERS="
        f"{int(cold.sum())}"
    )
    print(
        "MIN_HISTORY_3_PLAYERS="
        f"{int(shadow['minimum_history_3'].sum())}"
    )
    print(
        "RB_G2G_SHARE_AVAILABLE="
        f"{int(shadow['rb_g2g_share_available'].sum())}"
    )
    print(
        "TE_I10_SHARE_AVAILABLE="
        f"{int(shadow['te_i10_share_available'].sum())}"
    )
    print(
        "DVP_HISTORY_MIN="
        f"{int(shadow['dvp_history_games'].min())}"
    )
    print(
        "TEAM_HISTORY_MIN="
        f"{int(shadow['team_history_games'].min())}"
    )
    print(f"OUTPUT={OUTPUT}")
    print("PRODUCTION_INFLUENCE=NO")
    print("SOLVER_INFLUENCE=NO")
    print("PROJECTION_INFLUENCE=NO")
    print("UPDATER_INFLUENCE=NO")
    print("STEP_7I_B_BUILD_STATUS=PASS")


if __name__ == "__main__":
    main()
