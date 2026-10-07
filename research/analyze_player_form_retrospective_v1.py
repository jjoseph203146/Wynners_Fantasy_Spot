#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
USAGE = ROOT / "data/parquet/nfl_player_weekly_usage.parquet"
DVP = ROOT / "data/parquet/nfl_team_position_dvp.parquet"
OUT = ROOT / "data/research/player_form_retrospective_v1.parquet"
SUMMARY = ROOT / "data/research/player_form_retrospective_v1_summary.csv"

VALID_POS = {"QB", "RB", "WR", "TE"}


def direction(x):
    if pd.isna(x):
        return "UNAVAILABLE"
    if x > 0:
        return "UP"
    if x < 0:
        return "DOWN"
    return "FLAT"


def main():
    u = pd.read_parquet(USAGE).copy()
    d = pd.read_parquet(DVP).copy()

    # Offensive fantasy positions only.
    u["position"] = u["position"].astype(str).str.upper()
    u = u[u["position"].isin(VALID_POS)].copy()

    # Regular season only for first calibration pass.
    if "season_type" in u.columns:
        u = u[u["season_type"].astype(str).eq("REG")].copy()

    u["season"] = pd.to_numeric(u["season"], errors="coerce")
    u["week"] = pd.to_numeric(u["week"], errors="coerce")
    u = u.dropna(subset=["season", "week", "player_id"])
    u["season"] = u["season"].astype(int)
    u["week"] = u["week"].astype(int)

    # Use actual observed player/game data.
    numeric = [
        "fanduel_points", "opportunities", "offense_pct",
        "targets", "carries", "receptions",
        "rushing_yards", "receiving_yards",
        "rushing_tds", "receiving_tds",
    ]
    for c in numeric:
        if c in u.columns:
            u[c] = pd.to_numeric(u[c], errors="coerce")

    # One row per player/game.
    key = ["season", "week", "game_id", "player_id"]
    dupes = int(u.duplicated(key).sum())
    if dupes:
        raise RuntimeError(f"Duplicate player/game rows found: {dupes}")

    u = u.sort_values(
        ["player_id", "season", "week", "game_id"]
    ).reset_index(drop=True)

    rows = []

    # Historical reconstruction is season-bounded.
    for (player_id, season), g in u.groupby(
        ["player_id", "season"], sort=False
    ):
        g = g.sort_values(["week", "game_id"]).reset_index(drop=True)

        for i in range(len(g)):
            current = g.iloc[i]
            prior = g.iloc[:i]

            # Need 3 prior observed games for the first useful state.
            if len(prior) < 3:
                continue

            p3 = prior.tail(3)
            p5 = prior.tail(5)

            fd3 = pd.to_numeric(
                p3["fanduel_points"], errors="coerce"
            ).dropna()
            fd5 = pd.to_numeric(
                p5["fanduel_points"], errors="coerce"
            ).dropna()

            op3 = pd.to_numeric(
                p3["opportunities"], errors="coerce"
            ).dropna()
            op5 = pd.to_numeric(
                p5["opportunities"], errors="coerce"
            ).dropna()

            if len(fd3) < 3 or len(op3) < 3:
                continue

            fd_avg_3 = fd3.mean()
            fd_avg_5 = fd5.mean() if len(fd5) else np.nan

            op_avg_3 = op3.mean()
            op_avg_5 = op5.mean() if len(op5) else np.nan

            # Match Stage-1 semantics:
            # recent 3-game average minus available 5-game average.
            production_trend = (
                fd_avg_3 - fd_avg_5
                if pd.notna(fd_avg_5) else np.nan
            )
            opportunity_trend = (
                op_avg_3 - op_avg_5
                if pd.notna(op_avg_5) else np.nan
            )

            prod_dir = direction(production_trend)
            opp_dir = direction(opportunity_trend)

            actual_fd = current.get("fanduel_points", np.nan)

            # Compare next-game result to its own pregame baselines.
            rebound_vs_3 = (
                actual_fd - fd_avg_3
                if pd.notna(actual_fd) else np.nan
            )
            rebound_vs_5 = (
                actual_fd - fd_avg_5
                if pd.notna(actual_fd) and pd.notna(fd_avg_5)
                else np.nan
            )

            rows.append({
                "season": int(current["season"]),
                "week": int(current["week"]),
                "game_id": current["game_id"],
                "player_id": player_id,
                "player_name": current.get("player_display_name"),
                "team": current.get("team"),
                "opponent_team": current.get("opponent_team"),
                "position": current.get("position"),

                "history_games": len(prior),
                "fd_avg_3": fd_avg_3,
                "fd_avg_5": fd_avg_5,
                "production_trend": production_trend,
                "production_direction": prod_dir,

                "opportunities_avg_3": op_avg_3,
                "opportunities_avg_5": op_avg_5,
                "opportunity_trend": opportunity_trend,
                "opportunity_direction": opp_dir,

                "state": f"{prod_dir}__{opp_dir}",

                "actual_fd": actual_fd,
                "actual_opportunities": current.get("opportunities"),
                "actual_targets": current.get("targets"),
                "actual_carries": current.get("carries"),
                "actual_receptions": current.get("receptions"),
                "actual_rushing_yards": current.get("rushing_yards"),
                "actual_receiving_yards": current.get("receiving_yards"),
                "actual_rushing_tds": current.get("rushing_tds"),
                "actual_receiving_tds": current.get("receiving_tds"),

                "fd_change_vs_avg3": rebound_vs_3,
                "fd_change_vs_avg5": rebound_vs_5,
                "beat_avg3": (
                    bool(actual_fd > fd_avg_3)
                    if pd.notna(actual_fd) else False
                ),
                "beat_avg5": (
                    bool(actual_fd > fd_avg_5)
                    if pd.notna(actual_fd) and pd.notna(fd_avg_5)
                    else False
                ),
            })

    r = pd.DataFrame(rows)

    if r.empty:
        raise RuntimeError("No retrospective rows produced.")

    # ------------------------------------------------------------
    # Attach PRECOMPUTED historical DvP for that target week.
    # The DvP table itself is already week-specific/pregame.
    # ------------------------------------------------------------
    d["position"] = d["position"].astype(str).str.upper()
    d = d[d["position"].isin(VALID_POS)].copy()

    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["week"] = pd.to_numeric(d["week"], errors="coerce")
    d = d.dropna(subset=["season", "week"])
    d["season"] = d["season"].astype(int)
    d["week"] = d["week"].astype(int)

    dvp_cols = [
        "season", "week", "defense_team", "position",
        "history_games",
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
    d = d[[c for c in dvp_cols if c in d.columns]].copy()

    d = d.rename(columns={
        "defense_team": "opponent_team",
        "history_games": "dvp_history_games",
    })

    r = r.merge(
        d,
        on=["season", "week", "opponent_team", "position"],
        how="left",
        validate="many_to_one",
    )

    # Preserve raw DvP evidence. No ELITE/POOR threshold yet.
    r["dvp_fd_direction"] = r["fd_allowed_trend"].apply(direction)
    r["dvp_opportunity_direction"] = (
        r["opportunity_allowed_trend"].apply(direction)
    )

    # Minimum chronology sanity:
    # every state is based only on rows preceding target row
    # inside the same player-season.
    r["chronology_status"] = "PRIOR_GAMES_ONLY"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    r.to_parquet(OUT, index=False)

    summary = (
        r.groupby(
            ["position", "production_direction", "opportunity_direction"],
            dropna=False
        )
        .agg(
            players_games=("player_id", "size"),
            avg_actual_fd=("actual_fd", "mean"),
            median_actual_fd=("actual_fd", "median"),
            avg_change_vs_3=("fd_change_vs_avg3", "mean"),
            median_change_vs_3=("fd_change_vs_avg3", "median"),
            beat_avg3_rate=("beat_avg3", "mean"),
            avg_dvp_fd_allowed_3=("fd_allowed_avg_3", "mean"),
            avg_dvp_opportunities_3=(
                "opportunities_allowed_avg_3", "mean"
            ),
        )
        .reset_index()
        .sort_values(
            ["production_direction", "opportunity_direction", "position"]
        )
    )

    summary.to_csv(SUMMARY, index=False)

    print("\nSTAGE 2 RETROSPECTIVE V1")
    print("=" * 80)
    print("Rows:", len(r))
    print("Seasons:", sorted(r["season"].unique().tolist()))
    print("\nPosition counts:")
    print(r["position"].value_counts().to_string())

    print("\nProduction x Opportunity:")
    print(pd.crosstab(
        r["production_direction"],
        r["opportunity_direction"],
        margins=True
    ).to_string())

    print("\nKEY DOWN COHORTS:")
    down = r[r["production_direction"].eq("DOWN")]
    key = (
        down.groupby("opportunity_direction")
        .agg(
            N=("player_id", "size"),
            avg_actual_fd=("actual_fd", "mean"),
            median_actual_fd=("actual_fd", "median"),
            avg_change_vs_3=("fd_change_vs_avg3", "mean"),
            median_change_vs_3=("fd_change_vs_avg3", "median"),
            beat_avg3_rate=("beat_avg3", "mean"),
        )
        .sort_index()
    )
    print(key.to_string())

    print("\nDOWN COHORTS BY POSITION:")
    key_pos = (
        down.groupby(["position", "opportunity_direction"])
        .agg(
            N=("player_id", "size"),
            avg_change_vs_3=("fd_change_vs_avg3", "mean"),
            beat_avg3_rate=("beat_avg3", "mean"),
        )
    )
    print(key_pos.to_string())

    print("\nDvP coverage:")
    print(
        r["dvp_history_games"]
        .notna()
        .value_counts(dropna=False)
        .rename(index={True: "AVAILABLE", False: "MISSING"})
        .to_string()
    )

    print("\nOutput:")
    print(OUT)
    print(SUMMARY)

    print("\nSAFETY:")
    print("Production influence: NONE")
    print("Solver influence: NONE")
    print("Eligibility influence: NONE")
    print("Classification thresholds: NONE")
    print("Chronology: PRIOR GAMES ONLY")


if __name__ == "__main__":
    main()
