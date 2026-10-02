#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")

SRC = ROOT / "processed" / "forecast_prospective_evaluation_v1.csv"

OUT_JSON = ROOT / "processed" / "wfs_prediction_accuracy_v1.json"
OUT_CSV = ROOT / "processed" / "wfs_prediction_accuracy_v1.csv"

VERSION = "WFS_PREDICTION_ACCURACY_V1"


def main():
    print("=" * 72)
    print(VERSION)
    print("=" * 72)

    if not SRC.is_file():
        raise SystemExit(f"FAIL: missing {SRC}")

    df = pd.read_csv(SRC)

    required = {
        "game_id",
        "week",
        "away_team",
        "home_team",
        "pred_winner",
        "actual_winner",
    }

    missing = required - set(df.columns)

    if missing:
        raise SystemExit(
            f"FAIL: missing required columns: {sorted(missing)}"
        )

    graded = df[
        df["pred_winner"].notna()
        & df["actual_winner"].notna()
    ].copy()

    if graded.empty:
        print("NO GRADED GAMES")
        return

    graded["pred_winner"] = (
        graded["pred_winner"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    graded["actual_winner"] = (
        graded["actual_winner"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    graded["correct"] = (
        graded["pred_winner"] == graded["actual_winner"]
    )

    graded["result"] = graded["correct"].map({
        True: "WIN",
        False: "LOSS",
    })

    weekly_rows = []

    print()
    print("WEEKLY RECORD")
    print("-" * 72)

    season_wins = 0
    season_losses = 0

    weeks = sorted(
        int(w)
        for w in graded["week"].dropna().unique()
    )

    for week in weeks:
        w = graded[graded["week"] == week]

        wins = int(w["correct"].sum())
        losses = int(len(w) - wins)
        total = wins + losses

        pct = (
            wins / total * 100.0
            if total
            else 0.0
        )

        season_wins += wins
        season_losses += losses

        weekly_rows.append({
            "week": week,
            "wins": wins,
            "losses": losses,
            "games": total,
            "accuracy_pct": round(pct, 1),
        })

        print(
            f"Week {week:>2}: "
            f"{wins}-{losses} "
            f"({pct:.1f}%)"
        )

    season_total = season_wins + season_losses

    season_pct = (
        season_wins / season_total * 100.0
        if season_total
        else 0.0
    )

    print()
    print("SEASON RECORD")
    print("-" * 72)

    print(
        f"{season_wins}-{season_losses} "
        f"({season_pct:.1f}%) "
        f"| {season_total} graded games"
    )

    print()
    print("GAME-BY-GAME")
    print("-" * 72)

    game_rows = []

    sort_cols = ["week", "game_id"]

    for _, r in graded.sort_values(sort_cols).iterrows():

        game = {
            "game_id": str(r["game_id"]),
            "week": int(r["week"]),
            "away_team": str(r["away_team"]),
            "home_team": str(r["home_team"]),
            "pred_winner": str(r["pred_winner"]),
            "actual_winner": str(r["actual_winner"]),
            "correct": bool(r["correct"]),
            "result": str(r["result"]),
        }

        for col in (
            "pred_home_points",
            "pred_away_points",
            "pred_home_margin",
            "home_score",
            "away_score",
        ):
            if col in graded.columns:
                value = r.get(col)

                if pd.notna(value):
                    try:
                        game[col] = float(value)
                    except Exception:
                        game[col] = str(value)

        game_rows.append(game)

        print(
            f"Week {game['week']:>2} | "
            f"{game['away_team']} @ {game['home_team']} | "
            f"WFS {game['pred_winner']} | "
            f"Actual {game['actual_winner']} | "
            f"{game['result']}"
        )

    generated_at = datetime.now(timezone.utc).isoformat()

    payload = {
        "version": VERSION,
        "generated_at_utc": generated_at,
        "source": str(SRC.relative_to(ROOT)),
        "season": {
            "wins": season_wins,
            "losses": season_losses,
            "games": season_total,
            "accuracy_pct": round(season_pct, 1),
        },
        "weeks": weekly_rows,
        "games": game_rows,
    }

    OUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    weekly_df = pd.DataFrame(weekly_rows)

    weekly_df.to_csv(
        OUT_CSV,
        index=False,
    )

    print()
    print(f"JSON: {OUT_JSON}")
    print(f"CSV : {OUT_CSV}")
    print()
    print("PASS: WFS PREDICTION ACCURACY REPORT")


if __name__ == "__main__":
    main()
