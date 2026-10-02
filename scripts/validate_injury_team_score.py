from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
RQ = ROOT / "processed/forecast_v1_team_replacement_quality.csv"
DB = ROOT / "data/nfl.db"

r = pd.read_csv(RQ)

with sqlite3.connect(DB) as con:
    g = pd.read_sql_query("""
        SELECT game_id, season, week,
               away_team, home_team,
               away_score, home_score,
               spread_line, total_line
        FROM games
        WHERE season BETWEEN 2023 AND 2025
          AND completed = 1
          AND total_line IS NOT NULL
          AND spread_line IS NOT NULL
    """, con)

home = g[[
    "game_id","season","week","home_team",
    "home_score","spread_line","total_line"
]].copy()
home["team"] = home["home_team"]
home["actual_points"] = home["home_score"]
home["market_implied_points"] = (
    home["total_line"] - home["spread_line"]
) / 2.0

away = g[[
    "game_id","season","week","away_team",
    "away_score","spread_line","total_line"
]].copy()
away["team"] = away["away_team"]
away["actual_points"] = away["away_score"]
away["market_implied_points"] = (
    away["total_line"] + away["spread_line"]
) / 2.0

scores = pd.concat([
    home[["game_id","season","week","team",
          "actual_points","market_implied_points"]],
    away[["game_id","season","week","team",
          "actual_points","market_implied_points"]],
], ignore_index=True)

d = r.merge(
    scores,
    on=["game_id","season","week","team"],
    how="inner",
    validate="one_to_one",
)

d["score_residual"] = (
    d["actual_points"] - d["market_implied_points"]
)

features = [
    c for c in d.columns
    if c.startswith("replacement_")
    and c not in {
        "replacement_status",
    }
]

features = [
    c for c in features
    if pd.api.types.is_numeric_dtype(d[c])
]

d = d[d["replacement_qb_injury_count"] > 0].copy()
train = d[d["season"].isin([2023, 2024])].copy()
test = d[d["season"] == 2025].copy()

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    Ridge(alpha=10.0),
)

model.fit(train[features], train["score_residual"])
pred = model.predict(test[features])

baseline = np.zeros(len(test))

model_mae = mean_absolute_error(
    test["score_residual"], pred
)
baseline_mae = mean_absolute_error(
    test["score_residual"], baseline
)

print("ROWS", len(d))
print("TRAIN", len(train))
print("TEST", len(test))
print("FEATURES", len(features))
print("BASELINE_MAE", round(baseline_mae, 4))
print("MODEL_MAE", round(model_mae, 4))
print(
    "MAE_IMPROVEMENT",
    round(baseline_mae - model_mae, 4),
)
print(
    "PRED_MEAN",
    round(float(np.mean(pred)), 4),
)
print(
    "PRED_STD",
    round(float(np.std(pred)), 4),
)
