#!/bin/bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
PYTHON="$ROOT/venv/bin/python"

SOURCE="$ROOT/data/parquet/nfl_current_fanduel_expectation_gav2.parquet"
PRODUCTION="$ROOT/data/parquet/current_unified_fanduel_expectation.parquet"
MANIFEST="$ROOT/data/parquet/current_unified_fanduel_expectation_manifest.json"

run() {
    echo "===== $1 ====="
    shift
    "$@"
}

cd "$ROOT"

case "${1:-}" in
    ""|--build-only|--publish-only) ;;
    *) echo "Unknown Stage24 lifecycle mode: $1" >&2; exit 2 ;;
esac

if [ "${1:-}" != "--publish-only" ]; then
run "CURRENT TEAM ENVIRONMENT" \
    "$PYTHON" current_team_environment.py

run "CURRENT OFFENSIVE MODEL MATRIX" \
    "$PYTHON" current_offensive_model_matrix.py

run "CURRENT MATCHUP INTELLIGENCE SHADOW REFRESH" \
    "$PYTHON" scripts/run_current_mi_shadow_refresh_v1.py

run "CURRENT OFFENSIVE STAT FORECAST" \
    "$PYTHON" current_offensive_stat_forecast.py

run "CURRENT OFFENSIVE RECONCILIATION V3" \
    "$PYTHON" scripts/build_offensive_team_reconciliation_shadow_v3.py

run "CURRENT KICKER MATRIX" \
    "$PYTHON" current_kicker_stat_forecast.py --build-current-matrix

run "CURRENT KICKER STAT FORECAST" \
    "$PYTHON" current_kicker_stat_forecast.py --score-current-matrix

run "CURRENT DST POSTGAME ACTUALS" \
    "$PYTHON" current_dst_postgame_actuals.py --season 2026

run "CURRENT DST FEATURES" \
    "$PYTHON" dst_current_features.py

run "CURRENT DST STAT FORECAST" \
    "$PYTHON" current_dst_stat_forecast.py

run "CURRENT UNIFIED STAT FORECAST" \
    "$PYTHON" current_unified_stat_forecast.py

run "CURRENT FANDUEL EXPECTATION" \
    "$PYTHON" current_fanduel_expectation.py --current

run "CURRENT GAV2" \
    "$PYTHON" current_fanduel_gav2.py --current
fi

if [ "${1:-}" = "--build-only" ]; then
    echo "CURRENT_STAGE24_CANDIDATE=PASS; publication deferred to bound V3 handoff"
    exit 0
fi

echo "===== VALIDATE + ATOMICALLY PUBLISH SOLVER AUTHORITY ====="

"$PYTHON" - "$SOURCE" "$PRODUCTION" "$MANIFEST" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import tempfile

import pandas as pd

source = Path(sys.argv[1])
production = Path(sys.argv[2])
manifest = Path(sys.argv[3])

CONTRACT = "WFS_FANDUEL_EXPECTATION_PUBLISH_V1"
GAV2_CONTRACT = "WFS_GLOBAL_PLAYER_AVAILABILITY_V2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


if not source.is_file():
    raise RuntimeError(f"Missing GAV2 source: {source}")

df = pd.read_parquet(source)

required = {
    "game_id",
    "season",
    "week",
    "team",
    "expected_fanduel_points",
}

missing = sorted(required - set(df.columns))
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

if df.empty:
    raise RuntimeError("GAV2 source is empty")

season = sorted(
    df["season"].dropna().astype(int).unique().tolist()
)
week = sorted(
    df["week"].dropna().astype(int).unique().tolist()
)

games = int(df["game_id"].nunique())
teams = int(df["team"].nunique())

if len(season) != 1:
    raise RuntimeError(
        f"Expected one season, found {season}"
    )

if len(week) != 1:
    raise RuntimeError(
        f"Expected one week, found {week}"
    )

if games <= 0 or teams <= 0:
    raise RuntimeError(
        f"Invalid inventory: games={games} teams={teams}"
    )

if df["expected_fanduel_points"].isna().any():
    raise RuntimeError(
        "Null expected_fanduel_points detected"
    )

if (
    df["game_id"].isna().any()
    or df["team"].isna().any()
):
    raise RuntimeError(
        "Null game/team metadata detected"
    )

source_sha = sha256(source)

production.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with tempfile.NamedTemporaryFile(
    prefix=production.name + ".",
    suffix=".tmp",
    dir=production.parent,
    delete=False,
) as tmp:
    tmp_path = Path(tmp.name)

try:
    df.to_parquet(tmp_path, index=False)

    check = pd.read_parquet(tmp_path)

    if len(check) != len(df):
        raise RuntimeError(
            "Temporary publication row mismatch: "
            f"{len(check)} != {len(df)}"
        )

    os.replace(tmp_path, production)

finally:
    if tmp_path.exists():
        tmp_path.unlink()

production_sha = sha256(production)

if production_sha != source_sha:
    raise RuntimeError(
        "Published artifact SHA does not match "
        "validated GAV2 source"
    )

prior_manifest = {}

if manifest.is_file():
    try:
        prior_manifest = json.loads(
            manifest.read_text()
        )
    except Exception:
        prior_manifest = {}

payload = {
    "contract": CONTRACT,
    "status": "PRODUCTION_CURRENT_GAV2",
    "published_at_utc":
        datetime.now(timezone.utc).isoformat(),
    "current_path": str(production),
    "current_sha256": production_sha,
    "production_current_path": str(production),
    "production_current_sha256": production_sha,
    "source_path": str(source),
    "source_sha256": source_sha,
    "gav2_contract": GAV2_CONTRACT,
    "projection_column":
        "expected_fanduel_points",
    "projection_policy": {
        "fallback": False,
        "missing_projection": "FAIL_CLOSED",
    },
    "identity_policy": {
        "classic_players": "EXACT_GSIS_ID",
        "classic_dst": "EXACT_CANONICAL_TEAM",
        "fuzzy_matching": False,
    },
    "rows": int(len(df)),
    "games": games,
    "teams": teams,
    "season": season[0],
    "week": week[0],
    "prior_manifest_current_sha256":
        prior_manifest.get("current_sha256"),
}

with tempfile.NamedTemporaryFile(
    mode="w",
    prefix=manifest.name + ".",
    suffix=".tmp",
    dir=manifest.parent,
    delete=False,
) as tmp:
    manifest_tmp = Path(tmp.name)

    json.dump(
        payload,
        tmp,
        indent=2,
        sort_keys=True,
    )

    tmp.write("\n")
    tmp.flush()
    os.fsync(tmp.fileno())

try:
    os.replace(
        manifest_tmp,
        manifest,
    )
finally:
    if manifest_tmp.exists():
        manifest_tmp.unlink()

verify_manifest = json.loads(
    manifest.read_text()
)

if (
    verify_manifest.get("current_sha256")
    != production_sha
):
    raise RuntimeError(
        "Published manifest SHA verification failed"
    )

print(f"PUBLISHED={production}")
print(f"MANIFEST={manifest}")
print(f"ROWS={len(df)}")
print(f"GAMES={games}")
print(f"TEAMS={teams}")
print(f"SEASON={season[0]}")
print(f"WEEK={week[0]}")
print(f"SHA256={production_sha}")
print("MANIFEST_SHA_MATCH=PASS")
print("CURRENT_STAGE24_SOLVER_AUTHORITY=PASS")
PY

echo "===== CURRENT STAGE24 REFRESH PASS ====="
