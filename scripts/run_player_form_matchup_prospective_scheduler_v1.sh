#!/usr/bin/env bash

# WFS Player Form + Matchup Prospective Scheduler V1
#
# Research-only.
#
# Purpose:
#   Preserve immutable pre-kickoff player form / role / matchup evidence
#   for later prospective grading.
#
# This scheduler MUST NOT:
#   - publish production projections
#   - modify solver authority
#   - modify eligibility
#   - modify GPP scores
#   - modify UI
#   - restart services
#   - touch NFL Live
#
# Failure policy:
#   Research failure is logged and isolated from production.

set -uo pipefail

ROOT="/home/mwynn/nfl_data_engine"
PYTHON="$ROOT/venv/bin/python"

STAGE1="$ROOT/research/build_player_form_matchup_shadow_v1.py"
CAPTURE="$ROOT/research/capture_player_form_matchup_prospective_v1.py"

STAGE1_ARTIFACT="$ROOT/data/research/player_form_matchup_shadow_v1.parquet"
CAPTURE_DIR="$ROOT/data/research/prospective/player_form_matchup_v1"

LOCK="$ROOT/data/research/prospective/player_form_matchup_v1/.capture_scheduler.lock"

cd "$ROOT" || exit 0

mkdir -p "$CAPTURE_DIR"

# ------------------------------------------------------------
# Prevent overlapping cron executions.
# Research failure must never affect production.
# ------------------------------------------------------------
exec 9>"$LOCK"

if ! flock -n 9; then
    echo "$(date -u +%FT%TZ) CAPTURE_SCHEDULER=SKIP reason=LOCKED"
    exit 0
fi


# ------------------------------------------------------------
# Inspect the current Stage-1 artifact.
#
# Return codes:
#   0 = target is pregame and capture does not yet exist
#   10 = immutable capture already exists
#   11 = no usable Stage-1 artifact
#   12 = target has reached/passed kickoff
# ------------------------------------------------------------
inspect_target() {
"$PYTHON" - "$STAGE1_ARTIFACT" "$CAPTURE_DIR" <<'PY'
import sys
from pathlib import Path

import pandas as pd


stage1_path = Path(sys.argv[1])
capture_dir = Path(sys.argv[2])

if not stage1_path.is_file():
    print("INSPECT=NO_STAGE1")
    sys.exit(11)

try:
    d = pd.read_parquet(stage1_path)
except Exception as exc:
    print(f"INSPECT=STAGE1_READ_FAILED error={exc}")
    sys.exit(11)

required = {
    "season",
    "week",
    "game_id",
    "target_kickoff_utc",
}

missing = sorted(required - set(d.columns))

if missing:
    print(f"INSPECT=STAGE1_MISSING_COLUMNS columns={missing}")
    sys.exit(11)

if d.empty:
    print("INSPECT=STAGE1_EMPTY")
    sys.exit(11)

targets = d[
    ["season", "week", "game_id", "target_kickoff_utc"]
].drop_duplicates()

season_values = sorted(
    pd.to_numeric(
        targets["season"],
        errors="coerce",
    ).dropna().astype(int).unique().tolist()
)

week_values = sorted(
    pd.to_numeric(
        targets["week"],
        errors="coerce",
    ).dropna().astype(int).unique().tolist()
)

if len(season_values) != 1 or len(week_values) != 1:
    print(
        "INSPECT=AMBIGUOUS_TARGET "
        f"seasons={season_values} weeks={week_values}"
    )
    sys.exit(11)

season = season_values[0]
week = week_values[0]

kickoff = pd.to_datetime(
    targets["target_kickoff_utc"],
    utc=True,
    errors="coerce",
)

if kickoff.isna().any():
    print("INSPECT=INVALID_KICKOFF")
    sys.exit(11)

now = pd.Timestamp.now(tz="UTC")

capture = (
    capture_dir
    / f"{season}_week_{week:02d}_pregame.parquet"
)

manifest = (
    capture_dir
    / f"{season}_week_{week:02d}_pregame_manifest.json"
)

print(
    f"TARGET={season}-W{week:02d} "
    f"games={targets['game_id'].nunique()} "
    f"earliest_kickoff={kickoff.min().isoformat()}"
)

# A partial immutable pair is abnormal.
if capture.exists() != manifest.exists():
    print(
        "INSPECT=PARTIAL_CAPTURE "
        f"parquet={capture.exists()} "
        f"manifest={manifest.exists()}"
    )
    sys.exit(11)

if capture.exists() and manifest.exists():
    print("INSPECT=ALREADY_CAPTURED")
    sys.exit(10)

# Capture script itself performs the final strict kickoff gate.
if (kickoff <= now).any():
    print(
        "INSPECT=TARGET_STARTED "
        f"now={now.isoformat()}"
    )
    sys.exit(12)

print("INSPECT=READY")
sys.exit(0)
PY
}


echo "$(date -u +%FT%TZ) CAPTURE_SCHEDULER=START"

inspect_target
inspect_rc=$?

case "$inspect_rc" in

    10)
        echo "$(date -u +%FT%TZ) CAPTURE_SCHEDULER=OK reason=ALREADY_CAPTURED"
        exit 0
        ;;

    12)
        echo "$(date -u +%FT%TZ) CAPTURE_SCHEDULER=SKIP reason=TARGET_STARTED"
        exit 0
        ;;

    11)
        # Stage-1 may simply not represent a usable current target yet.
        # Attempt one research-only rebuild below.
        echo "$(date -u +%FT%TZ) CAPTURE_SCHEDULER=INFO reason=STAGE1_NOT_READY"
        ;;

    0)
        echo "$(date -u +%FT%TZ) CAPTURE_SCHEDULER=READY"
        ;;

    *)
        echo "$(date -u +%FT%TZ) CAPTURE_SCHEDULER=ERROR reason=INSPECT_RC_$inspect_rc"
        exit 0
        ;;
esac


# ------------------------------------------------------------
# Rebuild Stage 1 from current authoritative research inputs.
# This remains shadow/research-only.
# ------------------------------------------------------------
echo "$(date -u +%FT%TZ) STAGE1_REBUILD=START"

if ! "$PYTHON" "$STAGE1"; then
    echo "$(date -u +%FT%TZ) STAGE1_REBUILD=FAILED"
    exit 0
fi

echo "$(date -u +%FT%TZ) STAGE1_REBUILD=PASS"


# ------------------------------------------------------------
# Re-inspect AFTER rebuilding Stage 1.
# The current target may have changed.
# ------------------------------------------------------------
inspect_target
inspect_rc=$?

case "$inspect_rc" in

    10)
        echo "$(date -u +%FT%TZ) CAPTURE_SCHEDULER=OK reason=ALREADY_CAPTURED_AFTER_REBUILD"
        exit 0
        ;;

    12)
        echo "$(date -u +%FT%TZ) CAPTURE_SCHEDULER=SKIP reason=TARGET_STARTED_AFTER_REBUILD"
        exit 0
        ;;

    0)
        ;;

    *)
        echo "$(date -u +%FT%TZ) CAPTURE_SCHEDULER=SKIP reason=NOT_READY_AFTER_REBUILD rc=$inspect_rc"
        exit 0
        ;;
esac


# ------------------------------------------------------------
# Immutable prospective capture.
# Capture script independently validates:
#   hashes
#   exact identity
#   schedule binding
#   kickoff
#   completion status
#   immutability
# ------------------------------------------------------------
echo "$(date -u +%FT%TZ) IMMUTABLE_CAPTURE=START"

if ! "$PYTHON" "$CAPTURE"; then
    echo "$(date -u +%FT%TZ) IMMUTABLE_CAPTURE=FAILED"
    exit 0
fi

echo "$(date -u +%FT%TZ) IMMUTABLE_CAPTURE=PASS"
echo "$(date -u +%FT%TZ) CAPTURE_SCHEDULER=PASS"

exit 0
