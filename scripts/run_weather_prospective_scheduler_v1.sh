#!/usr/bin/env bash

set -u

ROOT="/home/mwynn/nfl_data_engine"
PYTHON="$ROOT/venv/bin/python"
CONTROLLER="$ROOT/scripts/run_weather_prospective_capture_v1.py"

LOG="$ROOT/logs/weather_prospective_scheduler.log"
LOCK="$ROOT/weather_prospective_scheduler.lock"

EXPECTED_CONTROLLER_SHA="119d9f76619659d465e4484704954d18944a2be711a5a459e5f0bfaff1b542ec"

MODE="${1:---dry-run}"

case "$MODE" in
    --dry-run|--execute)
        ;;
    *)
        echo "INVALID_MODE=$MODE"
        exit 64
        ;;
esac

cd "$ROOT" || exit 1
mkdir -p "$ROOT/logs"

exec 9>"$LOCK"

if ! flock -n 9; then
    {
        echo "============================================================"
        echo "WEATHER_SCHEDULER_SKIP=$(date -Is)"
        echo "SKIP_REASON=ALREADY_RUNNING"
        echo "============================================================"
    } >> "$LOG"
    exit 0
fi

{
    echo "============================================================"
    echo "WEATHER_SCHEDULER_RUN_START=$(date -Is)"
    echo "MODE=$MODE"

    ACTUAL_CONTROLLER_SHA="$(
        sha256sum "$CONTROLLER" |
        awk '{print $1}'
    )"

    echo "EXPECTED_CONTROLLER_SHA=$EXPECTED_CONTROLLER_SHA"
    echo "ACTUAL_CONTROLLER_SHA=$ACTUAL_CONTROLLER_SHA"

    if [ "$ACTUAL_CONTROLLER_SHA" != "$EXPECTED_CONTROLLER_SHA" ]; then
        echo "CONTROLLER_HASH=FAIL"
        echo "WEATHER_SCHEDULER=FAIL_CLOSED"
        echo "WEATHER_SCHEDULER_RUN_END=$(date -Is)"
        echo "============================================================"
        exit 70
    fi

    echo "CONTROLLER_HASH=PASS"

    CONTEXT="$(
        "$PYTHON" - <<'PY'
from wfs_schedule_context import resolve_schedule_week_context

ctx = resolve_schedule_week_context()

season = int(ctx.season)
planning = int(ctx.planning_week)

upcoming = (
    int(ctx.upcoming_week)
    if ctx.upcoming_week is not None
    else None
)

active = (
    int(ctx.active_game_week)
    if ctx.active_game_week is not None
    else None
)

if season <= 0:
    raise RuntimeError("Invalid resolved season.")

if not 1 <= planning <= 18:
    raise RuntimeError("Invalid planning week.")

if upcoming is not None and not 1 <= upcoming <= 18:
    raise RuntimeError("Invalid upcoming week.")

print(f"SEASON={season}")
print(f"PLANNING_WEEK={planning}")
print(
    "ACTIVE_GAME_WEEK="
    + (str(active) if active is not None else "NONE")
)
print(
    "UPCOMING_WEEK="
    + (str(upcoming) if upcoming is not None else "NONE")
)

weeks = [planning]

if upcoming is not None and upcoming not in weeks:
    weeks.append(upcoming)

print(
    "CANDIDATE_WEEKS="
    + ",".join(str(x) for x in weeks)
)
PY
    )"

    CONTEXT_RC=$?

    if [ "$CONTEXT_RC" -ne 0 ]; then
        echo "SCHEDULE_RESOLUTION=FAIL"
        echo "WEATHER_SCHEDULER=FAIL_CLOSED"
        echo "WEATHER_SCHEDULER_RUN_END=$(date -Is)"
        echo "============================================================"
        exit 71
    fi

    echo "$CONTEXT"

    SEASON="$(
        printf '%s\n' "$CONTEXT" |
        awk -F= '/^SEASON=/{print $2}'
    )"

    CANDIDATE_WEEKS="$(
        printf '%s\n' "$CONTEXT" |
        awk -F= '/^CANDIDATE_WEEKS=/{print $2}'
    )"

    if [ -z "$SEASON" ] || [ -z "$CANDIDATE_WEEKS" ]; then
        echo "SCHEDULE_RESOLUTION=FAIL"
        echo "WEATHER_SCHEDULER=FAIL_CLOSED"
        echo "WEATHER_SCHEDULER_RUN_END=$(date -Is)"
        echo "============================================================"
        exit 72
    fi

    echo "SCHEDULE_RESOLUTION=PASS"

    IFS=',' read -r -a WEEKS <<< "$CANDIDATE_WEEKS"

    OVERALL_RC=0

    for WEEK in "${WEEKS[@]}"; do

        echo
        echo "------------------------------------------------------------"
        echo "CONTROLLER_SEASON=$SEASON"
        echo "CONTROLLER_WEEK=$WEEK"
        echo "------------------------------------------------------------"

        "$PYTHON" "$CONTROLLER" \
            --season "$SEASON" \
            --week "$WEEK" \
            "$MODE"

        RC=$?

        echo "CONTROLLER_WEEK=$WEEK"
        echo "CONTROLLER_RC=$RC"

        case "$RC" in
            0)
                ;;
            3|5)
                # Expected fail-closed/no-op states:
                # 3 = no checkpoint in execution window
                # 5 = checkpoint already ledgered
                ;;
            *)
                OVERALL_RC="$RC"
                ;;
        esac

    done

    if [ "$OVERALL_RC" -eq 0 ]; then
        echo "WEATHER_SCHEDULER=PASS"
    else
        echo "WEATHER_SCHEDULER=FAIL"
    fi

    echo "WEATHER_SCHEDULER_RUN_END=$(date -Is)"
    echo "============================================================"

    exit "$OVERALL_RC"

} >> "$LOG" 2>&1
