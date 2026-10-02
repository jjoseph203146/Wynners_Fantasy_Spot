#!/bin/bash

set -u

PROJECT_DIR="/home/mwynn/nfl_data_engine"
PYTHON="$PROJECT_DIR/venv/bin/python"
UPDATER="$PROJECT_DIR/updater.py"
RP2A_REFRESH="$PROJECT_DIR/load_fanduel_dimensions_readiness.py"
RP2B_REFRESH_GATE="$PROJECT_DIR/scripts/run_rp2b_refresh_gate.py"
PLAYER_IDENTITY="$PROJECT_DIR/player_identity.py"
INJURY_INGEST="$PROJECT_DIR/fanduel_injury_ingest_v2.py"
INJURY_CONSENSUS="$PROJECT_DIR/injury_consensus.py"
FANDUEL_SLATE_INGEST="$PROJECT_DIR/fanduel_slate_ingest_v2.py"
CURRENT_SLATE_FEATURES="$PROJECT_DIR/current_slate_features.py"
PRODUCTION_PROJECTION="$PROJECT_DIR/production_projection.py"
CURRENT_STAGE24_REFRESH="$PROJECT_DIR/scripts/run_current_stage24_refresh.sh"
V3_RECONCILIATION_PROMOTION="$PROJECT_DIR/scripts/promote_offensive_reconciliation_v3.py"
CURRENT_UNIFIED_STAT_FORECAST="$PROJECT_DIR/current_unified_stat_forecast.py"
STAT_FORECAST_PUBLISH="$PROJECT_DIR/scripts/wfs_stat_forecast_publish_v1.py"
STAT_FORECAST_CANDIDATE="$PROJECT_DIR/data/parquet/nfl_current_unified_stat_forecasts.parquet"
FANDUEL_OPEN_GATE="$PROJECT_DIR/scripts/fanduel_no_open_slate_gate.py"
FANDUEL_PLAYER_POOL="$PROJECT_DIR/fanduel_player_pool.py"
FANDUEL_PROJECTION_ATTACH="$PROJECT_DIR/fanduel_slate_projection_attach_v5.py"
SOLVER_READY="$PROJECT_DIR/fanduel_solver_ready_pool.py"
HISTORICAL_ROLLOVER_GATE="$PROJECT_DIR/scripts/run_historical_rollover_gate_v1.py"
FEEDBACK_REFRESH="$PROJECT_DIR/nfl_2026_feedback_refresh.py"
FORECAST_BUILD="$PROJECT_DIR/scripts/build_forecast_live_core_v1.py"
FORECAST_V3="$PROJECT_DIR/scripts/run_forecast_live_core_v3.py"
FORECAST_CAPTURE="$PROJECT_DIR/scripts/capture_forecast_snapshot_v2.py"
FORECAST_IMPACT="$PROJECT_DIR/scripts/build_forecast_live_impact_v1.py"
FORECAST_REPLACEMENT="$PROJECT_DIR/scripts/build_forecast_live_replacement_quality_v1.py"
FORECAST_INJURY_SHADOW="$PROJECT_DIR/scripts/run_forecast_live_injury_shadow_v1.py"
FORECAST_INJURY_PUBLISH="$PROJECT_DIR/scripts/publish_forecast_live_injury_adjusted_v1.py"
FORECAST_EVALUATOR="$PROJECT_DIR/scripts/evaluate_forecast_prospective_v1.py"
WEEKLY_REVIEW_CONTROLLER="$PROJECT_DIR/scripts/wfs_weekly_review_controller_v1.py"
LOCK_FILE="$PROJECT_DIR/nfl_updater.lock"
CRON_LOG="$PROJECT_DIR/logs/cron.log"

cd "$PROJECT_DIR" || exit 1
mkdir -p "$PROJECT_DIR/logs"

log() {
    echo "$*" >> "$CRON_LOG"
}

wait_for_live_quiet() {
    local stage="$1"
    local waited=0

    log "Waiting for LIVE quiet window before $stage..."

    while pgrep -f '[l]ive_concurrency_gate.py' >/dev/null 2>&1; do
        sleep 1
        waited=$((waited + 1))

        if [ "$waited" -ge 120 ]; then
            log "LIVE quiet-window timeout before $stage."
            return 1
        fi
    done

    log "LIVE quiet window acquired for $stage after ${waited}s."
    return 0
}

run_stage() {
    local name="$1"
    local script="$2"
    local failure_code="$3"

    log "------------------------------------------------------------"
    log "Starting $name: $(date)"

    "$PYTHON" "$script" >> "$CRON_LOG" 2>&1
    local rc=$?

    if [ "$rc" -ne 0 ]; then
        log "$name FAILED with exit code $rc."
        return "$failure_code"
    fi

    log "$name completed successfully: $(date)"
    return 0
}

run_fanduel_ingest_with_retry() {
    local attempt=1
    local max_attempts=3
    local rc=0

    while [ "$attempt" -le "$max_attempts" ]; do
        log "------------------------------------------------------------"
        log "Starting FanDuel injury ingest attempt $attempt/$max_attempts: $(date)"

        "$PYTHON" "$INJURY_INGEST" >> "$CRON_LOG" 2>&1
        rc=$?

        if [ "$rc" -eq 0 ]; then
            log "FanDuel injury ingest completed successfully on attempt $attempt/$max_attempts: $(date)"
            return 0
        fi

        log "FanDuel injury ingest attempt $attempt/$max_attempts FAILED with exit code $rc."

        if [ "$attempt" -eq "$max_attempts" ]; then
            log "FanDuel injury ingest exhausted $max_attempts attempts."
            return 21
        fi

        if [ "$attempt" -eq 1 ]; then
            log "Retrying FanDuel injury ingest in 15 seconds."
            sleep 15
        else
            log "Retrying FanDuel injury ingest in 30 seconds."
            sleep 30
        fi

        attempt=$((attempt + 1))
    done

    return 21
}

run_pipeline() {
    log "------------------------------------------------------------"
    log "Starting core updater: $(date)"

    "$PYTHON" "$UPDATER" >> "$CRON_LOG" 2>&1
    local rc=$?

    if [ "$rc" -ne 0 ]; then
        log "Core updater FAILED with exit code $rc."
        return 10
    fi

    log "Core updater completed successfully: $(date)"

    if ! wait_for_live_quiet "player identity rebuild"; then
        return 36
    fi

    run_stage "Player identity rebuild" "$PLAYER_IDENTITY" 29
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    run_stage "RP-2A dimensions/readiness refresh" "$RP2A_REFRESH" 49
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    run_stage "RP-2B guarded refresh" "$RP2B_REFRESH_GATE" 37
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "prospective forecast evaluation"; then
        return 35
    fi

    run_stage "Prospective forecast evaluation" "$FORECAST_EVALUATOR" 28
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi
    if ! wait_for_live_quiet "weekly postgame review controller"; then
        return 48
    fi
    run_stage "Weekly postgame review controller" "$WEEKLY_REVIEW_CONTROLLER" 48
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "FanDuel injury ingest"; then
        return 31
    fi
    run_fanduel_ingest_with_retry
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "injury consensus"; then
        return 32
    fi
    run_stage "Injury consensus" "$INJURY_CONSENSUS" 22
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "current starter verification"; then
        return 51
    fi
    run_stage "Current starter verification (RotoWire corroboration)" \
        "$PROJECT_DIR/scripts/run_current_starter_verification.py" 51
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "FanDuel schedule-aware slate ingest"; then
        return 37
    fi
    run_stage "FanDuel schedule-aware slate ingest" "$FANDUEL_SLATE_INGEST" 30
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "historical rollover prerequisite"; then
        return 50
    fi
    run_stage "Historical rollover prerequisite" "$HISTORICAL_ROLLOVER_GATE" 50
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi
    if ! wait_for_live_quiet "current slate feature rebuild"; then
        return 38
    fi
    run_stage "Current slate feature rebuild" "$CURRENT_SLATE_FEATURES" 31
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "production projection rebuild"; then
        return 39
    fi
    run_stage "Production projection rebuild" "$PRODUCTION_PROJECTION" 32
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "Forecast LIVE CORE build"; then
        return 25
    fi
    run_stage "Forecast LIVE CORE build" "$FORECAST_BUILD" 25
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi
    if ! wait_for_live_quiet "Forecast LIVE CORE V3"; then
        return 26
    fi
    run_stage "Forecast LIVE CORE V3" "$FORECAST_V3" 26
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi
    if ! wait_for_live_quiet "current Stage24 refresh"; then
        return 44
    fi

    log "------------------------------------------------------------"
    log "Starting current Stage24 refresh: $(date)"

    "$CURRENT_STAGE24_REFRESH" --build-only >> "$CRON_LOG" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        log "Current Stage24 refresh FAILED with exit code $rc."
        return 44
    fi

    log "Current Stage24 refresh completed successfully: $(date)"

    if ! wait_for_live_quiet "V3 offensive reconciliation promotion"; then
        return 45
    fi

    PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$PROJECT_DIR/scripts/refresh_offensive_reconciliation_current_adaptive_team_v1.py" >> "$CRON_LOG" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        log "Current adaptive pair refresh FAILED with exit code $rc."
        return 45
    fi

    # Explicit immutable candidate handoff; never select a "latest" bundle.
    local v3_bundle
    v3_bundle="$PROJECT_DIR/data/model_candidates/offensive_reconciliation_v3/$(date -u +%Y%m%dT%H%M%SZ)_$$"
    "$PYTHON" "$PROJECT_DIR/scripts/current_classic_publication_lifecycle.py" \
        refresh-evidence >> "$CRON_LOG" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        log "Adaptive evidence refresh FAILED with exit code $rc."
        return 45
    fi
    # The adaptive refresh above publishes a new matrix-bound team budget pair.
    # Rebuild the V3 shadow against that exact adaptive generation before
    # immutable preflight; otherwise the Stage24 shadow remains bound to the
    # prior adaptive generation and attribution validation must fail closed.
    PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
        "$PROJECT_DIR/scripts/build_offensive_team_reconciliation_shadow_v3.py" \
        >> "$CRON_LOG" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        log "V3 adaptive-bound rebuild FAILED with exit code $rc."
        return 45
    fi

    "$PYTHON" "$PROJECT_DIR/scripts/preflight_offensive_reconciliation_v3.py" \
        --bundle "$v3_bundle" >> "$CRON_LOG" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        log "V3 immutable preflight FAILED with exit code $rc."
        return 45
    fi
    local v3_season v3_week
    if ! read -r v3_season v3_week < <("$PYTHON" -c \
        'import json,sys; m=json.load(open(sys.argv[1])); print(m["season"],m["week"])' \
        "$v3_bundle/validation.json"); then
        log "V3 immutable handoff metadata could not be read."
        return 45
    fi
    "$PYTHON" "$V3_RECONCILIATION_PROMOTION" \
        --validation "$v3_bundle/validation.json" \
        --season "$v3_season" --week "$v3_week" >> "$CRON_LOG" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        log "V3 bound promotion FAILED with exit code $rc."
        return 45
    fi

    if ! wait_for_live_quiet "V3 unified stat forecast rebuild"; then
        return 46
    fi

    "$PYTHON" "$CURRENT_UNIFIED_STAT_FORECAST" \
        --v3-validation "$v3_bundle/validation.json" \
        --output "$v3_bundle/unified.parquet" >> "$CRON_LOG" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return 46
    fi

    if ! wait_for_live_quiet "V3 public stat forecast publish"; then
        return 47
    fi

    log "------------------------------------------------------------"
    log "Starting V3 public stat forecast publish: $(date)"

    "$PYTHON" "$PROJECT_DIR/scripts/current_classic_publication_lifecycle.py" \
        publish-pair --bundle "$v3_bundle" >> "$CRON_LOG" 2>&1
    rc=$?

    if [ "$rc" -ne 0 ]; then
        log "V3 public stat forecast publish FAILED with exit code $rc."
        return 47
    fi

    log "V3 public stat forecast publish completed successfully: $(date)"

    if ! wait_for_live_quiet "FanDuel solver-open lifecycle gate"; then
        return 43
    fi

    log "------------------------------------------------------------"
    log "Starting FanDuel solver-open lifecycle gate: $(date)"

    "$PYTHON" "$FANDUEL_OPEN_GATE" >> "$CRON_LOG" 2>&1
    rc=$?

    if [ "$rc" -eq 10 ]; then
        log "FanDuel lifecycle state: NO_OPEN_SLATES."
        log "Stale downstream FanDuel solver inventory cleared successfully."
        log "Skipping FanDuel player pool, projection attach, and solver-ready rebuild."
    elif [ "$rc" -eq 0 ]; then
        log "FanDuel lifecycle state: OPEN_SLATES_PRESENT."
        log "Continuing downstream FanDuel solver build."

        if ! wait_for_live_quiet "FanDuel player pool rebuild"; then
            return 40
        fi
        run_stage "FanDuel player pool rebuild" "$FANDUEL_PLAYER_POOL" 33
        rc=$?
        if [ "$rc" -ne 0 ]; then
            return "$rc"
        fi

        if ! wait_for_live_quiet "FanDuel slate projection attach"; then
            return 41
        fi
        run_stage "FanDuel slate projection attach" "$FANDUEL_PROJECTION_ATTACH" 34
        rc=$?
        if [ "$rc" -ne 0 ]; then
            return "$rc"
        fi

        if ! wait_for_live_quiet "solver-ready rebuild"; then
            return 42
        fi
        run_stage "Solver-ready rebuild" "$SOLVER_READY" 35
        rc=$?
        if [ "$rc" -ne 0 ]; then
            return "$rc"
        fi
    else
        log "FanDuel solver-open lifecycle gate FAILED with exit code $rc."
        return 43
    fi

    # Classic/FanDuel authority publication is complete.
    # Release exclusive authority lock before downstream analytics.
    flock -u 9
    log "Classic/FanDuel authority lock released: $(date)"

    if ! wait_for_live_quiet "2026 feedback refresh"; then
        log "NFL hourly pipeline authority completed; downstream analytics deferred for LIVE."
        return 34
    fi

    run_stage "2026 feedback refresh" "$FEEDBACK_REFRESH" 24
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi



    run_stage "Forecast snapshot capture" "$FORECAST_CAPTURE" 27
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    run_stage "Forecast LIVE Impact V1" "$FORECAST_IMPACT" 44
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    run_stage "Forecast LIVE Replacement V1" "$FORECAST_REPLACEMENT" 45
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    run_stage "Forecast injury counterfactual shadow" "$FORECAST_INJURY_SHADOW" 46
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    run_stage "Forecast injury-adjusted companion publish" "$FORECAST_INJURY_PUBLISH" 47
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    return 0
}

log "============================================================"
log "NFL hourly pipeline launcher started: $(date)"

# Hold the existing updater lock for the ENTIRE core + injury chain.
exec 9>"$LOCK_FILE"

# Wait up to 60 seconds for an in-flight LIVE read cycle to release
# its shared nfl_updater.lock. Once acquired exclusively, this lock
# protects the complete updater + injury chain from new LIVE DB reads.
if ! flock -w 60 9; then
    log "Updater skipped because shared updater/LIVE lock remained busy for 60 seconds."
    log "NFL hourly pipeline deferred because LIVE lock remained busy."
    log "Finished: $(date)"
    log ""
    exit 1
fi

run_pipeline
EXIT_CODE=$?

# The pipeline normally releases FD9 after Classic/FanDuel authority
# publication. It can also return earlier while FD9 is still held.
# Always release our launcher lock before any guarded recovery attempt.
flock -u 9
log "Updater launcher lock released after pipeline return: $(date)"

# FanDuel Classic auto-recovery is eligible only after an otherwise
# successful hourly pipeline. The recovery controller is one-shot,
# fail-closed, and acquires the canonical updater lock through V4.3.
if [ "$EXIT_CODE" -eq 0 ]; then
    log "FanDuel Classic stale-source auto-recovery check started."

    "$PYTHON" "$PROJECT_DIR/scripts/fanduel_classic_autorecovery_v1.py"
    RECOVERY_RC=$?

    if [ "$RECOVERY_RC" -ne 0 ]; then
        log "FanDuel Classic stale-source auto-recovery FAILED with exit code $RECOVERY_RC."
        EXIT_CODE="$RECOVERY_RC"
    else
        log "FanDuel Classic stale-source auto-recovery check completed successfully."
    fi
fi

if [ "$EXIT_CODE" -eq 0 ]; then
    log "NFL hourly pipeline completed successfully."
else
    log "NFL hourly pipeline exited with code: $EXIT_CODE"
fi

log "Finished: $(date)"
log ""

exit "$EXIT_CODE"
