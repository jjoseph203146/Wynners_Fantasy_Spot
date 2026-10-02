PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS weather_prospective_capture_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,

    controller_version TEXT NOT NULL,

    season INTEGER NOT NULL,
    week INTEGER NOT NULL,

    checkpoint TEXT NOT NULL,
    checkpoint_hours INTEGER NOT NULL,

    scheduled_trigger_at_utc TEXT NOT NULL,
    target_kickoff_at_utc TEXT NOT NULL,

    target_game_count INTEGER NOT NULL,
    weekly_schedule_game_count INTEGER NOT NULL,

    capture_scope TEXT NOT NULL,
    evaluation_scope TEXT NOT NULL,

    attempt_started_at_utc TEXT,
    capture_completed_at_utc TEXT,

    v1_captured_at TEXT,

    v2_feature_version TEXT,
    v3_shadow_version TEXT,

    status TEXT NOT NULL,
    failure_reason TEXT,

    production_influence_allowed INTEGER NOT NULL
        CHECK (
            production_influence_allowed = 0
        ),

    UNIQUE (
        season,
        week,
        checkpoint,
        target_kickoff_at_utc
    )
);

CREATE TABLE IF NOT EXISTS weather_prospective_capture_targets (
    run_id INTEGER NOT NULL,
    game_id TEXT NOT NULL,

    eligible_for_checkpoint_evaluation INTEGER NOT NULL
        CHECK (
            eligible_for_checkpoint_evaluation = 1
        ),

    PRIMARY KEY (
        run_id,
        game_id
    ),

    FOREIGN KEY (
        run_id
    )
    REFERENCES weather_prospective_capture_runs (
        run_id
    )
    ON DELETE RESTRICT
);
