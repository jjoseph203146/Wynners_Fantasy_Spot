from datetime import datetime
import sys
import traceback

from config import (
    CURRENT_SEASON,
    LOG_DIR,
)

from database import initialize_database

from schedule import (
    download_schedule,
    update_schedule_database,
    export_schedule,
    print_summary,
)

from boxscores import (
    initialize_boxscore_tables,
    download_player_stats,
    download_team_stats,
    update_player_stats,
    update_team_stats,
    audit_game_ids,
    export_boxscores,
    print_boxscore_summary,
)

from fanduel_scoring import (
    initialize_scoring_columns,
    download_scoring_source,
    update_fanduel_scores,
    audit_scoring,
    print_scoring_sanity_check,
    export_top_scores,
    refresh_player_exports,
)

from current_context import (
    run_current_context_update,
)


# =========================================================
# SETTINGS
# =========================================================

LOG_FILE = LOG_DIR / "nfl_updater.log"


# =========================================================
# LOGGING
# =========================================================

class Tee:

    def __init__(
        self,
        terminal,
        logfile,
    ):
        self.terminal = terminal
        self.logfile = logfile

    def write(
        self,
        message,
    ):
        self.terminal.write(
            message
        )

        self.logfile.write(
            message
        )

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def start_logging():

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logfile = open(
        LOG_FILE,
        "a",
        buffering=1,
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(
        original_stdout,
        logfile,
    )

    sys.stderr = Tee(
        original_stderr,
        logfile,
    )

    return (
        logfile,
        original_stdout,
        original_stderr,
    )


# =========================================================
# HELPERS
# =========================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def timestamp():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def print_step_runtime(
    step_name,
    started,
):

    elapsed = (
        datetime.now()
        - started
    ).total_seconds()

    print()
    print(
        f"{step_name} runtime: "
        f"{elapsed:.1f} seconds"
    )


# =========================================================
# SCHEDULE
# =========================================================

def update_current_schedule():

    started = datetime.now()

    section(
        f"STEP 1 — {CURRENT_SEASON} NFL SCHEDULE"
    )

    df = download_schedule(
        season=CURRENT_SEASON
    )

    update_schedule_database(
        df
    )

    export_schedule()

    print_summary()

    print_step_runtime(
        "Schedule",
        started,
    )


# =========================================================
# CURRENT CONTEXT
# =========================================================

def update_context():

    started = datetime.now()

    section(
        "STEP 2 — NFL CONTEXT DATA"
    )

    result = (
        run_current_context_update()
    )

    print_step_runtime(
        "Context",
        started,
    )

    return result


# =========================================================
# BOXSCORES
# =========================================================

def update_current_boxscores():

    started = datetime.now()

    section(
        f"STEP 3 — {CURRENT_SEASON} NFL BOXSCORES"
    )

    initialize_boxscore_tables()

    print()
    print(
        f"Checking nflverse for "
        f"{CURRENT_SEASON} player stats..."
    )

    try:

        player_df = (
            download_player_stats(
                [CURRENT_SEASON]
            )
        )

    except Exception as exc:

        print()
        print(
            f"No usable {CURRENT_SEASON} "
            f"player statistics are "
            f"available yet."
        )

        print(
            f"Source response: {exc}"
        )

        print_step_runtime(
            "Box scores",
            started,
        )

        return False

    if (
        player_df is None
        or player_df.empty
    ):

        print()
        print(
            f"No {CURRENT_SEASON} player "
            f"box-score rows are "
            f"available yet."
        )

        print_step_runtime(
            "Box scores",
            started,
        )

        return False

    print()
    print(
        f"Checking nflverse for "
        f"{CURRENT_SEASON} team stats..."
    )

    try:

        team_df = (
            download_team_stats(
                [CURRENT_SEASON]
            )
        )

    except Exception as exc:

        print()
        print(
            f"No usable {CURRENT_SEASON} "
            f"team statistics are "
            f"available yet."
        )

        print(
            f"Source response: {exc}"
        )

        print_step_runtime(
            "Box scores",
            started,
        )

        return False

    if (
        team_df is None
        or team_df.empty
    ):

        print()
        print(
            f"No {CURRENT_SEASON} team "
            f"box-score rows are "
            f"available yet."
        )

        print_step_runtime(
            "Box scores",
            started,
        )

        return False

    update_player_stats(
        player_df
    )

    update_team_stats(
        team_df
    )

    audit_game_ids()

    export_boxscores()

    print_boxscore_summary()

    print_step_runtime(
        "Box scores",
        started,
    )

    return True


# =========================================================
# FANDUEL SCORING
# =========================================================

def update_current_fanduel_scoring():

    started = datetime.now()

    section(
        f"STEP 4 — {CURRENT_SEASON} FANDUEL SCORING"
    )

    initialize_scoring_columns()

    try:

        scoring_df = (
            download_scoring_source(
                [CURRENT_SEASON]
            )
        )

    except Exception as exc:

        print()
        print(
            f"No usable {CURRENT_SEASON} "
            f"FanDuel scoring source "
            f"is available yet."
        )

        print(
            f"Source response: {exc}"
        )

        print_step_runtime(
            "FanDuel scoring",
            started,
        )

        return False

    if (
        scoring_df is None
        or scoring_df.empty
    ):

        print()
        print(
            f"No {CURRENT_SEASON} player "
            f"statistics are available "
            f"to score."
        )

        print_step_runtime(
            "FanDuel scoring",
            started,
        )

        return False

    update_fanduel_scores(
        scoring_df
    )

    audit_scoring()

    print_scoring_sanity_check()

    export_top_scores()

    refresh_player_exports()

    print_step_runtime(
        "FanDuel scoring",
        started,
    )

    return True


# =========================================================
# FINAL AUDIT
# =========================================================

def final_database_audit():

    started = datetime.now()

    section(
        "STEP 5 — FINAL DATABASE AUDIT"
    )

    audit_game_ids()

    audit_scoring()

    print_scoring_sanity_check()

    print_summary()

    print_boxscore_summary()

    print_step_runtime(
        "Final audit",
        started,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    (
        logfile,
        original_stdout,
        original_stderr,
    ) = start_logging()

    started = datetime.now()

    try:

        section(
            "NFL DATA ENGINE — AUTOMATIC UPDATER"
        )

        print(
            f"Started: {timestamp()}"
        )

        print(
            f"Current NFL season: "
            f"{CURRENT_SEASON}"
        )

        print(
            f"Log file: {LOG_FILE}"
        )

        # -------------------------------------------------
        # DATABASE
        # -------------------------------------------------

        initialize_database()

        initialize_boxscore_tables()

        initialize_scoring_columns()

        # -------------------------------------------------
        # STEP 1
        # -------------------------------------------------

        update_current_schedule()

        # -------------------------------------------------
        # STEP 2
        # -------------------------------------------------

        context_result = (
            update_context()
        )

        # -------------------------------------------------
        # STEP 3
        # -------------------------------------------------

        boxscores_available = (
            update_current_boxscores()
        )

        # -------------------------------------------------
        # STEP 4
        # -------------------------------------------------

        if boxscores_available:

            scoring_available = (
                update_current_fanduel_scoring()
            )

        else:

            scoring_available = False

            section(
                f"STEP 4 — "
                f"{CURRENT_SEASON} "
                f"FANDUEL SCORING"
            )

            print(
                f"Skipped because no "
                f"{CURRENT_SEASON} "
                f"box-score data is "
                f"available yet."
            )

        # -------------------------------------------------
        # STEP 5
        # -------------------------------------------------

        final_database_audit()

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        ended = datetime.now()

        duration = (
            ended
            - started
        ).total_seconds()

        section(
            "NFL DATA UPDATE SUCCESSFUL"
        )

        print(
            f"Finished: {timestamp()}"
        )

        print(
            f"Runtime: "
            f"{duration:.1f} seconds"
        )

        print()

        print(
            f"{CURRENT_SEASON} schedule: "
            f"UPDATED"
        )

        print(
            f"{CURRENT_SEASON} depth charts: "
            f"{'AVAILABLE' if context_result['depth_available'] else 'NOT AVAILABLE'}"
        )

        print(
            f"New depth-chart rows: "
            f"{context_result['depth_new_rows']}"
        )

        print(
            f"{CURRENT_SEASON} rosters: "
            f"{'AVAILABLE' if context_result['rosters_available'] else 'NOT YET AVAILABLE'}"
        )

        print(
            f"{CURRENT_SEASON} injuries: "
            f"{'AVAILABLE' if context_result['injuries_available'] else 'NOT YET AVAILABLE'}"
        )

        print(
            f"{CURRENT_SEASON} snap counts: "
            f"{'AVAILABLE' if context_result['snaps_available'] else 'NOT YET AVAILABLE'}"
        )

        if boxscores_available:

            print(
                f"{CURRENT_SEASON} box scores: "
                f"UPDATED"
            )

        else:

            print(
                f"{CURRENT_SEASON} box scores: "
                f"NOT YET AVAILABLE"
            )

        if scoring_available:

            print(
                f"{CURRENT_SEASON} "
                f"FanDuel scores: UPDATED"
            )

        else:

            print(
                f"{CURRENT_SEASON} "
                f"FanDuel scores: "
                f"NOT YET AVAILABLE"
            )

        print()

        print(
            "Historical database remains intact."
        )

        print(
            f"Log: {LOG_FILE}"
        )

    except Exception as exc:

        section(
            "NFL DATA UPDATE FAILED"
        )

        print(
            f"Failed: {timestamp()}"
        )

        print()

        print(
            f"Error: {exc}"
        )

        print()

        traceback.print_exc()

        raise

    finally:

        sys.stdout.flush()
        sys.stderr.flush()

        sys.stdout = original_stdout
        sys.stderr = original_stderr

        logfile.close()


if __name__ == "__main__":
    main()
