#!/usr/bin/env python3
"""
WFS WEEKLY REVIEW CONTROLLER V1

Purpose
-------
Schedule-driven, exactly-once weekly postgame review orchestration.

Authority
---------
- data/nfl.db games table determines regular-season week completion.
- A week is review-eligible only when every scheduled game has:
    completed = 1
    away_score IS NOT NULL
    home_score IS NOT NULL
- Postgame grading calculations are unchanged; both graders use exact REG week scope.
- Existing prospective forecast evaluation is verified, not rebuilt here.
- No model training, projection modification, optimizer modification,
  service restart, or historical forecast mutation is permitted.

State
-----
processed/wfs_weekly_review_state_v1.json

Week 1 is bootstrapped externally by the installer as an already-reviewed
historical baseline so installation cannot accidentally rerun development
or historical postgame construction stages.
"""

from pathlib import Path
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


ROOT = Path("/home/mwynn/nfl_data_engine")
PYTHON = ROOT / "venv" / "bin" / "python"

NFL_DB = ROOT / "data" / "nfl.db"

STATE_PATH = (
    ROOT
    / "processed"
    / "wfs_weekly_review_state_v1.json"
)

GAME_GRADER = ROOT / "nfl_postgame_grader_r2.py"
PLAYER_GRADER = ROOT / "nfl_postgame_player_grader.py"

FORECAST_EVAL_AUDIT = (
    ROOT
    / "processed"
    / "forecast_prospective_evaluation_v1_audit.json"
)

SEASON = 2026

EXPECTED_COMPONENT_SHA256 = {
    str(GAME_GRADER):
        "81a824fd11bc1f6013101a92882ca4760e114fc2671deb636e10f749799950a3",
    str(PLAYER_GRADER):
        "163aeb7de5711eea6c45ab954d0e3df5914d5859e99aa54e6f9ebb36ebc29e05",
}


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def fail(message):
    print(f"WEEKLY_REVIEW_STATUS=FAIL")
    print(f"ERROR={message}")
    raise SystemExit(1)


def load_state():
    if not STATE_PATH.exists():
        fail(
            "weekly review state missing; "
            "installer bootstrap required"
        )

    try:
        state = json.loads(
            STATE_PATH.read_text()
        )
    except Exception as exc:
        fail(
            f"unable to read weekly review state: {exc}"
        )

    if state.get("contract") != (
        "WFS_WEEKLY_REVIEW_CONTROLLER_V1"
    ):
        fail("weekly review state contract mismatch")

    if int(state.get("season", -1)) != SEASON:
        fail("weekly review state season mismatch")

    reviews = state.get("reviews")

    if not isinstance(reviews, dict):
        fail("weekly review state reviews invalid")

    return state


def atomic_write_json(path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    tmp = Path(tmp_name)

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                payload,
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(
            tmp,
            path,
        )

    finally:
        if tmp.exists():
            tmp.unlink()


def verify_component_contract():
    for raw_path, expected in (
        EXPECTED_COMPONENT_SHA256.items()
    ):
        path = Path(raw_path)

        if not path.exists():
            fail(
                f"required component missing: {path}"
            )

        actual = sha256(path)

        if actual != expected:
            fail(
                "component SHA mismatch: "
                f"{path.name} "
                f"expected={expected} "
                f"actual={actual}"
            )


def select_week_games(conn, season, week):
    """Exact REG week authority; never hide unfinished games by filtering them out."""
    rows = conn.execute(
        """
        SELECT game_id, season, week, game_date, gametime,
               away_team, home_team, away_score, home_score, completed
        FROM games
        WHERE season=? AND game_type='REG' AND week=?
        ORDER BY game_id
        """,
        (season, week),
    ).fetchall()
    ids = [row["game_id"] for row in rows]
    if (
        not rows
        or any(not isinstance(gid, str) or not gid.strip() for gid in ids)
        or len(set(ids)) != len(ids)
        or any(
            row["completed"] != 1
            or row["away_score"] is None
            or row["home_score"] is None
            for row in rows
        )
    ):
        raise RuntimeError(f"Season {season} week {week}: incomplete or invalid REG game set")
    return rows


def read_week_game_ids(week):
    with sqlite3.connect(f"file:{NFL_DB}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        return [row["game_id"] for row in select_week_games(con, SEASON, week)]


def verify_week_scope(output, week, game_ids):
    reports = [line[len("WEEK_SCOPE="):] for line in output.splitlines()
               if line.startswith("WEEK_SCOPE=")]
    if len(reports) != 1:
        fail("grader must report exactly one week scope")
    try:
        report = json.loads(reports[0])
    except (TypeError, ValueError):
        fail("grader week scope is malformed")
    expected = {"season": SEASON, "week": week, "game_ids": game_ids}
    if (
        not isinstance(report, dict)
        or type(report.get("season")) is not int
        or type(report.get("week")) is not int
        or report != expected
    ):
        fail("grader week scope does not match authoritative schedule")


def read_week_status():
    uri = (
        f"file:{NFL_DB}"
        "?mode=ro"
    )

    with sqlite3.connect(
        uri,
        uri=True,
    ) as con:
        con.row_factory = sqlite3.Row

        rows = con.execute(
            """
            SELECT
                week,
                COUNT(*) AS scheduled,
                SUM(
                    CASE
                        WHEN completed=1
                        THEN 1 ELSE 0
                    END
                ) AS completed,
                SUM(
                    CASE
                        WHEN completed=1
                         AND away_score IS NOT NULL
                         AND home_score IS NOT NULL
                        THEN 1 ELSE 0
                    END
                ) AS final_scores
            FROM games
            WHERE season=?
              AND game_type='REG'
            GROUP BY week
            ORDER BY week
            """,
            (SEASON,),
        ).fetchall()

    if not rows:
        fail("no regular-season schedule rows found")

    result = []

    for row in rows:
        result.append(
            {
                "week": int(row["week"]),
                "scheduled": int(
                    row["scheduled"] or 0
                ),
                "completed": int(
                    row["completed"] or 0
                ),
                "final_scores": int(
                    row["final_scores"] or 0
                ),
            }
        )

    return result


def verify_forecast_evaluation():
    if not FORECAST_EVAL_AUDIT.exists():
        fail(
            "prospective forecast evaluation "
            "audit missing"
        )

    try:
        audit = json.loads(
            FORECAST_EVAL_AUDIT.read_text()
        )
    except Exception as exc:
        fail(
            "unable to read prospective forecast "
            f"evaluation audit: {exc}"
        )

    if str(
        audit.get("status", "")
    ).upper() != "PASS":
        fail(
            "prospective forecast evaluation "
            "audit is not PASS"
        )

    return sha256(
        FORECAST_EVAL_AUDIT
    )


def run_component(label, script, week):
    cmd = [
        str(PYTHON),
        str(script),
        "--season",
        str(SEASON),
        "--week",
        str(week),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    output = proc.stdout or ""

    print(
        f"===== {label} OUTPUT BEGIN ====="
    )
    print(output.rstrip())
    print(
        f"===== {label} OUTPUT END ====="
    )

    if proc.returncode != 0:
        fail(
            f"{label} failed with "
            f"exit code {proc.returncode}"
        )

    return output


def verify_game_grader_output(output):
    if (
        output.splitlines().count("NFL_POSTGAME_1C_R2_STATUS=PASS") != 1
    ):
        fail(
            "game grader PASS marker missing"
        )

    if (
        output.splitlines().count("PRODUCTION_DATABASE_WRITES=0") != 1
    ):
        fail(
            "game grader read-only marker missing"
        )


def verify_player_grader_output(output):
    if (
        output.splitlines().count("NFL_POSTGAME_1D_D_B_STATUS=PASS") != 1
    ):
        fail(
            "player grader PASS marker missing"
        )

    if (
        output.splitlines().count("PRODUCTION_DATABASE_WRITES=0") != 1
    ):
        fail(
            "player grader read-only marker missing"
        )


def main():
    print(
        "WFS_WEEKLY_REVIEW_CONTROLLER_V1"
    )

    verify_component_contract()

    state = load_state()
    reviews = state["reviews"]

    week_status = read_week_status()

    eligible = []

    for row in week_status:
        week = row["week"]
        scheduled = row["scheduled"]
        completed = row["completed"]
        finals = row["final_scores"]

        complete = (
            scheduled > 0
            and completed == scheduled
            and finals == scheduled
        )

        print(
            f"WEEK={week}|"
            f"SCHEDULED={scheduled}|"
            f"COMPLETED={completed}|"
            f"FINAL_SCORES={finals}|"
            f"COMPLETE={int(complete)}"
        )

        if complete:
            key = str(week)

            existing = reviews.get(
                key,
                {},
            )

            if (
                existing.get("status")
                == "REVIEW_COMPLETE"
            ):
                print(
                    f"WEEK={week}|"
                    "ACTION=ALREADY_REVIEWED"
                )
            else:
                eligible.append(row)

    if not eligible:
        print(
            "WEEKLY_REVIEW_ACTION=NOOP"
        )
        print(
            "WEEKLY_REVIEW_STATUS=PASS"
        )
        return 0

    # Process completed unreviewed weeks in chronological order.
    for row in eligible:
        week = row["week"]

        print(
            f"WEEK={week}|"
            "ACTION=RUN_REVIEW"
        )

        # Prospective evaluation is already executed earlier
        # in run_updater.sh. Verify its durable PASS artifact.
        forecast_eval_sha = (
            verify_forecast_evaluation()
        )

        game_ids = read_week_game_ids(week)

        game_output = run_component(
            "GAME_POSTGAME_GRADER",
            GAME_GRADER,
            week,
        )
        verify_game_grader_output(
            game_output
        )

        verify_week_scope(game_output, week, game_ids)

        player_output = run_component(
            "PLAYER_POSTGAME_GRADER",
            PLAYER_GRADER,
            week,
        )
        verify_player_grader_output(
            player_output
        )

        verify_week_scope(player_output, week, game_ids)

        # Re-read schedule after grading. Never close a week
        # based solely on the initial observation.
        fresh = {
            r["week"]: r
            for r in read_week_status()
        }[week]

        if not (
            fresh["scheduled"] > 0
            and fresh["completed"]
                == fresh["scheduled"]
            and fresh["final_scores"]
                == fresh["scheduled"]
        ):
            fail(
                f"Week {week} completion state "
                "changed during review"
            )

        if read_week_game_ids(week) != game_ids:
            fail(f"Week {week} game membership changed during review")

        reviews[str(week)] = {
            "status": "REVIEW_COMPLETE",
            "season": SEASON,
            "week": week,
            "scheduled_games":
                fresh["scheduled"],
            "completed_games":
                fresh["completed"],
            "final_score_games":
                fresh["final_scores"],
            "reviewed_at_utc":
                now_utc(),
            "game_grader_sha256":
                sha256(GAME_GRADER),
            "player_grader_sha256":
                sha256(PLAYER_GRADER),
            "forecast_evaluation_audit_sha256":
                forecast_eval_sha,
            "authority":
                "schedule_all_games_final",
            "production_mutation":
                False,
        }

        state["updated_at_utc"] = (
            now_utc()
        )

        atomic_write_json(
            STATE_PATH,
            state,
        )

        print(
            f"WEEK={week}|"
            "REVIEW_STATUS=REVIEW_COMPLETE"
        )

    print(
        "WEEKLY_REVIEW_STATUS=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
