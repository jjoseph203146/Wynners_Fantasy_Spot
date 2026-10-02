#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BUILDER_PATH = (
    ROOT
    / "scripts"
    / "build_offensive_reconciliation_current_adaptive_team_v1.py"
)

MATRIX_PATH = (
    ROOT
    / "data"
    / "parquet"
    / "nfl_current_offensive_model_matrix.parquet"
)

DB_PATH = ROOT / "data" / "nfl.db"

OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_current_adaptive_team_v1.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_current_adaptive_team_v1_audit.json"
)

BACKUP_ROOT = (
    ROOT
    / "backups"
    / "adaptive_current_runtime"
)


def fail(message: str) -> None:
    raise RuntimeError(
        "FAIL_CLOSED: " + message
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "adaptive_current_builder",
        BUILDER_PATH,
    )

    if spec is None or spec.loader is None:
        fail("could not load adaptive builder")

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def schedule_context(
    season: int,
    week: int,
):
    uri = (
        f"file:{DB_PATH.resolve()}"
        "?mode=ro"
    )

    with sqlite3.connect(
        uri,
        uri=True,
    ) as conn:
        games = pd.read_sql_query(
            """
            SELECT
                game_id,
                home_team,
                away_team,
                completed
            FROM games
            WHERE season = ?
              AND week = ?
              AND game_type = 'REG'
            """,
            conn,
            params=(season, week),
        )

    if games.empty:
        fail(
            "authoritative schedule returned "
            "no target games"
        )

    unfinished = games.loc[
        games["completed"].fillna(0).ne(1)
    ].copy()

    expected = {
        (
            str(row.game_id),
            str(team),
        )
        for row in unfinished.itertuples()
        for team in (
            row.home_team,
            row.away_team,
        )
    }

    return unfinished, expected


def validate_candidate(
    csv_path: Path,
    audit_path: Path,
):
    if not csv_path.is_file():
        fail("candidate CSV missing")

    if not audit_path.is_file():
        fail("candidate audit missing")

    candidate = pd.read_csv(csv_path)

    try:
        audit = json.loads(
            audit_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        fail(
            f"candidate audit unreadable: {exc}"
        )

    if audit.get("status") != (
        "PASS_HARD_CONTRACTS"
    ):
        fail(
            "candidate audit is not "
            "PASS_HARD_CONTRACTS"
        )

    hard_failures = audit.get(
        "hard_failures"
    )

    if not isinstance(
        hard_failures,
        dict,
    ):
        fail("hard_failures missing")

    if any(
        int(value) != 0
        for value in hard_failures.values()
    ):
        fail(
            "candidate contains hard failures"
        )

    try:
        season = int(
            audit["target_season"]
        )
        week = int(
            audit["target_week"]
        )
    except Exception:
        fail(
            "target season/week missing "
            "from candidate audit"
        )

    required_columns = {
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",
    }

    missing_columns = (
        required_columns
        - set(candidate.columns)
    )

    if missing_columns:
        fail(
            "candidate columns missing: "
            + ",".join(
                sorted(missing_columns)
            )
        )

    if candidate.duplicated(
        ["game_id", "team"]
    ).any():
        fail(
            "duplicate candidate "
            "(game_id, team) keys"
        )

    if not candidate[
        "season"
    ].eq(season).all():
        fail(
            "candidate season mismatch"
        )

    if not candidate[
        "week"
    ].eq(week).all():
        fail(
            "candidate week mismatch"
        )

    unfinished, expected = (
        schedule_context(
            season,
            week,
        )
    )

    actual = {
        (
            str(row.game_id),
            str(row.team),
        )
        for row in candidate.itertuples()
    }

    if actual != expected:
        extra = sorted(
            actual - expected
        )
        missing = sorted(
            expected - actual
        )

        fail(
            "unfinished schedule coverage "
            f"mismatch; extra={extra}; "
            f"missing={missing}"
        )

    if int(
        audit.get("rows", -1)
    ) != len(candidate):
        fail(
            "candidate/audit row "
            "count mismatch"
        )

    candidate_games = int(
        candidate["game_id"].nunique()
    )

    if int(
        audit.get("games", -1)
    ) != candidate_games:
        fail(
            "candidate/audit game "
            "count mismatch"
        )

    expected_games = int(
        unfinished["game_id"].nunique()
    )

    if candidate_games != expected_games:
        fail(
            "candidate unfinished game "
            "count mismatch"
        )

    source_hashes = audit.get(
        "source_hashes",
        {},
    )

    matrix_key = str(MATRIX_PATH)

    if matrix_key not in source_hashes:
        fail(
            "matrix source hash missing "
            "from audit"
        )

    matrix_hash = sha256(
        MATRIX_PATH
    )

    if (
        source_hashes[matrix_key]
        != matrix_hash
    ):
        fail(
            "candidate matrix source "
            "hash mismatch"
        )

    if audit.get(
        "sqlite_modified"
    ) is not False:
        fail(
            "candidate audit does not "
            "prove sqlite_modified=false"
        )

    if audit.get(
        "production_modified"
    ) is not False:
        fail(
            "candidate audit does not "
            "prove production_modified=false"
        )

    return {
        "season": season,
        "week": week,
        "rows": len(candidate),
        "games": candidate_games,
        "keys": len(actual),
        "matrix_hash": matrix_hash,
    }


def promote_pair(
    candidate_csv: Path,
    candidate_audit: Path,
):
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    backup = (
        BACKUP_ROOT
        / timestamp
    )

    backup.mkdir(
        parents=True,
        exist_ok=False,
    )

    old_csv_exists = OUTPUT.exists()
    old_audit_exists = (
        AUDIT_OUTPUT.exists()
    )

    if old_csv_exists:
        shutil.copy2(
            OUTPUT,
            backup / OUTPUT.name,
        )

    if old_audit_exists:
        shutil.copy2(
            AUDIT_OUTPUT,
            backup / AUDIT_OUTPUT.name,
        )

    staged_csv = OUTPUT.with_name(
        OUTPUT.name
        + f".new.{os.getpid()}"
    )

    staged_audit = (
        AUDIT_OUTPUT.with_name(
            AUDIT_OUTPUT.name
            + f".new.{os.getpid()}"
        )
    )

    try:
        shutil.copy2(
            candidate_csv,
            staged_csv,
        )

        shutil.copy2(
            candidate_audit,
            staged_audit,
        )

        os.replace(
            staged_csv,
            OUTPUT,
        )

        try:
            os.replace(
                staged_audit,
                AUDIT_OUTPUT,
            )
        except Exception:
            if old_csv_exists:
                shutil.copy2(
                    backup / OUTPUT.name,
                    OUTPUT,
                )
            else:
                OUTPUT.unlink(
                    missing_ok=True
                )

            raise

    except Exception:
        staged_csv.unlink(
            missing_ok=True
        )
        staged_audit.unlink(
            missing_ok=True
        )
        raise

    return backup


def main() -> None:
    for required in (
        BUILDER_PATH,
        MATRIX_PATH,
        DB_PATH,
    ):
        if not required.is_file():
            fail(
                f"required input missing: "
                f"{required}"
            )

    builder = load_builder()

    with tempfile.TemporaryDirectory(
        prefix="adaptive-current-",
        dir=str(ROOT / "processed"),
    ) as temp_dir:
        temp = Path(temp_dir)

        candidate_csv = (
            temp / OUTPUT.name
        )

        candidate_audit = (
            temp / AUDIT_OUTPUT.name
        )

        builder.OUTPUT = candidate_csv
        builder.AUDIT_OUTPUT = (
            candidate_audit
        )

        builder.main()

        facts = validate_candidate(
            candidate_csv,
            candidate_audit,
        )

        backup = promote_pair(
            candidate_csv,
            candidate_audit,
        )

    final_facts = validate_candidate(
        OUTPUT,
        AUDIT_OUTPUT,
    )

    if final_facts != facts:
        fail(
            "post-promotion validation "
            "facts changed"
        )

    print(
        "ADAPTIVE_CURRENT_REFRESH_STATUS="
        "PASS"
    )
    print(
        f"ADAPTIVE_ROWS="
        f"{facts['rows']}"
    )
    print(
        f"ADAPTIVE_GAMES="
        f"{facts['games']}"
    )
    print(
        f"ADAPTIVE_KEYS="
        f"{facts['keys']}"
    )
    print(
        f"TARGET_SEASON="
        f"{facts['season']}"
    )
    print(
        f"TARGET_WEEK="
        f"{facts['week']}"
    )
    print(
        f"MATRIX_SHA256="
        f"{facts['matrix_hash']}"
    )
    print(
        f"BACKUP={backup}"
    )
    print(
        "SQLITE_MODIFIED=FALSE"
    )
    print(
        "PRODUCTION_MODIFIED=FALSE"
    )


if __name__ == "__main__":
    main()
