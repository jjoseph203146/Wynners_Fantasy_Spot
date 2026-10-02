#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

READINESS = (
    ROOT
    / "data/fanduel/single_game/manifest/slate_readiness_r1.json"
)

STATE = (
    ROOT
    / "data/csv/fanduel_solver_inventory_state.json"
)

MAIN_CSV = (
    ROOT
    / "data/fanduel/slates/Main.csv"
)


def fail_closed(reason: str) -> int:
    print("CLASSIC_STALE_DETECTOR=FAIL_CLOSED")
    print(f"REASON={reason}")
    print("RECOVERY_AUTHORIZED=FALSE")
    return 1


def main() -> int:
    print("=" * 72)
    print("FANDUEL CLASSIC STALE SOURCE DETECTOR V1")
    print("=" * 72)

    for path in (READINESS, STATE, MAIN_CSV):
        if not path.is_file():
            return fail_closed(
                f"MISSING_REQUIRED_FILE:{path}"
            )

    try:
        readiness = json.loads(
            READINESS.read_text(encoding="utf-8")
        )
        state = json.loads(
            STATE.read_text(encoding="utf-8")
        )
        local = pd.read_csv(MAIN_CSV)
    except Exception as exc:
        return fail_closed(
            f"INPUT_READ_FAILED:{type(exc).__name__}:{exc}"
        )

    slates = readiness.get("slates")

    if not isinstance(slates, list):
        return fail_closed(
            "READINESS_SLATES_NOT_LIST"
        )

    main_matches = [
        s
        for s in slates
        if isinstance(s, dict)
        and str(
            s.get("public_slate_name", "")
        ).strip().lower() == "main"
    ]

    print(
        f"MAIN_MATCH_COUNT={len(main_matches)}"
    )

    if len(main_matches) != 1:
        return fail_closed(
            "MAIN_READINESS_CARDINALITY"
        )

    main = main_matches[0]

    if "gameInfo" not in local.columns:
        return fail_closed(
            "LOCAL_MAIN_GAMEINFO_MISSING"
        )

    if "salary" not in local.columns:
        return fail_closed(
            "LOCAL_MAIN_SALARY_MISSING"
        )

    canonical_games = main.get(
        "canonical_games"
    )

    if not isinstance(canonical_games, list):
        return fail_closed(
            "READINESS_CANONICAL_GAMES_NOT_LIST"
        )

    expected_games = {
        str(x).strip()
        for x in canonical_games
        if str(x).strip()
    }

    actual_games = {
        str(x).strip()
        for x in local["gameInfo"].dropna()
        if str(x).strip()
    }

    salary = pd.to_numeric(
        local["salary"]
        .astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )

    positive_salary_rows = int(
        (salary > 0).sum()
    )

    try:
        solver_open_rows = int(
            state.get(
                "solver_open_rows",
                -1,
            )
        )

        readiness_salary_rows = int(
            main.get(
                "numeric_salary_count",
                0,
            )
        )

        readiness_data_rows = int(
            main.get(
                "data_row_count",
                0,
            )
        )
    except (TypeError, ValueError):
        return fail_closed(
            "INVALID_NUMERIC_EVIDENCE"
        )

    inventory_no_open = (
        state.get("status")
        == "NO_OPEN_SLATES"
        and solver_open_rows == 0
        and state.get("authority")
        == "fanduel_slate_pool"
    )

    main_ready = (
        main.get("classification")
        == "CLASSIC"
        and main.get(
            "production_eligible"
        ) is True
        and main.get("ready") is True
        and readiness_salary_rows > 0
        and readiness_data_rows > 0
        and bool(expected_games)
        and bool(main.get("csv_sha256"))
        and bool(main.get("staging_file"))
    )

    membership_mismatch = (
        actual_games != expected_games
    )

    salary_contradiction = (
        readiness_salary_rows > 0
        and positive_salary_rows == 0
    )

    print(
        "CONDITION_INVENTORY_NO_OPEN="
        f"{inventory_no_open}"
    )
    print(
        "CONDITION_MAIN_READY="
        f"{main_ready}"
    )
    print(
        "CONDITION_MEMBERSHIP_MISMATCH="
        f"{membership_mismatch}"
    )
    print(
        "CONDITION_SALARY_CONTRADICTION="
        f"{salary_contradiction}"
    )

    print(
        f"READINESS_GAME_COUNT="
        f"{len(expected_games)}"
    )
    print(
        f"LOCAL_GAME_COUNT="
        f"{len(actual_games)}"
    )
    print(
        "READINESS_NUMERIC_SALARY_ROWS="
        f"{readiness_salary_rows}"
    )
    print(
        "LOCAL_POSITIVE_SALARY_ROWS="
        f"{positive_salary_rows}"
    )

    stale_proven = (
        inventory_no_open
        and main_ready
        and membership_mismatch
        and salary_contradiction
    )

    if stale_proven:
        print(
            "CLASSIC_STALE_DETECTOR="
            "STALE_SOURCE_PROVEN"
        )
        print(
            "CLASSIFICATION="
            "FANDUEL_CLASSIC_SOURCE_STALE"
        )
        print(
            "RECOVERY_CANDIDATE=TRUE"
        )
        print(
            "RECOVERY_EXECUTED=FALSE"
        )
        return 10

    print(
        "CLASSIC_STALE_DETECTOR="
        "NO_STALE_SOURCE_PROVEN"
    )
    print(
        "RECOVERY_CANDIDATE=FALSE"
    )
    print(
        "RECOVERY_EXECUTED=FALSE"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            "CLASSIC_STALE_DETECTOR="
            "FAIL_CLOSED"
        )
        print(
            f"UNHANDLED_ERROR="
            f"{type(exc).__name__}:{exc}"
        )
        print(
            "RECOVERY_AUTHORIZED=FALSE"
        )
        raise SystemExit(1)
