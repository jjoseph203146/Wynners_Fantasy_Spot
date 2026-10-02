#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from guardian_lifecycle import run_lifecycle_checks
from guardian_classification import classify_failures


ROOT = Path(__file__).resolve().parent
PROD_DATA = ROOT / "data"
PROD_PARQUET = PROD_DATA / "parquet"

FILES = [
    "current_unified_stat_forecasts.parquet",
    "current_unified_stat_forecasts_manifest.json",
    "current_unified_fanduel_expectation.parquet",
    "current_unified_fanduel_expectation_manifest.json",
    "current_starter_verification.parquet",
    "current_starter_verification_manifest.json",
]

PROD_FD = (
    PROD_PARQUET
    / "current_unified_fanduel_expectation.parquet"
)

PROD_FD_MANIFEST = (
    PROD_PARQUET
    / "current_unified_fanduel_expectation_manifest.json"
)

PROD_LKG = (
    PROD_DATA
    / "guardian"
    / "guardian_lkg.json"
)

PROD_AUDIT = (
    PROD_DATA
    / "guardian"
    / "guardian_audit.jsonl"
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def result_map(results: list[dict]) -> dict[str, dict]:
    return {
        str(row["check"]): row
        for row in results
    }


def main() -> int:
    print("=" * 72)
    print(
        "WFS GUARDIAN V2.6 PHASE 3 — "
        "ACTUAL VALIDATOR ISOLATION TEST"
    )
    print("=" * 72)

    production_before = {
        "fd": sha256(PROD_FD),
        "fd_manifest": sha256(PROD_FD_MANIFEST),
        "lkg": sha256(PROD_LKG),
        "audit": sha256(PROD_AUDIT),
    }

    if (
        production_before["fd"] is None
        or production_before["fd_manifest"] is None
    ):
        raise SystemExit(
            "FAIL: required production publication missing"
        )

    # Use the current schedule authority established by Guardian.
    # This is test input only; no database writes occur.
    schedule = {
        "season": 2026,
        "week": 2,
        "unfinished_games": 15,
        "unfinished_teams": 30,
        "unfinished_game_ids": set(),
    }

    # Derive active game IDs from the real stat manifest so the test
    # does not hardcode individual NFL game identifiers.
    import json

    stat_manifest = json.loads(
        (
            PROD_PARQUET
            / "current_unified_stat_forecasts_manifest.json"
        ).read_text()
    )

    states = stat_manifest.get("game_states", [])

    if not isinstance(states, list) or not states:
        raise SystemExit(
            "FAIL: stat manifest game_states unavailable"
        )

    active_game_ids = set()

    for row in states:
        if not isinstance(row, dict):
            continue

        game_id = row.get("game_id")

        if game_id is not None:
            active_game_ids.add(str(game_id))

    if len(active_game_ids) != 15:
        raise SystemExit(
            "FAIL: expected 15 active game IDs, got "
            + str(len(active_game_ids))
        )

    schedule["unfinished_game_ids"] = active_game_ids

    with tempfile.TemporaryDirectory(
        prefix="wfs_guardian_actual_validator_"
    ) as tmp:
        fixture_root = Path(tmp)
        fixture_parquet = (
            fixture_root / "data" / "parquet"
        )
        fixture_parquet.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name in FILES:
            source = PROD_PARQUET / name

            if not source.is_file():
                raise SystemExit(
                    f"FAIL: required fixture source missing: {source}"
                )

            shutil.copy2(
                source,
                fixture_parquet / name,
            )

        # --------------------------------------------------------
        # CONTROL: actual production validator against clean copies
        # --------------------------------------------------------

        control = run_lifecycle_checks(
            fixture_root,
            schedule,
        )

        control_map = result_map(control)

        print("CONTROL_RESULTS:")

        for row in control:
            print(
                f"  {row['check']:<36} "
                f"{row['status']:<5} "
                f"{row['detail']}"
            )

        control_failures = [
            row
            for row in control
            if row["status"] == "FAIL"
        ]

        if control_failures:
            raise SystemExit(
                "FAIL: clean fixture did not pass "
                "actual lifecycle validator"
            )

        print("CONTROL_VALIDATOR=PASS")

        # --------------------------------------------------------
        # FAULT: corrupt copied FanDuel publication only
        # --------------------------------------------------------

        fixture_fd = (
            fixture_parquet
            / "current_unified_fanduel_expectation.parquet"
        )

        with fixture_fd.open("ab") as f:
            f.write(
                b"WFS_GUARDIAN_ACTUAL_VALIDATOR_FAULT"
            )

        fault = run_lifecycle_checks(
            fixture_root,
            schedule,
        )

        fault_map = result_map(fault)

        print("-" * 72)
        print("FAULT_RESULTS:")

        for row in fault:
            print(
                f"  {row['check']:<36} "
                f"{row['status']:<5} "
                f"{row['detail']}"
            )

        fd_result = fault_map.get(
            "FANDUEL_PUBLICATION_LIFECYCLE"
        )

        if fd_result is None:
            raise SystemExit(
                "FAIL: FanDuel lifecycle result missing"
            )

        if fd_result["status"] != "FAIL":
            raise SystemExit(
                "FAIL: actual Guardian validator "
                "did not reject corrupted publication"
            )

        classification = classify_failures(fault)

        print(
            "ACTUAL_DETECTOR_STATUS=",
            fd_result["status"],
        )
        print(
            "FAILURE_CATEGORIES=",
            classification["categories"],
        )

        if "PUBLICATION" not in classification["categories"]:
            raise SystemExit(
                "FAIL: publication fault was not "
                "classified as PUBLICATION"
            )

        print("ACTUAL_VALIDATOR_FAULT_DETECTION=PASS")

    production_after = {
        "fd": sha256(PROD_FD),
        "fd_manifest": sha256(PROD_FD_MANIFEST),
        "lkg": sha256(PROD_LKG),
        "audit": sha256(PROD_AUDIT),
    }

    print("-" * 72)

    for name in production_before:
        unchanged = (
            production_before[name]
            == production_after[name]
        )

        print(
            f"PRODUCTION_{name.upper()}_UNCHANGED="
            f"{unchanged}"
        )

        if not unchanged:
            raise SystemExit(
                f"FAIL: production {name} changed"
            )

    print("PRODUCTION_ACTION=NONE")
    print("ACTUAL_VALIDATOR_USED=TRUE")
    print("V26_ACTUAL_VALIDATOR_TEST=PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
