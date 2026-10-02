#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from guardian_cross_stage import run_cross_stage
from guardian_classification import classify_failures


ROOT = Path(__file__).resolve().parent
PROD_PARQUET = ROOT / "data" / "parquet"

FILES = [
    "nfl_current_offensive_stat_forecasts.parquet",
    "nfl_current_kicker_stat_forecasts.parquet",
    "nfl_current_dst_stat_forecasts.parquet",
    "nfl_current_unified_stat_forecasts.parquet",
    "nfl_current_fanduel_expectation.parquet",
    "nfl_current_fanduel_expectation_gav2.parquet",
    "current_unified_fanduel_expectation.parquet",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def production_hashes() -> dict[str, str]:
    values = {}

    for name in FILES:
        path = PROD_PARQUET / name

        if not path.is_file():
            raise SystemExit(
                f"FAIL: production source missing: {path}"
            )

        values[name] = sha256(path)

    return values


def copy_fixture(root: Path) -> Path:
    p = root / "data" / "parquet"
    p.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        shutil.copy2(
            PROD_PARQUET / name,
            p / name,
        )

    return p


def build_schedule() -> dict:
    offense = pd.read_parquet(
        PROD_PARQUET
        / "nfl_current_offensive_stat_forecasts.parquet"
    )

    game_ids = {
        str(v)
        for v in offense["game_id"].dropna().unique()
    }

    teams = {
        str(v)
        for v in offense["team"].dropna().unique()
    }

    if len(game_ids) != 15:
        raise SystemExit(
            f"FAIL: expected 15 active games, got {len(game_ids)}"
        )

    if len(teams) != 30:
        raise SystemExit(
            f"FAIL: expected 30 active teams, got {len(teams)}"
        )

    return {
        "season": 2026,
        "week": 2,
        "unfinished_games": len(game_ids),
        "unfinished_teams": len(teams),
        "unfinished_game_ids": game_ids,
    }


def get_result(
    results: list[dict],
    check: str,
) -> dict:
    matches = [
        row
        for row in results
        if row["check"] == check
    ]

    if len(matches) != 1:
        raise SystemExit(
            f"FAIL: expected exactly one {check}, "
            f"got {len(matches)}"
        )

    return matches[0]


def show(label: str, results: list[dict]) -> None:
    print("-" * 72)
    print(label)

    for row in results:
        if (
            row["status"] == "FAIL"
            or row["check"]
            in {
                "OFFENSE_IDENTITY_CONTRACT",
                "COMPONENT_IDENTITY_HANDOFF",
                "CROSS_STAGE_RECONCILIATION",
            }
        ):
            print(
                f"  {row['check']:<38} "
                f"{row['status']:<5} "
                f"{row['detail']}"
            )


def main() -> int:
    print("=" * 72)
    print(
        "WFS GUARDIAN V2.6 PHASE 5 — "
        "ACTUAL CROSS-STAGE FIXTURE TEST"
    )
    print("=" * 72)

    before = production_hashes()
    schedule = build_schedule()

    print(
        "PRODUCTION_FILES_HASHED=",
        len(before),
    )

    # ============================================================
    # CONTROL
    # ============================================================

    with tempfile.TemporaryDirectory(
        prefix="wfs_guardian_xstage_control_"
    ) as tmp:
        fixture_root = Path(tmp)
        copy_fixture(fixture_root)

        results = run_cross_stage(
            fixture_root,
            schedule,
        )

        show(
            "CONTROL — CLEAN COPIED STAGES",
            results,
        )

        failures = [
            row
            for row in results
            if row["status"] == "FAIL"
        ]

        if failures:
            raise SystemExit(
                "FAIL: clean cross-stage fixture "
                "did not pass production validator"
            )

        print("CONTROL_CROSS_STAGE=PASS")

    # ============================================================
    # TEST 1 — IDENTITY
    #
    # Duplicate one offense player_id inside the copied offense
    # stage. This must violate the stage-local identity contract.
    # ============================================================

    with tempfile.TemporaryDirectory(
        prefix="wfs_guardian_xstage_identity_"
    ) as tmp:
        fixture_root = Path(tmp)
        p = copy_fixture(fixture_root)

        offense_path = (
            p
            / "nfl_current_offensive_stat_forecasts.parquet"
        )

        offense = pd.read_parquet(offense_path)

        if len(offense) < 2:
            raise SystemExit(
                "FAIL: offense fixture has fewer than 2 rows"
            )

        original_second = str(
            offense.loc[
                offense.index[1],
                "player_id",
            ]
        )

        duplicate_id = offense.loc[
            offense.index[0],
            "player_id",
        ]

        offense.loc[
            offense.index[1],
            "player_id",
        ] = duplicate_id

        offense.to_parquet(
            offense_path,
            index=False,
        )

        if str(duplicate_id) == original_second:
            raise SystemExit(
                "FAIL: selected offense rows already "
                "shared the same player_id"
            )

        results = run_cross_stage(
            fixture_root,
            schedule,
        )

        show(
            "FAULT TEST 1 — DUPLICATE OFFENSE IDENTITY",
            results,
        )

        identity = get_result(
            results,
            "OFFENSE_IDENTITY_CONTRACT",
        )

        if identity["status"] != "FAIL":
            raise SystemExit(
                "FAIL: duplicate offense identity accepted"
            )

        classification = classify_failures(
            results
        )

        print(
            "IDENTITY_FAILURE_CATEGORIES=",
            classification["categories"],
        )

        if "IDENTITY" not in classification["categories"]:
            raise SystemExit(
                "FAIL: duplicate identity not classified "
                "as IDENTITY"
            )

        print(
            "ACTUAL_IDENTITY_DETECTION=PASS"
        )

    # ============================================================
    # TEST 2 — HANDOFF
    #
    # Change one copied offense player_id to a brand-new unique ID.
    # This preserves offense-local uniqueness and inventory, but
    # breaks exact component -> unified identity reconciliation.
    # ============================================================

    with tempfile.TemporaryDirectory(
        prefix="wfs_guardian_xstage_handoff_"
    ) as tmp:
        fixture_root = Path(tmp)
        p = copy_fixture(fixture_root)

        offense_path = (
            p
            / "nfl_current_offensive_stat_forecasts.parquet"
        )

        offense = pd.read_parquet(offense_path)

        existing = {
            str(v)
            for v in offense["player_id"].dropna()
        }

        synthetic_id = "WFS-V26-HANDOFF-FAULT"

        if synthetic_id in existing:
            raise SystemExit(
                "FAIL: synthetic handoff ID already exists"
            )

        offense.loc[
            offense.index[0],
            "player_id",
        ] = synthetic_id

        offense.to_parquet(
            offense_path,
            index=False,
        )

        results = run_cross_stage(
            fixture_root,
            schedule,
        )

        show(
            "FAULT TEST 2 — COMPONENT HANDOFF MISMATCH",
            results,
        )

        local_identity = get_result(
            results,
            "OFFENSE_IDENTITY_CONTRACT",
        )

        handoff = get_result(
            results,
            "COMPONENT_IDENTITY_HANDOFF",
        )

        if local_identity["status"] != "PASS":
            raise SystemExit(
                "FAIL: handoff fixture also broke "
                "stage-local identity"
            )

        if handoff["status"] != "FAIL":
            raise SystemExit(
                "FAIL: component handoff mismatch accepted"
            )

        classification = classify_failures(
            results
        )

        print(
            "HANDOFF_FAILURE_CATEGORIES=",
            classification["categories"],
        )

        if "HANDOFF" not in classification["categories"]:
            raise SystemExit(
                "FAIL: boundary mismatch not classified "
                "as HANDOFF"
            )

        if "IDENTITY" in classification["categories"]:
            raise SystemExit(
                "FAIL: clean local identity was incorrectly "
                "classified as IDENTITY"
            )

        print(
            "LOCAL_IDENTITY_REMAINED_VALID=PASS"
        )
        print(
            "ACTUAL_HANDOFF_DETECTION=PASS"
        )

    # ============================================================
    # PRODUCTION IMMUTABILITY
    # ============================================================

    after = production_hashes()

    changed = [
        name
        for name in FILES
        if before[name] != after[name]
    ]

    print("-" * 72)
    print(
        "PRODUCTION_PARQUETS_UNCHANGED=",
        not changed,
    )

    if changed:
        raise SystemExit(
            "FAIL: production parquet changed: "
            + ",".join(changed)
        )

    print("PRODUCTION_LKG_MUTATED=FALSE")
    print("PRODUCTION_AUDIT_MUTATED=FALSE")
    print("PRODUCTION_ACTION=NONE")
    print("ACTUAL_CROSS_STAGE_VALIDATOR_USED=TRUE")
    print("V26_CROSS_STAGE_FIXTURE_TEST=PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
