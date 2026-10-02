#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import io
import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

V3_PATH = (
    ROOT
    / "processed"
    / "offensive_team_reconciliation_shadow_v3.csv"
)

VALIDATION_PATH = (
    ROOT
    / "processed"
    / "offensive_team_reconciliation_v3_validation_audit.json"
)

PRODUCTION_PARQUET = (
    ROOT
    / "data"
    / "parquet"
    / "nfl_production_projection.parquet"
)

PRODUCTION_CSV = (
    ROOT
    / "data"
    / "csv"
    / "nfl_production_projection.csv"
)

AUDIT_DIR = (
    ROOT
    / "processed"
)

EXPECTED_V3_SHA256 = (
    "285b84a38907a32eb0c2b0f87ad366127dceea498acb8c8acf55b326326c6b88"
)

EXPECTED_COUNTS = {
    "total": 574,
    "PRIMARY_QB": 32,
    "ACTIVE_ROTATION": 398,
    "CONTINGENCY_QB": 51,
    "UNAVAILABLE": 93,
    "pool": 430,
}

POOL_ROLES = {
    "PRIMARY_QB",
    "ACTIVE_ROTATION",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(
        f"FAIL_CLOSED: {message}"
    )


def fanduel_points_from_v3(
    df: pd.DataFrame,
) -> pd.Series:

    required = [
        "reconciled_passing_yards",
        "reconciled_passing_tds",
        "reconciled_interceptions",
        "reconciled_rushing_yards",
        "reconciled_rushing_tds",
        "reconciled_receptions",
        "reconciled_receiving_yards",
        "reconciled_receiving_tds",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        fail(
            "V3 missing FanDuel scoring columns: "
            + ", ".join(missing)
        )

    numeric = {}

    for column in required:
        numeric[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if numeric[column].isna().any():
            fail(
                f"non-numeric/null V3 scoring values in {column}"
            )

        values = numeric[column].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            fail(
                f"non-finite V3 scoring values in {column}"
            )

    # Official FanDuel Classic offensive scoring:
    # passing yards: 0.04
    # passing TD: 4
    # interception: -1
    # rushing yards: 0.1
    # rushing TD: 6
    # reception: 0.5
    # receiving yards: 0.1
    # receiving TD: 6
    #
    # No 300-yard passing bonus.
    # No 100-yard rushing/receiving bonus.
    #
    # Fumbles lost are intentionally not altered here because
    # V3 does not reconcile fumble-loss projections.

    return (
        numeric["reconciled_passing_yards"] * 0.04
        + numeric["reconciled_passing_tds"] * 4.0
        - numeric["reconciled_interceptions"] * 1.0
        + numeric["reconciled_rushing_yards"] * 0.10
        + numeric["reconciled_rushing_tds"] * 6.0
        + numeric["reconciled_receptions"] * 0.50
        + numeric["reconciled_receiving_yards"] * 0.10
        + numeric["reconciled_receiving_tds"] * 6.0
    )


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--production-parquet", type=Path, default=PRODUCTION_PARQUET)
    parser.add_argument("--production-csv", type=Path, default=PRODUCTION_CSV)
    parser.add_argument("--backup-root", type=Path, default=ROOT / "backups")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    source_path = V3_PATH
    validation_path = VALIDATION_PATH
    production_parquet = args.production_parquet
    production_csv = args.production_csv
    expected_hash = EXPECTED_V3_SHA256
    expected_counts = EXPECTED_COUNTS
    expected_matches = 406
    bound_v3 = bound_production = binding = None
    if args.validation:
        try:
            from scripts.preflight_offensive_reconciliation_v3 import load_validated_candidate
        except ModuleNotFoundError:
            from preflight_offensive_reconciliation_v3 import load_validated_candidate
        if args.season is None or args.week is None:
            fail("bound promotion requires explicit season and week")
        bound_v3, bound_production, binding = load_validated_candidate(
            args.validation, args.season, args.week,
        )
        source_path = args.validation.parent / binding["candidate_path"]
        validation_path = args.validation
        expected_hash = binding["candidate_sha256"]
        expected_counts = {"total": binding["row_count"], "pool": binding["active_pool_count"], **binding["role_counts"]}
        expected_matches = binding["production_match_count"]
        if (sha256(production_parquet) != binding["inputs"]["baseline.parquet"]
                or sha256(production_csv) != binding["inputs"]["baseline.csv"]):
            fail("production baseline changed after preflight")

    print("=" * 100)
    print("WFS OFFENSIVE RECONCILIATION V3 CONTROLLED PROMOTION")
    print("=" * 100)

    # ------------------------------------------------------
    # Required artifacts
    # ------------------------------------------------------

    for path in [
        source_path,
        validation_path,
        production_parquet,
        production_csv,
    ]:
        if not path.exists():
            fail(
                f"missing required artifact: {path}"
            )

    # ------------------------------------------------------
    # Frozen V3 source hash
    # ------------------------------------------------------

    # Bound mode already hashed and parsed the same in-memory bytes.
    source_bytes = None if binding else source_path.read_bytes()
    source_hash = binding["candidate_sha256"] if binding else hashlib.sha256(source_bytes).hexdigest()

    print(
        f"V3_SOURCE_SHA256={source_hash}"
    )

    if source_hash != expected_hash:
        fail(
            "V3 source hash changed since validated preflight"
        )

    # ------------------------------------------------------
    # Validation audit
    # ------------------------------------------------------

    validation = binding if binding else json.loads(validation_path.read_bytes())

    status = validation.get(
        "status"
    )

    if status != "PASS_HARD_CONTRACTS":
        fail(
            f"V3 validation status is {status!r}"
        )

    hard_failures = validation.get(
        "hard_failures",
        {}
    )

    if not isinstance(
        hard_failures,
        dict,
    ):
        fail(
            "validation hard_failures is not a dictionary"
        )

    nonzero_failures = {
        key: value
        for key, value in hard_failures.items()
        if int(value) != 0
    }

    if nonzero_failures:
        fail(
            f"validation hard failures present: "
            f"{nonzero_failures}"
        )

    print(
        "VALIDATION_STATUS=PASS_HARD_CONTRACTS"
    )

    # ------------------------------------------------------
    # Load V3
    # ------------------------------------------------------

    v3 = bound_v3 if binding else pd.read_csv(io.BytesIO(source_bytes), low_memory=False)

    if len(v3) != expected_counts["total"]:
        fail(
            f"V3 rows {len(v3)} != "
            f"{expected_counts['total']}"
        )

    required_v3 = [
        "game_id",
        "player_id",
        "reconciliation_role",
    ]

    missing = [
        column
        for column in required_v3
        if column not in v3.columns
    ]

    if missing:
        fail(
            "V3 missing required columns: "
            + ", ".join(missing)
        )

    role_counts = (
        v3["reconciliation_role"]
        .fillna("<NULL>")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    for role in [
        "PRIMARY_QB",
        "ACTIVE_ROTATION",
        "CONTINGENCY_QB",
        "UNAVAILABLE",
    ]:
        actual = int(
            role_counts.get(
                role,
                0,
            )
        )

        expected = expected_counts[
            role
        ]

        if actual != expected:
            fail(
                f"{role} rows {actual} != {expected}"
            )

    pool = v3[
        v3[
            "reconciliation_role"
        ].isin(
            POOL_ROLES
        )
    ].copy()

    if len(pool) != expected_counts["pool"]:
        fail(
            f"reconciliation pool rows "
            f"{len(pool)} != "
            f"{expected_counts['pool']}"
        )

    if pool[
        [
            "game_id",
            "player_id",
        ]
    ].duplicated().any():
        fail(
            "duplicate game_id/player_id keys in V3 pool"
        )

    # ------------------------------------------------------
    # Calculate V3 FanDuel projection
    # ------------------------------------------------------

    pool[
        "v3_reconciled_fd_points"
    ] = fanduel_points_from_v3(
        pool
    )

    fd_values = pool[
        "v3_reconciled_fd_points"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        fd_values
    ).all():
        fail(
            "V3 FanDuel projections contain non-finite values"
        )

    if (
        pool[
            "v3_reconciled_fd_points"
        ]
        <
        0
    ).any():
        fail(
            "V3 FanDuel projections contain negative values"
        )

    # ------------------------------------------------------
    # Load production
    # ------------------------------------------------------

    production = bound_production if binding else pd.read_parquet(production_parquet)

    production_before_rows = len(
        production
    )

    required_prod = [
        "game_id",
        "player_id",
        "ridge_projection",
        "production_status",
    ]

    missing = [
        column
        for column in required_prod
        if column not in production.columns
    ]

    if missing:
        fail(
            "production missing required columns: "
            + ", ".join(missing)
        )

    if production[
        [
            "game_id",
            "player_id",
        ]
    ].duplicated().any():
        fail(
            "duplicate game_id/player_id keys in production"
        )

    # ------------------------------------------------------
    # Exact authoritative join
    # ------------------------------------------------------

    attach = pool[
        [
            "game_id",
            "player_id",
            "reconciliation_role",
            "v3_reconciled_fd_points",
        ]
    ].copy()

    merged = production.merge(
        attach,
        on=[
            "game_id",
            "player_id",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(merged) != production_before_rows:
        fail(
            "production row count changed during V3 join"
        )

    matched_mask = merged[
        "v3_reconciled_fd_points"
    ].notna()

    matched_rows = int(
        matched_mask.sum()
    )

    print(
        f"PRODUCTION_ROWS={len(merged)}"
    )
    print(
        f"V3_POOL_ROWS={len(pool)}"
    )
    print(
        f"MATCHED_PRODUCTION_ROWS={matched_rows}"
    )

    # Exact result established during preflight.
    if matched_rows != expected_matches:
        fail(
            f"expected {expected_matches} production/V3 matches; "
            f"found {matched_rows}"
        )

    # ------------------------------------------------------
    # Preserve frozen Ridge output and publish V3 projection
    # ------------------------------------------------------

    if "ridge_projection_pre_v3" in merged.columns:
        fail(
            "ridge_projection_pre_v3 already exists; "
            "refusing ambiguous repeat promotion"
        )

    merged[
        "ridge_projection_pre_v3"
    ] = merged[
        "ridge_projection"
    ]

    merged[
        "offensive_reconciliation_version"
    ] = np.where(
        matched_mask,
        "V3",
        None,
    )

    merged[
        "offensive_reconciliation_role"
    ] = merged[
        "reconciliation_role"
    ]

    merged[
        "offensive_reconciliation_fd_points"
    ] = merged[
        "v3_reconciled_fd_points"
    ]

    # Only MODEL_READY rows can replace the frozen Ridge
    # production point projection.
    promote_mask = (
        matched_mask
        &
        (
            merged[
                "production_status"
            ]
            ==
            "MODEL_READY"
        )
    )

    promoted_rows = int(
        promote_mask.sum()
    )

    if promoted_rows <= 0:
        fail(
            "no MODEL_READY rows eligible for V3 promotion"
        )

    merged.loc[
        promote_mask,
        "ridge_projection",
    ] = merged.loc[
        promote_mask,
        "v3_reconciled_fd_points",
    ]

    # Do NOT overwrite injury_adjusted_projection here.
    # That field retains the semantics of the existing
    # live injury reforecast system.

    merged.drop(
        columns=[
            "reconciliation_role",
            "v3_reconciled_fd_points",
        ],
        inplace=True,
    )

    # ------------------------------------------------------
    # Post-merge contracts
    # ------------------------------------------------------

    if len(merged) != production_before_rows:
        fail(
            "post-promotion row count changed"
        )

    if merged[
        [
            "game_id",
            "player_id",
        ]
    ].duplicated().any():
        fail(
            "post-promotion duplicate keys"
        )

    promoted_values = pd.to_numeric(
        merged.loc[
            promote_mask,
            "ridge_projection",
        ],
        errors="coerce",
    )

    if promoted_values.isna().any():
        fail(
            "promoted ridge projections contain NULL"
        )

    if not np.isfinite(
        promoted_values.to_numpy(
            dtype=float
        )
    ).all():
        fail(
            "promoted ridge projections contain non-finite values"
        )

    if (
        promoted_values
        <
        0
    ).any():
        fail(
            "promoted ridge projections contain negative values"
        )

    # ------------------------------------------------------
    # Backup
    # ------------------------------------------------------

    if args.validate_only:
        print(json.dumps({"status": "PASS_VALIDATED_ONLY", "candidate_id": binding["candidate_id"] if binding else "FROZEN_LEGACY",
                          "source_sha256": source_hash, "matched_rows": matched_rows}, sort_keys=True))
        return

    if binding and (sha256(production_parquet) != binding["inputs"]["baseline.parquet"]
                    or sha256(production_csv) != binding["inputs"]["baseline.csv"]):
        fail("production baseline changed before publication")

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    backup_dir = (
        args.backup_root
        / f"v3_production_promotion_{timestamp}"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    backup_parquet = (
        backup_dir
        / production_parquet.name
    )

    backup_csv = (
        backup_dir
        / production_csv.name
    )

    shutil.copy2(
        production_parquet,
        backup_parquet,
    )

    shutil.copy2(
        production_csv,
        backup_csv,
    )

    source_script = Path(
        __file__
    ).resolve()

    shutil.copy2(
        source_script,
        backup_dir
        / source_script.name,
    )

    before_parquet_hash = sha256(
        production_parquet
    )

    before_csv_hash = sha256(
        production_csv
    )

    # ------------------------------------------------------
    # Atomic publication
    # ------------------------------------------------------

    temp_parquet = None
    temp_csv = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=production_parquet.parent,
            prefix=".v3_promote_",
            suffix=".parquet",
            delete=False,
        ) as handle:
            temp_parquet = Path(
                handle.name
            )

        with tempfile.NamedTemporaryFile(
            dir=production_csv.parent,
            prefix=".v3_promote_",
            suffix=".csv",
            delete=False,
        ) as handle:
            temp_csv = Path(
                handle.name
            )

        merged.to_parquet(
            temp_parquet,
            index=False,
        )

        merged.to_csv(
            temp_csv,
            index=False,
        )

        verify_parquet = pd.read_parquet(
            temp_parquet
        )

        verify_csv = pd.read_csv(
            temp_csv,
            low_memory=False,
        )

        if len(
            verify_parquet
        ) != production_before_rows:
            fail(
                "temporary parquet row-count verification failed"
            )

        if len(
            verify_csv
        ) != production_before_rows:
            fail(
                "temporary CSV row-count verification failed"
            )

        os.replace(
            temp_parquet,
            production_parquet,
        )

        temp_parquet = None

        os.replace(
            temp_csv,
            production_csv,
        )

        temp_csv = None

        # --------------------------------------------------
        # Re-read live publication
        # --------------------------------------------------

        published = pd.read_parquet(
            production_parquet
        )

        if len(
            published
        ) != production_before_rows:
            fail(
                "published production row count changed"
            )

        required_new = [
            "ridge_projection_pre_v3",
            "offensive_reconciliation_version",
            "offensive_reconciliation_role",
            "offensive_reconciliation_fd_points",
        ]

        missing_new = [
            column
            for column in required_new
            if column not in published.columns
        ]

        if missing_new:
            fail(
                "published artifact missing V3 columns: "
                + ", ".join(missing_new)
            )

        published_v3 = int(
            (
                published[
                    "offensive_reconciliation_version"
                ]
                ==
                "V3"
            ).sum()
        )

        if published_v3 != matched_rows:
            fail(
                f"published V3 marker count "
                f"{published_v3} != {matched_rows}"
            )

        after_parquet_hash = sha256(
            production_parquet
        )

        after_csv_hash = sha256(
            production_csv
        )

        manifest = {
            "version":
                "WFS_OFFENSIVE_RECONCILIATION_V3_PRODUCTION_PROMOTION",
            "timestamp_utc":
                timestamp,
            "status":
                "PASS_PROMOTED",
            "source":
                str(source_path),
            "source_sha256":
                source_hash,
            "validation":
                str(validation_path),
            "validation_status":
                status,
            "production_parquet":
                str(production_parquet),
            "production_csv":
                str(production_csv),
            "production_rows":
                production_before_rows,
            "v3_rows":
                len(v3),
            "v3_pool_rows":
                len(pool),
            "matched_production_rows":
                matched_rows,
            "promoted_model_ready_rows":
                promoted_rows,
            "backup_directory":
                str(backup_dir),
            "before_parquet_sha256":
                before_parquet_hash,
            "after_parquet_sha256":
                after_parquet_hash,
            "before_csv_sha256":
                before_csv_hash,
            "after_csv_sha256":
                after_csv_hash,
            "sqlite_modified":
                False,
        }

        manifest_path = (
            backup_dir
            / "v3_production_promotion_manifest.json"
        )

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                manifest,
                handle,
                indent=2,
                sort_keys=True,
            )

        print()
        print("=" * 100)
        print("V3 PRODUCTION PROMOTION SUCCESS")
        print("=" * 100)
        print(
            f"V3_POOL_ROWS={len(pool)}"
        )
        print(
            f"MATCHED_PRODUCTION_ROWS={matched_rows}"
        )
        print(
            f"PROMOTED_MODEL_READY_ROWS={promoted_rows}"
        )
        print(
            f"BACKUP={backup_dir}"
        )
        print(
            f"MANIFEST={manifest_path}"
        )
        print(
            f"production_parquet_SHA256={after_parquet_hash}"
        )
        print(
            "SQLITE_MODIFIED=FALSE"
        )
        print(
            "STATUS=PASS_PROMOTED"
        )
        print("=" * 100)

    except Exception:

        print()
        print(
            "PROMOTION FAILURE DETECTED — ROLLING BACK"
        )

        shutil.copy2(
            backup_parquet,
            production_parquet,
        )

        shutil.copy2(
            backup_csv,
            production_csv,
        )

        if temp_parquet is not None:
            temp_parquet.unlink(
                missing_ok=True
            )

        if temp_csv is not None:
            temp_csv.unlink(
                missing_ok=True
            )

        print(
            "ROLLBACK_COMPLETE=TRUE"
        )
        print(
            "SQLITE_MODIFIED=FALSE"
        )

        raise


if __name__ == "__main__":
    main()
