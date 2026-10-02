#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DB = ROOT / "data/nfl.db"

CURRENT_SOURCE = (
    ROOT
    / "data/parquet"
    / "nfl_current_fanduel_expectation.parquet"
)

CURRENT_OUTPUT = (
    ROOT
    / "data/parquet"
    / "nfl_current_fanduel_expectation_gav2.parquet"
)

FROZEN_SOURCE = (
    ROOT
    / "data/model_candidates/fanduel_expectation"
    / "stage24j_c_r7_r14h_c_20260912T195227Z"
    / "current_unified_fanduel_expectation_535.parquet"
)

FROZEN_EXPECTED = (
    ROOT
    / "data/sandbox/gav2_solver_propagation_20260913T030333Z"
    / "current_unified_fanduel_expectation_gav2.parquet"
)

CONTRACT = "WFS_GLOBAL_PLAYER_AVAILABILITY_V2"
IDENTITY = "EXACT_GSIS_ID"
PROJECTION_COLUMN = "expected_fanduel_points"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_authority() -> pd.DataFrame:
    if not DB.is_file():
        raise RuntimeError(f"GAV2_DATABASE_MISSING:{DB}")

    with sqlite3.connect(DB) as conn:
        exists = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table'
              AND name='injury_consensus_current'
            """
        ).fetchone()

        if not exists:
            raise RuntimeError(
                "GAV2_AUTHORITY_TABLE_MISSING:"
                "injury_consensus_current"
            )

        cols = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(injury_consensus_current)"
            ).fetchall()
        }

        required = {
            "gsis_id",
            "injury_gate",
        }

        missing = sorted(required - cols)
        if missing:
            raise RuntimeError(
                f"GAV2_AUTHORITY_SCHEMA_MISSING:{missing}"
            )

        optional = [
            c
            for c in [
                "player_name",
                "team",
                "position",
                "injury_gate_reason",
            ]
            if c in cols
        ]

        select_cols = [
            "gsis_id",
            "injury_gate",
            *optional,
        ]

        authority = pd.read_sql_query(
            "SELECT "
            + ", ".join(
                f'"{c}"' for c in select_cols
            )
            + " FROM injury_consensus_current",
            conn,
        )

    if authority.empty:
        raise RuntimeError(
            "GAV2_AUTHORITY_EMPTY"
        )

    authority["gsis_id"] = (
        authority["gsis_id"]
        .map(clean)
    )

    if authority["gsis_id"].eq("").any():
        raise RuntimeError(
            "GAV2_BLANK_GSIS_ID"
        )

    dup = authority[
        authority.duplicated(
            ["gsis_id"],
            keep=False,
        )
    ]

    if not dup.empty:
        ids = sorted(
            dup["gsis_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise RuntimeError(
            "GAV2_DUPLICATE_GSIS_ID:"
            + ",".join(ids[:20])
        )

    authority["injury_gate"] = (
        authority["injury_gate"]
        .map(clean)
        .str.upper()
    )

    valid = {"ALLOW", "BLOCK"}

    invalid = sorted(
        set(authority["injury_gate"])
        - valid
    )

    if invalid:
        raise RuntimeError(
            f"GAV2_INVALID_INJURY_GATE:{invalid}"
        )

    return authority


def validate_source(df: pd.DataFrame) -> None:
    required = {
        "game_id",
        "team",
        "position",
        "entity_id",
        "entity_name",
        "source_component",
        PROJECTION_COLUMN,
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            f"STAGE24_SOURCE_SCHEMA_MISSING:{missing}"
        )

    if df.empty:
        raise RuntimeError(
            "STAGE24_SOURCE_EMPTY"
        )

    points = pd.to_numeric(
        df[PROJECTION_COLUMN],
        errors="coerce",
    )

    if points.isna().any():
        raise RuntimeError(
            "STAGE24_SOURCE_PROJECTION_NULL"
        )

    if not np.isfinite(
        points.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "STAGE24_SOURCE_PROJECTION_NONFINITE"
        )

    if (points < 0).any():
        raise RuntimeError(
            "STAGE24_SOURCE_PROJECTION_NEGATIVE"
        )

    keys = [
        "game_id",
        "team",
        "position",
        "entity_id",
    ]

    if df.duplicated(keys).any():
        raise RuntimeError(
            "STAGE24_SOURCE_DUPLICATE_IDENTITY"
        )


def apply_gav2(
    source: pd.DataFrame,
    authority: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    out = source.copy(deep=True)

    offense = (
        out["source_component"]
        .astype(str)
        .str.upper()
        .eq("OFFENSE")
    )

    offense_ids = (
        out.loc[offense, "entity_id"]
        .map(clean)
    )

    if offense_ids.eq("").any():
        raise RuntimeError(
            "GAV2_OFFENSE_BLANK_ENTITY_ID"
        )

    block_ids = set(
        authority.loc[
            authority["injury_gate"].eq("BLOCK"),
            "gsis_id",
        ].astype(str)
    )

    allow_ids = set(
        authority.loc[
            authority["injury_gate"].eq("ALLOW"),
            "gsis_id",
        ].astype(str)
    )

    overlap = block_ids & allow_ids
    if overlap:
        raise RuntimeError(
            "GAV2_CONFLICTING_AUTHORITY:"
            + ",".join(sorted(overlap)[:20])
        )

    target = pd.Series(
        False,
        index=out.index,
    )

    target.loc[offense] = (
        offense_ids.isin(block_ids).to_numpy()
    )

    before = pd.to_numeric(
        out[PROJECTION_COLUMN],
        errors="raise",
    ).astype(float)

    positive_before = (
        target
        & before.gt(0.0)
    )

    already_zero = (
        target
        & before.eq(0.0)
    )

    out.loc[
        target,
        PROJECTION_COLUMN,
    ] = 0.0

    after = pd.to_numeric(
        out[PROJECTION_COLUMN],
        errors="raise",
    ).astype(float)

    if not out.loc[target, PROJECTION_COLUMN].eq(0.0).all():
        raise RuntimeError(
            "GAV2_BLOCK_PROJECTION_SURVIVED"
        )

    nonblock = ~target

    if not np.array_equal(
        before.loc[nonblock].to_numpy(),
        after.loc[nonblock].to_numpy(),
        equal_nan=True,
    ):
        raise RuntimeError(
            "GAV2_NONBLOCK_EXPECTATION_CHANGED"
        )

    metadata_cols = [
        c
        for c in source.columns
        if c != PROJECTION_COLUMN
    ]

    if not source[metadata_cols].equals(
        out[metadata_cols]
    ):
        raise RuntimeError(
            "GAV2_SOURCE_METADATA_CHANGED"
        )

    source_offense_ids = set(
        offense_ids.astype(str)
    )

    present_blocks = (
        block_ids & source_offense_ids
    )

    absent_blocks = (
        block_ids - source_offense_ids
    )

    audit = {
        "contract": CONTRACT,
        "identity": IDENTITY,
        "authority": (
            "production injury_consensus_current"
        ),
        "authority_rows": int(len(authority)),
        "gav2_blocks": int(len(block_ids)),
        "gav2_allows": int(len(allow_ids)),
        "stage24_block_targets": int(
            len(present_blocks)
        ),
        "stage24_absent_block_ids": int(
            len(absent_blocks)
        ),
        "stage24_already_zero": int(
            already_zero.sum()
        ),
        "stage24_meaningful_zeroings": int(
            positive_before.sum()
        ),
        "model_expectation_recomputed": False,
        "nonblock_expectations_changed": False,
        "display_name_identity": False,
        "fuzzy_matching": False,
        "projection_column": PROJECTION_COLUMN,
    }

    return out, audit


def regression() -> None:
    source = pd.read_parquet(FROZEN_SOURCE)
    expected = pd.read_parquet(FROZEN_EXPECTED)

    validate_source(source)
    validate_source(expected)

    authority = load_authority()

    actual, audit = apply_gav2(
        source,
        authority,
    )

    keys = [
        "game_id",
        "team",
        "position",
        "entity_id",
    ]

    a = actual.sort_values(
        keys
    ).reset_index(drop=True)

    e = expected.sort_values(
        keys
    ).reset_index(drop=True)

    if list(a.columns) != list(e.columns):
        raise RuntimeError(
            "GAV2_REGRESSION_COLUMN_DRIFT"
        )

    if len(a) != len(e):
        raise RuntimeError(
            "GAV2_REGRESSION_ROW_DRIFT"
        )

    if not a[keys].equals(e[keys]):
        raise RuntimeError(
            "GAV2_REGRESSION_IDENTITY_DRIFT"
        )

    ap = pd.to_numeric(
        a[PROJECTION_COLUMN],
        errors="raise",
    ).to_numpy(float)

    ep = pd.to_numeric(
        e[PROJECTION_COLUMN],
        errors="raise",
    ).to_numpy(float)

    delta = np.abs(ap - ep)

    mismatches = int(
        (delta > 1e-12).sum()
    )

    print("FROZEN_ROWS =", len(a))
    print(
        "MAX_ABS_DELTA =",
        float(delta.max())
        if len(delta)
        else 0.0,
    )
    print(
        "MISMATCH_GT_1E_12 =",
        mismatches,
    )
    print(
        "GAV2_BLOCK_TARGETS =",
        audit["stage24_block_targets"],
    )
    print(
        "GAV2_MEANINGFUL_ZEROINGS =",
        audit["stage24_meaningful_zeroings"],
    )

    if mismatches:
        raise RuntimeError(
            "GAV2_FROZEN_REGRESSION_FAILED"
        )

    print(
        "CLASSIC_GAV2_FROZEN_REGRESSION=PASS"
    )


def atomic_parquet(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        df.to_parquet(
            tmp_path,
            index=False,
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def current() -> None:
    source = pd.read_parquet(
        CURRENT_SOURCE
    )

    validate_source(source)

    authority = load_authority()

    out, audit = apply_gav2(
        source,
        authority,
    )

    atomic_parquet(
        out,
        CURRENT_OUTPUT,
    )

    print("CURRENT_ROWS =", len(out))
    print(
        "CURRENT_GAMES =",
        out["game_id"].nunique(),
    )
    print(
        "CURRENT_TEAMS =",
        out["team"].nunique(),
    )
    print(
        "GAV2_AUTHORITY_ROWS =",
        audit["authority_rows"],
    )
    print(
        "GAV2_BLOCK_IDS =",
        audit["gav2_blocks"],
    )
    print(
        "STAGE24_BLOCK_TARGETS =",
        audit["stage24_block_targets"],
    )
    print(
        "STAGE24_ABSENT_BLOCK_IDS =",
        audit["stage24_absent_block_ids"],
    )
    print(
        "STAGE24_ALREADY_ZERO =",
        audit["stage24_already_zero"],
    )
    print(
        "STAGE24_MEANINGFUL_ZEROINGS =",
        audit["stage24_meaningful_zeroings"],
    )
    print(
        "OUTPUT =",
        CURRENT_OUTPUT,
    )
    print(
        "SHA256 =",
        sha256(CURRENT_OUTPUT),
    )
    print(
        "CURRENT_CLASSIC_GAV2=PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--regression-week1",
        action="store_true",
    )

    group.add_argument(
        "--current",
        action="store_true",
    )

    args = parser.parse_args()

    if args.regression_week1:
        regression()
    else:
        current()


if __name__ == "__main__":
    main()
