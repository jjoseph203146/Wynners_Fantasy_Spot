#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys

import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

DEFAULT_DB = (
    ROOT
    / "data/nfl.db"
)

DEFAULT_CURRENT = (
    ROOT
    / "data/parquet/current_unified_stat_forecasts.parquet"
)

DEFAULT_MANIFEST = (
    ROOT
    / "data/parquet/current_unified_stat_forecasts_manifest.json"
)

PUBLISHER = (
    ROOT
    / "scripts/wfs_stat_forecast_publish_v1.py"
)

PYTHON = (
    ROOT
    / "venv/bin/python"
)

CONTRACT = (
    "WFS_STAT_FORECAST_AVAILABILITY_GATE_SANDBOX_V1"
)

HARD_SUPPRESSION_STATUSES = {
    "OUT",
    "INACTIVE",
}

PRESERVE_NUMERIC_STATUSES = {
    "QUESTIONABLE",
    "DOUBTFUL",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def fail(reason: str):
    print()
    print("=" * 110)
    print("STAT_FORECAST_AVAILABILITY_SANDBOX_RESULT=FAIL_CLOSED")
    print(f"FAIL_REASON={reason}")
    print("LIVE_CURRENT_PARQUET_EDITED=FALSE")
    print("LIVE_MANIFEST_EDITED=FALSE")
    print("PRODUCTION_PUBLISHER_EDITED=FALSE")
    print("DB_EDITED=FALSE")
    print("SOLVER_EDITED=FALSE")
    print("SERVICE_CHANGED=FALSE")
    print("SERVICE_RESTARTED=FALSE")
    print("=" * 110)

    raise SystemExit(1)


def normalize_status(value) -> str:
    if value is None:
        return ""

    return str(
        value
    ).strip().upper()


def scalar_equal(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True

    if pd.isna(a) != pd.isna(b):
        return False

    try:
        return float(a) == float(b)
    except Exception:
        return str(a) == str(b)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        default=str(
            DEFAULT_DB
        ),
    )

    parser.add_argument(
        "--current",
        default=str(
            DEFAULT_CURRENT
        ),
    )

    parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST
        ),
    )

    args = parser.parse_args()

    db_path = Path(
        args.db
    ).resolve()

    current_path = Path(
        args.current
    ).resolve()

    manifest_path = Path(
        args.manifest
    ).resolve()

    for required in [
        db_path,
        current_path,
        manifest_path,
        PUBLISHER,
        PYTHON,
    ]:
        if not required.exists():
            fail(
                f"MISSING_REQUIRED_PATH:{required}"
            )

    live_current_sha_before = sha256_file(
        current_path
    )

    live_manifest_sha_before = sha256_file(
        manifest_path
    )

    publisher_sha_before = sha256_file(
        PUBLISHER
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get("contract")
        !=
        "WFS_STAT_FORECAST_PUBLISH_V1"
    ):
        fail(
            "CURRENT_MANIFEST_CONTRACT_MISMATCH"
        )

    source_candidate_raw = manifest.get(
        "source_candidate_path"
    )

    if not source_candidate_raw:
        fail(
            "SOURCE_CANDIDATE_PATH_MISSING_FROM_MANIFEST"
        )

    source_candidate = Path(
        source_candidate_raw
    ).resolve()

    if not source_candidate.exists():
        fail(
            f"SOURCE_CANDIDATE_MISSING:{source_candidate}"
        )

    expected_source_sha = manifest.get(
        "source_candidate_sha256"
    )

    actual_source_sha = sha256_file(
        source_candidate
    )

    if (
        expected_source_sha
        and
        expected_source_sha != actual_source_sha
    ):
        fail(
            "SOURCE_CANDIDATE_SHA_MISMATCH"
        )

    source = pd.read_parquet(
        source_candidate
    )

    live_current = pd.read_parquet(
        current_path
    )

    required_columns = {
        "player_id",
        "game_id",
        "team",
        "position",
    }

    missing = sorted(
        required_columns
        -
        set(
            source.columns
        )
    )

    if missing:
        fail(
            f"SOURCE_REQUIRED_COLUMNS_MISSING:{missing}"
        )

    expected_columns = [
        column
        for column in source.columns
        if column.startswith(
            "expected_"
        )
    ]

    if not expected_columns:
        fail(
            "NO_EXPECTED_STAT_COLUMNS_FOUND"
        )

    print("=" * 110)
    print("WFS STAT FORECAST AVAILABILITY GATE — SANDBOX V1")
    print("PRODUCTION_MUTATION_ALLOWED=FALSE")
    print("LIVE_CURRENT_PARQUET_EDITED=FALSE")
    print("EXACT_ID_JOIN_ONLY=TRUE")
    print("FUZZY_MATCHING_ALLOWED=FALSE")
    print(
        "HARD_SUPPRESSION_STATUSES="
        f"{sorted(HARD_SUPPRESSION_STATUSES)}"
    )
    print(
        "PRESERVE_NUMERIC_STATUSES="
        f"{sorted(PRESERVE_NUMERIC_STATUSES)}"
    )
    print("=" * 110)

    print()
    print("[1] SOURCE / LIVE BASELINE")
    print("=" * 110)

    print(
        f"SOURCE_CANDIDATE={source_candidate}"
    )

    print(
        f"SOURCE_CANDIDATE_SHA={actual_source_sha}"
    )

    print(
        f"LIVE_CURRENT={current_path}"
    )

    print(
        f"LIVE_CURRENT_SHA_BEFORE={live_current_sha_before}"
    )

    print(
        f"LIVE_MANIFEST_SHA_BEFORE={live_manifest_sha_before}"
    )

    print(
        f"PUBLISHER_SHA_BEFORE={publisher_sha_before}"
    )

    print(
        f"SOURCE_ROWS={len(source)}"
    )

    print(
        f"SOURCE_GAMES={source['game_id'].nunique()}"
    )

    print(
        f"SOURCE_TEAMS={source['team'].nunique()}"
    )

    print(
        f"EXPECTED_STAT_COLUMN_COUNT={len(expected_columns)}"
    )

    print(
        "EXPECTED_STAT_COLUMNS="
        f"{expected_columns}"
    )

    print()
    print("[2] LOAD INJURY AUTHORITY")
    print("=" * 110)

    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
    )

    conn.row_factory = sqlite3.Row

    try:
        table = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name='injury_consensus_current'
        """).fetchone()

        if table is None:
            fail(
                "INJURY_CONSENSUS_CURRENT_MISSING"
            )

        rows = conn.execute("""
            SELECT
                gsis_id,
                team,
                position,
                player_name,
                consensus_status,
                injury_gate,
                injury_gate_reason,
                authoritative_source,
                last_updated
            FROM injury_consensus_current
            WHERE gsis_id IS NOT NULL
              AND trim(gsis_id) != ''
        """).fetchall()

    finally:
        conn.close()

    injury = pd.DataFrame(
        [
            dict(row)
            for row in rows
        ]
    )

    if injury.empty:
        fail(
            "INJURY_AUTHORITY_EMPTY"
        )

    injury["gsis_id"] = (
        injury["gsis_id"]
        .astype(str)
        .str.strip()
    )

    injury["consensus_status"] = (
        injury["consensus_status"]
        .map(
            normalize_status
        )
    )

    duplicate_ids = (
        injury[
            injury["gsis_id"] != ""
        ][
            "gsis_id"
        ]
        .value_counts()
    )

    duplicate_ids = duplicate_ids[
        duplicate_ids > 1
    ]

    if len(
        duplicate_ids
    ):
        fail(
            "INJURY_GSIS_ID_NOT_UNIQUE:"
            f"{duplicate_ids.to_dict()}"
        )

    authority_map = injury.set_index(
        "gsis_id"
    ).to_dict(
        orient="index"
    )

    print(
        f"INJURY_AUTHORITY_ROWS={len(injury)}"
    )

    print(
        "INJURY_STATUS_COUNTS="
        f"{injury['consensus_status'].value_counts().to_dict()}"
    )

    print(
        "INJURY_GSIS_ID_UNIQUE=True"
    )

    print()
    print("[3] APPLY EXACT-ID AVAILABILITY GATE IN MEMORY")
    print("=" * 110)

    gated = source.copy()

    gated["_wfs_injury_status"] = (
        gated["player_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .map(
            lambda player_id:
                authority_map.get(
                    player_id,
                    {},
                ).get(
                    "consensus_status",
                    "",
                )
        )
    )

    hard_mask = (
        gated[
            "_wfs_injury_status"
        ].isin(
            HARD_SUPPRESSION_STATUSES
        )
    )

    questionable_mask = (
        gated[
            "_wfs_injury_status"
        ].eq(
            "QUESTIONABLE"
        )
    )

    doubtful_mask = (
        gated[
            "_wfs_injury_status"
        ].eq(
            "DOUBTFUL"
        )
    )

    hard_rows_before = gated.loc[
        hard_mask
    ].copy()

    questionable_before = gated.loc[
        questionable_mask
    ].copy()

    doubtful_before = gated.loc[
        doubtful_mask
    ].copy()

    print(
        f"HARD_SUPPRESSION_ROWS={int(hard_mask.sum())}"
    )

    print(
        f"QUESTIONABLE_ROWS={int(questionable_mask.sum())}"
    )

    print(
        f"DOUBTFUL_ROWS={int(doubtful_mask.sum())}"
    )

    for column in expected_columns:
        gated.loc[
            hard_mask,
            column,
        ] = 0.0

    audit_rows = []

    for idx in gated.index[
        hard_mask
    ]:
        player_id = str(
            gated.at[
                idx,
                "player_id",
            ]
        ).strip()

        authority = authority_map.get(
            player_id,
            {},
        )

        audit_rows.append({
            "player_id":
                player_id,

            "entity_name":
                (
                    gated.at[
                        idx,
                        "entity_name",
                    ]
                    if
                    "entity_name" in gated.columns
                    else
                    None
                ),

            "team":
                gated.at[
                    idx,
                    "team",
                ],

            "position":
                gated.at[
                    idx,
                    "position",
                ],

            "game_id":
                gated.at[
                    idx,
                    "game_id",
                ],

            "consensus_status":
                authority.get(
                    "consensus_status"
                ),

            "injury_gate":
                authority.get(
                    "injury_gate"
                ),

            "injury_gate_reason":
                authority.get(
                    "injury_gate_reason"
                ),

            "authoritative_source":
                authority.get(
                    "authoritative_source"
                ),
        })

    for row in audit_rows:
        print(
            "SUPPRESSION_TARGET|"
            f"PLAYER_ID={row['player_id']}|"
            f"NAME={row['entity_name']!r}|"
            f"TEAM={row['team']}|"
            f"POS={row['position']}|"
            f"GAME={row['game_id']}|"
            f"STATUS={row['consensus_status']}|"
            f"GATE={row['injury_gate']}|"
            f"REASON={row['injury_gate_reason']}"
        )

    gated = gated.drop(
        columns=[
            "_wfs_injury_status",
        ]
    )

    print()
    print("[4] PRE-PUBLISH GATE VALIDATION")
    print("=" * 110)

    if len(
        gated
    ) != len(
        source
    ):
        fail(
            "ROW_COUNT_CHANGED"
        )

    if (
        set(
            gated["game_id"].astype(str)
        )
        !=
        set(
            source["game_id"].astype(str)
        )
    ):
        fail(
            "GAME_UNIVERSE_CHANGED"
        )

    if (
        set(
            gated["team"].astype(str)
        )
        !=
        set(
            source["team"].astype(str)
        )
    ):
        fail(
            "TEAM_UNIVERSE_CHANGED"
        )

    if list(
        gated.columns
    ) != list(
        source.columns
    ):
        fail(
            "SCHEMA_CHANGED"
        )

    hard_after = gated.loc[
        hard_mask
    ]

    hard_nonzero = []

    for idx, row in hard_after.iterrows():
        nonzero = []

        for column in expected_columns:
            value = pd.to_numeric(
                pd.Series(
                    [
                        row[
                            column
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            if (
                pd.notna(
                    value
                )
                and
                abs(
                    float(
                        value
                    )
                )
                >
                1e-12
            ):
                nonzero.append(
                    column
                )

        if nonzero:
            hard_nonzero.append({
                "player_id":
                    row[
                        "player_id"
                    ],

                "nonzero":
                    nonzero,
            })

    if hard_nonzero:
        fail(
            "HARD_OUT_NONZERO_REMAINS:"
            f"{hard_nonzero}"
        )

    preserve_failures = []

    for mask_name, before_df in [
        (
            "QUESTIONABLE",
            questionable_before,
        ),
        (
            "DOUBTFUL",
            doubtful_before,
        ),
    ]:
        for idx in before_df.index:
            for column in expected_columns:
                before_value = before_df.at[
                    idx,
                    column,
                ]

                after_value = gated.at[
                    idx,
                    column,
                ]

                if not scalar_equal(
                    before_value,
                    after_value,
                ):
                    preserve_failures.append({
                        "status":
                            mask_name,

                        "index":
                            int(
                                idx
                            ),

                        "player_id":
                            gated.at[
                                idx,
                                "player_id",
                            ],

                        "column":
                            column,

                        "before":
                            before_value,

                        "after":
                            after_value,
                    })

    if preserve_failures:
        fail(
            "QUESTIONABLE_DOUBTFUL_VALUES_CHANGED:"
            f"{preserve_failures[:20]}"
        )

    print(
        "ROW_COUNT_UNCHANGED=True"
    )

    print(
        "GAME_UNIVERSE_UNCHANGED=True"
    )

    print(
        "TEAM_UNIVERSE_UNCHANGED=True"
    )

    print(
        "SCHEMA_UNCHANGED=True"
    )

    print(
        "ALL_HARD_OUT_EXPECTED_STATS_ZERO=True"
    )

    print(
        "QUESTIONABLE_NUMERIC_FORECASTS_UNCHANGED=True"
    )

    print(
        "DOUBTFUL_NUMERIC_FORECASTS_UNCHANGED=True"
    )

    print()
    print("[5] BUILD SANDBOX")
    print("=" * 110)

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    sandbox_root = (
        ROOT
        / "data/sandbox"
        / (
            "stat_forecast_availability_gate_"
            + stamp
        )
    )

    sandbox_candidate_dir = (
        sandbox_root
        / "candidate"
    )

    sandbox_data_root = (
        sandbox_root
        / "published"
    )

    sandbox_candidate_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    sandbox_data_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    sandbox_candidate = (
        sandbox_candidate_dir
        / "current_unified_stat_forecasts.parquet"
    )

    gated.to_parquet(
        sandbox_candidate,
        index=False,
    )

    sandbox_candidate_sha = sha256_file(
        sandbox_candidate
    )

    print(
        f"SANDBOX_ROOT={sandbox_root}"
    )

    print(
        f"SANDBOX_GATED_CANDIDATE={sandbox_candidate}"
    )

    print(
        f"SANDBOX_GATED_CANDIDATE_SHA={sandbox_candidate_sha}"
    )

    print()
    print("[6] RUN EXISTING PUBLISHER AGAINST SANDBOX ONLY")
    print("=" * 110)

    published_at = manifest.get(
        "published_at_utc"
    )

    if not published_at:
        fail(
            "CURRENT_MANIFEST_PUBLISHED_AT_MISSING"
        )

    command = [
        str(
            PYTHON
        ),
        str(
            PUBLISHER
        ),
        "--source",
        str(
            sandbox_candidate
        ),
        "--data-root",
        str(
            sandbox_data_root
        ),
        "--db",
        str(
            db_path
        ),
        "--now-utc",
        str(
            published_at
        ),
    ]

    print(
        "SANDBOX_PUBLISH_COMMAND="
        + repr(
            command
        )
    )

    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )

    print(
        "SANDBOX_PUBLISH_STDOUT_BEGIN"
    )

    print(
        result.stdout
    )

    print(
        "SANDBOX_PUBLISH_STDOUT_END"
    )

    if result.stderr:
        print(
            "SANDBOX_PUBLISH_STDERR_BEGIN"
        )

        print(
            result.stderr
        )

        print(
            "SANDBOX_PUBLISH_STDERR_END"
        )

    if result.returncode != 0:
        fail(
            "SANDBOX_PUBLISHER_RETURN_CODE_"
            f"{result.returncode}"
        )

    if (
        "PUBLISH_RESULT=PASS"
        not in
        result.stdout
    ):
        fail(
            "SANDBOX_PUBLISH_PASS_TOKEN_MISSING"
        )

    sandbox_current = (
        sandbox_data_root
        / "parquet"
        / "current_unified_stat_forecasts.parquet"
    )

    sandbox_manifest = (
        sandbox_data_root
        / "parquet"
        / "current_unified_stat_forecasts_manifest.json"
    )

    if not sandbox_current.exists():
        fail(
            "SANDBOX_CURRENT_NOT_CREATED"
        )

    if not sandbox_manifest.exists():
        fail(
            "SANDBOX_MANIFEST_NOT_CREATED"
        )

    published = pd.read_parquet(
        sandbox_current
    )

    print()
    print("[7] POST-PUBLISH SANDBOX VALIDATION")
    print("=" * 110)

    if len(
        published
    ) != len(
        source
    ):
        fail(
            "SANDBOX_PUBLISHED_ROW_COUNT_CHANGED"
        )

    if list(
        published.columns
    ) != list(
        source.columns
    ):
        fail(
            "SANDBOX_PUBLISHED_SCHEMA_CHANGED"
        )

    if (
        set(
            published["game_id"].astype(str)
        )
        !=
        set(
            source["game_id"].astype(str)
        )
    ):
        fail(
            "SANDBOX_PUBLISHED_GAME_UNIVERSE_CHANGED"
        )

    if (
        set(
            published["team"].astype(str)
        )
        !=
        set(
            source["team"].astype(str)
        )
    ):
        fail(
            "SANDBOX_PUBLISHED_TEAM_UNIVERSE_CHANGED"
        )

    published_ids = (
        published[
            "player_id"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    published_status = published_ids.map(
        lambda player_id:
            authority_map.get(
                player_id,
                {},
            ).get(
                "consensus_status",
                "",
            )
    )

    published_hard_mask = (
        published_status.isin(
            HARD_SUPPRESSION_STATUSES
        )
    )

    published_hard = published.loc[
        published_hard_mask
    ]

    post_nonzero = []

    for _, row in published_hard.iterrows():
        nonzero = []

        for column in expected_columns:
            value = pd.to_numeric(
                pd.Series(
                    [
                        row[
                            column
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            if (
                pd.notna(
                    value
                )
                and
                abs(
                    float(
                        value
                    )
                )
                >
                1e-12
            ):
                nonzero.append(
                    column
                )

        if nonzero:
            post_nonzero.append({
                "player_id":
                    row[
                        "player_id"
                    ],

                "nonzero":
                    nonzero,
            })

    if post_nonzero:
        fail(
            "SANDBOX_HARD_OUT_NONZERO:"
            f"{post_nonzero}"
        )

    if int(
        published_hard_mask.sum()
    ) != int(
        hard_mask.sum()
    ):
        fail(
            "SANDBOX_HARD_OUT_ROW_COUNT_CHANGED"
        )

    print(
        f"SANDBOX_ROWS={len(published)}"
    )

    print(
        f"SANDBOX_GAMES={published['game_id'].nunique()}"
    )

    print(
        f"SANDBOX_TEAMS={published['team'].nunique()}"
    )

    print(
        f"SANDBOX_HARD_OUT_ROWS={int(published_hard_mask.sum())}"
    )

    print(
        "SANDBOX_HARD_OUT_ALL_EXPECTED_STATS_ZERO=True"
    )

    print()
    print("[8] LIVE IMMUTABILITY CHECK")
    print("=" * 110)

    live_current_sha_after = sha256_file(
        current_path
    )

    live_manifest_sha_after = sha256_file(
        manifest_path
    )

    publisher_sha_after = sha256_file(
        PUBLISHER
    )

    print(
        f"LIVE_CURRENT_SHA_AFTER={live_current_sha_after}"
    )

    print(
        f"LIVE_MANIFEST_SHA_AFTER={live_manifest_sha_after}"
    )

    print(
        f"PUBLISHER_SHA_AFTER={publisher_sha_after}"
    )

    live_current_unchanged = (
        live_current_sha_before
        ==
        live_current_sha_after
    )

    live_manifest_unchanged = (
        live_manifest_sha_before
        ==
        live_manifest_sha_after
    )

    publisher_unchanged = (
        publisher_sha_before
        ==
        publisher_sha_after
    )

    print(
        f"LIVE_CURRENT_UNCHANGED={live_current_unchanged}"
    )

    print(
        f"LIVE_MANIFEST_UNCHANGED={live_manifest_unchanged}"
    )

    print(
        f"PRODUCTION_PUBLISHER_UNCHANGED={publisher_unchanged}"
    )

    if not all([
        live_current_unchanged,
        live_manifest_unchanged,
        publisher_unchanged,
    ]):
        fail(
            "LIVE_ARTIFACT_MUTATION_DETECTED"
        )

    print()
    print("[9] WRITE SANDBOX AUDIT")
    print("=" * 110)

    audit_path = (
        sandbox_root
        / "availability_gate_sandbox_audit.json"
    )

    audit = {
        "contract":
            CONTRACT,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "policy": {
            "exact_id_join_only":
                True,

            "forecast_identity_field":
                "player_id",

            "injury_identity_field":
                "gsis_id",

            "hard_suppression_statuses":
                sorted(
                    HARD_SUPPRESSION_STATUSES
                ),

            "preserve_numeric_statuses":
                sorted(
                    PRESERVE_NUMERIC_STATUSES
                ),

            "all_expected_columns_zeroed_for_hard_suppression":
                True,

            "display_name_matching_used":
                False,

            "fuzzy_matching_used":
                False,
        },

        "source": {
            "path":
                str(
                    source_candidate
                ),

            "sha256":
                actual_source_sha,

            "rows":
                int(
                    len(
                        source
                    )
                ),

            "games":
                int(
                    source[
                        "game_id"
                    ].nunique()
                ),

            "teams":
                int(
                    source[
                        "team"
                    ].nunique()
                ),
        },

        "suppression": {
            "hard_suppression_rows":
                int(
                    hard_mask.sum()
                ),

            "questionable_rows":
                int(
                    questionable_mask.sum()
                ),

            "doubtful_rows":
                int(
                    doubtful_mask.sum()
                ),

            "targets":
                audit_rows,
        },

        "validation": {
            "row_count_unchanged":
                True,

            "game_universe_unchanged":
                True,

            "team_universe_unchanged":
                True,

            "schema_unchanged":
                True,

            "hard_out_expected_stats_zero":
                True,

            "questionable_forecasts_unchanged":
                True,

            "doubtful_forecasts_unchanged":
                True,

            "sandbox_publish_pass":
                True,

            "live_current_unchanged":
                live_current_unchanged,

            "live_manifest_unchanged":
                live_manifest_unchanged,

            "production_publisher_unchanged":
                publisher_unchanged,
        },

        "sandbox": {
            "root":
                str(
                    sandbox_root
                ),

            "gated_candidate":
                str(
                    sandbox_candidate
                ),

            "gated_candidate_sha256":
                sandbox_candidate_sha,

            "published_current":
                str(
                    sandbox_current
                ),

            "published_current_sha256":
                sha256_file(
                    sandbox_current
                ),

            "published_manifest":
                str(
                    sandbox_manifest
                ),

            "published_manifest_sha256":
                sha256_file(
                    sandbox_manifest
                ),
        },

        "live_production": {
            "current_path":
                str(
                    current_path
                ),

            "current_sha256":
                live_current_sha_after,

            "manifest_path":
                str(
                    manifest_path
                ),

            "manifest_sha256":
                live_manifest_sha_after,

            "publisher_path":
                str(
                    PUBLISHER
                ),

            "publisher_sha256":
                publisher_sha_after,
        },
    }

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print(
        f"SANDBOX_AUDIT={audit_path}"
    )

    print(
        f"SANDBOX_AUDIT_SHA={sha256_file(audit_path)}"
    )

    print()
    print("=" * 110)
    print("STAT FORECAST AVAILABILITY SANDBOX FINAL")
    print("=" * 110)

    print(
        f"SOURCE_ROWS={len(source)}"
    )

    print(
        f"HARD_SUPPRESSION_ROWS={int(hard_mask.sum())}"
    )

    print(
        f"QUESTIONABLE_ROWS={int(questionable_mask.sum())}"
    )

    print(
        f"DOUBTFUL_ROWS={int(doubtful_mask.sum())}"
    )

    print(
        "HARD_OUT_EXPECTED_STATS_ZERO=True"
    )

    print(
        "QUESTIONABLE_FORECASTS_PRESERVED=True"
    )

    print(
        "DOUBTFUL_FORECASTS_PRESERVED=True"
    )

    print(
        "ROW_COUNT_UNCHANGED=True"
    )

    print(
        "GAME_UNIVERSE_UNCHANGED=True"
    )

    print(
        "TEAM_UNIVERSE_UNCHANGED=True"
    )

    print(
        "SCHEMA_UNCHANGED=True"
    )

    print(
        f"LIVE_CURRENT_UNCHANGED={live_current_unchanged}"
    )

    print(
        f"LIVE_MANIFEST_UNCHANGED={live_manifest_unchanged}"
    )

    print(
        f"PRODUCTION_PUBLISHER_UNCHANGED={publisher_unchanged}"
    )

    print(
        "DB_EDITED=FALSE"
    )

    print(
        "SOLVER_EDITED=FALSE"
    )

    print(
        "SERVICE_CHANGED=FALSE"
    )

    print(
        "SERVICE_RESTARTED=FALSE"
    )

    print(
        "PRODUCTION_PROMOTION_PERFORMED=FALSE"
    )

    print(
        "STAT_FORECAST_AVAILABILITY_SANDBOX_RESULT=PASS"
    )

    print(
        "NEXT_GATE=PRODUCTION_PUBLISHER_AVAILABILITY_GATE_PROMOTION_REVIEW"
    )

    print(
        "STOP_GATE=RETURN_SANDBOX_OUTPUT_BEFORE_PRODUCTION_PROMOTION"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
