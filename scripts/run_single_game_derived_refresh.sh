#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
PY="$ROOT/venv/bin/python"

L="$ROOT/NFL-POSTGAME-1F-I-L-R2.py"
R="$ROOT/NFL-POSTGAME-1F-I-R1.py"
X="$ROOT/NFL-POSTGAME-1F-I-X-R1.py"

EXPECTED_L="ed22cb9d2afe12d135a0a277e71d89fe8a17d6e4731ece6c6382b3259b73f2b2"
EXPECTED_R="314befefa30b6a2284d2936411ae56b2a98a4f59b7b26be8b35ba8c0587f56cb"
EXPECTED_X="8866e050e9355fad7e384720ff8e669894eff77f136a1533eb22ffdf2cbd2582"

IDENTITY="$ROOT/data/fanduel/single_game/derived/single_game_identity_pool.parquet"
BRIDGE="$ROOT/data/fanduel/single_game/derived/single_game_projection_bridge_r1.parquet"
FINAL="$ROOT/data/fanduel/single_game/derived/single_game_projection_pool.parquet"

verify_sha() {
    local file="$1"
    local expected="$2"
    local label="$3"

    if [ ! -f "$file" ]; then
        echo "FAIL_CLOSED_MISSING_${label}=$file"
        exit 2
    fi

    local actual

    actual="$(sha256sum "$file" | awk '{print $1}')"

    if [ "$actual" != "$expected" ]; then
        echo "FAIL_CLOSED_${label}_SHA_MISMATCH=TRUE"
        echo "EXPECTED=$expected"
        echo "ACTUAL=$actual"
        exit 2
    fi
}

run_stage() {
    local label="$1"
    local script="$2"
    local rc=0

    echo
    echo "============================================================"
    echo "SINGLE_GAME_STAGE=$label"
    echo "SCRIPT=$script"
    echo "============================================================"

    set +e

    "$PY" -u "$script"
    rc=$?

    set -e

    if [ "$rc" -ne 0 ]; then
        echo "FAIL_CLOSED_STAGE=$label"
        echo "STAGE_RC=$rc"
        exit 2
    fi

    echo "STAGE_${label}=PASS"
}

echo "============================================================"
echo "WFS SINGLE-GAME DERIVED REFRESH"
echo "============================================================"
echo "STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

test -x "$PY" || {
    echo "FAIL_CLOSED_PYTHON_MISSING=$PY"
    exit 2
}

verify_sha "$L" "$EXPECTED_L" "L_R2"
verify_sha "$R" "$EXPECTED_R" "R1"
verify_sha "$X" "$EXPECTED_X" "X_R1"

run_stage "IDENTITY_L_R2" "$L"

test -s "$IDENTITY" || {
    echo "FAIL_CLOSED_IDENTITY_OUTPUT_MISSING=TRUE"
    exit 2
}

run_stage "BRIDGE_R1" "$R"

test -s "$BRIDGE" || {
    echo "FAIL_CLOSED_BRIDGE_OUTPUT_MISSING=TRUE"
    exit 2
}

run_stage "FINAL_X_R1" "$X"

test -s "$FINAL" || {
    echo "FAIL_CLOSED_FINAL_OUTPUT_MISSING=TRUE"
    exit 2
}

echo
echo "============================================================"
echo "GAV2 POST-BUILD VALIDATION"
echo "============================================================"

"$PY" - "$ROOT/data/nfl.db" "$FINAL" <<'PY_GAV2'
import sqlite3
import sys
from pathlib import Path

import pandas as pd


db = Path(sys.argv[1])
final = Path(sys.argv[2])


with sqlite3.connect(
    f"file:{db.resolve()}?mode=ro",
    uri=True,
) as conn:

    authority = pd.read_sql_query(
        """
        SELECT
            gsis_id,
            injury_gate
        FROM injury_consensus_current
        """,
        conn,
    )


authority["gsis_id"] = (
    authority["gsis_id"]
    .fillna("")
    .astype(str)
    .str.strip()
)

authority["injury_gate"] = (
    authority["injury_gate"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)


if authority.empty:
    raise SystemExit(
        "FAIL_CLOSED_GAV2_AUTHORITY_EMPTY"
    )

if authority["gsis_id"].eq("").any():
    raise SystemExit(
        "FAIL_CLOSED_GAV2_BLANK_GSIS_ID"
    )

if authority["gsis_id"].duplicated().any():
    raise SystemExit(
        "FAIL_CLOSED_GAV2_DUPLICATE_GSIS_ID"
    )


invalid = sorted(
    set(authority["injury_gate"])
    - {
        "ALLOW",
        "BLOCK",
    }
)

if invalid:
    raise SystemExit(
        "FAIL_CLOSED_GAV2_INVALID_GATE="
        + ",".join(invalid)
    )


block_ids = set(
    authority.loc[
        authority["injury_gate"].eq("BLOCK"),
        "gsis_id",
    ]
)


df = pd.read_parquet(
    final
)


required = {
    "player_id",
    "projection",
    "mvp_projection",
}

missing = sorted(
    required
    - set(df.columns)
)

if missing:
    raise SystemExit(
        "FAIL_CLOSED_SHOWDOWN_GAV2_FIELDS="
        + ",".join(missing)
    )


ids = (
    df["player_id"]
    .fillna("")
    .astype(str)
    .str.strip()
)

target = ids.isin(
    block_ids
)


projection = pd.to_numeric(
    df.loc[
        target,
        "projection",
    ],
    errors="coerce",
).fillna(0.0)

mvp = pd.to_numeric(
    df.loc[
        target,
        "mvp_projection",
    ],
    errors="coerce",
).fillna(0.0)


projection_nonzero = int(
    projection
    .abs()
    .gt(1e-12)
    .sum()
)

mvp_nonzero = int(
    mvp
    .abs()
    .gt(1e-12)
    .sum()
)


print(
    f"GAV2_AUTHORITY_ROWS={len(authority)}"
)

print(
    f"GAV2_BLOCK_IDS={len(block_ids)}"
)

print(
    f"SHOWDOWN_GAV2_BLOCK_ROWS={int(target.sum())}"
)

print(
    "SHOWDOWN_GAV2_BLOCK_PROJECTION_NONZERO="
    f"{projection_nonzero}"
)

print(
    "SHOWDOWN_GAV2_BLOCK_MVP_NONZERO="
    f"{mvp_nonzero}"
)


if projection_nonzero:
    raise SystemExit(
        "FAIL_CLOSED_SHOWDOWN_GAV2_PROJECTION_NONZERO"
    )

if mvp_nonzero:
    raise SystemExit(
        "FAIL_CLOSED_SHOWDOWN_GAV2_MVP_NONZERO"
    )


print(
    "SHOWDOWN_GAV2_POST_BUILD_VALIDATION=PASS"
)
PY_GAV2

echo
echo "FINAL_POOL=$FINAL"
echo "FINAL_POOL_SHA256=$(sha256sum "$FINAL" | awk '{print $1}')"
echo "FINAL_POOL_MTIME=$(stat -c '%y' "$FINAL")"
echo "FINISHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SHOWDOWN_GAV2_REGENERATION_PERSISTENCE=ACTIVE"
echo "SINGLE_GAME_DERIVED_REFRESH=PASS"
