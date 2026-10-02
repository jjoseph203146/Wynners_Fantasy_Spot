#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PYTHON = ROOT / "venv/bin/python"
BUILDER = ROOT / "research/build_matchup_intelligence_current_shadow_v1.py"

MATRIX = ROOT / "data/parquet/nfl_current_offensive_model_matrix.parquet"
SHADOW = ROOT / "data/research/matchup_intelligence_current_shadow_v1.parquet"
MANIFEST = ROOT / "data/research/matchup_intelligence_current_shadow_v1_manifest.json"

KEYS = [
    "game_id",
    "player_id",
    "team",
    "opponent_team",
    "position",
]

APPLICABLE = {"RB", "WR", "TE"}

CONTRACT = "WFS_MATCHUP_INTELLIGENCE_CURRENT_SHADOW_V1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value, reason):
    if not value:
        raise RuntimeError("MI_CURRENT_SHADOW_REFRESH_FAIL_CLOSED:" + reason)


def main():
    require(MATRIX.is_file(), "MATRIX_MISSING")
    require(BUILDER.is_file(), "BUILDER_MISSING")

    matrix = pd.read_parquet(MATRIX)

    require(not matrix.empty, "MATRIX_EMPTY")
    require(set(KEYS).issubset(matrix.columns), "MATRIX_IDENTITY_COLUMNS")
    require(not matrix[KEYS].isna().any().any(), "MATRIX_NULL_IDENTITY")
    require(
        not matrix.duplicated(["game_id", "player_id"]).any(),
        "MATRIX_DUPLICATE_IDENTITY",
    )

    seasons = sorted(
        matrix["season"].dropna().astype(int).unique().tolist()
    )
    weeks = sorted(
        matrix["week"].dropna().astype(int).unique().tolist()
    )

    require(len(seasons) == 1, "MATRIX_SEASON_BINDING")
    require(len(weeks) == 1, "MATRIX_WEEK_BINDING")

    # Builder writes the canonical shadow. It is deterministic and already
    # validates its own historical/evidence contracts.
    result = subprocess.run(
        [str(PYTHON), str(BUILDER)],
        cwd=str(ROOT),
        check=False,
    )

    require(result.returncode == 0, "BUILDER_FAILED")
    require(SHADOW.is_file(), "SHADOW_MISSING_AFTER_BUILD")

    shadow = pd.read_parquet(SHADOW)

    require(not shadow.empty, "SHADOW_EMPTY")
    require(set(KEYS).issubset(shadow.columns), "SHADOW_IDENTITY_COLUMNS")
    require(not shadow[KEYS].isna().any().any(), "SHADOW_NULL_IDENTITY")
    require(
        not shadow.duplicated(["game_id", "player_id"]).any(),
        "SHADOW_DUPLICATE_IDENTITY",
    )

    require(
        len(shadow) == len(matrix),
        f"ROW_COUNT:{len(shadow)}:{len(matrix)}",
    )

    # Exact one-way identity coverage. Evidence must cover every current
    # matrix identity; it may never create or substitute forecast identities.
    joined = matrix[KEYS].merge(
        shadow[KEYS],
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )

    require(
        len(joined) == len(matrix),
        "IDENTITY_ROW_COUNT_CHANGED",
    )

    require(
        joined["_merge"].eq("both").all(),
        "CURRENT_MATRIX_NOT_EXACTLY_COVERED",
    )

    applicable = joined["position"].isin(APPLICABLE)

    require(
        joined.loc[applicable, "_merge"].eq("both").all(),
        "APPLICABLE_IDENTITY_NOT_EXACTLY_COVERED",
    )

    shadow_sha = sha256(SHADOW)
    matrix_sha = sha256(MATRIX)

    payload = {
        "contract": CONTRACT,
        "status": "CURRENT_SHADOW_VALIDATED",
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": seasons[0],
        "week": weeks[0],
        "matrix_path": str(MATRIX),
        "matrix_sha256": matrix_sha,
        "shadow_path": str(SHADOW),
        "shadow_sha256": shadow_sha,
        "rows": int(len(shadow)),
        "matrix_rows": int(len(matrix)),
        "applicable_rows": int(applicable.sum()),
        "exact_identity_rows": int(joined["_merge"].eq("both").sum()),
        "exact_applicable_identity_rows": int(
            joined.loc[applicable, "_merge"].eq("both").sum()
        ),
        "identity_keys": KEYS,
        "identity_policy": "EXACT_CURRENT_MATRIX_COVERAGE",
        "historical_policy": "BUILDER_PRIOR_MASK_STRICTLY_BEFORE_TARGET_WEEK",
    }

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=MANIFEST.name + ".",
        suffix=".tmp",
        dir=MANIFEST.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())

    try:
        os.replace(tmp_path, MANIFEST)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    verify = json.loads(MANIFEST.read_text())

    require(
        verify.get("contract") == CONTRACT,
        "MANIFEST_CONTRACT",
    )
    require(
        verify.get("shadow_sha256") == sha256(SHADOW),
        "MANIFEST_SHADOW_HASH",
    )
    require(
        verify.get("matrix_sha256") == sha256(MATRIX),
        "MANIFEST_MATRIX_HASH",
    )

    print("MI_CURRENT_SHADOW_REFRESH=PASS")
    print(f"TARGET={seasons[0]}_WEEK_{weeks[0]}")
    print(f"MATRIX_ROWS={len(matrix)}")
    print(f"SHADOW_ROWS={len(shadow)}")
    print(f"EXACT_IDENTITY_ROWS={int(joined['_merge'].eq('both').sum())}")
    print(f"APPLICABLE_ROWS={int(applicable.sum())}")
    print(
        "EXACT_APPLICABLE_IDENTITY_ROWS="
        f"{int(joined.loc[applicable, '_merge'].eq('both').sum())}"
    )
    print(f"MATRIX_SHA256={matrix_sha}")
    print(f"SHADOW_SHA256={shadow_sha}")
    print(f"MANIFEST={MANIFEST}")


if __name__ == "__main__":
    main()
