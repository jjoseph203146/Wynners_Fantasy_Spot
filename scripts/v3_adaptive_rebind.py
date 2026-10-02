"""Immutable adaptive-input equivalence proof; never fit models or alter forecasts."""

from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUILDER = (
    ROOT
    / "scripts/build_offensive_reconciliation_current_adaptive_team_v1.py"
)
NFL_DB = ROOT / "data/nfl.db"

LEGACY_SHA = (
    "bef80cc4a0f2dcc36bfa90707a91a6adc058dac5582a76378e0c5249bca78c93"
)

CONTRACT = "WFS_V3_ADAPTIVE_INPUT_EQUIVALENCE_REBIND_V3"


def require(value, message):
    if not value:
        raise RuntimeError("FAIL_CLOSED: adaptive rebind " + message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def legacy_validator():
    import importlib.util

    path = (
        ROOT
        / "scripts/preflight_offensive_reconciliation_v3_legacy_v1.py"
    )

    require(
        digest(path.read_bytes()) == LEGACY_SHA,
        "historical validator hash",
    )

    spec = importlib.util.spec_from_file_location(
        "_v3_original_validator",
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def consumed_contract():
    """Extract and execute only the builder's actual projection, never main/fit."""

    source = BUILDER.read_bytes()
    tree = ast.parse(source)

    features = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "FEATURES"
            for t in n.targets
        )
    )

    main = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )

    projection = [
        copy.deepcopy(n)
        for n in main.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id in {"key_columns", "current_features"}
            for t in n.targets
        )
    ]

    require(
        len(projection) == 3,
        "builder projection structure changed",
    )

    keys = ast.literal_eval(projection[0].value)

    contract = {
        "builder_sha256": digest(source),
        "identity_columns": keys,
        "feature_columns": features,
        "projection_ast": ast.dump(
            ast.Module(body=projection, type_ignores=[]),
            include_attributes=False,
        ),
        "numeric_comparison": "EXACT_FLOAT64_NO_TOLERANCE",
        "null_comparison": "EXACT_MASK",
        "row_comparison": (
            "EXACT_SORTED_ACTIVE_TEAM_KEYS_WITH_AUTHORITATIVE_"
            "COMPLETED_GAME_RETIREMENT"
        ),
        "effective_input_hash_encoding": (
            "SORTED_COLUMNS_AND_ROWS_JSON_FLOAT64_HEX_NULL_V1"
        ),
        "completed_game_authority": "data/nfl.db:games.completed",
        "unconsumed_player_columns": [
            "player_id",
            "position",
            "active_flag",
            "injury_flag",
        ],
    }

    contract_hash = digest(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )

    code = compile(
        ast.fix_missing_locations(
            ast.Module(body=projection, type_ignores=[])
        ),
        str(BUILDER),
        "exec",
    )

    return contract, contract_hash, code


def project(data, season, week, contract, code):
    frame = pd.read_parquet(io.BytesIO(data))

    require(
        not frame.empty
        and frame.season.eq(season).all()
        and frame.week.eq(week).all(),
        "matrix target",
    )

    require(
        not frame[["game_id", "player_id"]].isna().any().any()
        and not frame.duplicated(["game_id", "player_id"]).any(),
        "matrix player identity",
    )

    ns = {
        "current": frame,
        "FEATURES": contract["feature_columns"],
        "TARGET_SEASON": season,
        "TARGET_WEEK": week,
    }

    exec(code, {"__builtins__": {}}, ns)

    projected = ns["current_features"].copy()
    keys = contract["identity_columns"]

    require(
        list(projected.columns)
        == keys + contract["feature_columns"],
        "consumed columns/order",
    )

    require(
        not projected[keys].isna().any().any()
        and not projected.duplicated(["game_id", "team"]).any(),
        "team identity",
    )

    for col in ["game_id", "team", "opponent_team"]:
        require(
            projected[col].map(
                lambda v: isinstance(v, str) and bool(v.strip())
            ).all(),
            "string identity",
        )

    for col in ["season", "week"]:
        values = pd.to_numeric(
            projected[col],
            errors="raise",
        )

        require(
            np.isfinite(values).all()
            and values.eq(values.astype("int64")).all(),
            "integer identity",
        )

        projected[col] = values.astype("int64")

    for col in contract["feature_columns"]:
        projected[col] = pd.to_numeric(
            projected[col],
            errors="raise",
        ).astype("float64")

        require(
            not np.isinf(projected[col]).any(),
            "infinite feature",
        )

    return projected.sort_values(keys).reset_index(drop=True)


def effective_input_hash(frame, contract):
    """Lossless float encoding with explicit nulls and ordered identities."""

    keys = contract["identity_columns"]
    rows = []

    for values in frame.itertuples(index=False, name=None):
        identity = list(values[: len(keys)])

        features = [
            None if pd.isna(v) else float(v).hex()
            for v in values[len(keys) :]
        ]

        rows.append(identity + features)

    return digest(
        json.dumps(
            {
                "columns": list(frame.columns),
                "rows": rows,
            },
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    )


def historical_bundle(files):
    legacy = legacy_validator()
    meta = json.loads(files["original/validation.json"])

    require(
        meta["contract_version"] == legacy.CONTRACT
        and meta["schema_version"] == legacy.SCHEMA,
        "historical contract",
    )

    require(
        meta["validator_sha256"] == LEGACY_SHA
        and meta["status"] == "PASS_HARD_CONTRACTS",
        "historical validation",
    )

    require(
        meta["candidate_path"] == "candidate.csv"
        and set(meta["inputs"])
        == set(legacy.INPUTS) | {"context.json"},
        "historical inputs",
    )

    blobs = {
        name: files["original/" + name]
        for name in meta["inputs"]
    }

    for name, data in blobs.items():
        require(
            digest(data) == meta["inputs"][name],
            "historical hash:" + name,
        )

    require(
        digest(blobs["candidate.csv"])
        == meta["candidate_sha256"],
        "historical candidate",
    )

    _, _, facts = legacy.validate_candidate(
        blobs,
        meta["season"],
        meta["week"],
    )

    require(
        all(meta.get(k) == v for k, v in facts.items()),
        "historical validation facts",
    )

    return meta, blobs


def authoritative_completed_games(season, week):
    """
    Return exact game IDs authoritatively marked completed for target week.

    Fail closed if the local authoritative schedule database or required
    schema is unavailable.
    """

    require(NFL_DB.exists(), "completed-game authority missing")

    connection = sqlite3.connect(
        f"file:{NFL_DB}?mode=ro",
        uri=True,
    )

    try:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(games)"
            )
        }

        require(
            {
                "game_id",
                "season",
                "week",
                "away_team",
                "home_team",
                "completed",
            }.issubset(columns),
            "completed-game authority schema",
        )

        rows = connection.execute(
            """
            SELECT
                game_id,
                away_team,
                home_team,
                completed
            FROM games
            WHERE season = ?
              AND week = ?
            """,
            (int(season), int(week)),
        ).fetchall()

    finally:
        connection.close()

    completed = {}

    for game_id, away_team, home_team, flag in rows:
        require(
            isinstance(game_id, str) and bool(game_id.strip()),
            "completed-game game_id",
        )

        require(
            isinstance(away_team, str) and bool(away_team.strip())
            and isinstance(home_team, str)
            and bool(home_team.strip()),
            "completed-game team identity",
        )

        require(
            flag in (0, 1),
            "completed-game flag",
        )

        if int(flag) == 1:
            completed[game_id] = {
                "game_id": game_id,
                "away_team": away_team,
                "home_team": home_team,
            }

    return completed


def reconcile_identity_lifecycle(before, after, keys, season, week):
    """
    Permit historical identities to retire only when the exact game is
    authoritatively completed. New identities remain prohibited.
    """

    before_index = pd.MultiIndex.from_frame(before[keys])
    after_index = pd.MultiIndex.from_frame(after[keys])

    before_set = set(before_index.tolist())
    after_set = set(after_index.tolist())

    added = sorted(after_set - before_set)
    removed = sorted(before_set - after_set)

    require(
        not added,
        "new consumed identity appeared",
    )

    if not removed:
        return (
            before.reset_index(drop=True),
            after.reset_index(drop=True),
            [],
        )

    completed = authoritative_completed_games(
        season,
        week,
    )

    key_pos = {name: i for i, name in enumerate(keys)}

    require(
        {
            "game_id",
            "season",
            "week",
            "team",
            "opponent_team",
        }.issubset(key_pos),
        "identity lifecycle key schema",
    )

    removed_by_game = {}

    for identity in removed:
        game_id = identity[key_pos["game_id"]]
        row_season = identity[key_pos["season"]]
        row_week = identity[key_pos["week"]]
        team = identity[key_pos["team"]]
        opponent = identity[key_pos["opponent_team"]]

        require(
            int(row_season) == int(season)
            and int(row_week) == int(week),
            "retired identity target",
        )

        require(
            game_id in completed,
            "unexplained consumed identity removal",
        )

        authority = completed[game_id]

        expected = {
            (authority["away_team"], authority["home_team"]),
            (authority["home_team"], authority["away_team"]),
        }

        require(
            (team, opponent) in expected,
            "retired identity team mismatch",
        )

        removed_by_game.setdefault(
            game_id,
            set(),
        ).add((team, opponent))

    exclusions = []

    for game_id, pairs in sorted(removed_by_game.items()):
        authority = completed[game_id]

        expected = {
            (authority["away_team"], authority["home_team"]),
            (authority["home_team"], authority["away_team"]),
        }

        require(
            pairs == expected,
            "partial completed-game identity removal",
        )

        exclusions.append(
            {
                "game_id": game_id,
                "season": int(season),
                "week": int(week),
                "away_team": authority["away_team"],
                "home_team": authority["home_team"],
                "completed": 1,
                "removed_team_identity_count": 2,
            }
        )

    survivor_mask = before_index.isin(after_set)

    before_survivors = (
        before.loc[survivor_mask]
        .sort_values(keys)
        .reset_index(drop=True)
    )

    after_survivors = (
        after.sort_values(keys)
        .reset_index(drop=True)
    )

    require(
        before_survivors[keys].equals(
            after_survivors[keys]
        ),
        "surviving consumed identity changed",
    )

    return before_survivors, after_survivors, exclusions


def equivalence(files):
    meta, old = historical_bundle(files)

    contract, contract_hash, code = consumed_contract()

    historical = project(
        old["matrix.parquet"],
        meta["season"],
        meta["week"],
        contract,
        code,
    )

    current = project(
        files["current_matrix.parquet"],
        meta["season"],
        meta["week"],
        contract,
        code,
    )

    keys = contract["identity_columns"]
    features = contract["feature_columns"]

    before, after, exclusions = reconcile_identity_lifecycle(
        historical,
        current,
        keys,
        meta["season"],
        meta["week"],
    )

    require(
        before[features].isna().equals(
            after[features].isna()
        ),
        "null semantics changed",
    )

    require(
        before[features].equals(
            after[features]
        ),
        "consumed numerical feature changed",
    )

    old_hash = effective_input_hash(
        before,
        contract,
    )

    current_hash = effective_input_hash(
        after,
        contract,
    )

    require(
        old_hash == current_hash,
        "effective input digest changed",
    )

    if len(before):
        difference = (
            before[features] - after[features]
        ).abs().max().max()
    else:
        difference = 0.0

    return {
        "contract": CONTRACT,
        "original_validated_matrix_hash": digest(
            old["matrix.parquet"]
        ),
        "current_matrix_hash": digest(
            files["current_matrix.parquet"]
        ),
        "original_validation_hash": digest(
            files["original/validation.json"]
        ),
        "original_adaptive_hash": digest(
            old["adaptive.csv"]
        ),
        "original_adaptive_output_hash": digest(
            old["adaptive.csv"]
        ),
        "original_adaptive_audit_hash": digest(
            old["adaptive_audit.json"]
        ),
        "adaptive_identity_columns": keys,
        "adaptive_consumed_feature_columns": features,
        "adaptive_consumed_contract_hash": contract_hash,
        "old_effective_input_hash": old_hash,
        "current_effective_input_hash": current_hash,
        "effective_input_rows": len(before),
        "effective_input_columns": len(before.columns),
        "historical_effective_input_rows": len(historical),
        "current_effective_input_rows": len(current),
        "completed_game_exclusion_count": len(exclusions),
        "completed_game_exclusions": exclusions,
        "equivalence_method": (
            "RECONSTRUCT_CURRENT_BUILDER_PROJECTION_"
            "EXACT_ACTIVE_IDENTITIES_FLOAT64_VALUES_AND_NULL_MASK_"
            "WITH_AUTHORITATIVE_COMPLETED_GAME_RETIREMENT"
        ),
        "equivalence_scope": (
            "VALIDATED_ADAPTIVE_MODEL_INPUT_ONLY_"
            "NOT_FULL_MATRIX_OR_AVAILABILITY"
        ),
        "adaptive_consumed_contract": contract,
        "season": meta["season"],
        "week": meta["week"],
        "old_rows": len(historical),
        "current_rows": len(current),
        "compared_rows": len(before),
        "identity_match": True,
        "consumed_column_match": True,
        "max_numeric_diff": float(difference),
        "semantic_equivalence_result": "PASS",
    }


def validate_rebind(
    archive,
    matrix,
    adaptive,
    audit,
    season,
    week,
):
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        require(
            len(z.namelist())
            == len(set(z.namelist())),
            "duplicate archive entries",
        )

        files = {
            name: z.read(name)
            for name in z.namelist()
        }

    legacy = legacy_validator()

    required = {
        "proof.json",
        "current_matrix.parquet",
        "original/validation.json",
    } | {
        "original/" + name
        for name in set(legacy.INPUTS) | {"context.json"}
    }

    require(
        set(files) == required,
        "archive inputs",
    )

    proof = json.loads(files["proof.json"])
    facts = equivalence(files)

    require(
        all(proof.get(k) == v for k, v in facts.items()),
        "proof facts",
    )

    require(
        bool(proof.get("rebind_reason"))
        and bool(proof.get("rebind_timestamp")),
        "proof metadata",
    )

    datetime.fromisoformat(
        proof["rebind_timestamp"]
    )

    require(
        proof.get("created_at")
        == proof["rebind_timestamp"],
        "creation timestamp",
    )

    require(
        datetime.fromisoformat(
            proof["created_at"]
        ).utcoffset()
        is not None,
        "creation timestamp timezone",
    )

    require(
        (facts["season"], facts["week"])
        == (season, week),
        "current target",
    )

    require(
        digest(matrix)
        == facts["current_matrix_hash"],
        "current matrix hash",
    )

    require(
        digest(adaptive)
        == facts["original_adaptive_hash"],
        "adaptive bytes changed",
    )

    require(
        digest(audit)
        == facts["original_adaptive_audit_hash"],
        "historical audit changed",
    )

    return facts


def create_rebind(
    original_validation,
    current_matrix,
    output,
):
    original_validation, current_matrix, output = map(
        Path,
        (
            original_validation,
            current_matrix,
            output,
        ),
    )

    require(
        not output.exists(),
        "output already exists",
    )

    legacy = legacy_validator()

    _, _, meta = legacy.load_validated_candidate(
        original_validation
    )

    files = {
        "original/validation.json":
            original_validation.read_bytes(),
        "current_matrix.parquet":
            current_matrix.read_bytes(),
    }

    files.update(
        {
            "original/" + name:
                (
                    original_validation.parent / name
                ).read_bytes()
            for name in meta["inputs"]
        }
    )

    facts = equivalence(files)

    reason = (
        "EXACT_ADAPTIVE_CONSUMED_INPUT_EQUIVALENCE_"
        "WITH_AUTHORITATIVE_COMPLETED_GAME_RETIREMENT"
        if facts["completed_game_exclusion_count"]
        else
        "EXACT_ADAPTIVE_CONSUMED_INPUT_EQUIVALENCE_"
        "WITH_DISTINCT_MATRIX_HASHES"
    )

    proof = dict(
        facts,
        rebind_reason=reason,
        rebind_timestamp=datetime.now(
            timezone.utc
        ).isoformat(),
    )

    proof["created_at"] = proof["rebind_timestamp"]

    files["proof.json"] = (
        json.dumps(
            proof,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as z:
        for name, data in sorted(files.items()):
            z.writestr(name, data)

    archive = buffer.getvalue()

    validate_rebind(
        archive,
        files["current_matrix.parquet"],
        files["original/adaptive.csv"],
        files["original/adaptive_audit.json"],
        meta["season"],
        meta["week"],
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("xb") as handle:
        handle.write(archive)

    output.chmod(0o444)

    return proof
