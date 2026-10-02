from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(
    "/home/mwynn/nfl_data_engine"
)

FROZEN_V1 = (
    ROOT
    / "backups"
    / "forecast_live_core_v1_predictions_frozen"
    / "run_forecast_live_core_v1.py"
)

ACTIVE_LIVE_CORE = (
    ROOT
    / "processed"
    / "forecast_live_core_v1.csv"
)

ACTIVE_LIVE_CORE_AUDIT = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_audit.json"
)

EXPECTED_V1_RUNNER_SHA = (
    "e549bde1a420d9d0a013c8f0a6de144a"
    "5bdc59d25dd9dc8abe8184d007ccaa3c"
)

EXPECTED_BUILDER_SHA = (
    "6ab248295dcab64108397c7a074d5210e"
    "3f252827f1e23adf5a069607455df02"
)

EXPECTED_MANIFEST_SHA = (
    "417807fcf1fa5fa435a6336dea908ed45"
    "b3a87114d30a5f04ed7ceb950392aae"
)

EXPECTED_TRAINING_SHA = (
    "76aa9a35ea10d7a04518e709a410a5aa"
    "e7ca84dbadc5b7d0462f3e2d47ed75e1"
)

BUILDER = (
    ROOT
    / "scripts"
    / "build_forecast_live_core_v1.py"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def fail(message: str) -> None:
    raise SystemExit(
        f"FAIL | {message}"
    )


def validate_live_core_contract() -> str:
    required = (
        FROZEN_V1,
        ACTIVE_LIVE_CORE,
        ACTIVE_LIVE_CORE_AUDIT,
        BUILDER,
    )

    for path in required:
        if not path.is_file():
            fail(
                f"missing required file: {path}"
            )

    v1_sha = sha256(
        FROZEN_V1
    )

    if v1_sha != EXPECTED_V1_RUNNER_SHA:
        fail(
            "frozen V1 runner hash mismatch"
        )

    builder_sha = sha256(
        BUILDER
    )

    if builder_sha != EXPECTED_BUILDER_SHA:
        fail(
            "Live CORE builder hash mismatch"
        )

    try:
        audit = json.loads(
            ACTIVE_LIVE_CORE_AUDIT.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        fail(
            f"unable to read builder audit: {exc}"
        )

    if audit.get("status") != "PASS":
        fail(
            "builder audit status is not PASS"
        )

    if (
        audit.get("architecture")
        != "live_core_v1_frozen_exp004_semantics"
    ):
        fail(
            "builder architecture mismatch"
        )

    if (
        audit.get("builder")
        != "build_forecast_live_core_v1.py"
    ):
        fail(
            "builder identity mismatch"
        )

    if audit.get("live_season") != 2026:
        fail(
            "builder live season mismatch"
        )

    contract = audit.get(
        "frozen_contract",
        {},
    )

    expected_contract = {
        "coach_feature_count": 8,
        "core_feature_count": 76,
        "direct_feature_count": 6,
        "rolling_feature_count": 62,
        "semantic_equivalence_status":
            "76_OF_76_PROVEN",
    }

    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            fail(
                f"builder contract mismatch: {key}"
            )

    integrity = audit.get(
        "input_integrity",
        {},
    )

    if (
        integrity.get("manifest_sha256")
        != EXPECTED_MANIFEST_SHA
    ):
        fail(
            "builder manifest lineage mismatch"
        )

    if (
        integrity.get("training_sha256")
        != EXPECTED_TRAINING_SHA
    ):
        fail(
            "builder training lineage mismatch"
        )

    guards = audit.get(
        "guards",
        {},
    )

    required_guards = {
        "database_write": False,
        "impact_enabled": False,
        "replacement_enabled": False,
        "model_executed": False,
        "optimizer_modified": False,
        "temp_wind_zero_filled": False,
        "ui_modified": False,
    }

    for key, expected in required_guards.items():
        if guards.get(key) is not expected:
            fail(
                f"builder guard mismatch: {key}"
            )

    if (
        guards.get("injury_source_status")
        != "PENDING"
    ):
        fail(
            "injury source status mismatch"
        )

    output = audit.get(
        "output",
        {},
    )

    if output.get("team_row_count") != 544:
        fail(
            "builder team-row count mismatch"
        )

    if (
        output.get("expected_team_row_count")
        != 544
    ):
        fail(
            "builder expected-team-row mismatch"
        )

    if output.get("game_count") != 272:
        fail(
            "builder game count mismatch"
        )

    if (
        output.get("duplicate_game_team_keys")
        != 0
    ):
        fail(
            "duplicate game/team keys detected"
        )

    if (
        output.get("infinite_core_values")
        != 0
    ):
        fail(
            "infinite CORE values detected"
        )

    live_sha = sha256(
        ACTIVE_LIVE_CORE
    )

    if (
        output.get("csv_sha256")
        != live_sha
    ):
        fail(
            "Live CORE CSV does not match "
            "builder audit SHA"
        )

    market_ready_games = output.get(
        "market_ready_games"
    )

    market_ready_team_rows = output.get(
        "market_ready_team_rows"
    )

    if (
        not isinstance(
            market_ready_games,
            int,
        )
        or market_ready_games < 0
        or market_ready_games > 272
    ):
        fail(
            "invalid market-ready game count"
        )

    if (
        market_ready_team_rows
        != market_ready_games * 2
    ):
        fail(
            "market-ready team-row count mismatch"
        )

    return live_sha


def run_frozen_engine(
    live_sha: str,
) -> None:
    source = FROZEN_V1.read_text(
        encoding="utf-8"
    )

    code = compile(
        source,
        str(FROZEN_V1),
        "exec",
    )

    namespace = {
        "__name__": "__wfs_frozen_v1__",
        "__file__": str(FROZEN_V1),
    }

    exec(
        code,
        namespace,
    )

    if "main" not in namespace:
        fail(
            "frozen V1 main() not found"
        )

    # ---------------------------------------------------------------
    # Only live-input bindings change.
    #
    # Frozen model lineage, model implementation, feature contract,
    # reconstruction proof and prediction logic remain untouched.
    # ---------------------------------------------------------------

    namespace["LIVE_CORE_DIR"] = (
        ACTIVE_LIVE_CORE.parent
    )

    namespace["LIVE_CORE"] = (
        ACTIVE_LIVE_CORE
    )

    namespace["EXPECTED_LIVE_CORE"] = (
        live_sha
    )

    print()
    print(
        "PASS | frozen V1 runner hash"
    )

    print(
        "PASS | frozen builder hash"
    )

    print(
        "PASS | builder audit contract"
    )

    print(
        "PASS | active Live CORE SHA bound dynamically"
    )

    print(
        f"LIVE CORE | {ACTIVE_LIVE_CORE}"
    )

    print(
        f"LIVE SHA  | {live_sha}"
    )

    print()
    print(
        "EXECUTE | exact frozen inference engine"
    )

    namespace["main"]()


def main() -> None:
    print(
        "=" * 78
    )

    print(
        "WFS FORECAST CENTER — "
        "LIVE CORE INFERENCE V2"
    )

    print(
        "=" * 78
    )

    print(
        "MODE | REFRESH-CAPABLE WRAPPER "
        "AROUND FROZEN V1 ENGINE"
    )

    live_sha = (
        validate_live_core_contract()
    )

    run_frozen_engine(
        live_sha
    )


if __name__ == "__main__":
    main()
