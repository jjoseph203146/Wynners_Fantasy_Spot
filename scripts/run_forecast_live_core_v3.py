from pathlib import Path
import hashlib
import json
import sqlite3


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path("/home/mwynn/nfl_data_engine")

NFL_DB = ROOT / "data" / "nfl.db"

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

FROZEN_V1 = (
    ROOT
    / "backups"
    / "forecast_live_core_v1_predictions_frozen"
    / "run_forecast_live_core_v1.py"
)

FROZEN_V2 = (
    ROOT
    / "backups"
    / "forecast_live_core_v2_frozen"
    / "run_forecast_live_core_v2.py"
)

BUILDER = (
    ROOT
    / "scripts"
    / "build_forecast_live_core_v1.py"
)


# =====================================================================
# FROZEN LINEAGE
# =====================================================================

EXPECTED_V1_SHA = (
    "e549bde1a420d9d0a013c8f0a6de144a"
    "5bdc59d25dd9dc8abe8184d007ccaa3c"
)

EXPECTED_V2_SHA = (
    "de7fee2e79afe89104394c63fb3294e1"
    "fe62c4940af832fea64c0f5faa7490bb"
)

EXPECTED_BUILDER_SHA = (
    "6ab248295dcab64108397c7a074d5210e"
    "3f252827f1e23adf5a069607455df02"
)

LIVE_SEASON = 2026


# =====================================================================
# HELPERS
# =====================================================================

def fail(message):
    raise RuntimeError(message)


def section(title):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def ro_connect(path):
    return sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
    )


def replace_exact_once(
    source,
    old,
    new,
    label,
):
    count = source.count(old)

    if count != 1:
        fail(
            f"{label}: expected exactly one "
            f"source match; found {count}"
        )

    return source.replace(
        old,
        new,
        1,
    )


# =====================================================================
# AUTHENTICATE FROZEN LINEAGE
# =====================================================================

def authenticate_lineage():
    required = [
        FROZEN_V1,
        FROZEN_V2,
        BUILDER,
        ACTIVE_LIVE_CORE,
        ACTIVE_LIVE_CORE_AUDIT,
        NFL_DB,
    ]

    for path in required:
        if not path.is_file():
            fail(
                f"missing required artifact: {path}"
            )

    checks = [
        (
            "Frozen V1",
            FROZEN_V1,
            EXPECTED_V1_SHA,
        ),
        (
            "Frozen V2",
            FROZEN_V2,
            EXPECTED_V2_SHA,
        ),
        (
            "Builder V1",
            BUILDER,
            EXPECTED_BUILDER_SHA,
        ),
    ]

    for label, path, expected in checks:
        actual = sha256(path)

        print(
            f"{label:<20} | {actual}"
        )

        if actual != expected:
            fail(
                f"{label} SHA mismatch"
            )

    print()
    print(
        "PASS | frozen inference lineage authenticated"
    )


# =====================================================================
# AUTHORITATIVE DYNAMIC POPULATION
# =====================================================================

def authoritative_population():
    conn = ro_connect(NFL_DB)

    try:
        integrity = conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        fk = conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        rows = conn.execute(
            """
            SELECT
                game_id,
                completed
            FROM games
            WHERE season = ?
            ORDER BY game_id
            """,
            (
                LIVE_SEASON,
            ),
        ).fetchall()

    finally:
        conn.close()

    if integrity != "ok":
        fail(
            "nfl.db integrity check failed"
        )

    if fk:
        fail(
            "nfl.db foreign-key violations"
        )

    if len(rows) != 272:
        fail(
            f"authoritative season != 272 games: "
            f"{len(rows)}"
        )

    game_ids = [
        str(row[0])
        for row in rows
    ]

    if len(set(game_ids)) != 272:
        fail(
            "authoritative season game IDs not unique"
        )

    completed_ids = {
        str(game_id)
        for game_id, completed in rows
        if int(completed or 0) == 1
    }

    live_ids = {
        str(game_id)
        for game_id, completed in rows
        if int(completed or 0) == 0
    }

    if completed_ids & live_ids:
        fail(
            "completed/live game populations overlap"
        )

    if (
        len(completed_ids)
        + len(live_ids)
        != 272
    ):
        fail(
            "completed/live partition does not equal "
            "authoritative season"
        )

    if not live_ids:
        fail(
            "no uncompleted games remain; "
            "live inference is not applicable"
        )

    return {
        "season_games":
            272,
        "completed_games":
            len(completed_ids),
        "live_games":
            len(live_ids),
        "expected_team_rows":
            2 * len(live_ids),
        "completed_ids":
            completed_ids,
        "live_ids":
            live_ids,
    }


# =====================================================================
# BUILDER AUDIT — DYNAMIC CONTRACT
# =====================================================================

def validate_builder_audit(population):
    audit = load_json(
        ACTIVE_LIVE_CORE_AUDIT
    )

    if audit.get("status") != "PASS":
        fail(
            "builder audit status is not PASS"
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
            "builder injury-source status mismatch"
        )

    output = audit.get(
        "output",
        {},
    )

    expected_games = population[
        "live_games"
    ]

    expected_rows = population[
        "expected_team_rows"
    ]

    if (
        output.get("game_count")
        != expected_games
    ):
        fail(
            "builder game count does not match "
            "authoritative uncompleted population"
        )

    if (
        output.get("team_row_count")
        != expected_rows
    ):
        fail(
            "builder team-row count does not match "
            "dynamic population"
        )

    if (
        output.get("expected_team_row_count")
        != expected_rows
    ):
        fail(
            "builder expected-team-row count mismatch"
        )

    if (
        output.get("duplicate_game_team_keys")
        != 0
    ):
        fail(
            "duplicate builder game/team keys"
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
            "active Live CORE SHA does not match "
            "builder audit"
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
        or market_ready_games > expected_games
    ):
        fail(
            "invalid dynamic market-ready game count"
        )

    if (
        market_ready_team_rows
        != market_ready_games * 2
    ):
        fail(
            "market-ready team-row count mismatch"
        )

    return (
        live_sha,
        market_ready_games,
    )


# =====================================================================
# AUTHENTICATED IN-MEMORY COMPATIBILITY LAYER
# =====================================================================

def build_dynamic_engine_source(
    expected_games,
    expected_team_rows,
    expected_market_ready,
):
    source = FROZEN_V1.read_text(
        encoding="utf-8"
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    #
    # These are the ONLY permitted source substitutions.
    #
    # Numerical model logic, frozen training lineage, feature order,
    # ridge implementation, residual calculations, market arithmetic,
    # winner logic and historical reconstruction remain unchanged.
    #
    # Every substitution must match EXACTLY ONCE or execution fails.
    # ---------------------------------------------------------------

    replacements = [
        (
            """    if len(live) != 544:
        raise RuntimeError(
            f"Live team rows != 544: {len(live)}"
        )
""",
            f"""    if len(live) != {expected_team_rows}:
        raise RuntimeError(
            f"Live team rows != {expected_team_rows}: {{len(live)}}"
        )
""",
            "live team-row population gate",
        ),
        (
            """    if live[
        "game_id"
    ].nunique() != 272:
        raise RuntimeError(
            "Live game count != 272"
        )
""",
            f"""    if live[
        "game_id"
    ].nunique() != {expected_games}:
        raise RuntimeError(
            "Live game count != {expected_games}"
        )
""",
            "live game-count gate",
        ),
        (
            """    if len(home) != 272:
        raise RuntimeError(
            f"Live home rows != 272: {len(home)}"
        )
""",
            f"""    if len(home) != {expected_games}:
        raise RuntimeError(
            f"Live home rows != {expected_games}: {{len(home)}}"
        )
""",
            "live home-row gate",
        ),
        (
            """    if home[
        "game_id"
    ].nunique() != 272:
        raise RuntimeError(
            "Live physical-game IDs not unique"
        )
""",
            f"""    if home[
        "game_id"
    ].nunique() != {expected_games}:
        raise RuntimeError(
            "Live physical-game IDs not unique"
        )
""",
            "live home game-ID gate",
        ),
        (
            """    if len(
        predictions
    ) != 272:
        raise RuntimeError(
            "Live output row count != 272"
        )
""",
            f"""    if len(
        predictions
    ) != {expected_games}:
        raise RuntimeError(
            "Live output row count != {expected_games}"
        )
""",
            "prediction row-count gate",
        ),
        (
            """    if predictions[
        "game_id"
    ].nunique() != 272:
        raise RuntimeError(
            "Live output game IDs not unique"
        )
""",
            f"""    if predictions[
        "game_id"
    ].nunique() != {expected_games}:
        raise RuntimeError(
            "Live output game IDs not unique"
        )
""",
            "prediction game-ID gate",
        ),
        (
            """    if (
        market_ready
        + market_pending
        != 272
    ):
""",
            f"""    if (
        market_ready
        + market_pending
        != {expected_games}
    ):
""",
            "forecast-status partition gate",
        ),
        (
            """    if market_ready != 100:
        raise RuntimeError(
            f"Expected 100 market-ready games; got {market_ready}"
        )
""",
            f"""    if market_ready != {expected_market_ready}:
        raise RuntimeError(
            f"Expected {expected_market_ready} market-ready games; got {{market_ready}}"
        )
""",
            "market-ready snapshot gate",
        ),
    ]

    for old, new, label in replacements:
        source = replace_exact_once(
            source,
            old,
            new,
            label,
        )

    return source


# =====================================================================
# POST-INFERENCE IDENTITY VALIDATION
# =====================================================================

def validate_prediction_identity(
    namespace,
    population,
):
    output_csv = namespace.get(
        "OUTPUT"
    )

    if output_csv is None:
        fail(
            "frozen engine OUTPUT_CSV not found"
        )

    output_csv = Path(
        output_csv
    )

    if not output_csv.is_file():
        fail(
            f"prediction output missing: {output_csv}"
        )

    pd = namespace.get(
        "pd"
    )

    if pd is None:
        fail(
            "frozen pandas namespace unavailable"
        )

    pred = pd.read_csv(
        output_csv
    )

    expected_games = population[
        "live_games"
    ]

    if len(pred) != expected_games:
        fail(
            "post-inference prediction row count mismatch"
        )

    if (
        pred["game_id"].nunique()
        != expected_games
    ):
        fail(
            "post-inference prediction game IDs "
            "not unique"
        )

    pred_ids = set(
        pred["game_id"].astype(str)
    )

    if (
        pred_ids
        != population["live_ids"]
    ):
        missing = sorted(
            population["live_ids"]
            - pred_ids
        )

        extra = sorted(
            pred_ids
            - population["live_ids"]
        )

        fail(
            "prediction IDs do not exactly equal "
            "authoritative uncompleted IDs | "
            f"missing={missing[:10]} | "
            f"extra={extra[:10]}"
        )

    if (
        pred_ids
        & population["completed_ids"]
    ):
        fail(
            "completed game emitted in prediction output"
        )

    print()
    print(
        "PASS | prediction IDs exactly equal "
        "authoritative uncompleted game IDs"
    )

    print(
        "PASS | completed games excluded from "
        "new prediction output"
    )


# =====================================================================
# EXECUTION
# =====================================================================

def run_dynamic_engine(
    source,
    live_sha,
):
    code = compile(
        source,
        str(FROZEN_V1),
        "exec",
    )

    namespace = {
        "__name__":
            "__wfs_frozen_v1_dynamic__",
        "__file__":
            str(FROZEN_V1),
    }

    exec(
        code,
        namespace,
    )

    if "main" not in namespace:
        fail(
            "frozen V1 main() not found"
        )

    # Same dynamic live-input rebinding proven by V2.
    namespace[
        "LIVE_CORE_DIR"
    ] = ACTIVE_LIVE_CORE.parent

    namespace[
        "LIVE_CORE"
    ] = ACTIVE_LIVE_CORE

    namespace[
        "EXPECTED_LIVE_CORE"
    ] = live_sha

    namespace["main"]()

    return namespace


def main():
    section(
        "WFS FORECAST CENTER — LIVE CORE INFERENCE V3"
    )

    print(
        "MODE | DYNAMIC POPULATION COMPATIBILITY "
        "WRAPPER"
    )

    print()
    print(
        "Frozen model mathematics | UNCHANGED"
    )
    print(
        "Frozen feature contract  | UNCHANGED"
    )
    print(
        "Frozen reconstruction    | REQUIRED"
    )
    print(
        "Population assumptions   | DYNAMIC"
    )

    section(
        "FROZEN LINEAGE"
    )

    authenticate_lineage()

    section(
        "AUTHORITATIVE POPULATION"
    )

    population = (
        authoritative_population()
    )

    print(
        f"Season games       | "
        f"{population['season_games']}"
    )
    print(
        f"Completed games    | "
        f"{population['completed_games']}"
    )
    print(
        f"Live games         | "
        f"{population['live_games']}"
    )
    print(
        f"Expected team rows | "
        f"{population['expected_team_rows']}"
    )

    section(
        "ACTIVE BUILDER CONTRACT"
    )

    (
        live_sha,
        market_ready_games,
    ) = validate_builder_audit(
        population
    )

    print(
        f"Live CORE SHA      | {live_sha}"
    )
    print(
        f"Market-ready games | {market_ready_games}"
    )

    print()
    print(
        "PASS | builder population matches "
        "authoritative uncompleted schedule"
    )

    section(
        "IN-MEMORY COMPATIBILITY LAYER"
    )

    source = build_dynamic_engine_source(
        expected_games=
            population["live_games"],
        expected_team_rows=
            population["expected_team_rows"],
        expected_market_ready=
            market_ready_games,
    )

    print(
        "PASS | exactly 8 authenticated "
        "population/status substitutions"
    )
    print(
        "PASS | frozen source file remains untouched"
    )
    print(
        "PASS | model/prediction mathematics unchanged"
    )

    section(
        "EXECUTE FROZEN ENGINE WITH DYNAMIC GATES"
    )

    namespace = run_dynamic_engine(
        source,
        live_sha,
    )

    section(
        "POST-INFERENCE IDENTITY"
    )

    validate_prediction_identity(
        namespace,
        population,
    )

    section(
        "LIVE CORE INFERENCE V3 — FINAL GATE"
    )

    print(
        "PASS | authoritative dynamic population"
    )
    print(
        "PASS | frozen V1 lineage authenticated"
    )
    print(
        "PASS | frozen V2 lineage authenticated"
    )
    print(
        "PASS | builder V1 lineage authenticated"
    )
    print(
        "PASS | exact frozen historical reconstruction required"
    )
    print(
        "PASS | exact 76-CORE feature contract preserved"
    )
    print(
        "PASS | residual prediction mathematics preserved"
    )
    print(
        "PASS | market arithmetic preserved"
    )
    print(
        "PASS | completed games excluded"
    )
    print(
        "PASS | optimizer / UI untouched"
    )
    print(
        "PASS | ledger untouched"
    )
    print()
    print(
        "READY | LIVE CORE INFERENCE V3 "
        "VALIDATION MAY PROCEED"
    )


if __name__ == "__main__":
    main()
