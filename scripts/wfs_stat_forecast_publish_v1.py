#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

DEFAULT_DB = ROOT / "data/nfl.db"
DEFAULT_DATA_ROOT = ROOT / "data"

CURRENT_NAME = "current_unified_stat_forecasts.parquet"
CURRENT_MANIFEST_NAME = "current_unified_stat_forecasts_manifest.json"

SNAPSHOT_PARQUET_NAME = "stat_forecast_kickoff.parquet"
SNAPSHOT_MANIFEST_NAME = "stat_forecast_kickoff_manifest.json"

CONTRACT = "WFS_STAT_FORECAST_PUBLISH_V1"

AVAILABILITY_GATE_CONTRACT = (
    "WFS_STAT_FORECAST_AVAILABILITY_GATE_V1"
)

HARD_SUPPRESSION_STATUSES = {
    "OUT",
    "INACTIVE",
}

PRESERVE_NUMERIC_STATUSES = {
    "QUESTIONABLE",
    "DOUBTFUL",
}

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_now(value: str | None) -> datetime:
    if not value:
        return utc_now()

    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    dt = datetime.fromisoformat(text)

    if dt.tzinfo is None:
        raise RuntimeError(
            "--now-utc must include UTC offset or Z"
        )

    return dt.astimezone(UTC)


def kickoff_utc(
    game_date,
    gametime,
) -> datetime:

    date_text = str(game_date).strip()
    time_text = str(gametime).strip()

    if (
        not date_text
        or date_text.lower() in {"nan", "none"}
        or not time_text
        or time_text.lower() in {"nan", "none"}
    ):
        raise RuntimeError(
            f"Kickoff unavailable: "
            f"game_date={game_date!r} "
            f"gametime={gametime!r}"
        )

    date_part = pd.to_datetime(
        date_text,
        errors="raise",
    ).date()

    parsed_time = None

    for fmt in (
        "%H:%M",
        "%H:%M:%S",
        "%I:%M %p",
        "%I:%M%p",
    ):
        try:
            parsed_time = datetime.strptime(
                time_text,
                fmt,
            ).time()
            break
        except ValueError:
            pass

    if parsed_time is None:
        raise RuntimeError(
            f"Unsupported gametime format: {time_text!r}"
        )

    local_dt = datetime.combine(
        date_part,
        parsed_time,
        tzinfo=ET,
    )

    return local_dt.astimezone(UTC)


def atomic_json_write(
    obj: dict,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    os.close(fd)

    temp_path = Path(temp_name)

    try:
        temp_path.write_text(
            json.dumps(
                obj,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        os.replace(
            temp_path,
            path,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_parquet_write(
    df: pd.DataFrame,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp.parquet",
        dir=str(path.parent),
    )

    os.close(fd)

    temp_path = Path(temp_name)

    try:
        df.to_parquet(
            temp_path,
            index=False,
        )

        os.replace(
            temp_path,
            path,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()



def load_injury_availability_authority(
    db_path: Path,
) -> dict[str, dict]:
    """
    Read the structured injury consensus authority.

    Identity contract:
      forecast.player_id == injury_consensus_current.gsis_id

    No display-name matching.
    No fuzzy matching.
    No fallback identity.
    """
    conn = sqlite3.connect(
        f"file:{db_path.resolve()}?mode=ro",
        uri=True,
    )

    conn.row_factory = sqlite3.Row

    try:
        table = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name='injury_consensus_current'
            """
        ).fetchone()

        if table is None:
            raise RuntimeError(
                "injury_consensus_current missing"
            )

        rows = conn.execute(
            """
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
            """
        ).fetchall()

    finally:
        conn.close()

    authority = {}

    for row in rows:
        player_id = str(
            row["gsis_id"]
        ).strip()

        if not player_id:
            continue

        if player_id in authority:
            raise RuntimeError(
                "Duplicate injury authority gsis_id: "
                f"{player_id}"
            )

        status = str(
            row["consensus_status"]
            or
            ""
        ).strip().upper()

        authority[player_id] = {
            "gsis_id":
                player_id,

            "team":
                row["team"],

            "position":
                row["position"],

            "player_name":
                row["player_name"],

            "consensus_status":
                status,

            "injury_gate":
                row["injury_gate"],

            "injury_gate_reason":
                row["injury_gate_reason"],

            "authoritative_source":
                row["authoritative_source"],

            "last_updated":
                row["last_updated"],
        }

    return authority


def apply_availability_gate(
    source: pd.DataFrame,
    db_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply current playable-forecast availability authority.

    OUT / INACTIVE:
      all expected_* values -> 0.0

    QUESTIONABLE / DOUBTFUL:
      numerical forecast remains unchanged

    All other / unmatched:
      numerical forecast remains unchanged

    The upstream model candidate is never modified.
    """
    required = {
        "player_id",
        "game_id",
        "team",
        "position",
    }

    missing = sorted(
        required
        -
        set(
            source.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Availability gate source columns missing: "
            f"{missing}"
        )

    expected_columns = [
        column
        for column in source.columns
        if column.startswith(
            "expected_"
        )
    ]

    if not expected_columns:
        raise RuntimeError(
            "Availability gate found no expected_* columns"
        )

    authority = load_injury_availability_authority(
        db_path
    )

    gated = source.copy()

    statuses = (
        gated["player_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .map(
            lambda player_id:
                authority.get(
                    player_id,
                    {},
                ).get(
                    "consensus_status",
                    "",
                )
        )
    )

    # WFS_GLOBAL_PLAYER_AVAILABILITY_V2
    #
    # Hard availability authority is the consensus gate itself.
    # OUT / INACTIVE remain compatible status signals, while
    # roster/verified-secondary hard unavailability is represented
    # by injury_gate == BLOCK.
    gates = (
        gated["player_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .map(
            lambda player_id:
                str(
                    authority.get(
                        player_id,
                        {},
                    ).get(
                        "injury_gate",
                        "",
                    )
                    or ""
                ).strip().upper()
        )
    )

    hard_mask = (
        statuses.isin(
            HARD_SUPPRESSION_STATUSES
        )
        | gates.eq("BLOCK")
    )

    questionable_mask = statuses.eq(
        "QUESTIONABLE"
    )

    doubtful_mask = statuses.eq(
        "DOUBTFUL"
    )

    hard_targets = []

    for idx in gated.index[
        hard_mask
    ]:
        player_id = str(
            gated.at[
                idx,
                "player_id",
            ]
        ).strip()

        row_authority = authority.get(
            player_id,
            {},
        )

        hard_targets.append({
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
                row_authority.get(
                    "consensus_status"
                ),

            "injury_gate":
                row_authority.get(
                    "injury_gate"
                ),

            "injury_gate_reason":
                row_authority.get(
                    "injury_gate_reason"
                ),
        })

    for column in expected_columns:
        gated.loc[
            hard_mask,
            column,
        ] = 0.0

    audit = {
        "contract":
            AVAILABILITY_GATE_CONTRACT,

        "identity_join":
            "player_id==gsis_id",

        "display_name_matching_used":
            False,

        "fuzzy_matching_used":
            False,

        "hard_suppression_statuses":
            sorted(
                HARD_SUPPRESSION_STATUSES
            ),

        "preserve_numeric_statuses":
            sorted(
                PRESERVE_NUMERIC_STATUSES
            ),

        "expected_columns":
            expected_columns,

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

        "hard_targets":
            hard_targets,
    }

    return (
        gated,
        audit,
    )


def load_schedule(
    db_path: Path,
    game_ids: set[str],
) -> pd.DataFrame:

    conn = sqlite3.connect(
        f"file:{db_path.resolve()}?mode=ro",
        uri=True,
    )

    try:
        games = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                game_date,
                gametime,
                away_team,
                home_team,
                completed
            FROM games
            """,
            conn,
        )

    finally:
        conn.close()

    games["game_id"] = (
        games["game_id"]
        .astype(str)
    )

    games = games[
        games["game_id"].isin(
            game_ids
        )
    ].copy()

    if len(games) != len(game_ids):
        found = set(
            games["game_id"]
        )

        missing = sorted(
            game_ids - found
        )

        raise RuntimeError(
            f"Schedule coverage incomplete. "
            f"MISSING={missing}"
        )

    if games["game_id"].duplicated().any():
        raise RuntimeError(
            "Schedule game_id is not unique"
        )

    games["kickoff_utc"] = [
        kickoff_utc(
            row.game_date,
            row.gametime,
        )
        for row in games.itertuples(
            index=False
        )
    ]

    return games


def validate_source(
    df: pd.DataFrame,
):
    required = {
        "entity_type",
        "game_id",
        "team",
        "opponent_team",
        "player_id",
        "entity_name",
        "position",
        "model_group",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Source missing columns: {missing}"
        )

    if df.empty:
        raise RuntimeError(
            "Source forecast is empty"
        )

    if df["game_id"].isna().any():
        raise RuntimeError(
            "Source contains null game_id"
        )

    if df["team"].isna().any():
        raise RuntimeError(
            "Source contains null team"
        )

    stat_cols = [
        c
        for c in df.columns
        if c.startswith("expected_")
    ]

    if not stat_cols:
        raise RuntimeError(
            "Source contains no expected_* stat columns"
        )


def validate_game_rows(
    df: pd.DataFrame,
    game_id: str,
):
    g = df[
        df["game_id"]
        .astype(str)
        .eq(game_id)
    ]

    if g.empty:
        raise RuntimeError(
            f"No rows for game_id={game_id}"
        )

    offense = g[
        g["entity_type"]
        .eq("OFFENSE_PLAYER")
    ]

    kicker = g[
        g["entity_type"]
        .eq("KICKER")
    ]

    dst = g[
        g["entity_type"]
        .eq("DST")
    ]

    if (
        kicker["team"].nunique() != 2
        or len(kicker) != 2
    ):
        raise RuntimeError(
            f"{game_id}: kicker coverage invalid"
        )

    if (
        dst["team"].nunique() != 2
        or len(dst) != 2
    ):
        raise RuntimeError(
            f"{game_id}: DST coverage invalid"
        )

    if offense.empty:
        raise RuntimeError(
            f"{game_id}: offense coverage empty"
        )


def snapshot_paths(
    snapshot_root: Path,
    game_id: str,
):
    game_dir = (
        snapshot_root
        / game_id
    )

    return (
        game_dir
        / SNAPSHOT_PARQUET_NAME,
        game_dir
        / SNAPSHOT_MANIFEST_NAME,
    )


def validate_existing_snapshot(
    parquet_path: Path,
    manifest_path: Path,
    game_id: str,
) -> pd.DataFrame:

    if (
        parquet_path.exists()
        != manifest_path.exists()
    ):
        raise RuntimeError(
            f"{game_id}: partial snapshot pair exists"
        )

    if not parquet_path.exists():
        raise RuntimeError(
            f"{game_id}: snapshot does not exist"
        )

    manifest = json.loads(
        manifest_path.read_text()
    )

    expected_sha = manifest.get(
        "snapshot_sha256"
    )

    actual_sha = sha256_file(
        parquet_path
    )

    if expected_sha != actual_sha:
        raise RuntimeError(
            f"{game_id}: snapshot hash mismatch"
        )

    snap = pd.read_parquet(
        parquet_path
    )

    ids = set(
        snap["game_id"]
        .astype(str)
    )

    if ids != {game_id}:
        raise RuntimeError(
            f"{game_id}: snapshot contains wrong game IDs"
        )

    validate_game_rows(
        snap,
        game_id,
    )

    return snap


def create_snapshot(
    snapshot_df: pd.DataFrame,
    snapshot_root: Path,
    game_id: str,
    kickoff: datetime,
    now: datetime,
    prior_current_sha: str,
):
    parquet_path, manifest_path = snapshot_paths(
        snapshot_root,
        game_id,
    )

    parquet_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        parquet_path.exists()
        or manifest_path.exists()
    ):
        return validate_existing_snapshot(
            parquet_path,
            manifest_path,
            game_id,
        )

    validate_game_rows(
        snapshot_df,
        game_id,
    )

    atomic_parquet_write(
        snapshot_df,
        parquet_path,
    )

    snap_sha = sha256_file(
        parquet_path
    )

    manifest = {
        "contract":
            "WFS_STAT_FORECAST_KICKOFF_SNAPSHOT_V1",

        "game_id":
            game_id,

        "kickoff_utc":
            kickoff.isoformat(),

        "frozen_at_utc":
            now.isoformat(),

        "rows":
            int(
                len(snapshot_df)
            ),

        "snapshot_sha256":
            snap_sha,

        "source_prior_current_sha256":
            prior_current_sha,

        "immutable":
            True,

        "postgame_regeneration_allowed":
            False,
    }

    atomic_json_write(
        manifest,
        manifest_path,
    )

    return validate_existing_snapshot(
        parquet_path,
        manifest_path,
        game_id,
    )


def publish(
    source_path: Path,
    data_root: Path,
    db_path: Path,
    now: datetime,
    source_binding: Path | None = None,
):

    source_bytes = source_path.read_bytes()
    consumed_source_sha = hashlib.sha256(source_bytes).hexdigest()
    provenance = None
    validated_v3 = None
    if source_binding is not None:
        try:
            from scripts.preflight_offensive_reconciliation_v3 import load_validated_candidate
        except ModuleNotFoundError:
            from preflight_offensive_reconciliation_v3 import load_validated_candidate
        provenance = json.loads(source_binding.read_bytes())
        if (provenance.get("contract") != "WFS_V3_BOUND_UNIFIED_V1"
                or provenance.get("source_path") != str(source_path.resolve())
                or provenance.get("source_sha256") != consumed_source_sha):
            raise RuntimeError("FAIL_CLOSED: unified publication source binding mismatch")
        validated_v3, _, parent = load_validated_candidate(
            Path(provenance["validation_path"]), provenance["season"], provenance["week"],
        )
        if (parent["candidate_id"] != provenance["candidate_id"]
                or parent["candidate_sha256"] != provenance["v3_sha256"]):
            raise RuntimeError("FAIL_CLOSED: unified V3 parent mismatch")
    # Parse exactly the buffer whose digest was checked above.
    source = pd.read_parquet(io.BytesIO(source_bytes))
    if provenance and (len(source) != provenance["rows"] or not source.game_id.astype(str).str.startswith(
            f"{provenance['season']}_{provenance['week']:02d}_").all()):
        raise RuntimeError("FAIL_CLOSED: unified row count/season/week mismatch")

    validate_source(
        source
    )

    try:
        from qb_role_authority import (
            build_refreshable_authority,
            create_kickoff_authority,
            write_bound_authority,
            write_current_authority,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FAIL_CLOSED: QB role authority module unavailable"
        ) from exc

    source = source.copy()

    source, availability_gate_audit = (
        apply_availability_gate(
            source=source,
            db_path=db_path,
        )
    )

    print(
        "AVAILABILITY_GATE_CONTRACT="
        f"{availability_gate_audit['contract']}"
    )

    print(
        "AVAILABILITY_GATE_IDENTITY_JOIN="
        f"{availability_gate_audit['identity_join']}"
    )

    print(
        "AVAILABILITY_GATE_HARD_SUPPRESSION_ROWS="
        f"{availability_gate_audit['hard_suppression_rows']}"
    )

    print(
        "AVAILABILITY_GATE_QUESTIONABLE_ROWS="
        f"{availability_gate_audit['questionable_rows']}"
    )

    print(
        "AVAILABILITY_GATE_DOUBTFUL_ROWS="
        f"{availability_gate_audit['doubtful_rows']}"
    )

    for target in availability_gate_audit[
        "hard_targets"
    ]:
        print(
            "AVAILABILITY_SUPPRESSION_TARGET|"
            f"PLAYER_ID={target['player_id']}|"
            f"NAME={target['entity_name']!r}|"
            f"TEAM={target['team']}|"
            f"POS={target['position']}|"
            f"GAME={target['game_id']}|"
            f"STATUS={target['consensus_status']}|"
            f"GATE={target['injury_gate']}|"
            f"REASON={target['injury_gate_reason']}"
        )

    source["game_id"] = (
        source["game_id"]
        .astype(str)
    )

    game_ids = set(
        source["game_id"]
    )

    schedule = load_schedule(
        db_path,
        game_ids,
    )

    schedule_map = {
        str(row.game_id): row
        for row in schedule.itertuples(
            index=False
        )
    }

    for game_id in sorted(
        game_ids
    ):
        validate_game_rows(
            source,
            game_id,
        )

    parquet_dir = (
        data_root
        / "parquet"
    )

    snapshot_root = (
        data_root
        / "forecast_snapshots"
    )

    current_path = (
        parquet_dir
        / CURRENT_NAME
    )

    manifest_path = (
        parquet_dir
        / CURRENT_MANIFEST_NAME
    )

    lock_path = (
        parquet_dir
        / ".current_unified_stat_forecasts.lock"
    )

    parquet_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with lock_path.open(
        "a+"
    ) as lock_file:

        fcntl.flock(
            lock_file.fileno(),
            fcntl.LOCK_EX,
        )

        prior = None
        prior_sha = None

        if current_path.exists():

            prior = pd.read_parquet(
                current_path
            )

            validate_source(
                prior
            )

            prior["game_id"] = (
                prior["game_id"]
                .astype(str)
            )

            prior_sha = sha256_file(
                current_path
            )

        output_parts = []
        states = []

        snapshots_created = 0
        snapshots_reused = 0
        pregame_refreshed = 0

        for game_id in sorted(
            game_ids
        ):

            row = schedule_map[
                game_id
            ]

            kickoff = row.kickoff_utc

            started = (
                now >= kickoff
            )

            snap_path, snap_manifest = snapshot_paths(
                snapshot_root,
                game_id,
            )

            if started:

                if snap_path.exists():

                    frozen = validate_existing_snapshot(
                        snap_path,
                        snap_manifest,
                        game_id,
                    )

                    snapshot_meta = json.loads(
                        snap_manifest.read_text(
                            encoding="utf-8"
                        )
                    )

                    snapshot_source_sha = str(
                        snapshot_meta.get(
                            "source_prior_current_sha256"
                        )
                        or ""
                    ).strip().lower()

                    if not snapshot_source_sha:
                        raise RuntimeError(
                            f"{game_id}: frozen forecast snapshot "
                            "has no prior-current SHA for QB "
                            "authority recovery"
                        )

                    create_kickoff_authority(
                        game_id=game_id,
                        forecast_snapshot_df=frozen,
                        source_forecast_sha256=(
                            snapshot_source_sha
                        ),
                        data_root=data_root,
                        snapshot_root=snapshot_root,
                    )

                    snapshots_reused += 1

                else:

                    if prior is None:

                        raise RuntimeError(
                            f"{game_id}: kickoff already passed "
                            "but no prior canonical pregame artifact "
                            "exists. Refusing to create a snapshot "
                            "from a post-kickoff candidate."
                        )

                    prior_game = prior[
                        prior["game_id"]
                        .eq(game_id)
                    ].copy()

                    if prior_game.empty:

                        raise RuntimeError(
                            f"{game_id}: kickoff already passed "
                            "but prior canonical artifact has no rows "
                            "for this game."
                        )

                    validate_game_rows(
                        prior_game,
                        game_id,
                    )

                    frozen = create_snapshot(
                        snapshot_df=prior_game,
                        snapshot_root=snapshot_root,
                        game_id=game_id,
                        kickoff=kickoff,
                        now=now,
                        prior_current_sha=prior_sha,
                    )

                    create_kickoff_authority(
                        game_id=game_id,
                        forecast_snapshot_df=prior_game,
                        source_forecast_sha256=prior_sha,
                        data_root=data_root,
                        snapshot_root=snapshot_root,
                    )

                    snapshots_created += 1

                output_parts.append(
                    frozen
                )

                states.append(
                    {
                        "game_id":
                            game_id,

                        "kickoff_utc":
                            kickoff.isoformat(),

                        "state":
                            "FROZEN",

                        "snapshot_sha256":
                            sha256_file(
                                snap_path
                            ),
                    }
                )

            else:

                fresh = source[
                    source["game_id"]
                    .eq(game_id)
                ].copy()

                output_parts.append(
                    fresh
                )

                pregame_refreshed += 1

                states.append(
                    {
                        "game_id":
                            game_id,

                        "kickoff_utc":
                            kickoff.isoformat(),

                        "state":
                            "REFRESHABLE",
                    }
                )

        output = pd.concat(
            output_parts,
            ignore_index=True,
        )

        if len(output) != len(source):
            raise RuntimeError(
                f"Output row count changed: "
                f"{len(output)} != {len(source)}"
            )

        if (
            set(output["game_id"].astype(str))
            != game_ids
        ):
            raise RuntimeError(
                "Output game universe changed"
            )

        refreshable_game_ids = {
            str(state["game_id"])
            for state in states
            if state.get("state") == "REFRESHABLE"
        }

        if refreshable_game_ids:
            if validated_v3 is None:
                raise RuntimeError(
                    "FAIL_CLOSED: refreshable publication requires "
                    "validated V3 QB role authority"
                )

            current_qb_authority = (
                build_refreshable_authority(
                    validated_v3,
                    output,
                    refreshable_game_ids,
                )
            )
        else:
            current_qb_authority = None

        current_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        stage_fd, stage_name = tempfile.mkstemp(
            prefix=current_path.name + ".qb-stage.",
            suffix=".tmp.parquet",
            dir=str(current_path.parent),
        )
        os.close(stage_fd)
        staged_current_path = Path(stage_name)

        try:
            output.to_parquet(
                staged_current_path,
                index=False,
            )

            current_sha = sha256_file(
                staged_current_path
            )

            if refreshable_game_ids:
                qb_authority_path, qb_authority_manifest = (
                    write_bound_authority(
                        current_qb_authority,
                        forecast_df=output[
                            output["game_id"].astype(str).isin(
                                refreshable_game_ids
                            )
                        ].copy(),
                        forecast_sha256=current_sha,
                        data_root=data_root,
                    )
                )
            else:
                qb_authority_path = None
                qb_authority_manifest = None

            if current_qb_authority is None:
                from qb_role_authority import ROLE_COLUMNS

                current_qb_authority = pd.DataFrame(
                    columns=ROLE_COLUMNS
                )

                current_qb_forecast = None
            else:
                current_qb_forecast = output[
                    output["game_id"].astype(str).isin(
                        refreshable_game_ids
                    )
                ].copy()

            os.replace(
                staged_current_path,
                current_path,
            )

        finally:
            if staged_current_path.exists():
                staged_current_path.unlink()

        if sha256_file(current_path) != current_sha:
            raise RuntimeError(
                "FAIL_CLOSED: committed canonical forecast "
                "SHA differs from staged publication SHA"
            )

        (
            current_qb_sidecar_path,
            current_qb_sidecar_manifest,
        ) = write_current_authority(
            current_qb_authority,
            forecast_sha256=current_sha,
            data_root=data_root,
            forecast_df=current_qb_forecast,
        )

        current_manifest = {
            "contract":
                CONTRACT,

            "published_at_utc":
                now.isoformat(),

            "source_candidate_path":
                str(source_path),

            "source_candidate_sha256":
                consumed_source_sha,

            "validated_v3_parent": provenance,

            "current_sha256":
                current_sha,

            "availability_gate_contract":
                availability_gate_audit[
                    "contract"
                ],

            "availability_gate_identity_join":
                availability_gate_audit[
                    "identity_join"
                ],

            "availability_gate_hard_suppression_statuses":
                availability_gate_audit[
                    "hard_suppression_statuses"
                ],

            "availability_gate_preserve_numeric_statuses":
                availability_gate_audit[
                    "preserve_numeric_statuses"
                ],

            "availability_gate_hard_suppression_rows":
                availability_gate_audit[
                    "hard_suppression_rows"
                ],

            "availability_gate_questionable_rows":
                availability_gate_audit[
                    "questionable_rows"
                ],

            "availability_gate_doubtful_rows":
                availability_gate_audit[
                    "doubtful_rows"
                ],

            "rows":
                int(
                    len(output)
                ),

            "games":
                int(
                    output[
                        "game_id"
                    ].nunique()
                ),

            "teams":
                int(
                    output[
                        "team"
                    ].nunique()
                ),

            "pregame_games_refreshed":
                int(
                    pregame_refreshed
                ),

            "snapshots_created":
                int(
                    snapshots_created
                ),

            "snapshots_reused":
                int(
                    snapshots_reused
                ),

            "game_states":
                states,

            "post_kickoff_overwrite_allowed":
                False,

            "historical_regeneration_allowed":
                False,
        }

        atomic_json_write(
            current_manifest,
            manifest_path,
        )

        # Final manifest self-check.
        reread_manifest = json.loads(
            manifest_path.read_text()
        )

        if (
            reread_manifest[
                "current_sha256"
            ]
            != sha256_file(
                current_path
            )
        ):
            raise RuntimeError(
                "Current manifest SHA does not match current parquet"
            )

        print(
            f"CURRENT_PATH={current_path}"
        )

        print(
            f"CURRENT_MANIFEST={manifest_path}"
        )

        print(
            f"CURRENT_SHA={current_sha}"
        )

        print(
            f"ROWS={len(output)}"
        )

        print(
            f"GAMES={output['game_id'].nunique()}"
        )

        print(
            f"TEAMS={output['team'].nunique()}"
        )

        print(
            f"PREGAME_GAMES_REFRESHED="
            f"{pregame_refreshed}"
        )

        print(
            f"SNAPSHOTS_CREATED="
            f"{snapshots_created}"
        )

        print(
            f"SNAPSHOTS_REUSED="
            f"{snapshots_reused}"
        )

        print(
            "PUBLISH_RESULT=PASS"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-binding", type=Path)

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--data-root",
        default=str(
            DEFAULT_DATA_ROOT
        ),
    )

    parser.add_argument(
        "--db",
        default=str(
            DEFAULT_DB
        ),
    )

    parser.add_argument(
        "--now-utc",
        default=None,
        help=(
            "Testing only. ISO-8601 timestamp "
            "with UTC offset or Z."
        ),
    )

    args = parser.parse_args()

    publish(
        source_path=Path(
            args.source
        ).resolve(),

        data_root=Path(
            args.data_root
        ).resolve(),

        db_path=Path(
            args.db
        ).resolve(),

        now=parse_now(
            args.now_utc
        ),
        source_binding=args.source_binding,
    )


if __name__ == "__main__":
    main()
