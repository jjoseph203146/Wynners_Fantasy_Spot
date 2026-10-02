#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: Path):
    return json.loads(path.read_text())


def run_lifecycle_checks(
    root: Path,
    schedule: dict | None,
) -> list[dict]:
    results = []

    def record(name, status, detail):
        results.append(
            {
                "check": name,
                "status": status,
                "detail": detail,
            }
        )

    if schedule is None:
        record(
            "PUBLICATION_LIFECYCLE",
            "FAIL",
            "schedule authority unavailable",
        )
        return results

    p = root / "data" / "parquet"

    stat_artifact = p / "current_unified_stat_forecasts.parquet"
    stat_manifest = p / "current_unified_stat_forecasts_manifest.json"

    fd_artifact = p / "current_unified_fanduel_expectation.parquet"
    fd_manifest = p / "current_unified_fanduel_expectation_manifest.json"

    starter_artifact = p / "current_starter_verification.parquet"
    starter_manifest = p / "current_starter_verification_manifest.json"

    required = [
        stat_artifact,
        stat_manifest,
        fd_artifact,
        fd_manifest,
        starter_artifact,
        starter_manifest,
    ]

    missing = [str(x) for x in required if not x.is_file()]

    if missing:
        record(
            "PUBLICATION_LIFECYCLE",
            "FAIL",
            "missing=" + ",".join(missing),
        )
        return results

    try:
        stat = _load_json(stat_manifest)
        fd = _load_json(fd_manifest)
        starter = _load_json(starter_manifest)
    except Exception as exc:
        record(
            "PUBLICATION_LIFECYCLE",
            "FAIL",
            f"manifest_read_error={type(exc).__name__}:{exc}",
        )
        return results

    season = int(schedule["season"])
    week = int(schedule["week"])
    games = int(schedule["unfinished_games"])
    teams = int(schedule["unfinished_teams"])
    active_game_ids = {
        str(v)
        for v in schedule["unfinished_game_ids"]
    }

    # ---------------------------------------------------------
    # STAT publication lifecycle
    # ---------------------------------------------------------

    stat_hash = _sha256(stat_artifact)
    stat_declared_hash = stat.get("current_sha256")

    game_states = stat.get("game_states")

    if not isinstance(game_states, list):
        stat_state_ids = set()
        states_valid = False
    else:
        stat_state_ids = {
            str(row.get("game_id"))
            for row in game_states
            if isinstance(row, dict) and row.get("game_id")
        }
        states_valid = (
            len(game_states) == games
            and stat_state_ids == active_game_ids
        )

    stat_ok = (
        stat.get("contract") == "WFS_STAT_FORECAST_PUBLISH_V1"
        and int(stat.get("games", -1)) == games
        and int(stat.get("teams", -1)) == teams
        and stat_declared_hash == stat_hash
        and states_valid
        and stat.get("historical_regeneration_allowed") is False
        and stat.get("post_kickoff_overwrite_allowed") is False
    )

    record(
        "STAT_PUBLICATION_LIFECYCLE",
        "PASS" if stat_ok else "FAIL",
        (
            f"games={stat.get('games')}/{games} "
            f"teams={stat.get('teams')}/{teams} "
            f"game_states={len(stat_state_ids)}/{games} "
            f"game_id_diff={len(stat_state_ids ^ active_game_ids)} "
            f"hash_match={stat_declared_hash == stat_hash}"
        ),
    )

    # ---------------------------------------------------------
    # FanDuel production publication lifecycle
    # ---------------------------------------------------------

    fd_hash = _sha256(fd_artifact)

    fd_current = fd.get("current_sha256")
    fd_production = fd.get("production_current_sha256")
    fd_source = fd.get("source_sha256")

    fd_hash_chain_ok = (
        fd_current == fd_hash
        and fd_production == fd_hash
        and fd_source == fd_hash
    )

    fd_ok = (
        fd.get("contract") == "WFS_FANDUEL_EXPECTATION_PUBLISH_V1"
        and fd.get("status") == "PRODUCTION_CURRENT_GAV2"
        and int(fd.get("season", -1)) == season
        and int(fd.get("week", -1)) == week
        and int(fd.get("games", -1)) == games
        and int(fd.get("teams", -1)) == teams
        and fd_hash_chain_ok
    )

    record(
        "FANDUEL_PUBLICATION_LIFECYCLE",
        "PASS" if fd_ok else "FAIL",
        (
            f"season={fd.get('season')}/{season} "
            f"week={fd.get('week')}/{week} "
            f"games={fd.get('games')}/{games} "
            f"teams={fd.get('teams')}/{teams} "
            f"hash_chain={fd_hash_chain_ok}"
        ),
    )

    # ---------------------------------------------------------
    # Starter verification lifecycle
    #
    # Generation time is deliberately NOT used as a freshness gate.
    # This artifact is analysis-only and is validated by schedule
    # ownership plus its explicit non-production contract.
    # ---------------------------------------------------------

    starter_ok = (
        starter.get("contract")
        == "WFS_STARTER_VERIFICATION_CURRENT_V1"
        and int(starter.get("season", -1)) == season
        and int(starter.get("week", -1)) == week
        and str(starter.get("game_type", "")).upper() == "REG"
        and starter.get("identity_gate") == "PASS"
        and int(starter.get("identity_ambiguous_count", -1)) == 0
        and int(starter.get("identity_unresolved_count", -1)) == 0
        and starter.get("analysis_only") is True
        and starter.get("production_influence") is False
        and starter.get("solver_influence") is False
        and starter.get("forecast_mutation") is False
        and starter.get("database_mutation") is False
        and starter.get("depth_chart_mutation") is False
    )

    record(
        "STARTER_PUBLICATION_LIFECYCLE",
        "PASS" if starter_ok else "FAIL",
        (
            f"season={starter.get('season')}/{season} "
            f"week={starter.get('week')}/{week} "
            f"game_type={starter.get('game_type')} "
            f"identity_gate={starter.get('identity_gate')} "
            f"analysis_only={starter.get('analysis_only')} "
            f"production_influence={starter.get('production_influence')}"
        ),
    )

    all_ok = stat_ok and fd_ok and starter_ok

    record(
        "PUBLICATION_LIFECYCLE",
        "PASS" if all_ok else "FAIL",
        (
            f"season={season} week={week} "
            f"active_games={games} active_teams={teams}"
        ),
    )

    return results


def write_lkg_if_guardian_pass(
    root: Path,
    schedule: dict | None,
    guardian_results: list[dict],
) -> tuple[bool, str]:
    """
    Guardian-owned metadata write only.

    This does NOT:
      - alter production artifacts
      - alter manifests
      - alter operational databases
      - copy production artifacts
      - restore anything
      - restart anything
    """

    if schedule is None:
        return False, "schedule authority unavailable"

    failures = [
        row
        for row in guardian_results
        if row.get("status") == "FAIL"
    ]

    if failures:
        return (
            False,
            f"guardian_has_failures={len(failures)}",
        )

    p = root / "data" / "parquet"

    artifacts = {
        "stat_publication": (
            p / "current_unified_stat_forecasts.parquet"
        ),
        "fanduel_publication": (
            p / "current_unified_fanduel_expectation.parquet"
        ),
        "starter_verification": (
            p / "current_starter_verification.parquet"
        ),
        "stage24_gav2": (
            p / "nfl_current_fanduel_expectation_gav2.parquet"
        ),
    }

    artifact_state = {}

    for name, path in artifacts.items():
        if not path.is_file():
            return False, f"missing_lkg_artifact={path}"

        artifact_state[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }

    state = {
        "contract": "WFS_GUARDIAN_LKG_V1",
        "guardian_version": "V2.3",
        "verified_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "production_influence": False,
        "auto_repair": False,
        "auto_rollback": False,
        "auto_restart": False,
        "schedule": {
            "season": int(schedule["season"]),
            "week": int(schedule["week"]),
            "unfinished_games": int(
                schedule["unfinished_games"]
            ),
            "unfinished_teams": int(
                schedule["unfinished_teams"]
            ),
            "completed_same_week": int(
                schedule["completed_same_week"]
            ),
            "unfinished_game_ids": sorted(
                str(v)
                for v in schedule["unfinished_game_ids"]
            ),
        },
        "artifacts": artifact_state,
    }

    guardian_dir = root / "data" / "guardian"
    guardian_dir.mkdir(parents=True, exist_ok=True)

    target = guardian_dir / "guardian_lkg.json"
    temp = guardian_dir / (
        f".guardian_lkg.{os.getpid()}.tmp"
    )

    temp.write_text(
        json.dumps(
            state,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    os.replace(temp, target)

    return (
        True,
        (
            f"path={target} "
            f"season={schedule['season']} "
            f"week={schedule['week']} "
            f"artifacts={len(artifact_state)}"
        ),
    )
