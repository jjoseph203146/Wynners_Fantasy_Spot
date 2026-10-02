#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path, required: set[str]):
    if not path.is_file():
        return None, f"missing={path}"

    if path.stat().st_size == 0:
        return None, f"zero_byte={path}"

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return None, f"read_error={type(exc).__name__}:{exc}"

    missing = sorted(required - set(df.columns))
    if missing:
        return None, "missing_columns=" + ",".join(missing)

    return df, None


TEAM_ALIASES = {
    "LAR": "LA",
    "WSH": "WAS",
}


def _team(value) -> str:
    team = str(value).strip().upper()
    return TEAM_ALIASES.get(team, team)


def _inventory(df):
    return (
        len(df),
        int(df["game_id"].nunique(dropna=True)),
        int(df["team"].nunique(dropna=True)),
    )


def _fd_keys(df):
    return {
        tuple(
            "<NULL>" if pd.isna(v) else str(v)
            for v in row
        )
        for row in df[
            ["game_id", "team", "position", "entity_id"]
        ].itertuples(index=False, name=None)
    }


def run_cross_stage(
    root: Path,
    schedule: dict | None,
) -> list[dict]:
    """
    Guardian V2.2 read-only cross-stage reconciliation.

    Returns Guardian-style result dictionaries.
    Writes nothing.
    Changes nothing.
    """

    results = []

    def record(name, status, detail):
        results.append(
            {
                "check": name,
                "status": status,
                "detail": detail,
            }
        )

    parquet = root / "data" / "parquet"

    paths = {
        "offense": parquet / "nfl_current_offensive_stat_forecasts.parquet",
        "kicker": parquet / "nfl_current_kicker_stat_forecasts.parquet",
        "dst": parquet / "nfl_current_dst_stat_forecasts.parquet",
        "unified": parquet / "nfl_current_unified_stat_forecasts.parquet",
        "fd": parquet / "nfl_current_fanduel_expectation.parquet",
        "gav2": parquet / "nfl_current_fanduel_expectation_gav2.parquet",
        "published": parquet / "current_unified_fanduel_expectation.parquet",
    }

    specs = {
        "offense": {
            "game_id", "team", "player_id", "position"
        },
        "kicker": {
            "game_id", "team", "gsis_id"
        },
        "dst": {
            "game_id", "team"
        },
        "unified": {
            "entity_type",
            "game_id",
            "team",
            "player_id",
            "position",
        },
        "fd": {
            "game_id",
            "team",
            "position",
            "entity_id",
            "source_component",
            "expected_fanduel_points",
        },
        "gav2": {
            "game_id",
            "team",
            "position",
            "entity_id",
            "source_component",
            "expected_fanduel_points",
        },
        "published": {
            "game_id",
            "team",
            "position",
            "entity_id",
            "source_component",
            "expected_fanduel_points",
        },
    }

    frames = {}

    for name in paths:
        df, error = _load(paths[name], specs[name])

        if error:
            record(
                f"XSTAGE_LOAD_{name.upper()}",
                "FAIL",
                error,
            )
        else:
            frames[name] = df

    if len(frames) != len(paths):
        record(
            "CROSS_STAGE_RECONCILIATION",
            "FAIL",
            "one or more required stages unavailable",
        )
        return results

    if schedule is None:
        record(
            "CROSS_STAGE_RECONCILIATION",
            "FAIL",
            "schedule authority unavailable",
        )
        return results

    expected_games = int(schedule["unfinished_games"])
    expected_teams = int(schedule["unfinished_teams"])
    expected_game_ids = {
        str(v)
        for v in schedule["unfinished_game_ids"]
    }

    inventory_labels = {
        "offense": "OFFENSE_STAGE_INVENTORY",
        "kicker": "KICKER_STAGE_INVENTORY",
        "dst": "DST_STAGE_INVENTORY",
        "unified": "UNIFIED_STAGE_INVENTORY",
        "fd": "FD_STAGE_INVENTORY",
        "gav2": "GAV2_STAGE_INVENTORY",
        "published": "PUBLISHED_STAGE_INVENTORY",
    }

    inventory_ok = True

    for name, label in inventory_labels.items():
        df = frames[name]
        rows, games, teams = _inventory(df)

        actual_game_ids = {
            str(v)
            for v in df["game_id"].dropna().unique()
        }

        ok = (
            games == expected_games
            and teams == expected_teams
            and actual_game_ids == expected_game_ids
        )

        if not ok:
            inventory_ok = False

        record(
            label,
            "PASS" if ok else "FAIL",
            (
                f"rows={rows} "
                f"games={games}/{expected_games} "
                f"teams={teams}/{expected_teams} "
                f"game_id_diff="
                f"{len(actual_game_ids ^ expected_game_ids)}"
            ),
        )

    offense = frames["offense"]
    kicker = frames["kicker"]
    dst = frames["dst"]
    unified = frames["unified"]
    fd = frames["fd"]
    gav2 = frames["gav2"]
    published = frames["published"]

    # Exact upstream player identities.
    offense_ids = offense["player_id"]

    offense_id_ok = (
        not offense_ids.isna().any()
        and not offense_ids.astype(str).str.strip().eq("").any()
        and not offense_ids.duplicated().any()
    )

    record(
        "OFFENSE_IDENTITY_CONTRACT",
        "PASS" if offense_id_ok else "FAIL",
        (
            f"rows={len(offense)} "
            f"unique_ids={offense_ids.nunique(dropna=True)} "
            f"null={int(offense_ids.isna().sum())} "
            f"duplicates={int(offense_ids.duplicated().sum())}"
        ),
    )

    kicker_ids = kicker["gsis_id"]

    kicker_id_ok = (
        not kicker_ids.isna().any()
        and not kicker_ids.astype(str).str.strip().eq("").any()
        and not kicker_ids.duplicated().any()
    )

    record(
        "KICKER_IDENTITY_CONTRACT",
        "PASS" if kicker_id_ok else "FAIL",
        (
            f"rows={len(kicker)} "
            f"unique_ids={kicker_ids.nunique(dropna=True)} "
            f"null={int(kicker_ids.isna().sum())} "
            f"duplicates={int(kicker_ids.duplicated().sum())}"
        ),
    )

    kicker_counts = kicker.groupby("team").size()
    dst_counts = dst.groupby("team").size()

    kicker_team_ok = (
        len(kicker) == expected_teams
        and kicker["team"].nunique() == expected_teams
        and kicker_counts.eq(1).all()
    )

    dst_team_ok = (
        len(dst) == expected_teams
        and dst["team"].nunique() == expected_teams
        and dst_counts.eq(1).all()
    )

    record(
        "KICKER_ONE_PER_ACTIVE_TEAM",
        "PASS" if kicker_team_ok else "FAIL",
        (
            f"rows={len(kicker)} "
            f"teams={kicker['team'].nunique()} "
            f"expected={expected_teams}"
        ),
    )

    record(
        "DST_ONE_PER_ACTIVE_TEAM",
        "PASS" if dst_team_ok else "FAIL",
        (
            f"rows={len(dst)} "
            f"teams={dst['team'].nunique()} "
            f"expected={expected_teams}"
        ),
    )

    # Component -> Unified arithmetic.
    expected_entity_counts = {
        "OFFENSE_PLAYER": len(offense),
        "KICKER": len(kicker),
        "DST": len(dst),
    }

    actual_entity_counts = (
        unified["entity_type"]
        .fillna("<NULL>")
        .value_counts()
        .to_dict()
    )

    expected_unified_rows = sum(
        expected_entity_counts.values()
    )

    component_count_ok = (
        len(unified) == expected_unified_rows
        and actual_entity_counts == expected_entity_counts
    )

    record(
        "COMPONENT_TO_UNIFIED",
        "PASS" if component_count_ok else "FAIL",
        (
            f"offense={len(offense)} "
            f"kicker={len(kicker)} "
            f"dst={len(dst)} "
            f"expected={expected_unified_rows} "
            f"actual={len(unified)} "
            f"entity_counts={actual_entity_counts}"
        ),
    )

    u_offense = unified[
        unified["entity_type"] == "OFFENSE_PLAYER"
    ]

    u_kicker = unified[
        unified["entity_type"] == "KICKER"
    ]

    u_dst = unified[
        unified["entity_type"] == "DST"
    ]

    non_dst = unified[
        unified["entity_type"].isin(
            ["OFFENSE_PLAYER", "KICKER"]
        )
    ]

    non_dst_ids = non_dst["player_id"]

    non_dst_ok = (
        len(non_dst) == len(offense) + len(kicker)
        and not non_dst_ids.isna().any()
        and not non_dst_ids.astype(str).str.strip().eq("").any()
        and not non_dst_ids.duplicated().any()
    )

    record(
        "UNIFIED_NON_DST_IDENTITY",
        "PASS" if non_dst_ok else "FAIL",
        (
            f"rows={len(non_dst)} "
            f"null_ids={int(non_dst_ids.isna().sum())} "
            f"duplicates={int(non_dst_ids.duplicated().sum())}"
        ),
    )

    dst_nulls = int(u_dst["player_id"].isna().sum())

    dst_id_ok = (
        len(u_dst) == len(dst)
        and dst_nulls == len(u_dst)
    )

    record(
        "UNIFIED_DST_ID_CONVENTION",
        "PASS" if dst_id_ok else "FAIL",
        (
            f"dst_rows={len(u_dst)} "
            f"null_player_ids={dst_nulls}"
        ),
    )

    offense_source_keys = {
        (str(g), _team(t), str(pid))
        for g, t, pid in offense[
            ["game_id", "team", "player_id"]
        ].itertuples(index=False, name=None)
    }

    offense_unified_keys = {
        (str(g), _team(t), str(pid))
        for g, t, pid in u_offense[
            ["game_id", "team", "player_id"]
        ].itertuples(index=False, name=None)
    }

    kicker_source_keys = {
        (str(g), _team(t), str(pid))
        for g, t, pid in kicker[
            ["game_id", "team", "gsis_id"]
        ].itertuples(index=False, name=None)
    }

    kicker_unified_keys = {
        (str(g), _team(t), str(pid))
        for g, t, pid in u_kicker[
            ["game_id", "team", "player_id"]
        ].itertuples(index=False, name=None)
    }

    dst_source_keys = {
        (str(g), _team(t))
        for g, t in dst[
            ["game_id", "team"]
        ].itertuples(index=False, name=None)
    }

    dst_unified_keys = {
        (str(g), _team(t))
        for g, t in u_dst[
            ["game_id", "team"]
        ].itertuples(index=False, name=None)
    }

    component_identity_ok = (
        offense_source_keys == offense_unified_keys
        and kicker_source_keys == kicker_unified_keys
        and dst_source_keys == dst_unified_keys
    )

    record(
        "COMPONENT_IDENTITY_HANDOFF",
        "PASS" if component_identity_ok else "FAIL",
        (
            f"offense_diff="
            f"{len(offense_source_keys ^ offense_unified_keys)} "
            f"kicker_diff="
            f"{len(kicker_source_keys ^ kicker_unified_keys)} "
            f"dst_diff="
            f"{len(dst_source_keys ^ dst_unified_keys)}"
        ),
    )

    # Unified -> FanDuel component inventory.
    expected_fd_counts = {
        "OFFENSE": len(offense),
        "KICKER": len(kicker),
        "DST": len(dst),
    }

    actual_fd_counts = (
        fd["source_component"]
        .fillna("<NULL>")
        .value_counts()
        .to_dict()
    )

    fd_inventory_ok = (
        len(fd) == len(unified)
        and actual_fd_counts == expected_fd_counts
    )

    record(
        "UNIFIED_TO_FD_INVENTORY",
        "PASS" if fd_inventory_ok else "FAIL",
        (
            f"unified={len(unified)} "
            f"fd={len(fd)} "
            f"components={actual_fd_counts}"
        ),
    )

    entity_contract_ok = True

    for label, frame in (
        ("FD_ENTITY_ID_CONTRACT", fd),
        ("GAV2_ENTITY_ID_CONTRACT", gav2),
        ("PUBLISHED_ENTITY_ID_CONTRACT", published),
    ):
        ids = frame["entity_id"]

        ok = (
            not ids.isna().any()
            and not ids.astype(str).str.strip().eq("").any()
            and not ids.duplicated().any()
        )

        if not ok:
            entity_contract_ok = False

        record(
            label,
            "PASS" if ok else "FAIL",
            (
                f"rows={len(frame)} "
                f"unique_ids={ids.nunique(dropna=True)} "
                f"null={int(ids.isna().sum())} "
                f"duplicates={int(ids.duplicated().sum())}"
            ),
        )

    # Exact FD -> GAV2 -> publication identity preservation.
    fd_keys = _fd_keys(fd)
    gav2_keys = _fd_keys(gav2)
    published_keys = _fd_keys(published)

    fd_gav2_ok = (
        len(fd_keys) == len(fd)
        and len(gav2_keys) == len(gav2)
        and fd_keys == gav2_keys
    )

    record(
        "FD_TO_GAV2_IDENTITY_HANDOFF",
        "PASS" if fd_gav2_ok else "FAIL",
        (
            f"fd_keys={len(fd_keys)} "
            f"gav2_keys={len(gav2_keys)} "
            f"fd_minus_gav2={len(fd_keys - gav2_keys)} "
            f"gav2_minus_fd={len(gav2_keys - fd_keys)}"
        ),
    )

    gav2_publish_ok = (
        len(gav2_keys) == len(gav2)
        and len(published_keys) == len(published)
        and gav2_keys == published_keys
    )

    record(
        "GAV2_TO_PUBLISH_IDENTITY_HANDOFF",
        "PASS" if gav2_publish_ok else "FAIL",
        (
            f"gav2_keys={len(gav2_keys)} "
            f"published_keys={len(published_keys)} "
            f"gav2_minus_publish="
            f"{len(gav2_keys - published_keys)} "
            f"publish_minus_gav2="
            f"{len(published_keys - gav2_keys)}"
        ),
    )

    gav2_hash = _sha256(paths["gav2"])
    published_hash = _sha256(paths["published"])

    hash_ok = gav2_hash == published_hash

    record(
        "GAV2_TO_PUBLISH_SHA",
        "PASS" if hash_ok else "FAIL",
        (
            f"gav2={gav2_hash[:12]}... "
            f"published={published_hash[:12]}..."
        ),
    )

    critical_ok = all(
        (
            inventory_ok,
            offense_id_ok,
            kicker_id_ok,
            kicker_team_ok,
            dst_team_ok,
            component_count_ok,
            non_dst_ok,
            dst_id_ok,
            component_identity_ok,
            fd_inventory_ok,
            entity_contract_ok,
            fd_gav2_ok,
            gav2_publish_ok,
            hash_ok,
        )
    )

    record(
        "CROSS_STAGE_RECONCILIATION",
        "PASS" if critical_ok else "FAIL",
        (
            f"components={len(offense)}+{len(kicker)}+{len(dst)} "
            f"unified={len(unified)} "
            f"fd={len(fd)} "
            f"gav2={len(gav2)} "
            f"published={len(published)}"
        ),
    )

    return results
