from __future__ import annotations

import fcntl
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


TEAM_ALIASES = {
    "LA": "LAR",
    "WAS": "WSH",
    "JAC": "JAX",
}


STAGE24_SOURCE = "WFS_STAGE24_FANDUEL_EXPECTATION_V1"
STAGE24_AUTHORITY = "WFS_STAGE24_MODEL"
STAGE24_STATUS = "READY"


class Stage24AttachmentError(RuntimeError):
    pass


_CLASSIC_ROOT = Path(__file__).resolve().parent


def _classic_publication_snapshot():
    """Read bound publication bytes under the updater's reader lock; never build.

    Classic only. The shared Stage24 artifact and Showdown path are unchanged.
    Read the exact bytes whose digests are validated, not a second path read.
    """
    from current_fanduel_expectation import offense_points

    def require(condition, message):
        if not condition:
            raise Stage24AttachmentError("CLASSIC_PUBLICATION_" + message)

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    try:
        with (_CLASSIC_ROOT / "nfl_updater.lock").open("rb") as lock:
            # Fail closed while a writer is active; never create/truncate a lock.
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            directory = _CLASSIC_ROOT / "data/parquet"
            manifest_bytes = (directory / "current_unified_stat_forecasts_manifest.json").read_bytes()
            publication_bytes = (directory / "current_unified_stat_forecasts.parquet").read_bytes()
            stage_bytes = (directory / "current_unified_fanduel_expectation.parquet").read_bytes()
            manifest = json.loads(manifest_bytes)
            require(manifest["contract"] == "WFS_STAT_FORECAST_PUBLISH_V1", "CONTRACT")
            require(digest(publication_bytes) == manifest["current_sha256"], "SHA_MISMATCH")
            parent = manifest["validated_v3_parent"]
            require(parent["contract"] == "WFS_V3_BOUND_UNIFIED_V1", "BINDING_CONTRACT")
            require(manifest["source_candidate_path"] == parent["source_path"]
                    and manifest["source_candidate_sha256"] == parent["source_sha256"], "SOURCE_BINDING")
            source = Path(parent["source_path"])
            require(source.is_absolute(), "SOURCE_PATH")
            require(digest(source.read_bytes()) == parent["source_sha256"], "BOUND_SOURCE_SHA")
            require(json.loads(source.with_suffix(".binding.json").read_bytes()) == parent, "BOUND_METADATA")
            validation_path = Path(parent["validation_path"])
            validation = json.loads(validation_path.read_bytes())
            require(validation["contract_version"] == "WFS_V3_IMMUTABLE_PREFLIGHT_V1"
                    and validation["status"] == "PASS_HARD_CONTRACTS", "V3_VALIDATION")
            require(all(validation[k] == parent[k] for k in ("candidate_id", "season", "week")), "V3_PARENT")
            require(validation["candidate_sha256"] == parent["v3_sha256"], "V3_BINDING")
            candidate = validation_path.parent / validation["candidate_path"]
            require(digest(candidate.read_bytes()) == parent["v3_sha256"], "V3_SHA")
            stat = pd.read_parquet(io.BytesIO(publication_bytes))
            require(len(stat) == manifest["rows"] == parent["rows"], "ROW_COUNT")
            require(stat["game_id"].astype(str).str.startswith(
                f"{int(parent['season'])}_{int(parent['week']):02d}_"
            ).all(), "WEEK")
            offense = stat.loc[stat["entity_type"].eq("OFFENSE_PLAYER")].copy()
            required = {
                "player_id", "game_id", "team", "position", "expected_passing_yards",
                "expected_passing_tds", "expected_interceptions", "expected_rushing_yards",
                "expected_rushing_tds", "expected_receptions", "expected_receiving_yards",
                "expected_receiving_tds",
            }
            require(required.issubset(offense.columns) and not offense.empty, "OFFENSE_SCHEMA")
            require(offense["player_id"].map(lambda x: bool(re.fullmatch(r"00-\d{7}", str(x)))).all(), "GSIS_ID")
            require(not offense["player_id"].duplicated().any(), "DUPLICATE_GSIS")
            require(offense["team"].map(clean).ne("").all(), "TEAM")
            offense["_classic_points"] = offense.apply(offense_points, axis=1)
            require(offense["_classic_points"].map(finite).all()
                    and offense["_classic_points"].ge(0).all(), "POINTS")
            token = {
                "publication_sha256": digest(publication_bytes),
                "manifest_sha256": digest(manifest_bytes),
                "stage24_sha256": digest(stage_bytes),
                "bridge_sha256": digest(Path(__file__).read_bytes()),
                "scoring_sha256": digest(Path(offense_points.__code__.co_filename).read_bytes()),
            }
            return offense, parent, token, pd.read_parquet(io.BytesIO(stage_bytes))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise Stage24AttachmentError("CLASSIC_PUBLICATION_UNAVAILABLE_OR_INVALID") from exc


def classic_publication_fingerprint() -> dict:
    """Bind Classic portfolio reuse to the same validated runtime authorities."""
    return _classic_publication_snapshot()[2]


def _classic_publication_overlay(pool: pd.DataFrame, stage24: pd.DataFrame):
    offense, parent, token, persisted_stage = _classic_publication_snapshot()
    required = {"season", "week", "game_id", "position_identity", "injury_gsis_id",
                "team_solver", "solver_position", "solver_eligible"}
    if not required.issubset(pool.columns):
        raise Stage24AttachmentError("CLASSIC_PUBLICATION_POOL_SCHEMA")
    # The caller may have loaded Stage24 before the updater lock was obtained.
    # Reject a stale snapshot rather than mixing two publication generations.
    if not set(persisted_stage.columns).issubset(stage24.columns) or not stage24[
        persisted_stage.columns
    ].reset_index(drop=True).equals(persisted_stage.reset_index(drop=True)):
        raise Stage24AttachmentError("CLASSIC_PUBLICATION_STAGE24_SNAPSHOT_CHANGED")
    weeks = set(pool[["season", "week"]].itertuples(index=False, name=None))
    if not weeks or weeks != {(parent["season"], parent["week"])}:
        raise Stage24AttachmentError("CLASSIC_PUBLICATION_POOL_WEEK")
    # Retain existing Stage24 duplicate/missing/team checks for all components.
    _stage24_maps(stage24)
    points = offense.set_index("player_id")
    overlay = stage24.copy(deep=True)
    stage_offense = overlay["position"].isin(["QB", "RB", "WR", "TE"])
    if set(overlay.loc[stage_offense, "entity_id"]) != set(points.index):
        raise Stage24AttachmentError("CLASSIC_PUBLICATION_STAGE24_IDENTITY_SET")
    for idx, row in overlay.loc[stage_offense].iterrows():
        published = points.loc[row["entity_id"]]
        if (row["game_id"] != published["game_id"]
                or canon_team(row["team"]) != canon_team(published["team"])):
            raise Stage24AttachmentError("CLASSIC_PUBLICATION_STAGE24_IDENTITY_CONFLICT")
        overlay.at[idx, "expected_fanduel_points"] = published["_classic_points"]
        overlay.at[idx, "_stage24_projection"] = published["_classic_points"]
    eligible_offense = (
        pd.to_numeric(pool["solver_eligible"], errors="coerce").fillna(0).eq(1)
        & pool["solver_position"].isin(["QB", "RB", "WR", "TE"])
    )
    for _, row in pool.loc[eligible_offense].iterrows():
        gsis = clean(row["injury_gsis_id"])
        if gsis not in points.index or row["position_identity"] != "GSIS:" + gsis:
            raise Stage24AttachmentError("CLASSIC_PUBLICATION_POOL_IDENTITY")
        published = points.loc[gsis]
        if (row["game_id"] != published["game_id"]
                or canon_team(row["team_solver"]) != canon_team(published["team"])):
            raise Stage24AttachmentError("CLASSIC_PUBLICATION_POOL_IDENTITY_CONFLICT")
    return overlay, token


def clean(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def canon_team(value) -> str:
    team = clean(value).upper()
    return TEAM_ALIASES.get(team, team)


def finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def load_stage24(path: Path | str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise Stage24AttachmentError(
            f"STAGE24_FILE_MISSING:{path}"
        )

    df = pd.read_parquet(path).copy()

    required = {
        "game_id",
        "team",
        "position",
        "entity_id",
        "entity_name",
        "expected_fanduel_points",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise Stage24AttachmentError(
            "STAGE24_SCHEMA_MISSING:"
            + ",".join(missing)
        )

    df["_stage24_entity_id"] = (
        df["entity_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["_stage24_team"] = df["team"].map(
        canon_team
    )

    df["_stage24_game_id"] = (
        df["game_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["_stage24_position"] = (
        df["position"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    projection = pd.to_numeric(
        df["expected_fanduel_points"],
        errors="coerce",
    )

    if projection.isna().any():
        raise Stage24AttachmentError(
            "STAGE24_NONFINITE_PROJECTION"
        )

    if not np.isfinite(
        projection.to_numpy(dtype=float)
    ).all():
        raise Stage24AttachmentError(
            "STAGE24_NONFINITE_PROJECTION"
        )

    if (projection < 0).any():
        raise Stage24AttachmentError(
            "STAGE24_NEGATIVE_PROJECTION"
        )

    df["_stage24_projection"] = projection.astype(
        float
    )

    if (
        df["_stage24_entity_id"]
        .eq("")
        .any()
    ):
        raise Stage24AttachmentError(
            "STAGE24_BLANK_ENTITY_ID"
        )

    if (
        df["_stage24_game_id"]
        .eq("")
        .any()
    ):
        raise Stage24AttachmentError(
            "STAGE24_BLANK_GAME_ID"
        )

    return df


def _stage24_maps(
    stage24: pd.DataFrame,
):
    is_dst = (
        stage24["_stage24_position"]
        .isin(["DST", "D/ST", "DEF"])
    )

    players = stage24.loc[
        ~is_dst
    ].copy()

    dst = stage24.loc[
        is_dst
    ].copy()

    if players.duplicated(
        ["_stage24_entity_id"],
        keep=False,
    ).any():
        raise Stage24AttachmentError(
            "STAGE24_PLAYER_ID_DUPLICATE"
        )

    if dst.duplicated(
        ["_stage24_team"],
        keep=False,
    ).any():
        raise Stage24AttachmentError(
            "STAGE24_DST_TEAM_DUPLICATE"
        )

    player_projection = {
        clean(r["_stage24_entity_id"]):
        float(r["_stage24_projection"])
        for _, r in players.iterrows()
    }

    player_team = {
        clean(r["_stage24_entity_id"]):
        canon_team(r["_stage24_team"])
        for _, r in players.iterrows()
    }

    player_position = {
        clean(r["_stage24_entity_id"]):
        clean(r["_stage24_position"]).upper()
        for _, r in players.iterrows()
    }

    dst_projection = {
        canon_team(r["_stage24_team"]):
        float(r["_stage24_projection"])
        for _, r in dst.iterrows()
    }

    return (
        player_projection,
        player_team,
        player_position,
        dst_projection,
    )


def attach_classic_stage24(
    pool: pd.DataFrame,
    stage24: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Classic frozen Stage24J contract.

    Player projection identity:
        injury_gsis_id == Stage24 entity_id

    DST projection identity:
        exact canonical team

    Only projection_solver may change. Classic offensive points come from the
    bound current-stat publication; shared Stage24 remains the DST authority.
    """

    required = {
        "team_solver",
        "solver_position",
        "projection_solver",
        "solver_eligible",
        "injury_gsis_id",
    }

    missing = sorted(
        required - set(pool.columns)
    )

    if missing:
        raise Stage24AttachmentError(
            "CLASSIC_SCHEMA_MISSING:"
            + ",".join(missing)
        )

    stage24, publication_token = _classic_publication_overlay(pool, stage24)
    out = pool.copy(deep=True)

    (
        player_projection,
        player_team,
        player_position,
        dst_projection,
    ) = _stage24_maps(stage24)

    eligible = (
        pd.to_numeric(
            out["solver_eligible"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .eq(1)
    )

    eligible_idx = out.index[
        eligible
    ].tolist()

    matched = 0
    player_matches = 0
    dst_matches = 0
    team_mismatch = 0
    position_drift = 0
    missing_rows = []

    for idx in eligible_idx:
        row = out.loc[idx]

        team = canon_team(
            row["team_solver"]
        )

        position = clean(
            row["solver_position"]
        ).upper()

        gsis_id = clean(
            row["injury_gsis_id"]
        )

        is_dst = position in {
            "DST",
            "D/ST",
            "DEF",
        }

        value = None

        if is_dst:
            if team in dst_projection:
                value = dst_projection[team]
                dst_matches += 1
        else:
            if gsis_id in player_projection:
                value = player_projection[
                    gsis_id
                ]

                player_matches += 1

                if (
                    player_team[gsis_id]
                    != team
                ):
                    team_mismatch += 1

                stage_pos = player_position[
                    gsis_id
                ]

                if stage_pos != position:
                    position_drift += 1

        if value is None:
            missing_rows.append(
                {
                    "index": int(idx),
                    "team": team,
                    "solver_position": position,
                    "injury_gsis_id": gsis_id,
                }
            )
            continue

        if not finite(value):
            raise Stage24AttachmentError(
                "CLASSIC_STAGE24_NONFINITE:"
                + gsis_id
            )

        out.at[
            idx,
            "projection_solver",
        ] = float(value)

        matched += 1

    audit = {
        "current_stat_publication": publication_token,
        "eligible_rows": int(
            eligible.sum()
        ),
        "matched_rows": matched,
        "missing_rows": len(
            missing_rows
        ),
        "player_matches": player_matches,
        "dst_matches": dst_matches,
        "team_mismatch_rows": team_mismatch,
        "position_drift_rows": position_drift,
        "missing_detail": missing_rows,
    }

    if missing_rows:
        raise Stage24AttachmentError(
            "CLASSIC_MISSING_STAGE24_PROJECTION:"
            + str(len(missing_rows))
        )

    if team_mismatch:
        raise Stage24AttachmentError(
            "CLASSIC_TEAM_MISMATCH:"
            + str(team_mismatch)
        )

    return out, audit


def _build_stage24_game_map(
    stage24: pd.DataFrame,
):
    game_map = {}

    for game_id, group in stage24.groupby(
        "_stage24_game_id",
        sort=False,
    ):
        teams = sorted(
            set(
                group["_stage24_team"]
                .loc[
                    group[
                        "_stage24_team"
                    ].ne("")
                ]
                .tolist()
            )
        )

        if len(teams) != 2:
            raise Stage24AttachmentError(
                "STAGE24_BAD_GAME_TEAM_COUNT:"
                + clean(game_id)
            )

        pair = tuple(teams)

        if pair in game_map:
            raise Stage24AttachmentError(
                "STAGE24_DUPLICATE_GAME_PAIR:"
                + repr(pair)
            )

        game_map[pair] = clean(
            game_id
        )

    return game_map


def parse_showdown_pair(
    value,
):
    text = clean(value).upper()

    for token in (
        " VS. ",
        " VS ",
        "@",
        "-",
    ):
        if token not in text:
            continue

        parts = [
            canon_team(x)
            for x in text.split(token)
            if clean(x)
        ]

        if len(parts) == 2:
            return tuple(
                sorted(parts)
            )

    return None


def attach_showdown_stage24(
    pool: pd.DataFrame,
    stage24: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Frozen R9/R10 composite Showdown contract.

    Stage24 player identity:
        exact player_id + exact resolved Stage24 game

    Stage24 DST identity:
        exact canonical team + exact resolved Stage24 game

    Stage24-authoritative rows:
        projection
        mvp_projection
        projection_source
        projection_authority
        projection_status

    Independent WFS Showdown authority:
        preserved

    READY_EXTERNAL:
        preserved

    Outside current Stage24 game scope:
        preserved

    Any other current-scope identity:
        fail closed.
    """

    required = {
        "public_slate_name",
        "game",
        "player",
        "player_id",
        "team",
        "position",
        "salary",
        "projection",
        "projection_source",
        "projection_authority",
        "projection_status",
        "mvp_salary",
        "mvp_projection",
    }

    missing = sorted(
        required - set(pool.columns)
    )

    if missing:
        raise Stage24AttachmentError(
            "SHOWDOWN_SCHEMA_MISSING:"
            + ",".join(missing)
        )

    out = pool.copy(deep=True)

    game_map = _build_stage24_game_map(
        stage24
    )

    is_stage_dst = (
        stage24["_stage24_position"]
        .isin(["DST", "D/ST", "DEF"])
    )

    stage_players = stage24.loc[
        ~is_stage_dst
    ].copy()

    stage_dst = stage24.loc[
        is_stage_dst
    ].copy()

    if stage_players.duplicated(
        [
            "_stage24_entity_id",
            "_stage24_game_id",
        ],
        keep=False,
    ).any():
        raise Stage24AttachmentError(
            "SHOWDOWN_STAGE24_PLAYER_KEY_DUPLICATE"
        )

    if stage_dst.duplicated(
        [
            "_stage24_team",
            "_stage24_game_id",
        ],
        keep=False,
    ).any():
        raise Stage24AttachmentError(
            "SHOWDOWN_STAGE24_DST_KEY_DUPLICATE"
        )

    player_map = {
        (
            clean(r["_stage24_entity_id"]),
            clean(r["_stage24_game_id"]),
        ):
        float(r["_stage24_projection"])
        for _, r in stage_players.iterrows()
    }

    dst_map = {
        (
            canon_team(r["_stage24_team"]),
            clean(r["_stage24_game_id"]),
        ):
        float(r["_stage24_projection"])
        for _, r in stage_dst.iterrows()
    }

    stage24_rows = 0
    independent_wfs_rows = 0
    external_rows = 0
    outside_rows = 0
    unresolved = []

    for idx, row in out.iterrows():
        pair = parse_showdown_pair(
            row["game"]
        )

        if pair is None:
            pair = parse_showdown_pair(
                row["public_slate_name"]
            )

        game_id = game_map.get(
            pair
        )

        if not game_id:
            outside_rows += 1
            continue

        pid = clean(
            row["player_id"]
        )

        team = canon_team(
            row["team"]
        )

        position = clean(
            row["position"]
        ).upper()

        is_dst = position in {
            "DST",
            "D/ST",
            "DEF",
        }

        value = None

        if is_dst:
            value = dst_map.get(
                (
                    team,
                    game_id,
                )
            )
        else:
            value = player_map.get(
                (
                    pid,
                    game_id,
                )
            )

        if value is not None:
            if not finite(value):
                raise Stage24AttachmentError(
                    "SHOWDOWN_STAGE24_NONFINITE:"
                    + pid
                )

            out.at[
                idx,
                "projection",
            ] = float(value)

            out.at[
                idx,
                "mvp_projection",
            ] = float(value) * 1.5

            out.at[
                idx,
                "projection_source",
            ] = STAGE24_SOURCE

            out.at[
                idx,
                "projection_authority",
            ] = STAGE24_AUTHORITY

            out.at[
                idx,
                "projection_status",
            ] = STAGE24_STATUS

            stage24_rows += 1
            continue

        source = clean(
            row["projection_source"]
        )

        authority = clean(
            row["projection_authority"]
        )

        status = clean(
            row["projection_status"]
        )

        # Legitimate independent WFS Showdown authority.
        if (
            status == "READY"
            and authority
            == "WFS_SHOWDOWN_MODEL"
        ):
            if (
                not finite(
                    row["projection"]
                )
                or
                not finite(
                    row["mvp_projection"]
                )
            ):
                raise Stage24AttachmentError(
                    "SHOWDOWN_INDEPENDENT_WFS_NONFINITE:"
                    + pid
                )

            independent_wfs_rows += 1
            continue

        # Existing external authority is preserved exactly.
        if (
            status == "READY_EXTERNAL"
            and authority
            == "EXTERNAL_FANDUEL_SOURCE"
        ):
            if (
                not finite(
                    row["projection"]
                )
                or
                not finite(
                    row["mvp_projection"]
                )
            ):
                raise Stage24AttachmentError(
                    "SHOWDOWN_EXTERNAL_NONFINITE:"
                    + pid
                )

            external_rows += 1
            continue

        unresolved.append(
            {
                "index": int(idx),
                "player": clean(
                    row["player"]
                ),
                "player_id": pid,
                "team": team,
                "position": position,
                "game_id": game_id,
                "projection_source": source,
                "projection_authority": authority,
                "projection_status": status,
            }
        )

    if unresolved:
        raise Stage24AttachmentError(
            "SHOWDOWN_UNRESOLVED_CURRENT_SCOPE:"
            + repr(unresolved)
        )

    audit = {
        "total_rows": len(out),
        "stage24_rows": stage24_rows,
        "independent_wfs_rows": independent_wfs_rows,
        "external_rows": external_rows,
        "outside_scope_rows": outside_rows,
        "unresolved_rows": len(
            unresolved
        ),
        "unresolved_detail": unresolved,
    }

    return out, audit
