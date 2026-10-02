from __future__ import annotations

from pathlib import Path
from collections import Counter
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")
DB = ROOT / "data" / "nfl.db"
OUT_DIR = ROOT / "processed"

PLAYER_OUT = OUT_DIR / "forecast_live_player_replacement_quality_v1.csv"
TEAM_OUT = OUT_DIR / "forecast_live_team_replacement_quality_v1.csv"
AUDIT_OUT = OUT_DIR / "forecast_live_replacement_quality_v1_audit.json"

TARGET_SEASONS = (2026,)


# =============================================================================
# EXPLICIT POSITION VOCABULARY
# =============================================================================

# Injury feed -> depth-chart primary football positions.
DEPTH_POSITION_MAP = {
    "QB": {"QB"},
    "RB": {"RB"},
    "FB": {"FB"},
    "WR": {"WR"},
    "TE": {"TE"},
    "T": {"LT", "RT"},
    "G": {"LG", "RG"},
    "C": {"C"},
    "DE": {"LDE", "RDE"},
    "DT": {"LDT", "RDT", "NT"},
    "LB": {"WLB", "SLB", "MLB", "LILB", "RILB"},
    "CB": {"LCB", "RCB", "NB"},
    "S": {"FS", "SS"},
    "K": {"PK"},
    "P": {"P"},
    "LS": {"LS"},
}


# Injury feed -> historical usage family.
#
# Explicit mappings only. No fuzzy matching.
USAGE_POSITION_FAMILY = {
    "QB": {"QB"},
    "RB": {"RB", "HB"},
    "FB": {"FB"},
    "WR": {"WR"},
    "TE": {"TE"},

    "T": {"T", "OT"},
    "G": {"G"},
    "C": {"C"},

    "DE": {"DE"},
    "DT": {"DT", "NT", "DL"},

    "LB": {
        "LB",
        "OLB",
        "ILB",
        "MLB",
    },

    "CB": {"CB", "DB"},

    "S": {
        "S",
        "SAF",
        "FS",
        "SS",
        "DB",
    },

    "K": {"K"},
    "P": {"P"},
    "LS": {"LS"},
}


SPECIAL_TEAM_PRIMARY = {"K", "P", "LS"}


HISTORY_METRICS = [
    "offense_pct",
    "offense_snaps",
    "opportunities",
    "touches",
    "targets",
    "carries",
    "fanduel_points",
    "fanduel_per_snap",
    "fanduel_per_touch",
    "yards_per_opportunity",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def clean_id(value) -> str | None:
    if pd.isna(value):
        return None

    s = str(value).strip()

    if not s:
        return None

    return s


def clean_pos(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    )

    if not values.notna().any():
        return np.nan

    return float(values.mean())


def numeric_last(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.iloc[-1])


def history_tier(prior_games: int) -> str:
    if prior_games >= 5:
        return "FULL"
    if prior_games >= 3:
        return "PARTIAL"
    if prior_games >= 1:
        return "LIMITED"
    return "COLD"


def player_history(
    usage_by_player: dict[str, pd.DataFrame],
    player_id: str | None,
    target_kickoff_ns: int,
) -> dict:
    """
    All history here must be strictly before target kickoff.

    Cross-season history is allowed for quality estimation.
    Target/future games are never used.
    """

    base = {
        "prior_games": 0,
        "history_tier": "COLD",

        "snap_pct_last": np.nan,
        "snap_pct_avg_3": np.nan,
        "snap_pct_avg_5": np.nan,

        "snaps_last": np.nan,
        "snaps_avg_3": np.nan,
        "snaps_avg_5": np.nan,

        "opportunities_last": np.nan,
        "opportunities_avg_3": np.nan,
        "opportunities_avg_5": np.nan,

        "touches_last": np.nan,
        "touches_avg_3": np.nan,
        "touches_avg_5": np.nan,

        "targets_last": np.nan,
        "targets_avg_3": np.nan,
        "targets_avg_5": np.nan,

        "carries_last": np.nan,
        "carries_avg_3": np.nan,
        "carries_avg_5": np.nan,

        "fd_last": np.nan,
        "fd_avg_3": np.nan,
        "fd_avg_5": np.nan,

        "fd_per_snap_avg_3": np.nan,
        "fd_per_touch_avg_3": np.nan,
        "yards_per_opp_avg_3": np.nan,

        "last_prior_game_id": None,
        "last_prior_kickoff_ns": np.nan,
    }

    if player_id is None:
        return base

    frame = usage_by_player.get(player_id)

    if frame is None or frame.empty:
        return base

    hist = frame[
        frame["kickoff_ns"].lt(
            int(target_kickoff_ns)
        )
    ].copy()

    if hist.empty:
        return base

    if hist["kickoff_ns"].ge(
        int(target_kickoff_ns)
    ).any():
        raise RuntimeError(
            "Temporal leakage detected in player history."
        )

    hist = hist.sort_values(
        ["kickoff_ns", "game_id"]
    )

    last3 = hist.tail(3)
    last5 = hist.tail(5)

    n = len(hist)

    base.update({
        "prior_games": int(n),
        "history_tier": history_tier(n),

        "snap_pct_last":
            numeric_last(hist, "offense_pct"),
        "snap_pct_avg_3":
            numeric_mean(last3, "offense_pct"),
        "snap_pct_avg_5":
            numeric_mean(last5, "offense_pct"),

        "snaps_last":
            numeric_last(hist, "offense_snaps"),
        "snaps_avg_3":
            numeric_mean(last3, "offense_snaps"),
        "snaps_avg_5":
            numeric_mean(last5, "offense_snaps"),

        "opportunities_last":
            numeric_last(hist, "opportunities"),
        "opportunities_avg_3":
            numeric_mean(last3, "opportunities"),
        "opportunities_avg_5":
            numeric_mean(last5, "opportunities"),

        "touches_last":
            numeric_last(hist, "touches"),
        "touches_avg_3":
            numeric_mean(last3, "touches"),
        "touches_avg_5":
            numeric_mean(last5, "touches"),

        "targets_last":
            numeric_last(hist, "targets"),
        "targets_avg_3":
            numeric_mean(last3, "targets"),
        "targets_avg_5":
            numeric_mean(last5, "targets"),

        "carries_last":
            numeric_last(hist, "carries"),
        "carries_avg_3":
            numeric_mean(last3, "carries"),
        "carries_avg_5":
            numeric_mean(last5, "carries"),

        "fd_last":
            numeric_last(hist, "fanduel_points"),
        "fd_avg_3":
            numeric_mean(last3, "fanduel_points"),
        "fd_avg_5":
            numeric_mean(last5, "fanduel_points"),

        "fd_per_snap_avg_3":
            numeric_mean(last3, "fanduel_per_snap"),

        "fd_per_touch_avg_3":
            numeric_mean(last3, "fanduel_per_touch"),

        "yards_per_opp_avg_3":
            numeric_mean(
                last3,
                "yards_per_opportunity",
            ),

        "last_prior_game_id":
            str(hist.iloc[-1]["game_id"]),

        "last_prior_kickoff_ns":
            int(hist.iloc[-1]["kickoff_ns"]),
    })

    return base


def depth_candidate_2025(
    target_row,
    depth: pd.DataFrame,
    unavailable_lookup: dict,
) -> dict:
    result = {
        "candidate_source": "UNRESOLVED",
        "candidate_id": None,
        "candidate_name": None,
        "candidate_position": None,
        "candidate_depth_group": None,
        "candidate_depth_slot": np.nan,
        "candidate_depth_rank": np.nan,

        "injured_depth_found": 0,
        "primary_role_found": 0,
        "candidate_found": 0,
        "skipped_unavailable": 0,
    }

    if pd.isna(target_row.snapshot_dt):
        result["candidate_source"] = (
            "UNRESOLVED_NO_PREGAME_DEPTH"
        )
        return result

    chart = depth[
        depth["snapshot_dt"].eq(
            target_row.snapshot_dt
        )
        &
        depth["team"].eq(
            target_row.team
        )
    ].copy()

    injured_rows = chart[
        chart["gsis_id"].eq(
            target_row.gsis_id
        )
    ].copy()

    if injured_rows.empty:
        result["candidate_source"] = (
            "UNRESOLVED_INJURED_NOT_IN_DEPTH"
        )
        return result

    result["injured_depth_found"] = 1

    injury_pos = clean_pos(
        target_row.position
    )

    allowed = DEPTH_POSITION_MAP.get(
        injury_pos,
        set(),
    )

    if injury_pos in SPECIAL_TEAM_PRIMARY:
        primary = injured_rows[
            injured_rows["pos_abb"].isin(
                allowed
            )
            &
            injured_rows["pos_grp"].eq(
                "Special Teams"
            )
        ].copy()

    else:
        primary = injured_rows[
            injured_rows["pos_abb"].isin(
                allowed
            )
            &
            injured_rows["pos_grp"].ne(
                "Special Teams"
            )
        ].copy()

    if primary.empty:
        result["candidate_source"] = (
            "UNRESOLVED_PRIMARY_ROLE"
        )
        return result

    result["primary_role_found"] = 1

    blocked = unavailable_lookup.get(
        (
            int(target_row.season),
            int(target_row.week),
            str(target_row.team),
        ),
        set(),
    )

    possibilities = []

    for ir in primary.itertuples(index=False):

        injured_rank = pd.to_numeric(
            pd.Series([ir.pos_rank]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(injured_rank):
            continue

        lane = chart[
            chart["pos_grp"].eq(
                ir.pos_grp
            )
            &
            chart["pos_abb"].eq(
                ir.pos_abb
            )
            &
            chart["pos_slot"].eq(
                ir.pos_slot
            )
            &
            chart["gsis_id"].ne(
                target_row.gsis_id
            )
        ].copy()

        lane["rank_num"] = pd.to_numeric(
            lane["pos_rank"],
            errors="coerce",
        )

        lane = lane[
            lane["rank_num"].gt(
                injured_rank
            )
        ].sort_values(
            [
                "rank_num",
                "player_name",
                "gsis_id",
            ]
        )

        skipped = 0

        for candidate in lane.itertuples(
            index=False
        ):
            cid = clean_id(
                candidate.gsis_id
            )

            if cid is None:
                continue

            if cid in blocked:
                skipped += 1
                continue

            possibilities.append({
                "rank_gap":
                    float(
                        candidate.rank_num
                        - injured_rank
                    ),
                "candidate": candidate,
                "skipped": skipped,
            })

            # First available player in this
            # exact football depth lane.
            break

    if not possibilities:
        result["candidate_source"] = (
            "UNRESOLVED_NO_AVAILABLE_LOWER_DEPTH"
        )
        return result

    possibilities.sort(
        key=lambda x: (
            x["rank_gap"],
            int(x["candidate"].pos_slot),
            float(x["candidate"].rank_num),
            str(x["candidate"].gsis_id),
        )
    )

    best = possibilities[0]
    c = best["candidate"]

    result.update({
        "candidate_source": "DEPTH_2025",
        "candidate_id":
            clean_id(c.gsis_id),
        "candidate_name":
            c.player_name,
        "candidate_position":
            c.pos_abb,
        "candidate_depth_group":
            c.pos_grp,
        "candidate_depth_slot":
            c.pos_slot,
        "candidate_depth_rank":
            c.pos_rank,

        "candidate_found": 1,
        "skipped_unavailable":
            int(best["skipped"]),
    })

    return result


def fallback_candidate_pre2025(
    target_row,
    usage: pd.DataFrame,
    usage_by_player: dict[str, pd.DataFrame],
    unavailable_lookup: dict,
) -> dict:
    """
    Leakage-safe historical fallback.

    Important:
    - No target-week realized player population is used.
    - Candidate roster evidence must come from a strictly prior
      game for the SAME TEAM in the SAME SEASON.
    - This intentionally sacrifices coverage in early-season
      and roster-change situations rather than inventing
      pregame roster knowledge that did not exist historically.
    """

    result = {
        "candidate_source": "UNRESOLVED",
        "candidate_id": None,
        "candidate_name": None,
        "candidate_position": None,
        "candidate_depth_group": None,
        "candidate_depth_slot": np.nan,
        "candidate_depth_rank": np.nan,

        "injured_depth_found": 0,
        "primary_role_found": 0,
        "candidate_found": 0,
        "skipped_unavailable": 0,
    }

    injury_pos = clean_pos(
        target_row.position
    )

    allowed_positions = (
        USAGE_POSITION_FAMILY.get(
            injury_pos,
            set(),
        )
    )

    if not allowed_positions:
        result["candidate_source"] = (
            "UNRESOLVED_POSITION_FAMILY"
        )
        return result

    blocked = unavailable_lookup.get(
        (
            int(target_row.season),
            int(target_row.week),
            str(target_row.team),
        ),
        set(),
    )

    # Current-season, same-team, strictly prior rows only.
    #
    # We do not use previous-season team membership as roster
    # evidence because the player may have changed teams.
    pool_rows = usage[
        usage["season"].eq(
            int(target_row.season)
        )
        &
        usage["team"].eq(
            target_row.team
        )
        &
        usage["kickoff_ns"].lt(
            int(target_row.kickoff_ns)
        )
        &
        usage["position_clean"].isin(
            allowed_positions
        )
    ].copy()

    if pool_rows.empty:
        result["candidate_source"] = (
            "UNRESOLVED_NO_PRIOR_TEAM_POSITION"
        )
        return result

    injured_id = clean_id(
        target_row.gsis_id
    )

    candidates = []

    for candidate_id, grp in pool_rows.groupby(
        "player_id_clean",
        sort=True,
    ):
        cid = clean_id(candidate_id)

        if cid is None:
            continue

        if cid == injured_id:
            continue

        if cid in blocked:
            continue

        # Candidate must have at least one prior appearance
        # for this team in this season. That membership evidence
        # comes only from games before target kickoff.
        grp = grp.sort_values(
            ["kickoff_ns", "game_id"]
        )

        if grp.empty:
            continue

        hist = player_history(
            usage_by_player,
            cid,
            int(target_row.kickoff_ns),
        )

        last3_team = grp.tail(3)

        snap3 = numeric_mean(
            last3_team,
            "offense_pct",
        )

        opp3 = numeric_mean(
            last3_team,
            "opportunities",
        )

        fd3 = numeric_mean(
            last3_team,
            "fanduel_points",
        )

        # No weighted score.
        #
        # Deterministic lexicographic role ordering:
        # 1. more recent same-team appearance
        # 2. higher recent snap %
        # 3. higher recent opportunity
        # 4. higher recent FD production
        # 5. more prior NFL games
        # 6. stable GSIS ID tiebreak
        #
        # NaN becomes -inf so observed evidence wins.
        candidates.append({
            "candidate_id": cid,
            "candidate_name":
                grp.iloc[-1]["player_name"],
            "candidate_position":
                grp.iloc[-1]["position"],

            "last_team_kickoff_ns":
                int(
                    grp.iloc[-1]["kickoff_ns"]
                ),

            "snap3":
                float(snap3)
                if pd.notna(snap3)
                else -np.inf,

            "opp3":
                float(opp3)
                if pd.notna(opp3)
                else -np.inf,

            "fd3":
                float(fd3)
                if pd.notna(fd3)
                else -np.inf,

            "prior_games":
                int(hist["prior_games"]),
        })

    if not candidates:
        result["candidate_source"] = (
            "UNRESOLVED_NO_AVAILABLE_PRIOR_ROLE"
        )
        return result

    candidates.sort(
        key=lambda x: (
            -x["last_team_kickoff_ns"],
            -x["snap3"],
            -x["opp3"],
            -x["fd3"],
            -x["prior_games"],
            x["candidate_id"],
        )
    )

    best = candidates[0]

    result.update({
        "candidate_source":
            "PRIOR_TEAM_ROLE_FALLBACK",

        "candidate_id":
            best["candidate_id"],

        "candidate_name":
            best["candidate_name"],

        "candidate_position":
            best["candidate_position"],

        "candidate_found": 1,
    })

    return result


def prefix_history(
    prefix: str,
    hist: dict,
) -> dict:
    return {
        f"{prefix}_{k}": v
        for k, v in hist.items()
    }


def safe_delta(candidate, injured):
    c = pd.to_numeric(
        pd.Series([candidate]),
        errors="coerce",
    ).iloc[0]

    i = pd.to_numeric(
        pd.Series([injured]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(c) or pd.isna(i):
        return np.nan

    return float(c - i)


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 120)
    print("WFS NFL FORECAST V1 — BUILD REPLACEMENT QUALITY")
    print("=" * 120)

    print(f"Database : {DB}")
    print(f"Player   : {PLAYER_OUT}")
    print(f"Team     : {TEAM_OUT}")
    print(f"Audit    : {AUDIT_OUT}")
    print()
    print("SQLite mode: READ ONLY")

    con = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True,
    )

    try:

        # =====================================================================
        # DATABASE SAFETY
        # =====================================================================

        integrity = con.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        fk = con.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        if integrity != "ok":
            raise RuntimeError(
                f"Database integrity FAIL: {integrity}"
            )

        if fk:
            raise RuntimeError(
                f"Foreign key check FAIL: {len(fk)}"
            )

        print("DB integrity : PASS")
        print("Foreign keys : PASS")

        # =====================================================================
        # GAMES
        # =====================================================================

        games = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                game_date,
                gametime,
                home_team,
                away_team,
                completed
            FROM games
            WHERE season BETWEEN 2023 AND 2026
            """,
            con,
        )

        games["kickoff_ts"] = pd.to_datetime(
            games["game_date"].astype(str)
            + " "
            + games["gametime"]
                .fillna("23:59:59")
                .astype(str),
            utc=True,
            errors="coerce",
        )

        if games["kickoff_ts"].isna().any():
            raise RuntimeError(
                "Invalid game kickoff timestamp."
            )

        games["kickoff_ns"] = (
            games["kickoff_ts"].astype("int64")
        )

        if len(games) != 1127:
            raise RuntimeError(
                f"Expected 1127 chronology games, "
                f"found {len(games)}"
            )

        game_time = games[
            [
                "game_id",
                "kickoff_ts",
                "kickoff_ns",
            ]
        ].drop_duplicates()

        team_games_rows = []

        live_games = games[
            games["season"].eq(2026)
        ].copy()

        if len(live_games) != 272:
            raise RuntimeError(
                "Expected 272 live 2026 games, "
                f"found {len(live_games)}"
            )

        for r in live_games.itertuples(index=False):

            team_games_rows.extend([
                {
                    "game_id": r.game_id,
                    "season": int(r.season),
                    "week": int(r.week),
                    "team": r.home_team,
                    "opponent": r.away_team,
                    "kickoff_ts": r.kickoff_ts,
                    "kickoff_ns": int(r.kickoff_ns),
                },
                {
                    "game_id": r.game_id,
                    "season": int(r.season),
                    "week": int(r.week),
                    "team": r.away_team,
                    "opponent": r.home_team,
                    "kickoff_ts": r.kickoff_ts,
                    "kickoff_ns": int(r.kickoff_ns),
                },
            ])

        team_games = pd.DataFrame(
            team_games_rows
        )

        if len(team_games) != 544:
            raise RuntimeError(
                f"Expected 544 team-games, "
                f"found {len(team_games)}"
            )

        # =====================================================================
        # INJURIES
        # =====================================================================

        injuries = pd.read_sql_query(
            """
            SELECT
                season,
                week,
                team,
                gsis_id,
                full_name,
                position,
                report_status,
                practice_status,
                report_primary_injury
            FROM injuries
            WHERE season = 2026
              AND UPPER(
                    TRIM(
                        COALESCE(report_status, '')
                    )
                  )
                  IN ('OUT', 'DOUBTFUL')
            """,
            con,
        )

        injuries["gsis_id"] = (
            injuries["gsis_id"]
            .map(clean_id)
        )

        targets = injuries.merge(
            team_games,
            on=[
                "season",
                "week",
                "team",
            ],
            how="left",
            validate="many_to_one",
        )

        if targets["game_id"].isna().any():
            raise RuntimeError(
                "Injury -> game mapping incomplete."
            )

        if targets.empty:
            raise RuntimeError(
                f"Expected nonempty 2026 OUT/DOUBTFUL rows, "
                f"found {len(targets)}"
            )

        unavailable_lookup = {}

        for key, grp in targets.groupby(
            ["season", "week", "team"]
        ):
            unavailable_lookup[key] = set(
                x
                for x in grp["gsis_id"].tolist()
                if x is not None
            )

        # =====================================================================
        # USAGE / HISTORY
        # =====================================================================

        usage = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                player_id,
                player_name,
                position,
                team,
                fanduel_points,
                offense_snaps,
                offense_pct,
                carries,
                targets,
                touches,
                opportunities,
                target_share,
                air_yards_share,
                wopr,
                fanduel_per_snap,
                fanduel_per_touch,
                yards_per_opportunity
            FROM player_weekly_usage
            WHERE season BETWEEN 2023 AND 2026
            """,
            con,
        )

        usage["player_id_clean"] = (
            usage["player_id"].map(
                clean_id
            )
        )

        usage["position_clean"] = (
            usage["position"].map(
                clean_pos
            )
        )

        usage = usage.merge(
            game_time,
            on="game_id",
            how="left",
            validate="many_to_one",
        )

        if usage["kickoff_ns"].isna().any():
            raise RuntimeError(
                "Usage chronology incomplete."
            )

        usage = usage.sort_values(
            [
                "player_id_clean",
                "kickoff_ns",
                "game_id",
            ]
        )

        usage_by_player = {
            pid: grp.copy()
            for pid, grp in usage[
                usage["player_id_clean"].notna()
            ].groupby(
                "player_id_clean",
                sort=False,
            )
        }

        # =====================================================================
        # DEPTH SNAPSHOTS
        # =====================================================================

        snapshots = pd.read_sql_query(
            """
            SELECT DISTINCT
                snapshot_dt,
                team
            FROM depth_charts
            """,
            con,
        )

        snapshots["snapshot_ts"] = (
            pd.to_datetime(
                snapshots["snapshot_dt"],
                utc=True,
                errors="coerce",
            )
        )

        if snapshots["snapshot_ts"].isna().any():
            raise RuntimeError(
                "Invalid depth snapshot timestamp."
            )

        snapshots["snapshot_ns"] = (
            snapshots["snapshot_ts"]
            .astype("int64")
        )

        valid_snapshot_rows = []

        for team, gt in team_games.groupby(
            "team",
            sort=True,
        ):
            s = snapshots[
                snapshots["team"].eq(team)
            ].sort_values(
                "snapshot_ns"
            )

            sns = s["snapshot_ns"].to_numpy(
                dtype=np.int64
            )

            sdt = s["snapshot_dt"].to_numpy()

            for r in gt.itertuples(index=False):

                idx = (
                    np.searchsorted(
                        sns,
                        int(r.kickoff_ns),
                        side="left",
                    )
                    - 1
                )

                chosen = (
                    sdt[idx]
                    if idx >= 0
                    else None
                )

                valid_snapshot_rows.append({
                    "game_id": r.game_id,
                    "team": r.team,
                    "snapshot_dt": chosen,
                })

        valid_snapshots = pd.DataFrame(
            valid_snapshot_rows
        )

        depth = pd.read_sql_query(
            """
            SELECT
                snapshot_dt,
                team,
                player_name,
                gsis_id,
                pos_grp,
                pos_name,
                pos_abb,
                pos_slot,
                pos_rank
            FROM depth_charts
            """,
            con,
        )

        depth["gsis_id"] = (
            depth["gsis_id"]
            .map(clean_id)
        )

        # Attach selected historical snapshot.
        targets = targets.merge(
            valid_snapshots,
            on=[
                "game_id",
                "team",
            ],
            how="left",
            validate="many_to_one",
        )

        # =====================================================================
        # PLAYER-LEVEL BUILD
        # =====================================================================

        print()
        print("Building player-level replacement rows...")

        player_rows = []

        leakage_violations = 0

        for r in targets.sort_values(
            [
                "season",
                "week",
                "game_id",
                "team",
                "gsis_id",
            ]
        ).itertuples(index=False):

            injured_id = clean_id(
                r.gsis_id
            )

            injured_hist = player_history(
                usage_by_player,
                injured_id,
                int(r.kickoff_ns),
            )

            if int(r.season) >= 2025:
                candidate = depth_candidate_2025(
                    r,
                    depth,
                    unavailable_lookup,
                )
            else:
                candidate = fallback_candidate_pre2025(
                    r,
                    usage,
                    usage_by_player,
                    unavailable_lookup,
                )

            candidate_id = clean_id(
                candidate["candidate_id"]
            )

            candidate_hist = player_history(
                usage_by_player,
                candidate_id,
                int(r.kickoff_ns),
            )

            for h in (
                injured_hist,
                candidate_hist,
            ):
                last_ns = h[
                    "last_prior_kickoff_ns"
                ]

                if (
                    pd.notna(last_ns)
                    and
                    int(last_ns)
                    >= int(r.kickoff_ns)
                ):
                    leakage_violations += 1

            row = {
                "game_id": r.game_id,
                "season": int(r.season),
                "week": int(r.week),
                "team": r.team,
                "opponent": r.opponent,

                "injured_gsis_id":
                    injured_id,

                "injured_name":
                    r.full_name,

                "injury_position":
                    clean_pos(r.position),

                "report_status":
                    r.report_status,

                "practice_status":
                    r.practice_status,

                "primary_injury":
                    r.report_primary_injury,

                "candidate_source":
                    candidate[
                        "candidate_source"
                    ],

                "candidate_found":
                    int(
                        candidate[
                            "candidate_found"
                        ]
                    ),

                "candidate_gsis_id":
                    candidate_id,

                "candidate_name":
                    candidate[
                        "candidate_name"
                    ],

                "candidate_position":
                    candidate[
                        "candidate_position"
                    ],

                "candidate_depth_group":
                    candidate[
                        "candidate_depth_group"
                    ],

                "candidate_depth_slot":
                    candidate[
                        "candidate_depth_slot"
                    ],

                "candidate_depth_rank":
                    candidate[
                        "candidate_depth_rank"
                    ],

                "injured_depth_found":
                    int(
                        candidate[
                            "injured_depth_found"
                        ]
                    ),

                "primary_role_found":
                    int(
                        candidate[
                            "primary_role_found"
                        ]
                    ),

                "skipped_unavailable":
                    int(
                        candidate[
                            "skipped_unavailable"
                        ]
                    ),
            }

            row.update(
                prefix_history(
                    "injured",
                    injured_hist,
                )
            )

            row.update(
                prefix_history(
                    "replacement",
                    candidate_hist,
                )
            )

            # Raw replacement-vs-injured deltas.
            #
            # These are not weights and not point penalties.
            for metric in [
                "snap_pct_avg_3",
                "snaps_avg_3",
                "opportunities_avg_3",
                "touches_avg_3",
                "targets_avg_3",
                "carries_avg_3",
                "fd_avg_3",
                "fd_avg_5",
                "fd_per_snap_avg_3",
                "fd_per_touch_avg_3",
                "yards_per_opp_avg_3",
            ]:
                row[
                    f"replacement_minus_injured_{metric}"
                ] = safe_delta(
                    candidate_hist[
                        metric
                    ],
                    injured_hist[
                        metric
                    ],
                )

            row["replacement_history_known"] = int(
                candidate_hist[
                    "prior_games"
                ] >= 1
            )

            row["replacement_cold_start"] = int(
                candidate[
                    "candidate_found"
                ] == 1
                and
                candidate_hist[
                    "prior_games"
                ] == 0
            )

            row["replacement_full_history"] = int(
                candidate_hist[
                    "prior_games"
                ] >= 5
            )

            player_rows.append(row)

        player_df = pd.DataFrame(
            player_rows
        )

        if leakage_violations:
            raise RuntimeError(
                f"Temporal leakage violations: "
                f"{leakage_violations}"
            )

        if len(player_df) != len(targets):
            raise RuntimeError(
                f"Expected {len(targets)} player replacement rows, "
                f"found {len(player_df)}"
            )

        # =====================================================================
        # TEAM AGGREGATES
        # =====================================================================

        print("Building team-level replacement aggregates...")

        grouped_rows = []

        delta_metrics = [
            "snap_pct_avg_3",
            "snaps_avg_3",
            "opportunities_avg_3",
            "touches_avg_3",
            "targets_avg_3",
            "carries_avg_3",
            "fd_avg_3",
            "fd_avg_5",
            "fd_per_snap_avg_3",
            "fd_per_touch_avg_3",
            "yards_per_opp_avg_3",
        ]

        for (
            game_id,
            season,
            week,
            team,
            opponent,
        ), grp in player_df.groupby(
            [
                "game_id",
                "season",
                "week",
                "team",
                "opponent",
            ],
            sort=False,
        ):
            record = {
                "game_id": game_id,
                "season": int(season),
                "week": int(week),
                "team": team,
                "opponent": opponent,

                "replacement_injury_count":
                    int(len(grp)),

                "replacement_candidate_count":
                    int(
                        grp[
                            "candidate_found"
                        ].sum()
                    ),

                "replacement_unresolved_count":
                    int(
                        (
                            grp[
                                "candidate_found"
                            ]
                            == 0
                        ).sum()
                    ),

                "replacement_history_known_count":
                    int(
                        grp[
                            "replacement_history_known"
                        ].sum()
                    ),

                "replacement_cold_start_count":
                    int(
                        grp[
                            "replacement_cold_start"
                        ].sum()
                    ),

                "replacement_full_history_count":
                    int(
                        grp[
                            "replacement_full_history"
                        ].sum()
                    ),

                "replacement_injury_skip_count":
                    int(
                        (
                            grp[
                                "skipped_unavailable"
                            ]
                            > 0
                        ).sum()
                    ),

                "replacement_depth_source_count":
                    int(
                        (
                            grp[
                                "candidate_source"
                            ]
                            == "DEPTH_2025"
                        ).sum()
                    ),

                "replacement_prior_role_source_count":
                    int(
                        (
                            grp[
                                "candidate_source"
                            ]
                            == "PRIOR_TEAM_ROLE_FALLBACK"
                        ).sum()
                    ),

                "replacement_qb_injury_count":
                    int(
                        (
                            grp[
                                "injury_position"
                            ]
                            == "QB"
                        ).sum()
                    ),

                "replacement_qb_candidate_count":
                    int(
                        (
                            (
                                grp[
                                    "injury_position"
                                ]
                                == "QB"
                            )
                            &
                            (
                                grp[
                                    "candidate_found"
                                ]
                                == 1
                            )
                        ).sum()
                    ),
            }

            for metric in delta_metrics:
                col = (
                    "replacement_minus_injured_"
                    + metric
                )

                values = pd.to_numeric(
                    grp[col],
                    errors="coerce",
                )

                # Injury replacement is modeled relative to the
                # healthy/no-injury baseline. A replacement may
                # mitigate an injury cost to zero, but historical
                # replacement usage must not turn the injury event
                # itself into a positive team benefit.
                #
                # Preserve the raw player-level deltas for audit
                # and diagnostics; constrain only the model-facing
                # team aggregate.
                model_values = values.clip(upper=0.0)

                record[
                    f"replacement_delta_{metric}_sum"
                ] = (
                    float(model_values.sum())
                    if model_values.notna().any()
                    else np.nan
                )

                record[
                    f"replacement_delta_{metric}_mean"
                ] = (
                    float(model_values.mean())
                    if model_values.notna().any()
                    else np.nan
                )

            grouped_rows.append(
                record
            )

        injury_team = pd.DataFrame(
            grouped_rows
        )

        team_df = team_games[
            [
                "game_id",
                "season",
                "week",
                "team",
                "opponent",
            ]
        ].copy()

        team_df = team_df.merge(
            injury_team,
            on=[
                "game_id",
                "season",
                "week",
                "team",
                "opponent",
            ],
            how="left",
            validate="one_to_one",
        )

        count_cols = [
            c for c in team_df.columns
            if c.endswith("_count")
        ]

        for c in count_cols:
            team_df[c] = (
                pd.to_numeric(
                    team_df[c],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )

        if len(team_df) != len(team_games):
            raise RuntimeError(
                f"Expected {len(team_games)} team rows, "
                f"found {len(team_df)}"
            )

        if team_df.duplicated(
            ["game_id", "team"]
        ).any():
            raise RuntimeError(
                "Duplicate game/team in team output."
            )

        two_teams = (
            team_df.groupby("game_id")[
                "team"
            ]
            .nunique()
        )

        if not two_teams.eq(2).all():
            raise RuntimeError(
                "Every game must contain exactly "
                "two team rows."
            )

        # =====================================================================
        # AUDIT
        # =====================================================================

        source_counts = Counter(
            player_df[
                "candidate_source"
            ].fillna("NULL")
        )

        history_counts = Counter(
            player_df.loc[
                player_df["candidate_found"].eq(1),
                "replacement_history_tier",
            ].fillna("NULL")
        )

        by_season = {}

        for season, grp in player_df.groupby(
            "season"
        ):
            by_season[str(int(season))] = {
                "injury_rows":
                    int(len(grp)),

                "candidate_found":
                    int(
                        grp[
                            "candidate_found"
                        ].sum()
                    ),

                "unresolved":
                    int(
                        (
                            grp[
                                "candidate_found"
                            ]
                            == 0
                        ).sum()
                    ),

                "history_known":
                    int(
                        grp[
                            "replacement_history_known"
                        ].sum()
                    ),

                "cold_start":
                    int(
                        grp[
                            "replacement_cold_start"
                        ].sum()
                    ),
            }

        by_position = {}

        for pos, grp in player_df.groupby(
            "injury_position"
        ):
            by_position[str(pos)] = {
                "injury_rows":
                    int(len(grp)),

                "candidate_found":
                    int(
                        grp[
                            "candidate_found"
                        ].sum()
                    ),

                "history_known":
                    int(
                        grp[
                            "replacement_history_known"
                        ].sum()
                    ),

                "cold_start":
                    int(
                        grp[
                            "replacement_cold_start"
                        ].sum()
                    ),
            }

        audit = {
            "status": "PASS",

            "database": str(DB),

            "database_integrity": integrity,

            "foreign_keys": "PASS",

            "seasons": list(
                TARGET_SEASONS
            ),

            "physical_games":
                int(len(games)),

            "team_game_rows":
                int(len(team_df)),

            "injury_rows":
                int(len(player_df)),

            "candidate_found":
                int(
                    player_df[
                        "candidate_found"
                    ].sum()
                ),

            "candidate_unresolved":
                int(
                    (
                        player_df[
                            "candidate_found"
                        ]
                        == 0
                    ).sum()
                ),

            "candidate_source_counts":
                dict(
                    sorted(
                        source_counts.items()
                    )
                ),

            "replacement_history_tiers":
                dict(
                    sorted(
                        history_counts.items()
                    )
                ),

            "replacement_history_known":
                int(
                    player_df[
                        "replacement_history_known"
                    ].sum()
                ),

            "replacement_cold_start":
                int(
                    player_df[
                        "replacement_cold_start"
                    ].sum()
                ),

            "replacement_full_history":
                int(
                    player_df[
                        "replacement_full_history"
                    ].sum()
                ),

            "injury_cascade_skips":
                int(
                    (
                        player_df[
                            "skipped_unavailable"
                        ]
                        > 0
                    ).sum()
                ),

            "temporal_leakage_violations":
                int(leakage_violations),

            "by_season":
                by_season,

            "by_injury_position":
                by_position,

            "rules": {
                "identity":
                    "Exact GSIS IDs only",

                "fuzzy_matching":
                    False,

                "depth_2025":
                    (
                        "Latest snapshot strictly "
                        "before kickoff"
                    ),

                "depth_candidate":
                    (
                        "Same primary football "
                        "pos_grp + pos_abb + pos_slot; "
                        "first available greater pos_rank"
                    ),

                "availability":
                    (
                        "OUT and DOUBTFUL excluded "
                        "from replacement candidates"
                    ),

                "pre2025_fallback":
                    (
                        "Same-season, same-team, "
                        "strictly prior usage only"
                    ),

                "history":
                    (
                        "All candidate/injured history "
                        "strictly before target kickoff"
                    ),

                "questionable":
                    "Not treated as unavailable",

                "arbitrary_point_penalty":
                    False,

                "sqlite_writes":
                    False,
            },
        }

        # =====================================================================
        # WRITE FILES
        # =====================================================================

        player_df = player_df.sort_values(
            [
                "season",
                "week",
                "game_id",
                "team",
                "injured_gsis_id",
            ]
        ).reset_index(drop=True)

        team_df = team_df.sort_values(
            [
                "season",
                "week",
                "game_id",
                "team",
            ]
        ).reset_index(drop=True)

        player_df.to_csv(
            PLAYER_OUT,
            index=False,
        )

        team_df.to_csv(
            TEAM_OUT,
            index=False,
        )

        audit["outputs"] = {
            "player_csv":
                str(PLAYER_OUT),

            "team_csv":
                str(TEAM_OUT),

            "player_sha256":
                sha256_file(
                    PLAYER_OUT
                ),

            "team_sha256":
                sha256_file(
                    TEAM_OUT
                ),
        }

        with AUDIT_OUT.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                audit,
                f,
                indent=2,
                sort_keys=True,
            )

        # =====================================================================
        # CONSOLE REPORT
        # =====================================================================

        print()
        print("=" * 120)
        print("REPLACEMENT QUALITY V1 BUILD COMPLETE")
        print("=" * 120)

        print(
            f"Player rows       : "
            f"{len(player_df)}"
        )

        print(
            f"Team rows         : "
            f"{len(team_df)}"
        )

        print(
            f"Candidates found  : "
            f"{audit['candidate_found']}"
        )

        print(
            f"Unresolved        : "
            f"{audit['candidate_unresolved']}"
        )

        print(
            f"History known     : "
            f"{audit['replacement_history_known']}"
        )

        print(
            f"Cold replacements : "
            f"{audit['replacement_cold_start']}"
        )

        print(
            f"Cascade skips     : "
            f"{audit['injury_cascade_skips']}"
        )

        print(
            f"Leakage violations: "
            f"{audit['temporal_leakage_violations']}"
        )

        print()
        print("Candidate sources:")

        for key, value in sorted(
            source_counts.items()
        ):
            print(
                f"  {key:<40} {value}"
            )

        print()
        print("Replacement history tiers:")

        for key, value in sorted(
            history_counts.items()
        ):
            print(
                f"  {key:<12} {value}"
            )

        print()
        print("By season:")

        print(
            pd.DataFrame(
                audit["by_season"]
            ).T.to_string()
        )

        print()
        print("Outputs:")
        print(f"  {PLAYER_OUT}")
        print(f"  {TEAM_OUT}")
        print(f"  {AUDIT_OUT}")

        print()
        print("SHA256:")
        print(
            f"  player: "
            f"{audit['outputs']['player_sha256']}"
        )
        print(
            f"  team  : "
            f"{audit['outputs']['team_sha256']}"
        )

        print()
        print(
            "PASS | exact identity"
        )

        print(
            "PASS | deterministic candidate selection"
        )

        print(
            "PASS | target/future games excluded from history"
        )

        print(
            "PASS | SQLite read only"
        )

        print("=" * 120)

    finally:
        con.close()


if __name__ == "__main__":
    main()
