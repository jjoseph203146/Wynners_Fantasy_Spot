"""Fantasy League Hub presentation page.

Behavior-preserving extraction from app.py.

Sleeper/ESPN application helpers remain owned by app.py and are
injected here. This module does not alter projections, scoring,
solver authority, eligibility, Stage24, NFL Live, or data authority.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st


REPO_ROOT = None
DATABASE_PATH = None

_SLEEPER_LEAGUE = None
_SLEEPER_LEAGUE_MATCHUPS = None
_SLEEPER_LEAGUE_ROSTERS = None
_SLEEPER_LEAGUE_USERS = None
_SLEEPER_ROSTER_FOR_USER = None
_SLEEPER_USER = None
_SLEEPER_USER_LEAGUES = None
_WFS_SLEEPER_CONNECTIONS = None
_WFS_SLEEPER_DELETE_CONNECTION = None
_WFS_SLEEPER_UPSERT_CONNECTION = None
_SLEEPER_PLAYERS = None
_RENDER_ESPN_LEAGUE_HUB = None


def configure_fantasy_league_hub(
    *,
    repo_root,
    database_path,
    sleeper_league_fn,
    sleeper_league_matchups_fn,
    sleeper_league_rosters_fn,
    sleeper_league_users_fn,
    sleeper_roster_for_user_fn,
    sleeper_user_fn,
    sleeper_user_leagues_fn,
    wfs_sleeper_connections_fn,
    wfs_sleeper_delete_connection_fn,
    wfs_sleeper_upsert_connection_fn,
    sleeper_players_fn,
    render_espn_league_hub_fn,
):
    global REPO_ROOT
    global DATABASE_PATH
    global _SLEEPER_LEAGUE
    global _SLEEPER_LEAGUE_MATCHUPS
    global _SLEEPER_LEAGUE_ROSTERS
    global _SLEEPER_LEAGUE_USERS
    global _SLEEPER_ROSTER_FOR_USER
    global _SLEEPER_USER
    global _SLEEPER_USER_LEAGUES
    global _WFS_SLEEPER_CONNECTIONS
    global _WFS_SLEEPER_DELETE_CONNECTION
    global _WFS_SLEEPER_UPSERT_CONNECTION
    global _SLEEPER_PLAYERS
    global _RENDER_ESPN_LEAGUE_HUB

    REPO_ROOT = Path(repo_root).resolve()
    DATABASE_PATH = database_path

    _SLEEPER_LEAGUE = sleeper_league_fn
    _SLEEPER_LEAGUE_MATCHUPS = sleeper_league_matchups_fn
    _SLEEPER_LEAGUE_ROSTERS = sleeper_league_rosters_fn
    _SLEEPER_LEAGUE_USERS = sleeper_league_users_fn
    _SLEEPER_ROSTER_FOR_USER = sleeper_roster_for_user_fn
    _SLEEPER_USER = sleeper_user_fn
    _SLEEPER_USER_LEAGUES = sleeper_user_leagues_fn
    _WFS_SLEEPER_CONNECTIONS = wfs_sleeper_connections_fn
    _WFS_SLEEPER_DELETE_CONNECTION = wfs_sleeper_delete_connection_fn
    _WFS_SLEEPER_UPSERT_CONNECTION = wfs_sleeper_upsert_connection_fn
    _SLEEPER_PLAYERS = sleeper_players_fn
    _RENDER_ESPN_LEAGUE_HUB = render_espn_league_hub_fn


def render_sleeper_league_views(saved, season, week):
    """
    Provider-native Sleeper league views.

    Matchup, standings and roster data come directly from Sleeper.
    No ESPN-shaped translation and no fuzzy WFS identity resolution.
    """
    league_id = str(saved.get("league_id") or "").strip()

    try:
        roster_id = int(saved.get("team_id"))
    except (TypeError, ValueError):
        st.info("This Sleeper team does not have a valid roster assignment.")
        return

    if not league_id:
        return

    try:
        league = _SLEEPER_LEAGUE(league_id)
        users = _SLEEPER_LEAGUE_USERS(league_id)
        rosters = _SLEEPER_LEAGUE_ROSTERS(league_id)
        matchups = _SLEEPER_LEAGUE_MATCHUPS(league_id, week)
        players = _SLEEPER_PLAYERS()
    except Exception:
        st.info(
            "Sleeper league details are temporarily unavailable. "
            "Please try again shortly."
        )
        return

    if not league or not rosters:
        st.info("Sleeper league details are currently unavailable.")
        return

    roster_by_id = {}
    for roster in rosters:
        if not isinstance(roster, dict):
            continue
        try:
            rid = int(roster.get("roster_id"))
        except (TypeError, ValueError):
            continue
        roster_by_id[rid] = roster

    my_roster = roster_by_id.get(roster_id)
    if my_roster is None:
        st.info(
            "Your saved Sleeper roster could not be verified in this league."
        )
        return

    user_by_id = {
        str(user.get("user_id")): user
        for user in users
        if isinstance(user, dict) and user.get("user_id") is not None
    }

    def team_name_for(roster):
        owner_id = str(roster.get("owner_id") or "")
        user = user_by_id.get(owner_id, {})
        metadata = user.get("metadata") or {}

        return (
            str(metadata.get("team_name") or "").strip()
            or str(user.get("display_name") or "").strip()
            or f"Roster {roster.get('roster_id')}"
        )

    def player_info(player_id):
        pid = str(player_id or "")
        player = players.get(pid) or {}

        name = (
            str(player.get("full_name") or "").strip()
            or str(player.get("first_name") or "").strip()
            or f"Sleeper Player {pid}"
        )

        return {
            "id": pid,
            "name": name,
            "position": str(player.get("position") or "—"),
            "team": str(player.get("team") or "FA"),
            "status": str(player.get("status") or "Unknown"),
        }

    # WFS_SLEEPER_OUTLOOK_V1
    # Analysis-only WFS enrichment for Sleeper views.
    # Exact Sleeper ID -> GSIS only. No fuzzy/name fallback.
    wfs_outlook_by_sleeper_id = {}

    try:
        import sqlite3

        outlook_path = (
            REPO_ROOT
            / "data"
            / "parquet"
            / "current_unified_fanduel_expectation.parquet"
        )
        identity_path = (
            REPO_ROOT
            / "data"
            / "nfl.db"
        )

        outlook_df = pd.read_parquet(outlook_path)

        outlook_df = outlook_df[
            outlook_df["source_component"].astype(str).eq("OFFENSE")
            & pd.to_numeric(
                outlook_df["season"], errors="coerce"
            ).eq(int(season))
            & pd.to_numeric(
                outlook_df["week"], errors="coerce"
            ).eq(int(week))
            & outlook_df["source_contract"].astype(str).eq(
                "WFS_FANDUEL_OFFENSE_EXPECTATION_V1"
            )
        ].copy()

        if not outlook_df.empty:
            outlook_df["entity_id"] = (
                outlook_df["entity_id"].astype(str).str.strip()
            )

            valid_contract = (
                not outlook_df["entity_id"].duplicated().any()
                and outlook_df["entity_id"]
                .str.match(r"^00-\d{7}$")
                .all()
            )

            if not valid_contract:
                raise ValueError(
                    "WFS outlook identity contract failed"
                )

            outlook_by_gsis = {
                str(row.entity_id): {
                    "points": float(
                        row.expected_fanduel_points
                    ),
                    "team": str(
                        row.team or ""
                    ).strip(),
                    "opponent": str(
                        row.opponent_team or ""
                    ).strip(),
                }
                for row in outlook_df.itertuples(index=False)
            }

            with sqlite3.connect(identity_path) as conn:
                identity_rows = conn.execute(
                    """
                    SELECT sleeper_id, gsis_id
                    FROM player_identity
                    WHERE sleeper_id IS NOT NULL
                      AND gsis_id IS NOT NULL
                      AND TRIM(CAST(sleeper_id AS TEXT)) <> ''
                      AND TRIM(CAST(gsis_id AS TEXT)) <> ''
                    """
                ).fetchall()

            sleeper_to_gsis = {}
            duplicate_ids = set()

            for sleeper_id, gsis_id in identity_rows:
                sid = str(sleeper_id).strip()
                gid = str(gsis_id).strip()

                if sid in sleeper_to_gsis:
                    duplicate_ids.add(sid)
                else:
                    sleeper_to_gsis[sid] = gid

            if duplicate_ids:
                raise ValueError(
                    "Duplicate Sleeper identity mappings detected"
                )

            for sid, gid in sleeper_to_gsis.items():
                outlook = outlook_by_gsis.get(gid)

                if outlook is not None:
                    wfs_outlook_by_sleeper_id[sid] = outlook

    except Exception:
        # WFS enrichment fails closed without breaking Sleeper V1.
        wfs_outlook_by_sleeper_id = {}

    def wfs_outlook(player_id):
        return wfs_outlook_by_sleeper_id.get(
            str(player_id or "").strip()
        )

    def wfs_outlook_text(player_id):
        outlook = wfs_outlook(player_id)

        if outlook is None:
            return "WFS Outlook —"

        context = ""

        if outlook["team"] and outlook["opponent"]:
            context = (
                f" • {outlook['team']} "
                f"vs {outlook['opponent']}"
            )

        return (
            f"WFS Outlook {outlook['points']:.2f}"
            f"{context}"
        )

    # WFS_SLEEPER_CORE_V1
    # League-specific analysis-only projection using the validated
    # WFS stat forecast contract and Sleeper scoring settings.
    # Exact Sleeper ID -> GSIS only. No fuzzy/name fallback.
    #
    # "Core" is intentional: passing INTs, offensive 2PT conversions,
    # and fumbles lost are not forecast by the current stat contract.
    wfs_sleeper_core_by_sleeper_id = {}

    try:
        scoring_settings = league.get("scoring_settings")

        if (
            not isinstance(scoring_settings, dict)
            or not scoring_settings
        ):
            raise ValueError(
                "Sleeper league scoring settings unavailable"
            )

        stat_path = (
            REPO_ROOT
            / "data"
            / "parquet"
            / "nfl_current_unified_stat_forecasts.parquet"
        )

        stat_df = pd.read_parquet(stat_path)

        required_columns = {
            "entity_type",
            "player_id",
            "position",
            "expected_passing_yards",
            "expected_passing_tds",
            "expected_rushing_yards",
            "expected_rushing_tds",
            "expected_receptions",
            "expected_receiving_yards",
            "expected_receiving_tds",
        }

        if not required_columns.issubset(stat_df.columns):
            raise ValueError(
                "WFS Sleeper Core stat contract columns missing"
            )

        stat_df = stat_df[
            stat_df["entity_type"].astype(str).eq("OFFENSE_PLAYER")
        ].copy()

        if stat_df.empty:
            raise ValueError(
                "WFS Sleeper Core offense forecast is empty"
            )

        stat_df["player_id"] = (
            stat_df["player_id"].astype(str).str.strip()
        )

        valid_stat_identity = (
            not stat_df["player_id"].duplicated().any()
            and stat_df["player_id"]
            .str.match(r"^00-\d{7}$")
            .all()
        )

        if not valid_stat_identity:
            raise ValueError(
                "WFS Sleeper Core identity contract failed"
            )

        def _core_number(value):
            if pd.isna(value):
                return 0.0
            return float(value)

        def _core_points(row):
            position = str(row.position or "").strip().upper()

            if position == "QB":
                return (
                    _core_number(row.expected_passing_yards)
                    * float(scoring_settings.get("pass_yd", 0.0) or 0.0)
                    + _core_number(row.expected_passing_tds)
                    * float(scoring_settings.get("pass_td", 0.0) or 0.0)
                    + _core_number(row.expected_rushing_yards)
                    * float(scoring_settings.get("rush_yd", 0.0) or 0.0)
                    + _core_number(row.expected_rushing_tds)
                    * float(scoring_settings.get("rush_td", 0.0) or 0.0)
                )

            if position == "RB":
                return (
                    _core_number(row.expected_rushing_yards)
                    * float(scoring_settings.get("rush_yd", 0.0) or 0.0)
                    + _core_number(row.expected_rushing_tds)
                    * float(scoring_settings.get("rush_td", 0.0) or 0.0)
                    + _core_number(row.expected_receptions)
                    * float(scoring_settings.get("rec", 0.0) or 0.0)
                    + _core_number(row.expected_receiving_yards)
                    * float(scoring_settings.get("rec_yd", 0.0) or 0.0)
                    + _core_number(row.expected_receiving_tds)
                    * float(scoring_settings.get("rec_td", 0.0) or 0.0)
                )

            if position in {"WR", "TE"}:
                reception_value = float(
                    scoring_settings.get("rec", 0.0) or 0.0
                )

                if position == "TE":
                    reception_value += float(
                        scoring_settings.get(
                            "bonus_rec_te", 0.0
                        ) or 0.0
                    )

                return (
                    _core_number(row.expected_receptions)
                    * reception_value
                    + _core_number(row.expected_receiving_yards)
                    * float(scoring_settings.get("rec_yd", 0.0) or 0.0)
                    + _core_number(row.expected_receiving_tds)
                    * float(scoring_settings.get("rec_td", 0.0) or 0.0)
                )

            return None

        stat_by_gsis = {
            str(row.player_id): row
            for row in stat_df.itertuples(index=False)
        }

        # Reuse the exact identity map already validated by
        # WFS_SLEEPER_OUTLOOK_V1. If that boundary failed closed,
        # Core also fails closed.
        if "sleeper_to_gsis" not in locals():
            raise ValueError(
                "WFS Sleeper Core exact identity map unavailable"
            )

        for sid, gid in sleeper_to_gsis.items():
            row = stat_by_gsis.get(gid)

            if row is None:
                continue

            points = _core_points(row)

            if points is None:
                continue

            wfs_sleeper_core_by_sleeper_id[sid] = {
                "points": float(points),
            }

    except Exception:
        # Core enrichment fails closed without breaking Sleeper V1
        # or the already validated WFS Outlook surface.
        wfs_sleeper_core_by_sleeper_id = {}

    def wfs_sleeper_core_text(player_id):
        core = wfs_sleeper_core_by_sleeper_id.get(
            str(player_id or "").strip()
        )

        if core is None:
            return "Sleeper Core —"

        return f"Sleeper Core {core['points']:.2f}"

    # WFS_SLEEPER_START_SIT_V1
    # Analysis-only lineup decision intelligence built on the validated
    # Sleeper Core V1 projection layer.
    #
    # This does not change the user's Sleeper lineup. It compares the
    # provider-native current starters with the best legal Core lineup.
    # Exact Sleeper identity/Core availability is required; unavailable
    # players fail closed and are never assigned fabricated zero points.
    wfs_start_sit = {
        "available": False,
        "reason": "",
        "optimal_total": None,
        "current_total": None,
        "difference": None,
        "recommendations": [],
        "optimal_ids": set(),
        "replacement_costs": {},
        "median_cost": None,
        "p75_cost": None,
    }

    try:
        from ortools.sat.python import cp_model

        roster_positions = league.get("roster_positions") or []

        starter_slots = [
            str(slot).strip().upper()
            for slot in roster_positions
            if str(slot).strip().upper() != "BN"
        ]

        supported_slots = {
            "QB",
            "RB",
            "WR",
            "TE",
            "FLEX",
            "SUPER_FLEX",
        }

        if (
            not starter_slots
            or any(slot not in supported_slots for slot in starter_slots)
        ):
            raise ValueError(
                "Sleeper lineup contains unsupported starter slots"
            )

        roster_player_ids = [
            str(pid).strip()
            for pid in (my_roster.get("players") or [])
            if str(pid).strip()
        ]

        current_starter_ids = [
            str(pid).strip()
            for pid in (my_roster.get("starters") or [])
            if str(pid).strip()
        ]

        # WFS_SLEEPER_PLACEMENT_GATE_V1
        #
        # Reserve and taxi players remain provider-rostered,
        # but are not legal Start/Sit candidates.
        roster_player_set = set(roster_player_ids)

        reserve_player_ids = {
            str(pid).strip()
            for pid in (my_roster.get("reserve") or [])
            if str(pid).strip()
        }

        taxi_player_ids = {
            str(pid).strip()
            for pid in (my_roster.get("taxi") or [])
            if str(pid).strip()
        }

        if not reserve_player_ids.issubset(
            roster_player_set
        ):
            raise ValueError(
                "Sleeper reserve contains player outside roster"
            )

        if not taxi_player_ids.issubset(
            roster_player_set
        ):
            raise ValueError(
                "Sleeper taxi contains player outside roster"
            )

        if reserve_player_ids & taxi_player_ids:
            raise ValueError(
                "Sleeper reserve/taxi overlap"
            )

        startable_roster_player_ids = [
            pid
            for pid in roster_player_ids
            if pid not in reserve_player_ids
            and pid not in taxi_player_ids
        ]

        current_starter_set = set(
            current_starter_ids
        )

        if not current_starter_set.issubset(
            set(startable_roster_player_ids)
        ):
            raise ValueError(
                "Current Sleeper starters include "
                "reserve/taxi player"
            )

        if len(current_starter_ids) != len(starter_slots):
            raise ValueError(
                "Sleeper starter count does not match lineup slots"
            )

        def _decision_positions(player_id):
            player = players.get(str(player_id)) or {}

            raw_positions = player.get("fantasy_positions")

            if isinstance(raw_positions, (list, tuple, set)):
                positions = {
                    str(pos).strip().upper()
                    for pos in raw_positions
                    if str(pos).strip()
                }
            else:
                position = str(
                    player.get("position") or ""
                ).strip().upper()

                positions = {position} if position else set()

            return positions & {"QB", "RB", "WR", "TE"}

        def _decision_eligible(positions, slot):
            if slot in {"QB", "RB", "WR", "TE"}:
                return slot in positions

            if slot == "FLEX":
                return bool(
                    positions & {"RB", "WR", "TE"}
                )

            if slot == "SUPER_FLEX":
                return bool(
                    positions & {"QB", "RB", "WR", "TE"}
                )

            return False

        # WFS_SLEEPER_AVAILABILITY_GATE_V1
        #
        # Start/Sit candidates must pass the same current-player
        # availability principles used by GAV2. Core projections are
        # never rewritten or zeroed because of availability.
        #
        # Exact current season/week weekly_rosters is required for each
        # candidate. Hard roster unavailability blocks even when there
        # is no structured injury-consensus row. An exact current-week
        # injury-consensus BLOCK also removes the candidate.
        hard_roster_statuses = {
            "RES",
            "CUT",
            "DEV",
            "EXE",
            "INA",
            "INACTIVE",
            "RET",
            "RETIRED",
            "WAI",
            "WAIVED",
        }

        with sqlite3.connect(DATABASE_PATH) as availability_conn:
            roster_authority = pd.read_sql_query(
                """
                SELECT
                    gsis_id,
                    team,
                    status,
                    status_description_abbr,
                    updated_at
                FROM weekly_rosters
                WHERE season = ?
                  AND week = ?
                  AND gsis_id IS NOT NULL
                  AND TRIM(CAST(gsis_id AS TEXT)) <> ''
                """,
                availability_conn,
                params=[int(season), int(week)],
            )

            injury_authority = pd.read_sql_query(
                """
                SELECT
                    gsis_id,
                    injury_gate
                FROM injury_consensus_current
                WHERE season = ?
                  AND week = ?
                  AND game_type = 'REG'
                """,
                availability_conn,
                params=[int(season), int(week)],
            )

        roster_authority["gsis_id"] = (
            roster_authority["gsis_id"]
            .astype(str)
            .str.strip()
        )

        injury_authority["gsis_id"] = (
            injury_authority["gsis_id"]
            .astype(str)
            .str.strip()
        )

        # Match frozen GAV2 current-roster authority exactly:
        # latest valid updated_at record per exact GSIS controls state.
        roster_authority["_updated"] = pd.to_datetime(
            roster_authority["updated_at"],
            errors="coerce",
            utc=True,
        )

        if roster_authority["_updated"].isna().any():
            raise ValueError(
                "Current weekly roster contains invalid updated_at"
            )

        for column in (
            "team",
            "status",
            "status_description_abbr",
        ):
            roster_authority[column] = (
                roster_authority[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        roster_authority = roster_authority.sort_values(
            [
                "gsis_id",
                "_updated",
                "team",
                "status",
                "status_description_abbr",
            ],
            ascending=[
                True,
                False,
                True,
                True,
                True,
            ],
            kind="stable",
        )

        roster_authority = (
            roster_authority
            .drop_duplicates(
                subset=["gsis_id"],
                keep="first",
            )
            .drop(columns=["_updated"])
            .copy()
        )

        if roster_authority["gsis_id"].duplicated().any():
            raise ValueError(
                "Latest roster state is not singleton per GSIS"
            )

        if injury_authority["gsis_id"].duplicated().any():
            raise ValueError(
                "Current injury authority is not unique by GSIS"
            )

        roster_by_gsis = {
            str(row.gsis_id): row
            for row in roster_authority.itertuples(index=False)
        }

        injury_by_gsis = {
            str(row.gsis_id): row
            for row in injury_authority.itertuples(index=False)
        }

        blocked_decision_ids = set()
        decision_pool = []

        for pid in startable_roster_player_ids:
            core = wfs_sleeper_core_by_sleeper_id.get(pid)

            if core is None:
                continue

            gid = str(
                sleeper_to_gsis.get(pid) or ""
            ).strip()

            if not gid:
                continue

            roster_row = roster_by_gsis.get(gid)

            # Start/Sit is an action-oriented recommendation surface.
            # Missing current-week roster authority cannot establish
            # that a player is currently startable, so fail closed.
            if roster_row is None:
                blocked_decision_ids.add(pid)
                continue

            roster_status = str(
                roster_row.status or ""
            ).strip().upper()

            if roster_status in hard_roster_statuses:
                blocked_decision_ids.add(pid)
                continue

            injury_row = injury_by_gsis.get(gid)

            if injury_row is not None:
                injury_gate = str(
                    injury_row.injury_gate or ""
                ).strip().upper()

                if injury_gate == "BLOCK":
                    blocked_decision_ids.add(pid)
                    continue

            positions = _decision_positions(pid)

            if not positions:
                continue

            points = float(core["points"])

            decision_pool.append(
                {
                    "id": pid,
                    "positions": positions,
                    "points": points,
                }
            )

        if len(decision_pool) < len(starter_slots):
            raise ValueError(
                "Insufficient exact Core players for legal lineup"
            )

        def _solve_core_lineup(exclude_id=None):
            model = cp_model.CpModel()
            variables = {}

            for player_index, player in enumerate(decision_pool):
                if (
                    exclude_id is not None
                    and player["id"] == str(exclude_id)
                ):
                    continue

                for slot_index, slot in enumerate(starter_slots):
                    if _decision_eligible(
                        player["positions"],
                        slot,
                    ):
                        variables[
                            (player_index, slot_index)
                        ] = model.NewBoolVar(
                            f"wfs_ss_{player_index}_{slot_index}"
                        )

            for slot_index in range(len(starter_slots)):
                slot_vars = [
                    var
                    for (player_index, candidate_slot), var
                    in variables.items()
                    if candidate_slot == slot_index
                ]

                if not slot_vars:
                    return None

                model.Add(sum(slot_vars) == 1)

            for player_index in range(len(decision_pool)):
                player_vars = [
                    var
                    for (candidate_player, slot_index), var
                    in variables.items()
                    if candidate_player == player_index
                ]

                if player_vars:
                    model.Add(sum(player_vars) <= 1)

            model.Maximize(
                sum(
                    int(
                        round(
                            decision_pool[player_index]["points"]
                            * 100
                        )
                    )
                    * var
                    for (player_index, slot_index), var
                    in variables.items()
                )
            )

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 1.0
            solver.parameters.num_search_workers = 2

            status = solver.Solve(model)

            if status not in (
                cp_model.OPTIMAL,
                cp_model.FEASIBLE,
            ):
                return None

            lineup = []

            for slot_index, slot in enumerate(starter_slots):
                selected = None

                for (
                    player_index,
                    candidate_slot,
                ), var in variables.items():
                    if (
                        candidate_slot == slot_index
                        and solver.Value(var)
                    ):
                        selected = decision_pool[player_index]
                        break

                if selected is None:
                    return None

                lineup.append(
                    {
                        "slot": slot,
                        "id": selected["id"],
                        "points": selected["points"],
                    }
                )

            total = sum(
                player["points"]
                for player in lineup
            )

            return {
                "total": float(total),
                "lineup": lineup,
                "status": solver.StatusName(status),
            }

        optimal = _solve_core_lineup()

        if optimal is None:
            raise ValueError(
                "No legal WFS Sleeper Core lineup"
            )

        optimal_ids = {
            row["id"]
            for row in optimal["lineup"]
        }

        current_core_rows = [
            wfs_sleeper_core_by_sleeper_id.get(pid)
            for pid in current_starter_ids
        ]

        if any(row is None for row in current_core_rows):
            raise ValueError(
                "Current Sleeper starters are not fully Core-resolved"
            )

        current_total = sum(
            float(row["points"])
            for row in current_core_rows
        )

        replacement_costs = {}

        for row in optimal["lineup"]:
            pid = row["id"]

            alternative = _solve_core_lineup(
                exclude_id=pid
            )

            if alternative is None:
                continue

            replacement_costs[pid] = max(
                0.0,
                float(optimal["total"])
                - float(alternative["total"]),
            )

        cost_values = sorted(
            replacement_costs.values()
        )

        if not cost_values:
            raise ValueError(
                "No legal replacement-cost distribution"
            )

        cost_series = pd.Series(
            cost_values,
            dtype="float64",
        )

        median_cost = float(
            cost_series.quantile(0.50)
        )

        p75_cost = float(
            cost_series.quantile(0.75)
        )

        current_set = set(current_starter_ids)

        start_ids = [
            pid
            for pid in optimal_ids
            if pid not in current_set
        ]

        sit_ids = [
            pid
            for pid in current_starter_ids
            if pid not in optimal_ids
        ]

        recommendations = []

        for pid in start_ids:
            cost = replacement_costs.get(pid)

            if cost is None:
                strength = "Decision Edge"
            elif cost >= p75_cost:
                strength = "Clear Edge"
            elif cost >= median_cost:
                strength = "Meaningful Edge"
            else:
                strength = "Close Decision"

            recommendations.append(
                {
                    "action": "START",
                    "player_id": pid,
                    "replacement_cost": cost,
                    "strength": strength,
                }
            )

        for pid in sit_ids:
            recommendations.append(
                {
                    "action": "SIT",
                    "player_id": pid,
                    "replacement_cost": None,
                    "strength": None,
                }
            )

        recommendations.sort(
            key=lambda row: (
                0 if row["action"] == "START" else 1,
                -(
                    row["replacement_cost"]
                    if row["replacement_cost"] is not None
                    else -1.0
                ),
                player_info(row["player_id"])["name"].lower(),
            )
        )

        wfs_start_sit = {
            "available": True,
            "reason": "",
            "optimal_total": float(optimal["total"]),
            "current_total": float(current_total),
            "difference": float(
                optimal["total"] - current_total
            ),
            "recommendations": recommendations,
            "optimal_ids": optimal_ids,
            "replacement_costs": replacement_costs,
            "median_cost": median_cost,
            "p75_cost": p75_cost,
        }

    except Exception:
        # Start/Sit fails closed without changing provider-native Sleeper,
        # Outlook V1, Core V1, ESPN, or the FanDuel optimizer.
        wfs_start_sit = {
            "available": False,
            "reason": "",
            "optimal_total": None,
            "current_total": None,
            "difference": None,
            "recommendations": [],
            "optimal_ids": set(),
            "replacement_costs": {},
            "median_cost": None,
            "p75_cost": None,
        }

    # WFS_SLEEPER_START_SIT_UI_V1
    def render_wfs_start_sit():
        if not wfs_start_sit.get("available"):
            return

        recommendations = (
            wfs_start_sit.get("recommendations") or []
        )

        starts = [
            row
            for row in recommendations
            if row.get("action") == "START"
        ]

        sits = [
            row
            for row in recommendations
            if row.get("action") == "SIT"
        ]

        if not starts and not sits:
            return

        st.markdown("#### WFS Start/Sit")

        difference = wfs_start_sit.get("difference")

        if difference is not None:
            st.caption(
                "Projected lineup gain: "
                f"{float(difference):+.2f} Sleeper Core"
            )

        for row in starts:
            info = player_info(row.get("player_id"))
            name = info["name"]

            strength = str(
                row.get("strength") or ""
            ).strip()

            if strength:
                st.markdown(
                    f"**Start {name}** — {strength}"
                )
            else:
                st.markdown(
                    f"**Start {name}**"
                )

        for row in sits:
            info = player_info(row.get("player_id"))

            st.markdown(
                f"**Sit {info['name']}**"
            )

        st.caption(
            "WFS lineup recommendation based on "
            "current player outlook and your league settings."
        )

    def fpts_from_settings(settings):
        settings = settings or {}
        whole = settings.get("fpts", 0) or 0
        decimal = settings.get("fpts_decimal", 0) or 0

        try:
            return float(whole) + (float(decimal) / 100.0)
        except (TypeError, ValueError):
            return 0.0

    render_wfs_start_sit()
    st.divider()

    tabs = st.tabs(["🏈 Matchup", "📊 Standings", "👥 Roster"])

    # -------------------------
    # MATCHUP
    # -------------------------
    with tabs[0]:
        st.markdown(f"#### Week {week} Matchup")

        my_matchup = next(
            (
                row
                for row in matchups
                if isinstance(row, dict)
                and int(row.get("roster_id", -1)) == roster_id
            ),
            None,
        )

        if my_matchup is None:
            st.info(f"No Sleeper matchup is available for Week {week}.")
        else:
            matchup_id = my_matchup.get("matchup_id")

            opponent_matchup = next(
                (
                    row
                    for row in matchups
                    if isinstance(row, dict)
                    and row.get("matchup_id") == matchup_id
                    and int(row.get("roster_id", -1)) != roster_id
                ),
                None,
            )

            opponent_roster = None
            if opponent_matchup is not None:
                try:
                    opponent_roster = roster_by_id.get(
                        int(opponent_matchup.get("roster_id"))
                    )
                except (TypeError, ValueError):
                    opponent_roster = None

            my_name = team_name_for(my_roster)
            opponent_name = (
                team_name_for(opponent_roster)
                if opponent_roster
                else "Opponent"
            )

            my_points = float(my_matchup.get("points") or 0.0)
            opponent_points = (
                float(opponent_matchup.get("points") or 0.0)
                if opponent_matchup
                else 0.0
            )

            c1, c2 = st.columns(2)

            with c1:
                st.metric(my_name, f"{my_points:.2f}")

            with c2:
                st.metric(opponent_name, f"{opponent_points:.2f}")

            st.caption(
                "Scores shown here are Sleeper's current fantasy scores."
            )

            left, right = st.columns(2)

            def render_starters(column, label, matchup_row):
                with column:
                    st.markdown(f"**{label}**")

                    starter_ids = (
                        matchup_row.get("starters") or []
                        if matchup_row
                        else []
                    )
                    starter_points = (
                        matchup_row.get("starters_points") or []
                        if matchup_row
                        else []
                    )

                    for idx, player_id in enumerate(starter_ids):
                        info = player_info(player_id)

                        points = 0.0
                        if idx < len(starter_points):
                            try:
                                points = float(
                                    starter_points[idx] or 0.0
                                )
                            except (TypeError, ValueError):
                                points = 0.0

                        st.markdown(
                            f"**{info['name']}**  \n"
                            f"{info['position']} • {info['team']} "
                            f"— {points:.2f}  \n"
                            f"{wfs_outlook_text(player_id)}  \n"
                            f"{wfs_sleeper_core_text(player_id)}"
                        )

            render_starters(
                left,
                my_name,
                my_matchup,
            )

            render_starters(
                right,
                opponent_name,
                opponent_matchup,
            )

    # -------------------------
    # STANDINGS
    # -------------------------
    with tabs[1]:
        st.markdown("#### League Standings")

        standings = []

        for roster in rosters:
            if not isinstance(roster, dict):
                continue

            settings = roster.get("settings") or {}

            try:
                rid = int(roster.get("roster_id"))
            except (TypeError, ValueError):
                continue

            standings.append(
                {
                    "roster_id": rid,
                    "team": team_name_for(roster),
                    "wins": int(settings.get("wins") or 0),
                    "losses": int(settings.get("losses") or 0),
                    "ties": int(settings.get("ties") or 0),
                    "points": fpts_from_settings(settings),
                }
            )

        standings.sort(
            key=lambda row: (
                -row["wins"],
                row["losses"],
                -row["ties"],
                -row["points"],
                row["team"].lower(),
            )
        )

        for rank, row in enumerate(standings, start=1):
            marker = " • Your Team" if row["roster_id"] == roster_id else ""

            st.markdown(
                f"**{rank}. {row['team']}**{marker}  \n"
                f"{row['wins']}-{row['losses']}"
                f"{('-' + str(row['ties'])) if row['ties'] else ''}"
                f" • {row['points']:.2f} PF"
            )

    # -------------------------
    # ROSTER
    # -------------------------
    with tabs[2]:
        st.markdown(f"#### {team_name_for(my_roster)}")

        all_players = [
            str(pid)
            for pid in (my_roster.get("players") or [])
        ]

        starters = [
            str(pid)
            for pid in (my_roster.get("starters") or [])
        ]

        taxi = {
            str(pid)
            for pid in (my_roster.get("taxi") or [])
        }

        reserve = {
            str(pid)
            for pid in (my_roster.get("reserve") or [])
        }

        starter_set = set(starters)

        bench = [
            pid
            for pid in all_players
            if pid not in starter_set
            and pid not in taxi
            and pid not in reserve
        ]

        def render_player_group(title, ids):
            st.markdown(f"**{title}**")

            if not ids:
                st.caption("None")
                return

            for pid in ids:
                info = player_info(pid)

                st.markdown(
                    f"**{info['name']}**  \n"
                    f"{info['position']} • {info['team']} "
                    f"• {info['status']}  \n"
                    f"{wfs_outlook_text(pid)}  \n"
                    f"{wfs_sleeper_core_text(pid)}"
                )

        render_player_group("Starters", starters)
        st.divider()

        render_player_group("Bench", bench)

        if reserve:
            st.divider()
            render_player_group("Reserve", sorted(reserve))

        if taxi:
            st.divider()
            render_player_group("Taxi", sorted(taxi))


def render_sleeper_league_hub():
    """
    Sleeper Fantasy Football V1 connection surface.

    Read-only Sleeper API integration. Users connect by Sleeper username;
    no Sleeper password or authentication secret is collected by WFS.
    """
    try:
        user = dict(st.user)
    except Exception:
        user = {}

    user_sub = str(user.get("sub") or "").strip()
    if not user_sub:
        st.info("Sign in to WFS to connect a Sleeper fantasy league.")
        return

    try:
        from wfs_schedule_context import resolve_schedule_week_context

        schedule_context = resolve_schedule_week_context()
        season = int(schedule_context.season)
    except Exception:
        st.info(
            "Sleeper league connection is temporarily unavailable because "
            "the current NFL season could not be verified."
        )
        return

    st.markdown("### Sleeper Fantasy Football")
    st.caption(
        "Connect with your Sleeper username. WFS never asks for or stores "
        "your Sleeper password."
    )

    username = st.text_input(
        "Sleeper username",
        key="wfs_sleeper_username",
        placeholder="Enter your Sleeper username",
    )

    if st.button(
        "Find My Sleeper Leagues",
        key="wfs_sleeper_find_leagues",
        width="stretch",
    ):
        value = str(username or "").strip()

        if not value:
            st.warning("Enter your Sleeper username first.")
        else:
            try:
                sleeper_user = _SLEEPER_USER(value)

                if not sleeper_user:
                    st.warning(
                        "WFS could not find that Sleeper user. "
                        "Check the username and try again."
                    )
                else:
                    sleeper_user_id = str(
                        sleeper_user.get("user_id") or ""
                    ).strip()

                    leagues = _SLEEPER_USER_LEAGUES(
                        sleeper_user_id,
                        season,
                    )

                    valid_leagues = []

                    for league in leagues:
                        if not isinstance(league, dict):
                            continue

                        league_id = str(
                            league.get("league_id") or ""
                        ).strip()

                        if not league_id:
                            continue

                        if str(
                            league.get("sport") or ""
                        ).lower() != "nfl":
                            continue

                        if str(
                            league.get("season") or ""
                        ) != str(season):
                            continue

                        rosters = _SLEEPER_LEAGUE_ROSTERS(
                            league_id
                        )

                        my_roster = _SLEEPER_ROSTER_FOR_USER(
                            rosters,
                            sleeper_user_id,
                        )

                        if my_roster is None:
                            continue

                        roster_id = my_roster.get("roster_id")

                        if roster_id is None:
                            continue

                        metadata = (
                            sleeper_user.get("metadata") or {}
                        )

                        team_name = (
                            str(
                                metadata.get("team_name") or ""
                            ).strip()
                            or str(
                                sleeper_user.get(
                                    "display_name"
                                ) or ""
                            ).strip()
                            or str(
                                sleeper_user.get(
                                    "username"
                                ) or ""
                            ).strip()
                            or "My Sleeper Team"
                        )

                        valid_leagues.append(
                            {
                                "league_id": league_id,
                                "league_name": str(
                                    league.get("name")
                                    or (
                                        "Sleeper League "
                                        f"{league_id}"
                                    )
                                ),
                                "roster_id": int(roster_id),
                                "team_name": team_name,
                            }
                        )

                    st.session_state[
                        "wfs_sleeper_discovered_user_id"
                    ] = sleeper_user_id

                    st.session_state[
                        "wfs_sleeper_discovered_leagues"
                    ] = valid_leagues

                    if valid_leagues:
                        count = len(valid_leagues)
                        suffix = "" if count == 1 else "s"

                        st.success(
                            f"Found {count} Sleeper league"
                            f"{suffix} for {season}."
                        )
                    else:
                        st.info(
                            f"No {season} NFL leagues with an "
                            "owned roster were found for that "
                            "Sleeper user."
                        )

            except Exception:
                st.error(
                    "Sleeper is temporarily unavailable. "
                    "Please try again shortly."
                )

    discovered = st.session_state.get(
        "wfs_sleeper_discovered_leagues",
        [],
    )

    if discovered:
        options = {
            (
                f"{item['league_name']} • "
                f"Roster {item['roster_id']}"
            ): item
            for item in discovered
        }

        selected_label = st.selectbox(
            "Choose a Sleeper league",
            list(options),
            key="wfs_sleeper_discovered_choice",
        )

        selected = options[selected_label]

        if st.button(
            "Add Sleeper League",
            key="wfs_sleeper_add_league",
            width="stretch",
        ):
            existing = {
                str(row["league_id"])
                for row in _WFS_SLEEPER_CONNECTIONS(
                    user_sub,
                    season,
                )
            }

            if selected["league_id"] in existing:
                st.info(
                    "That Sleeper league is already saved "
                    "to your WFS account."
                )
            else:
                _WFS_SLEEPER_UPSERT_CONNECTION(
                    user_sub=user_sub,
                    season=season,
                    league_id=selected["league_id"],
                    league_name=selected["league_name"],
                    team_id=selected["roster_id"],
                    team_name=selected["team_name"],
                )

                st.success(
                    f"{selected['league_name']} added to "
                    "your WFS account."
                )

                st.rerun()

    st.markdown("### My Sleeper Teams")

    connections = _WFS_SLEEPER_CONNECTIONS(
        user_sub,
        season,
    )

    if not connections:
        st.info(
            f"You have no Sleeper leagues saved for {season}. "
            "Enter your Sleeper username above to connect one."
        )
        return

    for saved in connections:
        league_name = (
            saved.get("league_name")
            or f"Sleeper League {saved['league_id']}"
        )

        team_name = (
            saved.get("team_name")
            or "My Sleeper Team"
        )

        c1, c2 = st.columns([4, 1])

        with c1:
            st.markdown(f"**{league_name}**")
            st.caption(
                f"{team_name} • "
                f"League ID {saved['league_id']} • "
                f"Roster {saved.get('team_id')}"
            )

        with c2:
            if st.button(
                "Remove",
                key=(
                    "wfs_remove_sleeper_"
                    f"{season}_{saved['league_id']}"
                ),
                width="stretch",
            ):
                _WFS_SLEEPER_DELETE_CONNECTION(
                    user_sub,
                    season,
                    saved["league_id"],
                )
                st.rerun()

        render_sleeper_league_views(
            saved=saved,
            season=season,
            week=int(schedule_context.planning_week),
        )


def render_fantasy_league_hub():
    """
    WFS multi-provider Fantasy League Hub.

    Provider implementations remain isolated so adding Sleeper does not
    modify the validated ESPN Fantasy implementation.
    """
    st.markdown(
        """
        <div style="
            background:linear-gradient(
                110deg,
                #10233d 0%,
                #173b68 58%,
                #0b73f6 100%
            );
            border-radius:18px;
            padding:1.05rem 1.15rem;
            margin:0 0 .9rem;
            color:white;
            box-shadow:0 8px 24px rgba(15,55,105,.14);
        ">
          <div style="
              font-size:.68rem;
              font-weight:900;
              letter-spacing:.10em;
              color:#9fd0ff;
          ">
            WFS FANTASY FOOTBALL
          </div>

          <div style="
              font-size:1.55rem;
              font-weight:950;
              margin-top:.18rem;
          ">
            Fantasy League Hub
          </div>

          <div style="
              font-size:.84rem;
              color:#d8eaff;
              margin-top:.3rem;
              line-height:1.45;
          ">
            Connect your fantasy leagues, follow your teams and bring
            WFS football intelligence into your season.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    provider = st.radio(
        "Fantasy provider",
        ["ESPN", "Sleeper"],
        horizontal=True,
        key="wfs_fantasy_provider",
    )

    if provider == "Sleeper":
        render_sleeper_league_hub()
    else:
        _RENDER_ESPN_LEAGUE_HUB()
