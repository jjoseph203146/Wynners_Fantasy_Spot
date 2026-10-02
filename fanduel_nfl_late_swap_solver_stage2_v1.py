#!/usr/bin/env python3
"""
fanduel_nfl_late_swap_solver.py

Late Swap Stage 2 solver for Wynners Fantasy Spot.

Purpose
-------
Build a legal FanDuel NFL replacement lineup while preserving every
already-started/unknown roster cell in its exact original slot.

Design guardrails
-----------------
- Separate module; does not modify the normal production solver.
- Reuses the production solver's validated GPP objective helpers.
- No fuzzy matching.
- Salary cap is a ceiling, not a target.
- FanDuel hard max: 4 players from one NFL team.
- LOCKED and UNKNOWN current slots are exact-slot locks.
- Locked current occupants may be preserved without a fresh projection.
- Only SWAPPABLE players may be newly selected into replaceable slots.
- Players from LOCKED/UNKNOWN games cannot be newly introduced.
- Two-stage production objective is preserved:
    1) maximize WFS projection
    2) stay within allowed projection loss and maximize validated GPP rank
       with projection as deterministic tiebreak.
- Single-search-worker / seed 0 deterministic CP-SAT behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping

import pandas as pd
from ortools.sat.python import cp_model

import fanduel_nfl_ui_solver as prod


SALARY_CAP = prod.SALARY_CAP
FANDUEL_MAX_PLAYERS_PER_TEAM = prod.FANDUEL_MAX_PLAYERS_PER_TEAM
ROSTER_SLOTS = list(prod.ROSTER_SLOTS)
PROJECTION_SCALE = prod.PROJECTION_SCALE
GPP_PRIMARY_MULTIPLIER = prod.GPP_PRIMARY_MULTIPLIER

LATE_SWAP_SLOT_POSITIONS = {
    "QB": {"QB"},
    "RB1": {"RB"},
    "RB2": {"RB"},
    "WR1": {"WR"},
    "WR2": {"WR"},
    "WR3": {"WR"},
    "TE": {"TE"},
    "FLEX": {"RB", "WR", "TE"},
    "DST": {"DST"},
}

VALID_LOCK_STATES = {"LOCKED", "SWAPPABLE", "UNKNOWN"}


@dataclass(frozen=True)
class LateSwapSettings:
    qb_stack: int = 1
    bring_back: int = 0
    projection_loss_pct: float = 0.015
    salary_floor: int = 0
    max_team: int = 0


def _normalize_state(value) -> str:
    state = str(value or "").strip().upper()
    if state not in VALID_LOCK_STATES:
        raise ValueError(
            f"Invalid late-swap status {value!r}. "
            f"Expected one of {sorted(VALID_LOCK_STATES)}."
        )
    return state


def _prepare_pool(
    pool: pd.DataFrame,
    current_slot_keys: Mapping[str, str],
    slot_status: Mapping[str, str],
) -> pd.DataFrame:
    """
    Prepare a late-swap pool with two deterministic row classes:

    1) candidate rows:
       solver_eligible == 1 and therefore allowed to participate normally.

    2) preserve-only rows:
       the exact current occupant of a LOCKED/UNKNOWN roster slot, retained even
       when solver_eligible == 0 or the current projection is unavailable.

    Preserve-only rows can never be newly introduced into another slot. Because
    they are exact fixed cells in every feasible solution, a missing projection
    is represented as objective weight 0 without changing the optimizer's choice
    among SWAPPABLE candidates.
    """
    required = {
        "player_solver",
        "team_solver",
        "solver_position",
        "salary_solver",
        "projection_solver",
        "solver_eligible",
        "game_solver",
    }
    missing = sorted(required - set(pool.columns))
    if missing:
        raise ValueError(
            "Late-swap pool missing required column(s): " + ", ".join(missing)
        )

    missing_slots = [s for s in ROSTER_SLOTS if s not in current_slot_keys]
    if missing_slots:
        raise ValueError(
            "Current lineup is missing roster slot(s): "
            + ", ".join(missing_slots)
        )

    missing_status = [s for s in ROSTER_SLOTS if s not in slot_status]
    if missing_status:
        raise ValueError(
            "Late-swap status missing for slot(s): "
            + ", ".join(missing_status)
        )

    out = pool.copy()

    out["salary_solver"] = pd.to_numeric(
        out["salary_solver"], errors="coerce"
    )
    out["projection_solver"] = pd.to_numeric(
        out["projection_solver"], errors="coerce"
    )
    out["solver_eligible"] = (
        pd.to_numeric(out["solver_eligible"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if out["salary_solver"].isna().any():
        raise ValueError("Late-swap source pool contains invalid salary.")


    out["salary_solver"] = out["salary_solver"].astype(int)

    # Build exact UI keys before eligibility filtering so locked current
    # occupants can still be found and preserved.
    out["_ui_player_key"] = out.apply(prod.player_key_from_row, axis=1)

    if out["_ui_player_key"].duplicated().any():
        dup = out.loc[
            out["_ui_player_key"].duplicated(keep=False),
            [
                "_ui_player_key",
                "player_solver",
                "team_solver",
                "solver_position",
                "salary_solver",
                "game_solver",
            ],
        ]
        raise ValueError(
            "Duplicate UI player keys detected in late-swap source pool:\n"
            + dup.to_string(index=False)
        )

    current_locked_keys = set()
    for slot in ROSTER_SLOTS:
        state = _normalize_state(slot_status[slot])
        key = str(current_slot_keys[slot]).strip()
        if state in {"LOCKED", "UNKNOWN"}:
            current_locked_keys.add(key)

    out["_late_swap_candidate"] = out["solver_eligible"].eq(1)
    out["_late_swap_preserve_only"] = (
        out["_ui_player_key"].isin(current_locked_keys)
        & ~out["_late_swap_candidate"]
    )

    out = out.loc[
        out["_late_swap_candidate"]
        | out["_late_swap_preserve_only"]
    ].copy()
    # Validate positions only after narrowing to rows that can actually
    # participate in Late Swap:
    #   1) normal solver-eligible candidates, or
    #   2) exact LOCKED/UNKNOWN current occupants retained preserve-only.
    #
    # Unresolved-position rows that are neither candidates nor current locked
    # occupants remain quarantined upstream and must not block the whole slate.
    bad_working_position = ~out["solver_position"].astype(str).isin(
        prod.VALID_POSITIONS
    )
    if bad_working_position.any():
        bad = out.loc[
            bad_working_position,
            [
                "player_solver",
                "team_solver",
                "solver_position",
                "_late_swap_candidate",
                "_late_swap_preserve_only",
            ],
        ]
        raise ValueError(
            "Late-swap working pool contains invalid position:\n"
            + bad.to_string(index=False)
        )

    missing_locked = sorted(
        key for key in current_locked_keys
        if key not in set(out["_ui_player_key"].astype(str))
    )
    if missing_locked:
        raise ValueError(
            "Locked/unknown current lineup player(s) are missing from the "
            "slate source pool: "
            + ", ".join(missing_locked)
        )

    # SWAPPABLE candidate rows must always have a real current projection.
    bad_candidate_projection = (
        out["_late_swap_candidate"]
        & out["projection_solver"].isna()
    )
    if bad_candidate_projection.any():
        bad = out.loc[
            bad_candidate_projection,
            [
                "player_solver",
                "team_solver",
                "solver_position",
                "game_solver",
            ],
        ]
        raise ValueError(
            "Late-swap candidate pool contains invalid projection:\n"
            + bad.to_string(index=False)
        )

    # Fixed locked rows are constants in every feasible solution. Zero objective
    # weight is mathematically neutral and avoids fabricating a stale projection.
    out["_projection_display"] = out["projection_solver"]
    out.loc[
        out["_late_swap_preserve_only"]
        & out["projection_solver"].isna(),
        "projection_solver",
    ] = 0.0

    dup = out.duplicated(
        ["player_solver", "team_solver", "salary_solver"],
        keep=False,
    )
    if dup.any():
        raise ValueError(
            "Duplicate late-swap player/team/salary rows detected:\n"
            + out.loc[
                dup,
                [
                    "player_solver",
                    "team_solver",
                    "solver_position",
                    "salary_solver",
                    "game_solver",
                ],
            ].to_string(index=False)
        )

    # Build validated GPP objective fields only for rows that can actually
    # compete for SWAPPABLE slots. Preserve-only locked rows are fixed constants
    # and must not be required to have current GPP-score coverage.
    candidate_rows = out.loc[
        out["_late_swap_candidate"]
    ].copy()
    preserve_rows = out.loc[
        out["_late_swap_preserve_only"]
    ].copy()

    if not candidate_rows.empty:
        candidate_rows = prod.attach_validated_gpp_fields(candidate_rows)
        candidate_rows = prod.add_opponents(candidate_rows)
        candidate_rows = prod.build_gpp_objective_fields(candidate_rows)

    if not preserve_rows.empty:
        preserve_rows = prod.add_opponents(preserve_rows)
        preserve_rows["gpp_objective_signal"] = 0.0
        preserve_rows["gpp_objective_method"] = "LOCKED_PRESERVE_ONLY"
        preserve_rows["gpp_objective_rank_points"] = 0

    out = pd.concat(
        [candidate_rows, preserve_rows],
        ignore_index=True,
        sort=False,
    )

    if out.empty:
        raise ValueError("Late-swap working pool is empty after preparation.")

    out = out.sort_values(
        ["solver_position", "team_solver", "player_solver", "salary_solver"],
        kind="mergesort",
    ).reset_index(drop=True)

    return out


def _validate_current_lineup(
    pool: pd.DataFrame,
    current_slot_keys: Mapping[str, str],
    slot_status: Mapping[str, str],
) -> dict[str, int]:
    missing_slots = [s for s in ROSTER_SLOTS if s not in current_slot_keys]
    if missing_slots:
        raise ValueError(
            "Current lineup is missing roster slot(s): "
            + ", ".join(missing_slots)
        )

    missing_status = [s for s in ROSTER_SLOTS if s not in slot_status]
    if missing_status:
        raise ValueError(
            "Late-swap status missing for slot(s): "
            + ", ".join(missing_status)
        )

    key_to_index = dict(zip(pool["_ui_player_key"], pool.index))
    slot_to_index: dict[str, int] = {}

    for slot in ROSTER_SLOTS:
        key = str(current_slot_keys[slot]).strip()
        if not key:
            raise ValueError(f"Current lineup slot {slot} has no player key.")
        if key not in key_to_index:
            raise ValueError(
                f"Current lineup player for {slot} is not in the eligible "
                "solver-ready pool. Late swap fails closed."
            )

        i = int(key_to_index[key])
        position = str(pool.at[i, "solver_position"]).upper()
        if position not in LATE_SWAP_SLOT_POSITIONS[slot]:
            raise ValueError(
                f"Current lineup slot {slot} contains ineligible position "
                f"{position}."
            )
        slot_to_index[slot] = i
        _normalize_state(slot_status[slot])

    if len(set(slot_to_index.values())) != 9:
        raise ValueError(
            "Current lineup contains the same player in more than one slot."
        )

    total_salary = sum(
        int(pool.at[i, "salary_solver"]) for i in slot_to_index.values()
    )
    if total_salary > SALARY_CAP:
        raise ValueError(
            f"Current lineup salary ${total_salary:,} exceeds FanDuel cap."
        )

    team_counts = (
        pd.Series(
            [str(pool.at[i, "team_solver"]) for i in slot_to_index.values()],
            dtype="object",
        )
        .value_counts()
        .to_dict()
    )
    if team_counts and max(team_counts.values()) > FANDUEL_MAX_PLAYERS_PER_TEAM:
        raise ValueError(
            "Current lineup already violates FanDuel max-4-per-team legality."
        )

    return slot_to_index


def _build_model(
    pool: pd.DataFrame,
    slot_to_index: Mapping[str, int],
    slot_status: Mapping[str, str],
    player_status: Mapping[str, str],
    settings: LateSwapSettings,
    objective_mode: str,
    projection_floor_scaled: int | None = None,
):
    model = cp_model.CpModel()

    # Candidate variables are exact player -> exact roster slot.
    x = {}
    for i, row in pool.iterrows():
        pos = str(row["solver_position"]).upper()
        player_key = str(row["_ui_player_key"])
        pstate = _normalize_state(player_status.get(player_key, "UNKNOWN"))

        for slot in ROSTER_SLOTS:
            if pos not in LATE_SWAP_SLOT_POSITIONS[slot]:
                continue

            current_i = int(slot_to_index[slot])
            sstate = _normalize_state(slot_status[slot])

            # LOCKED or UNKNOWN roster cells are exact-slot locks.
            if sstate in {"LOCKED", "UNKNOWN"}:
                if i == current_i:
                    x[(i, slot)] = model.NewBoolVar(f"x_{i}_{slot}")
                continue

            # SWAPPABLE slot:
            # - only projection-ready candidate rows may be selected
            # - preserve-only rows from already-started/unknown games can never
            #   be newly introduced into any replaceable slot
            if (
                pstate == "SWAPPABLE"
                and bool(row["_late_swap_candidate"])
            ):
                x[(i, slot)] = model.NewBoolVar(f"x_{i}_{slot}")

    # One player in every slot.
    for slot in ROSTER_SLOTS:
        vars_for_slot = [v for (i, s), v in x.items() if s == slot]
        if not vars_for_slot:
            raise ValueError(
                f"No legal late-swap candidate exists for roster slot {slot}."
            )
        model.Add(sum(vars_for_slot) == 1)

    # Locked/unknown exact slot equality.
    for slot in ROSTER_SLOTS:
        state = _normalize_state(slot_status[slot])
        if state in {"LOCKED", "UNKNOWN"}:
            i = int(slot_to_index[slot])
            var = x.get((i, slot))
            if var is None:
                raise ValueError(
                    f"Exact locked player for {slot} is unavailable."
                )
            model.Add(var == 1)

    # Player can occupy at most one slot.
    selected = {}
    for i in pool.index:
        vars_for_player = [v for (j, _), v in x.items() if j == i]
        if vars_for_player:
            y = model.NewBoolVar(f"selected_{i}")
            model.Add(sum(vars_for_player) == y)
            selected[i] = y

    model.Add(sum(selected.values()) == 9)

    salary_expr = sum(
        int(pool.at[i, "salary_solver"]) * y
        for i, y in selected.items()
    )
    model.Add(salary_expr <= SALARY_CAP)
    if int(settings.salary_floor) > 0:
        model.Add(salary_expr >= int(settings.salary_floor))

    # FanDuel hard legality, optionally stricter WFS cap.
    effective_team_cap = FANDUEL_MAX_PLAYERS_PER_TEAM
    if int(settings.max_team) > 0:
        effective_team_cap = min(
            int(settings.max_team),
            FANDUEL_MAX_PLAYERS_PER_TEAM,
        )

    for team in sorted(pool["team_solver"].dropna().astype(str).unique()):
        idxs = pool.index[pool["team_solver"].eq(team)].tolist()
        vars_team = [selected[i] for i in idxs if i in selected]
        if vars_team:
            model.Add(sum(vars_team) <= effective_team_cap)

    # Preserve production QB-stack / bring-back semantics.
    qb_indices = pool.index[
        pool["solver_position"].astype(str).eq("QB")
    ].tolist()

    for qi in qb_indices:
        if qi not in selected:
            continue

        qteam = pool.at[qi, "team_solver"]
        opp = pool.at[qi, "opponent_solver"]

        stack_idxs = pool.index[
            pool["team_solver"].eq(qteam)
            & pool["solver_position"].isin(["WR", "TE"])
        ].tolist()
        stack_vars = [selected[i] for i in stack_idxs if i in selected]

        if len(stack_vars) < int(settings.qb_stack):
            model.Add(selected[qi] == 0)
        else:
            model.Add(
                sum(stack_vars)
                >= int(settings.qb_stack) * selected[qi]
            )

        if int(settings.bring_back):
            br_idxs = pool.index[
                pool["team_solver"].eq(opp)
                & pool["solver_position"].isin(["RB", "WR", "TE"])
            ].tolist()
            br_vars = [selected[i] for i in br_idxs if i in selected]
            if not br_vars:
                model.Add(selected[qi] == 0)
            else:
                model.Add(sum(br_vars) >= selected[qi])

    projection_expr = sum(
        int(round(float(pool.at[i, "projection_solver"]) * PROJECTION_SCALE))
        * y
        for i, y in selected.items()
    )

    gpp_expr = sum(
        int(pool.at[i, "gpp_objective_rank_points"]) * y
        for i, y in selected.items()
    )

    if projection_floor_scaled is not None:
        model.Add(projection_expr >= int(projection_floor_scaled))

    if objective_mode == "projection":
        model.Maximize(projection_expr)
    elif objective_mode == "gpp":
        model.Maximize(
            GPP_PRIMARY_MULTIPLIER * gpp_expr + projection_expr
        )
    else:
        raise ValueError(f"Unknown objective_mode: {objective_mode}")

    return model, x, selected


def _solve_model(model, x, pool: pd.DataFrame):
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen = []
    for (i, slot), var in x.items():
        if solver.Value(var) == 1:
            chosen.append((int(i), slot))

    if len(chosen) != 9:
        raise RuntimeError(
            f"Late-swap solver returned {len(chosen)} players instead of 9."
        )

    return chosen


def optimize_late_swap(
    pool: pd.DataFrame,
    current_slot_keys: Mapping[str, str],
    slot_status: Mapping[str, str],
    player_status: Mapping[str, str],
    settings: LateSwapSettings | None = None,
) -> dict:
    """
    Optimize a single FanDuel late-swap lineup.

    Parameters
    ----------
    pool:
        Selected-slate solver-ready DataFrame.
    current_slot_keys:
        Exact 9-slot mapping of roster slot -> production UI player key.
    slot_status:
        Exact 9-slot mapping of roster slot -> LOCKED/SWAPPABLE/UNKNOWN.
        UNKNOWN fails closed and is locked to the current exact slot.
    player_status:
        Mapping of production UI player key -> LOCKED/SWAPPABLE/UNKNOWN.
        Only SWAPPABLE players may be newly introduced.
    settings:
        LateSwapSettings using production-style stack/bring-back/frontier rules.

    Returns
    -------
    dict with lineup DataFrame and legality/audit metadata.
    """
    if settings is None:
        settings = LateSwapSettings()

    if not (0.0 <= float(settings.projection_loss_pct) < 1.0):
        raise ValueError("projection_loss_pct must be in [0, 1).")
    if int(settings.qb_stack) < 0:
        raise ValueError("qb_stack cannot be negative.")

    prepared = _prepare_pool(
        pool,
        current_slot_keys,
        slot_status,
    )
    slot_to_index = _validate_current_lineup(
        prepared,
        current_slot_keys,
        slot_status,
    )

    # Stage 1: projection optimum under the current late-swap state.
    model1, x1, _ = _build_model(
        prepared,
        slot_to_index,
        slot_status,
        player_status,
        settings,
        objective_mode="projection",
    )
    chosen1 = _solve_model(model1, x1, prepared)
    if chosen1 is None:
        raise ValueError(
            "No legal late-swap lineup exists under the current exact-slot locks."
        )

    optimum_projection_scaled = sum(
        int(
            round(
                float(prepared.at[i, "projection_solver"])
                * PROJECTION_SCALE
            )
        )
        for i, _ in chosen1
    )

    projection_floor_scaled = int(
        optimum_projection_scaled
        * (1.0 - float(settings.projection_loss_pct))
    )

    # Stage 2: production GPP objective inside projection frontier.
    model2, x2, _ = _build_model(
        prepared,
        slot_to_index,
        slot_status,
        player_status,
        settings,
        objective_mode="gpp",
        projection_floor_scaled=projection_floor_scaled,
    )
    chosen2 = _solve_model(model2, x2, prepared)
    if chosen2 is None:
        raise ValueError(
            "Projection-frontier late-swap solve became infeasible."
        )

    slot_order = {slot: n for n, slot in enumerate(ROSTER_SLOTS)}
    rows = []
    for i, slot in chosen2:
        r = prepared.loc[i]
        rows.append(
            {
                "slot": slot,
                "player_key": str(r["_ui_player_key"]),
                "player": str(r["player_solver"]),
                "team": str(r["team_solver"]),
                "position": str(r["solver_position"]),
                "salary": int(r["salary_solver"]),
                "projection": (
                    float(r["_projection_display"])
                    if pd.notna(r["_projection_display"])
                    else None
                ),
                "game": str(r["game_solver"]),
                "late_swap_status": _normalize_state(
                    player_status.get(
                        str(r["_ui_player_key"]),
                        "UNKNOWN",
                    )
                ),
            }
        )

    lineup = pd.DataFrame(rows).sort_values(
        "slot",
        key=lambda s: s.map(slot_order),
        kind="mergesort",
    ).reset_index(drop=True)

    # Hard audit: locked/unknown cells must be unchanged in exact slot.
    current_index_to_key = {
        slot: str(prepared.at[i, "_ui_player_key"])
        for slot, i in slot_to_index.items()
    }
    final_slot_to_key = dict(zip(lineup["slot"], lineup["player_key"]))

    exact_lock_ok = all(
        final_slot_to_key[slot] == current_index_to_key[slot]
        for slot in ROSTER_SLOTS
        if _normalize_state(slot_status[slot]) in {"LOCKED", "UNKNOWN"}
    )

    salary_total = int(lineup["salary"].sum())
    max_team_count = int(lineup["team"].value_counts().max())
    unique_players_ok = lineup["player_key"].nunique() == 9

    audit_ok = (
        exact_lock_ok
        and salary_total <= SALARY_CAP
        and max_team_count <= FANDUEL_MAX_PLAYERS_PER_TEAM
        and unique_players_ok
        and len(lineup) == 9
    )

    if not audit_ok:
        raise RuntimeError("Late-swap final legality audit failed.")

    changed_slots = []
    for slot in ROSTER_SLOTS:
        old_key = current_index_to_key[slot]
        new_key = final_slot_to_key[slot]
        if old_key != new_key:
            changed_slots.append(slot)

    return {
        "lineup": lineup,
        "audit_ok": True,
        "exact_lock_ok": exact_lock_ok,
        "salary_total": salary_total,
        "salary_remaining": SALARY_CAP - salary_total,
        "max_players_one_team": max_team_count,
        "team_limit": FANDUEL_MAX_PLAYERS_PER_TEAM,
        "projection_total": float(
            pd.to_numeric(
                lineup["projection"],
                errors="coerce",
            ).sum()
        ),
        "projection_optimum": optimum_projection_scaled / PROJECTION_SCALE,
        "projection_floor": projection_floor_scaled / PROJECTION_SCALE,
        "changed_slots": changed_slots,
        "changed_slot_count": len(changed_slots),
        "preserve_only_locked_rows": int(
            prepared["_late_swap_preserve_only"].sum()
        ),
        "projection_ready_candidate_rows": int(
            prepared["_late_swap_candidate"].sum()
        ),
        "locked_slots": [
            s
            for s in ROSTER_SLOTS
            if _normalize_state(slot_status[s]) in {"LOCKED", "UNKNOWN"}
        ],
        "swappable_slots": [
            s
            for s in ROSTER_SLOTS
            if _normalize_state(slot_status[s]) == "SWAPPABLE"
        ],
    }


def settings_from_solver_args(args) -> LateSwapSettings:
    """
    Convenience adapter for callers already holding the normal solver settings.
    Exposure/overlap settings intentionally do not apply to a single late-swap entry.
    """
    return LateSwapSettings(
        qb_stack=int(getattr(args, "qb_stack", 1)),
        bring_back=int(getattr(args, "bring_back", 0)),
        projection_loss_pct=float(
            getattr(args, "projection_loss_pct", 0.015)
        ),
        salary_floor=int(getattr(args, "salary_floor", 0)),
        max_team=int(getattr(args, "max_team", 0)),
    )
