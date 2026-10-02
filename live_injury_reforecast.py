from __future__ import annotations

"""
Live injury reforecast layer.

Purpose:
- Preserve baseline Ridge/player and game forecasts.
- Consume current authoritative injury consensus.
- Reallocate lost offensive opportunity using role evidence.
- Produce injury-adjusted player projections.
- Produce injury-adjusted team/game scoring forecasts.
- Fail closed when replacement evidence is insufficient.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


INACTIVE_STATUSES = {"OUT", "DOUBTFUL", "INACTIVE"}


@dataclass(frozen=True)
class InjuryAdjustment:
    player_id: str
    player_name: str
    team: str
    position: str
    baseline_projection: float
    adjusted_projection: float
    adjustment: float
    reason: str
import sqlite3
from pathlib import Path


NFL_DB = Path("data/nfl.db")


def load_injury_consensus(
    season: int,
    week: int,
    db_path: Path = NFL_DB,
) -> pd.DataFrame:
    """Load current authoritative injury consensus for one NFL week."""

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                season,
                week,
                team,
                gsis_id,
                position,
                player_name,
                consensus_status,
                injury_gate,
                injury_gate_reason,
                availability_risk,
                injury_risk,
                updated_at
            FROM injury_consensus_current
            WHERE season = ?
              AND week = ?
            """,
            conn,
            params=(season, week),
        )

    if df.empty:
        return df

    df["consensus_status"] = (
        df["consensus_status"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return df


def inactive_players(injuries: pd.DataFrame) -> pd.DataFrame:
    """Return only authoritative inactive offensive players."""

    if injuries.empty:
        return injuries.copy()

    status_block = injuries["consensus_status"].isin(
        INACTIVE_STATUSES
    )

    gate_block = (
        injuries["injury_gate"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .eq("BLOCK")
    )

    return injuries.loc[
        status_block | gate_block
    ].copy()

REPLACEMENT_HISTORY = Path(
    "processed/forecast_v1_player_replacement_quality.csv"
)


def load_replacement_history(
    path: Path = REPLACEMENT_HISTORY,
) -> pd.DataFrame:
    """Load leakage-safe historical replacement-role evidence."""

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)

    numeric_cols = [
        "candidate_found",
        "candidate_depth_rank",
        "replacement_history_known",
        "replacement_cold_start",
        "replacement_full_history",
        "replacement_opportunities_avg_3",
        "replacement_targets_avg_3",
        "replacement_carries_avg_3",
        "replacement_fd_avg_3",
        "replacement_fd_avg_5",
        "replacement_fd_per_snap_avg_3",
        "replacement_fd_per_touch_avg_3",
        "injured_opportunities_avg_3",
        "injured_targets_avg_3",
        "injured_carries_avg_3",
        "injured_fd_avg_3",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_replacement_profiles(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build conservative historical replacement profiles.

    Only rows with an identified candidate and known replacement
    history are allowed to influence a live projection. Cold-start
    evidence is retained elsewhere for audit but does not create
    workload.
    """

    if history.empty:
        return pd.DataFrame()

    required = {
        "team",
        "injury_position",
        "candidate_found",
        "replacement_history_known",
        "replacement_cold_start",
    }

    if not required.issubset(history.columns):
        return pd.DataFrame()

    usable = history.loc[
        history["injury_position"].isin({"QB", "RB", "WR", "TE"})
        &
        history["candidate_found"].eq(1)
        & history["replacement_history_known"].eq(1)
        & history["replacement_cold_start"].eq(0)
    ].copy()

    if usable.empty:
        return pd.DataFrame()

    metrics = [
        "replacement_opportunities_avg_3",
        "replacement_targets_avg_3",
        "replacement_carries_avg_3",
        "replacement_fd_avg_3",
        "replacement_fd_avg_5",
        "replacement_fd_per_snap_avg_3",
        "replacement_fd_per_touch_avg_3",
    ]

    metrics = [c for c in metrics if c in usable.columns]

    profiles = (
        usable.groupby(
            ["team", "injury_position"],
            as_index=False,
        )[metrics]
        .median()
    )

    counts = (
        usable.groupby(
            ["team", "injury_position"]
        )
        .size()
        .rename("replacement_evidence_games")
        .reset_index()
    )

    profiles = profiles.merge(
        counts,
        on=["team", "injury_position"],
        how="left",
        validate="one_to_one",
    )

    return profiles
def identify_live_replacements(
    slate: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map each inactive offensive player to the highest-ranked
    currently available teammate at the same position.
    """

    columns = [
        "injured_gsis_id",
        "injured_name",
        "team",
        "position",
        "injured_depth_rank",
        "replacement_gsis_id",
        "replacement_name",
        "replacement_depth_rank",
        "replacement_status",
    ]

    if slate.empty or injuries.empty:
        return pd.DataFrame(columns=columns)

    work = slate.copy()

    work["position"] = (
        work["position"].fillna("").astype(str).str.upper().str.strip()
    )
    work["team"] = (
        work["team"].fillna("").astype(str).str.upper().str.strip()
    )
    work["depth_rank"] = pd.to_numeric(
        work["depth_rank"], errors="coerce"
    )

    inactive = inactive_players(injuries)
    inactive = inactive.loc[
        inactive["position"].isin({"QB", "RB", "WR", "TE"})
        & ~inactive["injury_gate_reason"].fillna("").str.startswith(("ROSTER_RESERVE_STATUS:", "ROSTER_EXEMPT_STATUS:"))
    ].copy()

    inactive_ids = set(
        inactive["gsis_id"].dropna().astype(str)
    )

    rows = []

    for _, injured in inactive.iterrows():
        team = str(injured["team"]).upper().strip()
        position = str(injured["position"]).upper().strip()
        injured_id = str(injured["gsis_id"])

        group = work.loc[
            work["team"].eq(team)
            & work["position"].eq(position)
        ].copy()

        injured_match = group.loc[
            group["player_id"].astype(str).eq(injured_id)
        ]

        injured_rank = (
            injured_match["depth_rank"].iloc[0]
            if not injured_match.empty
            else pd.NA
        )

        candidates = group.loc[
            ~group["player_id"].astype(str).isin(inactive_ids)
        ].copy()

        candidates = candidates.loc[
            ~candidates["player_id"].astype(str).eq(injured_id)
        ]

        candidates = candidates.dropna(subset=["depth_rank"])
        candidates["depth_distance"] = candidates["depth_rank"] - injured_rank if pd.notna(injured_rank) else candidates["depth_rank"]
        candidates["promotion_side"] = (candidates["depth_distance"] < 0).astype(int)
        candidates["depth_distance"] = candidates["depth_distance"].abs()
        candidates = candidates.sort_values(
            ["promotion_side", "depth_distance", "depth_rank", "player_display_name"],
            kind="stable",
        )

        if candidates.empty:
            rows.append({
                "injured_gsis_id": injured_id,
                "injured_name": injured["player_name"],
                "team": team,
                "position": position,
                "injured_depth_rank": injured_rank,
                "replacement_gsis_id": pd.NA,
                "replacement_name": pd.NA,
                "replacement_depth_rank": pd.NA,
                "replacement_status": "NO_VALID_REPLACEMENT",
            })
            continue

        replacement = candidates.iloc[0]

        rows.append({
            "injured_gsis_id": injured_id,
            "injured_name": injured["player_name"],
            "team": team,
            "position": position,
            "injured_depth_rank": injured_rank,
            "replacement_gsis_id": str(replacement["player_id"]),
            "replacement_name": replacement["player_display_name"],
            "replacement_depth_rank": replacement["depth_rank"],
            "replacement_status": "IDENTIFIED",
        })

    return pd.DataFrame(rows, columns=columns)
def qualify_live_role_losses(
    slate: pd.DataFrame,
    replacements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Qualify injury events for live workload redistribution.

    Fail closed unless:
    - injured player has exact current-slate identity
    - replacement was identified
    - injured player has measurable recent offensive workload
    """

    if replacements.empty:
        work = replacements.copy()
        work["reforecast_eligible"] = pd.Series(index=work.index, dtype=bool)
        work["reforecast_reason"] = pd.Series(index=work.index, dtype=object)
        return work

    work = replacements.merge(
        slate,
        left_on="injured_gsis_id",
        right_on="player_id",
        how="left",
        suffixes=("", "_slate"),
    )

    work["opportunities_last"] = pd.to_numeric(
        work["opportunities_last"], errors="coerce"
    )
    work["targets_avg_5"] = pd.to_numeric(
        work["targets_avg_5"], errors="coerce"
    )
    work["fd_avg_5"] = pd.to_numeric(
        work["fd_avg_5"], errors="coerce"
    )

    exact_identity = work["player_id"].notna()

    measurable_role = (
        work["opportunities_last"].fillna(0).gt(0)
        | work["targets_avg_5"].fillna(0).gt(0)
        | work["fd_avg_5"].fillna(0).gt(0)
    )

    valid_replacement = work["replacement_status"].eq("IDENTIFIED")

    work["reforecast_eligible"] = (
        exact_identity
        & measurable_role
        & valid_replacement
    )

    work["reforecast_reason"] = "QUALIFIED_ROLE_LOSS"

    work.loc[
        ~exact_identity,
        "reforecast_reason",
    ] = "NO_CURRENT_SLATE_ROLE"

    work.loc[
        exact_identity & ~measurable_role,
        "reforecast_reason",
    ] = "NO_MEASURABLE_RECENT_ROLE"

    work.loc[
        exact_identity
        & measurable_role
        & ~valid_replacement,
        "reforecast_reason",
    ] = "NO_VALID_REPLACEMENT"

    return work
def apply_live_injury_reforecast(
    projections: pd.DataFrame,
    injuries: pd.DataFrame,
    replacement_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply validated live injury role adjustments.

    Ridge remains immutable as the baseline.
    Only qualified current role losses may create adjustments.
    """

    out = projections.copy()

    out["injury_adjusted_projection"] = pd.to_numeric(
        out["ridge_projection"], errors="coerce"
    )
    out["projection_adjustment"] = 0.0
    out["projection_adjustment_reason"] = "NO_LIVE_ADJUSTMENT"

    replacements = identify_live_replacements(out, injuries)
    qualified = qualify_live_role_losses(out, replacements)

    qualified = qualified.loc[
        qualified["reforecast_eligible"].eq(True)
    ].copy()

    audit_rows = []

    if qualified.empty:
        return out, pd.DataFrame(audit_rows)

    for _, event in qualified.iterrows():
        position = str(event["position"]).upper()
        team = str(event["team"]).upper()
        replacement_id = str(event["replacement_gsis_id"])

        # QB requires a dedicated passing-volume model.
        # Fail closed rather than treating QB rushing opportunity
        # as total quarterback workload.
        if position == "QB":
            audit_rows.append({
                "injured_player": event["injured_name"],
                "team": team,
                "position": position,
                "replacement_player": event["replacement_name"],
                "status": "SKIPPED_QB_REQUIRES_PASSING_MODEL",
                "projection_adjustment": 0.0,
            })
            continue

        profile = replacement_profiles.loc[
            replacement_profiles["team"].eq(team)
            & replacement_profiles["injury_position"].eq(position)
        ]

        if profile.empty:
            audit_rows.append({
                "injured_player": event["injured_name"],
                "team": team,
                "position": position,
                "replacement_player": event["replacement_name"],
                "status": "NO_REPLACEMENT_EVIDENCE",
                "projection_adjustment": 0.0,
            })
            continue

        profile = profile.iloc[0]

        replacement_mask = (
            out["player_id"].astype(str).eq(replacement_id)
        )

        if not replacement_mask.any():
            audit_rows.append({
                "injured_player": event["injured_name"],
                "team": team,
                "position": position,
                "replacement_player": event["replacement_name"],
                "status": "REPLACEMENT_NOT_IN_PROJECTION_POOL",
                "projection_adjustment": 0.0,
            })
            continue

        replacement = out.loc[replacement_mask].iloc[0]

        baseline_projection = pd.to_numeric(
            replacement["ridge_projection"], errors="coerce"
        )

        efficiency = pd.to_numeric(
            replacement["fd_per_touch_avg_3"], errors="coerce"
        )

        if pd.isna(efficiency) or efficiency <= 0:
            audit_rows.append({
                "injured_player": event["injured_name"],
                "team": team,
                "position": position,
                "replacement_player": event["replacement_name"],
                "status": "NO_VALID_REPLACEMENT_EFFICIENCY",
                "projection_adjustment": 0.0,
            })
            continue

        historical_opportunity = pd.to_numeric(
            profile.get("replacement_opportunities_avg_3"),
            errors="coerce",
        )

        current_opportunity = pd.to_numeric(
            replacement.get("opportunities_avg_5"),
            errors="coerce",
        )

        if pd.isna(current_opportunity) or current_opportunity <= 0:
            current_opportunity = pd.to_numeric(
                replacement.get("opportunities_last"),
                errors="coerce",
            )

        if pd.isna(current_opportunity):
            current_opportunity = 0.0

        if pd.isna(historical_opportunity):
            audit_rows.append({
                "injured_player": event["injured_name"],
                "team": team,
                "position": position,
                "replacement_player": event["replacement_name"],
                "status": "NO_VALID_OPPORTUNITY_EVIDENCE",
                "projection_adjustment": 0.0,
            })
            continue

        added_opportunity = max(
            0.0,
            float(historical_opportunity) - float(current_opportunity),
        )

        adjustment = added_opportunity * float(efficiency)

        adjusted_projection = (
            float(baseline_projection) + adjustment
        )

        out.loc[
            replacement_mask,
            "injury_adjusted_projection",
        ] = adjusted_projection

        out.loc[
            replacement_mask,
            "projection_adjustment",
        ] = adjustment

        out.loc[
            replacement_mask,
            "projection_adjustment_reason",
        ] = (
            f"{event['injured_name']} OUT; "
            f"validated {position} replacement role"
        )

        audit_rows.append({
            "injured_player": event["injured_name"],
            "team": team,
            "position": position,
            "replacement_player": event["replacement_name"],
            "baseline_projection": baseline_projection,
            "historical_replacement_opportunity": historical_opportunity,
            "current_replacement_opportunity": current_opportunity,
            "added_opportunity": added_opportunity,
            "replacement_efficiency": efficiency,
            "projection_adjustment": adjustment,
            "adjusted_projection": adjusted_projection,
            "status": "ADJUSTED",
        })

    return out, pd.DataFrame(audit_rows)
