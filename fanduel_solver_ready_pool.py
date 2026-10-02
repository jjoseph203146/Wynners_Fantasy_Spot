#!/usr/bin/env python3
"""
fanduel_solver_ready_pool.py

Build a deterministic, solver-ready FanDuel NFL slate pool from the stable
slate-specific projection attachment layer.

Architecture
------------
fanduel_slate_projection_pool
        +
stable offensive FanDuel pool
        ↓
exact salary-bearing slate universe
exact QB/RB/WR/TE position
D/ST position for defenses
V3 production projection authority for promoted MODEL_READY offense
existing slate projection authority for all other rows
        ↓
solver-ready slate pool

Important rules
---------------
- Selected FanDuel slate salary is authoritative for contest membership.
- No salary inference.
- No fuzzy matching.
- Offensive position authority is stable `fd_position`.
- Defense position is normalized to `DST`.
- V3-promoted MODEL_READY offensive players use the verified production
  `ridge_projection` carried by the stable FanDuel player pool.
- Non-V3 offensive rows retain the existing slate projection path.
- DST rows retain the existing slate projection path.
- Injury consensus remains a one-way eligibility gate.
- Rows with missing/invalid roster positions are blocked.
- This script does NOT modify upstream stable tables/files.

Outputs
-------
SQLite:
    fanduel_solver_ready_pool
    fanduel_solver_ready_manifest

CSV:
    data/csv/fanduel_solver_ready_pool.csv
    data/csv/fanduel_solver_ready_manifest.csv
    data/csv/audit_fanduel_solver_ready_gaps.csv

Parquet:
    data/parquet/fanduel_solver_ready_pool.parquet
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATABASE_PATH, CSV_DIR, PARQUET_DIR


SOURCE_TABLE = "fanduel_slate_projection_pool"
OUTPUT_TABLE = "fanduel_solver_ready_pool"
MANIFEST_TABLE = "fanduel_solver_ready_manifest"
INJURY_TABLE = "injury_consensus_current"
OFFENSE_TABLE = "fanduel_player_pool"

OFFENSE_FILE_CANDIDATES = [
    Path(PARQUET_DIR) / "nfl_fanduel_player_pool.parquet",
    Path(CSV_DIR) / "nfl_fanduel_player_pool.csv",
    Path(PARQUET_DIR) / "fanduel_player_pool.parquet",
    Path(CSV_DIR) / "fanduel_player_pool.csv",
]

OUT_CSV = Path(CSV_DIR) / "fanduel_solver_ready_pool.csv"
OUT_PARQUET = Path(PARQUET_DIR) / "fanduel_solver_ready_pool.parquet"
OUT_MANIFEST = Path(CSV_DIR) / "fanduel_solver_ready_manifest.csv"
OUT_GAPS = Path(CSV_DIR) / "audit_fanduel_solver_ready_gaps.csv"

VALID_OFFENSIVE_POSITIONS = {"QB", "RB", "WR", "TE"}
VALID_SOLVER_POSITIONS = {"QB", "RB", "WR", "TE", "DST"}

TEAM_ALIASES = {
    "ARI": "ARI",
    "ARZ": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BLT": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "CLV": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "GNB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAC": "JAX",
    "JAX": "JAX",
    "KC": "KC",
    "KAN": "KC",
    "LV": "LV",
    "LVR": "LV",
    "OAK": "LV",
    "LAC": "LAC",
    "SD": "LAC",
    "SDG": "LAC",
    "LA": "LA",
    "LAR": "LA",
    "STL": "LA",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NWE": "NE",
    "NO": "NO",
    "NOR": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SF": "SF",
    "SFO": "SF",
    "SEA": "SEA",
    "TB": "TB",
    "TAM": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
    "WSH": "WAS",
}


def section(title: str) -> None:
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def first_col(df: pd.DataFrame, candidates, required=False):
    lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        hit = lower.get(candidate.lower())
        if hit is not None:
            return hit

    if required:
        raise RuntimeError(
            f"Required column not found. Tried: {candidates}. "
            f"Available: {list(df.columns)}"
        )

    return None


def norm_text(value) -> str:
    if pd.isna(value):
        return ""

    s = unicodedata.normalize(
        "NFKD",
        str(value),
    ).encode(
        "ascii",
        "ignore",
    ).decode("ascii")

    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)

    return re.sub(r"\s+", " ", s).strip()


def norm_team(value):
    if pd.isna(value):
        return None

    token = re.sub(
        r"[^A-Z]",
        "",
        str(value).upper(),
    )

    return TEAM_ALIASES.get(
        token,
        token if token in set(TEAM_ALIASES.values()) else None,
    )


def norm_position(value):
    if pd.isna(value):
        return None

    token = (
        str(value)
        .strip()
        .upper()
        .replace("D/ST", "DST")
        .replace("DEF", "DST")
    )

    if token in VALID_SOLVER_POSITIONS:
        return token

    return None


def boolish(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().any():
        return numeric.fillna(0).astype(int)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "1",
                "true",
                "yes",
                "y",
                "ready",
                "eligible",
            }
        )
        .astype(int)
    )


def load_offense_source(conn: sqlite3.Connection):
    if table_exists(conn, OFFENSE_TABLE):
        return (
            pd.read_sql_query(
                f'SELECT * FROM "{OFFENSE_TABLE}"',
                conn,
            ),
            f"sqlite:{OFFENSE_TABLE}",
        )

    for path in OFFENSE_FILE_CANDIDATES:
        if not path.exists():
            continue

        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        if not df.empty:
            return df, str(path)

    raise RuntimeError(
        "Stable offensive FanDuel pool not found. Searched:\n"
        + "\n".join(
            f"  - {p}"
            for p in OFFENSE_FILE_CANDIDATES
        )
    )


def resolve_source_schema(df: pd.DataFrame):
    return {
        "slate": first_col(
            df,
            ["slate_name", "slate"],
            True,
        ),
        "slug": first_col(
            df,
            ["slate_slug", "slug"],
            True,
        ),
        "player": first_col(
            df,
            ["player", "fd_name", "player_name"],
            True,
        ),
        "team": first_col(
            df,
            ["team_internal", "team"],
            True,
        ),
        "salary": first_col(
            df,
            ["salary"],
            True,
        ),
        "game": first_col(
            df,
            ["game_key", "game"],
            True,
        ),
        "is_dst": first_col(
            df,
            ["is_dst"],
        ),
        "projection": first_col(
            df,
            [
                "model_projection",
                "internal_projection",
                "ridge_projection",
                "dst_projection",
            ],
            True,
        ),
        "projection_status": first_col(
            df,
            ["projection_status"],
            True,
        ),
        "optimizer_eligible": first_col(
            df,
            ["optimizer_eligible"],
        ),
        "identity": first_col(
            df,
            [
                "projection_identity",
                "identity_key",
                "player_id",
            ],
        ),
        "source_projection": first_col(
            df,
            [
                "source_fantasy_projection",
                "fantasy",
            ],
        ),
    }


def resolve_offense_schema(df: pd.DataFrame):
    return {
        "player": first_col(
            df,
            [
                "fd_name",
                "player_display_name",
                "player",
                "player_name",
            ],
            True,
        ),
        "team": first_col(
            df,
            [
                "team",
                "model_team",
                "team_internal",
            ],
            True,
        ),
        "position": first_col(
            df,
            [
                "fd_position",
                "model_position",
                "position",
                "pos",
            ],
            True,
        ),
        "identity": first_col(
            df,
            [
                "identity_key",
                "player_id",
                "gsis_id",
            ],
        ),
        "ridge_projection": first_col(
            df,
            ["ridge_projection"],
        ),
        "production_status": first_col(
            df,
            ["production_status"],
        ),
        "reconciliation_version": first_col(
            df,
            ["offensive_reconciliation_version"],
        ),
        "reconciliation_projection": first_col(
            df,
            ["offensive_reconciliation_fd_points"],
        ),
        "projection_adjustment": first_col(
            df,
            ["projection_adjustment"],
        ),
    }


def prepare_offense_lookup(df: pd.DataFrame, schema):
    out = df.copy()

    out["_player_norm"] = out[
        schema["player"]
    ].map(norm_text)

    out["_team_norm"] = out[
        schema["team"]
    ].map(norm_team)

    out["_position_norm"] = out[
        schema["position"]
    ].map(norm_position)

    if schema["identity"]:
        out["_identity"] = out[
            schema["identity"]
        ]
    else:
        out["_identity"] = ""

    if schema["ridge_projection"]:
        out["_offense_ridge_projection"] = pd.to_numeric(
            out[schema["ridge_projection"]],
            errors="coerce",
        )
    else:
        out["_offense_ridge_projection"] = np.nan

    if schema["production_status"]:
        out["_production_status"] = (
            out[schema["production_status"]]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        out["_production_status"] = ""

    if schema["reconciliation_version"]:
        out["_reconciliation_version"] = (
            out[schema["reconciliation_version"]]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        out["_reconciliation_version"] = ""

    if schema["reconciliation_projection"]:
        out["_reconciliation_projection"] = pd.to_numeric(
            out[schema["reconciliation_projection"]],
            errors="coerce",
        )
    else:
        out["_reconciliation_projection"] = np.nan

    if schema["projection_adjustment"]:
        out["_projection_adjustment"] = pd.to_numeric(
            out[schema["projection_adjustment"]],
            errors="coerce",
        ).fillna(0.0)
    else:
        out["_projection_adjustment"] = 0.0

    # Only exact QB/RB/WR/TE rows can provide offensive authority.
    out = out[
        out["_position_norm"].isin(
            VALID_OFFENSIVE_POSITIONS
        )
    ].copy()

    counts = (
        out.groupby(
            [
                "_player_norm",
                "_team_norm",
            ],
            dropna=False,
        )
        .size()
        .rename("_match_count")
        .reset_index()
    )

    out = out.merge(
        counts,
        on=[
            "_player_norm",
            "_team_norm",
        ],
        how="left",
    )

    # Require deterministic uniqueness.
    #
    # FanDuel players may occur on multiple overlapping slates. Those
    # duplicate slate rows are allowed only when their authoritative
    # position / identity / V3 projection state agree.
    rows = []

    for _, group in out.groupby(
        [
            "_player_norm",
            "_team_norm",
        ],
        dropna=False,
    ):
        positions = sorted(
            set(
                group[
                    "_position_norm"
                ]
                .dropna()
                .astype(str)
            )
        )

        identities = sorted(
            set(
                str(x)
                for x in group[
                    "_identity"
                ].dropna()
                if (
                    str(x).strip()
                    and str(x).lower() != "nan"
                )
            )
        )

        if len(positions) != 1:
            continue

        # Validate V3 production values across overlapping slate rows.
        v3_group = group[
            group["_reconciliation_version"].eq("V3")
            & group["_production_status"].eq("MODEL_READY")
        ].copy()

        v3_values = sorted(
            set(
                float(x)
                for x in v3_group[
                    "_offense_ridge_projection"
                ].dropna()
                if np.isfinite(float(x))
            )
        )

        if len(v3_values) > 1:
            raise RuntimeError(
                "Conflicting V3 production projections for exact "
                f"offensive identity: "
                f"{group.iloc[0]['_player_norm']} / "
                f"{group.iloc[0]['_team_norm']}"
            )

        v3_adjustments = sorted(
            set(
                float(x)
                for x in v3_group[
                    "_projection_adjustment"
                ].dropna()
                if np.isfinite(float(x))
            )
        )

        if len(v3_adjustments) > 1:
            raise RuntimeError(
                "Conflicting V3 projection adjustments for exact "
                f"offensive identity: "
                f"{group.iloc[0]['_player_norm']} / "
                f"{group.iloc[0]['_team_norm']}"
            )

        rec = group.iloc[0].copy()

        rec["_position_norm"] = positions[0]

        rec["_identity"] = (
            identities[0]
            if len(identities) == 1
            else (
                rec["_identity"]
                if len(identities) == 0
                else ""
            )
        )

        rec["_source_rows"] = len(group)
        rec["_position_conflict"] = 0

        # If any overlapping FanDuel row is an authoritative V3
        # MODEL_READY row, preserve that state in the collapsed lookup.
        if not v3_group.empty:
            rec["_reconciliation_version"] = "V3"
            rec["_production_status"] = "MODEL_READY"

            if len(v3_values) == 1:
                rec["_offense_ridge_projection"] = v3_values[0]

            if len(v3_adjustments) == 1:
                rec["_projection_adjustment"] = v3_adjustments[0]
            else:
                rec["_projection_adjustment"] = 0.0

            recon_values = sorted(
                set(
                    float(x)
                    for x in v3_group[
                        "_reconciliation_projection"
                    ].dropna()
                    if np.isfinite(float(x))
                )
            )

            if len(recon_values) > 1:
                raise RuntimeError(
                    "Conflicting V3 reconciliation projections for exact "
                    f"offensive identity: "
                    f"{group.iloc[0]['_player_norm']} / "
                    f"{group.iloc[0]['_team_norm']}"
                )

            if len(recon_values) == 1:
                rec["_reconciliation_projection"] = recon_values[0]

        rows.append(rec)

    if not rows:
        return pd.DataFrame(
            columns=[
                "_player_norm",
                "_team_norm",
                "_position_norm",
                "_identity",
                "_offense_ridge_projection",
                "_production_status",
                "_reconciliation_version",
                "_reconciliation_projection",
                "_projection_adjustment",
            ]
        )

    return pd.DataFrame(rows)


def canonical_gsis_from_identity(value) -> str:
    if pd.isna(value):
        return ""

    token = str(value).strip()

    if not token or token.lower() == "nan":
        return ""

    if token.startswith("GSIS:"):
        return token[5:].strip()

    return ""


def load_injury_consensus(
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    if not table_exists(
        conn,
        INJURY_TABLE,
    ):
        raise RuntimeError(
            f"Required injury consensus table not found: "
            f"{INJURY_TABLE}. "
            "Run injury_consensus.py first."
        )

    cols = {
        row[1]
        for row in conn.execute(
            f'PRAGMA table_info("{INJURY_TABLE}")'
        )
    }

    required = {
        "gsis_id",
        "injury_gate",
        "consensus_status",
        "injury_gate_reason",
        "injury_risk",
    }

    missing = sorted(
        required - cols
    )

    if missing:
        raise RuntimeError(
            "Injury consensus table missing required columns: "
            + ",".join(missing)
        )

    df = pd.read_sql_query(
        f"""
        SELECT
            gsis_id,
            injury_gate,
            consensus_status,
            injury_gate_reason,
            injury_risk
        FROM "{INJURY_TABLE}"
        """,
        conn,
    )

    if df.empty:
        raise RuntimeError(
            "Injury consensus table returned zero rows."
        )

    for col in [
        "gsis_id",
        "injury_gate",
        "consensus_status",
        "injury_gate_reason",
        "injury_risk",
    ]:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df["injury_gate"] = (
        df["injury_gate"]
        .str.upper()
    )

    df["consensus_status"] = (
        df["consensus_status"]
        .str.upper()
    )

    df["injury_risk"] = (
        df["injury_risk"]
        .str.upper()
    )

    if df["gsis_id"].eq("").any():
        raise RuntimeError(
            "Injury consensus contains blank GSIS IDs."
        )

    dupes = df.duplicated(
        ["gsis_id"],
        keep=False,
    )

    if dupes.any():
        bad = sorted(
            df.loc[
                dupes,
                "gsis_id",
            ]
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Injury consensus contains duplicate GSIS IDs: "
            + ",".join(bad)
        )

    invalid_gate = ~df[
        "injury_gate"
    ].isin(
        {
            "ALLOW",
            "BLOCK",
        }
    )

    if invalid_gate.any():
        bad = sorted(
            df.loc[
                invalid_gate,
                "injury_gate",
            ]
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Injury consensus contains invalid injury_gate values: "
            + ",".join(bad)
        )

    return df


def apply_injury_gate(
    solver: pd.DataFrame,
    injury_consensus: pd.DataFrame,
) -> pd.DataFrame:
    out = solver.copy()

    injury_map = {
        row["gsis_id"]: row
        for row in injury_consensus.to_dict(
            "records"
        )
    }

    out["injury_consensus_found"] = 0
    out["injury_gsis_id"] = ""
    out["injury_gate"] = ""
    out["injury_status"] = ""
    out["injury_gate_reason"] = ""
    out["injury_risk"] = ""

    for idx in out.index:
        if int(
            out.at[
                idx,
                "_is_dst",
            ]
        ) == 1:
            continue

        gsis_id = canonical_gsis_from_identity(
            out.at[
                idx,
                "position_identity",
            ]
        )

        if not gsis_id:
            continue

        out.at[
            idx,
            "injury_gsis_id",
        ] = gsis_id

        match = injury_map.get(
            gsis_id
        )

        if match is None:
            continue

        out.at[
            idx,
            "injury_consensus_found",
        ] = 1

        out.at[
            idx,
            "injury_gate",
        ] = match[
            "injury_gate"
        ]

        out.at[
            idx,
            "injury_status",
        ] = match[
            "consensus_status"
        ]

        out.at[
            idx,
            "injury_gate_reason",
        ] = match[
            "injury_gate_reason"
        ]

        out.at[
            idx,
            "injury_risk",
        ] = match[
            "injury_risk"
        ]

        # Injury logic is one-way: it may only remove eligibility.
        if match[
            "injury_gate"
        ] == "BLOCK":
            out.at[
                idx,
                "solver_status",
            ] = "BLOCKED_INJURY"

    # solver_status is authoritative.
    out["solver_eligible"] = (
        out["solver_status"]
        .eq("READY")
        .astype(int)
    )

    escapes = out[
        out["injury_gate"].eq("BLOCK")
        & out["solver_eligible"].eq(1)
    ]

    if not escapes.empty:
        raise RuntimeError(
            "Injury gate escape detected: "
            "BLOCK row remains solver eligible."
        )

    return out


def build_solver_pool(
    source: pd.DataFrame,
    source_schema,
    offense_lookup: pd.DataFrame,
):
    out = source.copy()

    out["_player_norm"] = out[
        source_schema["player"]
    ].map(norm_text)

    out["_team_norm"] = out[
        source_schema["team"]
    ].map(norm_team)

    out["_salary_num"] = pd.to_numeric(
        out[
            source_schema["salary"]
        ],
        errors="coerce",
    )

    # Preserve the original slate projection for audit/provenance.
    out["_source_projection_num"] = pd.to_numeric(
        out[
            source_schema["projection"]
        ],
        errors="coerce",
    )

    out["_projection_num"] = (
        out["_source_projection_num"]
        .copy()
    )

    out["solver_projection_authority"] = (
        "SLATE_PROJECTION"
    )

    if source_schema["is_dst"]:
        raw_dst = boolish(
            out[
                source_schema["is_dst"]
            ]
        ).eq(1)
    else:
        raw_dst = (
            out["_player_norm"]
            .str.contains(
                r"(?:\bdst\b|\bdefense\b|\bdef\b)",
                regex=True,
            )
        )

    out["_is_dst"] = (
        raw_dst.astype(int)
    )

    position_map = {
        (
            rec["_player_norm"],
            rec["_team_norm"],
        ): rec
        for rec in offense_lookup.to_dict(
            "records"
        )
    }

    out["solver_position"] = None
    out["position_match_method"] = ""
    out["position_identity"] = ""
    out["solver_status"] = "BLOCKED"

    out["v3_projection_applied"] = 0
    out["v3_projection_value"] = np.nan
    out["v3_projection_adjustment"] = 0.0
    out["v3_injury_rebased_projection"] = np.nan

    for idx in out.index:
        salary_ready = (
            pd.notna(
                out.at[
                    idx,
                    "_salary_num",
                ]
            )
            and float(
                out.at[
                    idx,
                    "_salary_num",
                ]
            ) > 0
        )

        # DST remains entirely on the existing slate projection path.
        if int(
            out.at[
                idx,
                "_is_dst",
            ]
        ) == 1:
            projection_ready = (
                str(
                    out.at[
                        idx,
                        source_schema[
                            "projection_status"
                        ],
                    ]
                ).upper()
                == "READY"
                and pd.notna(
                    out.at[
                        idx,
                        "_projection_num",
                    ]
                )
                and np.isfinite(
                    float(
                        out.at[
                            idx,
                            "_projection_num",
                        ]
                    )
                )
            )

            out.at[
                idx,
                "solver_position",
            ] = "DST"

            out.at[
                idx,
                "position_match_method",
            ] = "dst_exact_slate_row"

            out.at[
                idx,
                "solver_projection_authority",
            ] = "SLATE_PROJECTION_DST"

            if (
                projection_ready
                and salary_ready
            ):
                out.at[
                    idx,
                    "solver_status",
                ] = "READY"

            elif not projection_ready:
                out.at[
                    idx,
                    "solver_status",
                ] = "BLOCKED_PROJECTION"

            else:
                out.at[
                    idx,
                    "solver_status",
                ] = "BLOCKED_SALARY"

            continue

        key = (
            out.at[
                idx,
                "_player_norm",
            ],
            out.at[
                idx,
                "_team_norm",
            ],
        )

        match = position_map.get(
            key
        )

        if match is None:
            out.at[
                idx,
                "solver_status",
            ] = "BLOCKED_POSITION"

            continue

        out.at[
            idx,
            "solver_position",
        ] = match[
            "_position_norm"
        ]

        out.at[
            idx,
            "position_match_method",
        ] = "exact_name_team_fd_position"

        out.at[
            idx,
            "position_identity",
        ] = match.get(
            "_identity",
            "",
        )

        # -------------------------------------------------
        # V3 production projection handoff
        # -------------------------------------------------
        #
        # Only authoritative V3 + MODEL_READY offensive rows
        # override the legacy slate projection.
        #
        # The stable FanDuel pool's ridge_projection already
        # contains the verified V3 reconciled FanDuel value.
        #
        # We intentionally do NOT use live_projection here:
        # live_projection currently prefers the older
        # injury_adjusted_projection and would bypass V3.
        # -------------------------------------------------

        reconciliation_version = str(
            match.get(
                "_reconciliation_version",
                "",
            )
        ).strip().upper()

        production_status = str(
            match.get(
                "_production_status",
                "",
            )
        ).strip().upper()

        offense_projection = pd.to_numeric(
            pd.Series(
                [
                    match.get(
                        "_offense_ridge_projection",
                        np.nan,
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]

        reconciliation_projection = pd.to_numeric(
            pd.Series(
                [
                    match.get(
                        "_reconciliation_projection",
                        np.nan,
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]

        use_v3 = (
            reconciliation_version == "V3"
            and production_status == "MODEL_READY"
            and pd.notna(
                offense_projection
            )
            and np.isfinite(
                float(
                    offense_projection
                )
            )
        )

        if use_v3:
            # Fail closed if the carried production Ridge value
            # disagrees with the explicit V3 reconciliation value.
            if (
                pd.notna(
                    reconciliation_projection
                )
                and np.isfinite(
                    float(
                        reconciliation_projection
                    )
                )
                and abs(
                    float(
                        offense_projection
                    )
                    - float(
                        reconciliation_projection
                    )
                )
                > 1e-9
            ):
                raise RuntimeError(
                    "V3 production/reconciliation projection "
                    "mismatch for "
                    f"{out.at[idx, source_schema['player']]} / "
                    f"{out.at[idx, source_schema['team']]}"
                )

            projection_adjustment = pd.to_numeric(
                pd.Series(
                    [
                        match.get(
                            "_projection_adjustment",
                            0.0,
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            if pd.isna(projection_adjustment):
                projection_adjustment = 0.0

            if not np.isfinite(float(projection_adjustment)):
                raise RuntimeError(
                    "Non-finite V3 projection adjustment for "
                    f"{out.at[idx, source_schema['player']]} / "
                    f"{out.at[idx, source_schema['team']]}"
                )

            injury_rebased_projection = (
                float(offense_projection)
                + float(projection_adjustment)
            )

            if not np.isfinite(injury_rebased_projection):
                raise RuntimeError(
                    "Non-finite V3 injury-rebased projection for "
                    f"{out.at[idx, source_schema['player']]} / "
                    f"{out.at[idx, source_schema['team']]}"
                )

            out.at[
                idx,
                "_projection_num",
            ] = injury_rebased_projection

            out.at[
                idx,
                "v3_projection_applied",
            ] = 1

            out.at[
                idx,
                "v3_projection_value",
            ] = float(
                offense_projection
            )

            out.at[
                idx,
                "v3_projection_adjustment",
            ] = float(
                projection_adjustment
            )

            out.at[
                idx,
                "v3_injury_rebased_projection",
            ] = injury_rebased_projection

            out.at[
                idx,
                "solver_projection_authority",
            ] = "V3_PRODUCTION_RIDGE_PLUS_INJURY_ADJUSTMENT"

        else:
            out.at[
                idx,
                "solver_projection_authority",
            ] = "SLATE_PROJECTION_OFFENSE"

        projection_ready = (
            str(
                out.at[
                    idx,
                    source_schema[
                        "projection_status"
                    ],
                ]
            ).upper()
            == "READY"
            and pd.notna(
                out.at[
                    idx,
                    "_projection_num",
                ]
            )
            and np.isfinite(
                float(
                    out.at[
                        idx,
                        "_projection_num",
                    ]
                )
            )
        )

        if not salary_ready:
            out.at[
                idx,
                "solver_status",
            ] = "BLOCKED_SALARY"

        elif not projection_ready:
            out.at[
                idx,
                "solver_status",
            ] = "BLOCKED_PROJECTION"

        else:
            out.at[
                idx,
                "solver_status",
            ] = "READY"

    out["solver_eligible"] = (
        out["solver_status"]
        .eq("READY")
        .astype(int)
    )

    # Canonical solver-facing columns.
    # Preserve source columns as well.
    out["slate_name_solver"] = out[
        source_schema["slate"]
    ]

    out["slate_slug_solver"] = out[
        source_schema["slug"]
    ]

    out["player_solver"] = out[
        source_schema["player"]
    ]

    out["team_solver"] = out[
        "_team_norm"
    ]

    out["salary_solver"] = out[
        "_salary_num"
    ]

    out["game_solver"] = out[
        source_schema["game"]
    ]

    out["projection_solver"] = out[
        "_projection_num"
    ]

    # Give defenses a stable solver identity even if upstream
    # player identity is blank.
    out["solver_player_key"] = np.where(
        out["_is_dst"].eq(1),
        "DST:"
        + out[
            "_team_norm"
        ].fillna(""),
        out[
            "position_identity"
        ].astype(str),
    )

    return out


def build_manifest(
    df: pd.DataFrame,
):
    rows = []

    for (
        slate,
        slug,
    ), g in df.groupby(
        [
            "slate_name_solver",
            "slate_slug_solver",
        ],
        dropna=False,
    ):
        eligible = g[
            g[
                "solver_eligible"
            ].eq(1)
        ]

        dst = g[
            g[
                "_is_dst"
            ].eq(1)
        ]

        offense = g[
            g[
                "_is_dst"
            ].eq(0)
        ]

        counts = (
            eligible[
                "solver_position"
            ]
            .value_counts()
            .to_dict()
        )

        invalid_position_rows = int(
            eligible[
                "solver_position"
            ].isna().sum()
            + (
                ~eligible[
                    "solver_position"
                ].isin(
                    VALID_SOLVER_POSITIONS
                )
            ).sum()
        )

        rows.append(
            {
                "slate_name": slate,
                "slate_slug": slug,
                "contest_rows": len(g),
                "solver_eligible_rows": len(
                    eligible
                ),
                "solver_blocked_rows": int(
                    g[
                        "solver_eligible"
                    ].eq(0).sum()
                ),
                "offense_rows": len(
                    offense
                ),
                "dst_rows": len(
                    dst
                ),
                "eligible_qb": int(
                    counts.get(
                        "QB",
                        0,
                    )
                ),
                "eligible_rb": int(
                    counts.get(
                        "RB",
                        0,
                    )
                ),
                "eligible_wr": int(
                    counts.get(
                        "WR",
                        0,
                    )
                ),
                "eligible_te": int(
                    counts.get(
                        "TE",
                        0,
                    )
                ),
                "eligible_dst": int(
                    counts.get(
                        "DST",
                        0,
                    )
                ),
                "blocked_projection": int(
                    g[
                        "solver_status"
                    ]
                    .eq(
                        "BLOCKED_PROJECTION"
                    )
                    .sum()
                ),
                "blocked_position": int(
                    g[
                        "solver_status"
                    ]
                    .eq(
                        "BLOCKED_POSITION"
                    )
                    .sum()
                ),
                "blocked_salary": int(
                    g[
                        "solver_status"
                    ]
                    .eq(
                        "BLOCKED_SALARY"
                    )
                    .sum()
                ),
                "blocked_injury": int(
                    g[
                        "solver_status"
                    ]
                    .eq(
                        "BLOCKED_INJURY"
                    )
                    .sum()
                ),
                "invalid_eligible_position_rows":
                    invalid_position_rows,
                "v3_projection_rows": int(
                    g[
                        "v3_projection_applied"
                    ].eq(1).sum()
                ),
                "status": (
                    "PASS"
                    if (
                        len(
                            eligible
                        ) > 0
                        and counts.get(
                            "QB",
                            0,
                        ) >= 1
                        and counts.get(
                            "RB",
                            0,
                        ) >= 3
                        and counts.get(
                            "WR",
                            0,
                        ) >= 4
                        and counts.get(
                            "TE",
                            0,
                        ) >= 1
                        and counts.get(
                            "DST",
                            0,
                        ) >= 1
                        and invalid_position_rows
                        == 0
                    )
                    else "BLOCKED"
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "slate_name"
        )
        .reset_index(
            drop=True
        )
    )


def clean_export(
    df: pd.DataFrame,
):
    # Keep source provenance, including V3 authority fields,
    # but remove private normalization helpers.
    drop = [
        "_player_norm",
        "_team_norm",
        "_salary_num",
        "_projection_num",
        "_source_projection_num",
        "_is_dst",
    ]

    return df.drop(
        columns=[
            c
            for c in drop
            if c in df.columns
        ]
    )


def main():
    section(
        "FANDUEL SOLVER-READY SLATE POOL"
    )

    print(
        f"Database: {DATABASE_PATH}"
    )

    print(
        f"Projection source: {SOURCE_TABLE}"
    )

    print(
        "Position authority: stable offensive fd_position + DST"
    )

    print(
        "Offensive V3 projection authority: "
        "stable FanDuel player pool ridge_projection"
    )

    print(
        "Non-V3/DST projection authority: "
        "fanduel_slate_projection_pool"
    )

    print(
        "Slate salary authority: fanduel_slate_projection_pool"
    )

    print(
        "Fuzzy matching: disabled"
    )

    print(
        "Salary inference: disabled"
    )

    print(
        "Upstream stable files/tables: read only"
    )

    with sqlite3.connect(
        DATABASE_PATH
    ) as conn:
        if not table_exists(
            conn,
            SOURCE_TABLE,
        ):
            raise RuntimeError(
                f"Required SQLite table not found: "
                f"{SOURCE_TABLE}. "
                "Run fanduel_slate_projection_attach_v5.py first."
            )

        source = pd.read_sql_query(
            f'SELECT * FROM "{SOURCE_TABLE}"',
            conn,
        )

        offense, offense_source = (
            load_offense_source(
                conn
            )
        )

        source_schema = (
            resolve_source_schema(
                source
            )
        )

        offense_schema = (
            resolve_offense_schema(
                offense
            )
        )

        section(
            "RESOLVED SOURCES"
        )

        print(
            f"Projection rows: {len(source)}"
        )

        print(
            f"Offensive source: {offense_source}"
        )

        print(
            f"Offensive rows: {len(offense)}"
        )

        print()
        print(
            "Projection source schema:"
        )

        for (
            k,
            v,
        ) in source_schema.items():
            print(
                f"  {k:<22} -> {v}"
            )

        print()
        print(
            "Offensive schema:"
        )

        for (
            k,
            v,
        ) in offense_schema.items():
            print(
                f"  {k:<22} -> {v}"
            )

        offense_lookup = (
            prepare_offense_lookup(
                offense,
                offense_schema,
            )
        )

        injury_consensus = (
            load_injury_consensus(
                conn
            )
        )

        solver = build_solver_pool(
            source,
            source_schema,
            offense_lookup,
        )

        solver = apply_injury_gate(
            solver,
            injury_consensus,
        )

        manifest = build_manifest(
            solver
        )

        section(
            "SOLVER-READY COVERAGE BY SLATE"
        )

        print(
            manifest.to_string(
                index=False
            )
        )

        section(
            "GLOBAL STRUCTURAL AUDIT"
        )

        ready = solver[
            "solver_eligible"
        ].eq(1)

        blocked = ~ready

        print(
            f"Source contest rows:             {len(solver)}"
        )

        print(
            f"Solver eligible rows:           {int(ready.sum())}"
        )

        print(
            f"Solver blocked rows:            {int(blocked.sum())}"
        )

        print(
            "Blocked projection rows:        "
            f"{int(solver['solver_status'].eq('BLOCKED_PROJECTION').sum())}"
        )

        print(
            "Blocked position rows:          "
            f"{int(solver['solver_status'].eq('BLOCKED_POSITION').sum())}"
        )

        print(
            "Blocked salary rows:            "
            f"{int(solver['solver_status'].eq('BLOCKED_SALARY').sum())}"
        )

        print(
            "Blocked injury rows:            "
            f"{int(solver['solver_status'].eq('BLOCKED_INJURY').sum())}"
        )

        print(
            "Eligible rows missing position: "
            f"{int(solver.loc[ready, 'solver_position'].isna().sum())}"
        )

        print(
            "Eligible rows invalid position: "
            f"{int((~solver.loc[ready, 'solver_position'].isin(VALID_SOLVER_POSITIONS)).sum())}"
        )

        print(
            "Duplicate slate/player/team:    "
            f"{int(solver.duplicated(['slate_slug_solver', 'player_solver', 'team_solver', 'salary_solver']).sum())}"
        )

        print(
            "V3 projection rows:             "
            f"{int(solver['v3_projection_applied'].eq(1).sum())}"
        )

        v3_rows = solver[
            solver[
                "v3_projection_applied"
            ].eq(1)
        ].copy()

        if not v3_rows.empty:
            v3_base = pd.to_numeric(
                v3_rows[
                    "v3_projection_value"
                ],
                errors="coerce",
            )

            v3_adjustment = pd.to_numeric(
                v3_rows[
                    "v3_projection_adjustment"
                ],
                errors="coerce",
            ).fillna(0.0)

            v3_expected = (
                v3_base
                + v3_adjustment
            )

            v3_actual = pd.to_numeric(
                v3_rows[
                    "projection_solver"
                ],
                errors="coerce",
            )

            v3_gap = (
                v3_actual
                - v3_expected
            ).abs()

            v3_bad = int(
                (
                    v3_gap.isna()
                    | (
                        v3_gap
                        > 1e-9
                    )
                ).sum()
            )

            v3_max_gap = float(
                v3_gap.max()
            )

            v3_adjusted_rows = int(
                v3_adjustment
                .abs()
                .gt(1e-12)
                .sum()
            )

            v3_max_adjustment = float(
                v3_adjustment
                .abs()
                .max()
            )

        else:
            v3_bad = 0
            v3_max_gap = 0.0
            v3_adjusted_rows = 0
            v3_max_adjustment = 0.0

        print(
            "V3 injury-adjusted rows:        "
            f"{v3_adjusted_rows}"
        )

        print(
            "V3 max injury adjustment:       "
            f"{v3_max_adjustment:.12f}"
        )

        print(
            "V3 injury-rebase mismatches:    "
            f"{v3_bad}"
        )

        print(
            "V3 injury-rebase max gap:       "
            f"{v3_max_gap:.12f}"
        )

        if v3_bad:
            raise RuntimeError(
                "V3 injury-rebased projection handoff "
                "failed internal audit."
            )

        section(
            "GLOBAL ELIGIBLE POSITION COUNTS"
        )

        pos_counts = (
            solver.loc[
                ready,
                "solver_position",
            ]
            .value_counts()
            .rename_axis(
                "position"
            )
            .reset_index(
                name="rows"
            )
        )

        print(
            pos_counts.to_string(
                index=False
            )
        )

        gaps = solver[
            blocked
        ].copy()

        section(
            "BLOCKED ROW BREAKDOWN"
        )

        if gaps.empty:
            print(
                "No blocked rows."
            )
        else:
            print(
                gaps.groupby(
                    [
                        "solver_status"
                    ],
                    dropna=False,
                )
                .size()
                .rename(
                    "rows"
                )
                .reset_index()
                .to_string(
                    index=False
                )
            )

        exported = clean_export(
            solver
        )

        # Write derived outputs only.
        OUT_CSV.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUT_PARQUET.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        exported.to_csv(
            OUT_CSV,
            index=False,
        )

        manifest.to_csv(
            OUT_MANIFEST,
            index=False,
        )

        gap_cols = [
            c
            for c in [
                "slate_name_solver",
                "player_solver",
                "team_solver",
                "salary_solver",
                "game_solver",
                "solver_position",
                "projection_solver",
                "projection_status",
                "solver_projection_authority",
                "v3_projection_applied",
                "v3_projection_value",
                "solver_status",
                "position_match_method",
                "injury_consensus_found",
                "injury_gsis_id",
                "injury_gate",
                "injury_status",
                "injury_gate_reason",
                "injury_risk",
            ]
            if c in solver.columns
        ]

        gaps[
            gap_cols
        ].to_csv(
            OUT_GAPS,
            index=False,
        )

        exported.to_sql(
            OUTPUT_TABLE,
            conn,
            if_exists="replace",
            index=False,
        )

        manifest.to_sql(
            MANIFEST_TABLE,
            conn,
            if_exists="replace",
            index=False,
        )

        conn.execute(
            f'CREATE INDEX IF NOT EXISTS '
            f'idx_{OUTPUT_TABLE}_slate '
            f'ON {OUTPUT_TABLE}(slate_slug_solver)'
        )

        conn.execute(
            f'CREATE INDEX IF NOT EXISTS '
            f'idx_{OUTPUT_TABLE}_eligible '
            f'ON {OUTPUT_TABLE}'
            f'(slate_slug_solver, solver_eligible)'
        )

        conn.execute(
            f'CREATE INDEX IF NOT EXISTS '
            f'idx_{OUTPUT_TABLE}_position '
            f'ON {OUTPUT_TABLE}'
            f'(slate_slug_solver, solver_position)'
        )

        conn.commit()

        # Written last, after the SQLite commit: app.py keys its Streamlit
        # cache freshness token off this file's mtime/size, and treats SQLite
        # as the authoritative table. Writing it earlier opened a window
        # where the UI cache invalidated and re-queried SQLite before the
        # authoritative table finished being replaced.
        exported.to_parquet(
            OUT_PARQUET,
            index=False,
        )

    section(
        "EXPORTS"
    )

    print(
        f"Pool CSV:     {OUT_CSV}"
    )

    print(
        f"Pool Parquet: {OUT_PARQUET}"
    )

    print(
        f"Manifest:     {OUT_MANIFEST}"
    )

    print(
        f"Gap audit:    {OUT_GAPS}"
    )

    print(
        f"SQLite pool:  {OUTPUT_TABLE}"
    )

    print(
        f"SQLite audit: {MANIFEST_TABLE}"
    )

    section(
        "SOLVER-READY POOL COMPLETE"
    )

    if (
        manifest[
            "status"
        ]
        == "PASS"
    ).all():
        print(
            "ALL SLATES PASS "
            "position/projection/salary structural gates."
        )
    else:
        print(
            "One or more slates are BLOCKED. "
            "Review the manifest before solving."
        )

    print(
        "V3-promoted MODEL_READY offensive players use the "
        "verified production V3 projection plus the validated additive "
        "injury adjustment. Other offense and DST retain the existing "
        "projection path."
    )

    print(
        "Only exact-position, positive-salary, internally projected, "
        "injury-allowed rows are marked solver_eligible=1."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print(
            "=" * 112
        )
        print(
            "FANDUEL SOLVER-READY POOL FAILED"
        )
        print(
            "=" * 112
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )
        raise
