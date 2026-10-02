#!/usr/bin/env python3
"""
fanduel_solver_ready_pool.py

Build a deterministic, solver-ready FanDuel NFL slate pool from the stable
slate-specific projection attachment layer.

Architecture
------------
fanduel_slate_projection_pool
        +
stable offensive FanDuel pool (position authority only)
        ↓
exact salary-bearing slate universe
exact QB/RB/WR/TE position
D/ST position for defenses
internal model projection
        ↓
solver-ready slate pool

Important rules
---------------
- Selected FanDuel slate salary is authoritative for contest membership.
- No salary inference.
- No fuzzy matching.
- Offensive position authority is stable `fd_position`.
- Defense position is normalized to `DST`.
- Only rows with projection_status == READY can be optimizer eligible.
- Rows with missing/invalid roster positions are blocked.
- This script does NOT modify any upstream stable table/file.

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
    "ARI":"ARI","ARZ":"ARI","ATL":"ATL","BAL":"BAL","BLT":"BAL",
    "BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE",
    "CLV":"CLE","DAL":"DAL","DEN":"DEN","DET":"DET","GB":"GB",
    "GNB":"GB","HOU":"HOU","IND":"IND","JAC":"JAX","JAX":"JAX",
    "KC":"KC","KAN":"KC","LV":"LV","LVR":"LV","OAK":"LV",
    "LAC":"LAC","SD":"LAC","SDG":"LAC","LA":"LA","LAR":"LA",
    "STL":"LA","MIA":"MIA","MIN":"MIN","NE":"NE","NWE":"NE",
    "NO":"NO","NOR":"NO","NYG":"NYG","NYJ":"NYJ","PHI":"PHI",
    "PIT":"PIT","SF":"SF","SFO":"SF","SEA":"SEA","TB":"TB",
    "TAM":"TB","TEN":"TEN","WAS":"WAS","WSH":"WAS",
}


def section(title: str) -> None:
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


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
    s = unicodedata.normalize("NFKD", str(value)).encode(
        "ascii", "ignore"
    ).decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_team(value):
    if pd.isna(value):
        return None
    token = re.sub(r"[^A-Z]", "", str(value).upper())
    return TEAM_ALIASES.get(token, token if token in set(TEAM_ALIASES.values()) else None)


def norm_position(value):
    if pd.isna(value):
        return None
    token = str(value).strip().upper().replace("D/ST", "DST").replace("DEF", "DST")
    if token in VALID_SOLVER_POSITIONS:
        return token
    return None


def boolish(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "ready", "eligible"})
        .astype(int)
    )


def load_offense_source(conn: sqlite3.Connection):
    if table_exists(conn, OFFENSE_TABLE):
        return (
            pd.read_sql_query(f'SELECT * FROM "{OFFENSE_TABLE}"', conn),
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
        + "\n".join(f"  - {p}" for p in OFFENSE_FILE_CANDIDATES)
    )


def resolve_source_schema(df: pd.DataFrame):
    return {
        "slate": first_col(df, ["slate_name", "slate"], True),
        "slug": first_col(df, ["slate_slug", "slug"], True),
        "player": first_col(df, ["player", "fd_name", "player_name"], True),
        "team": first_col(df, ["team_internal", "team"], True),
        "salary": first_col(df, ["salary"], True),
        "game": first_col(df, ["game_key", "game"], True),
        "is_dst": first_col(df, ["is_dst"]),
        "projection": first_col(
            df,
            ["model_projection", "internal_projection", "ridge_projection", "dst_projection"],
            True,
        ),
        "projection_status": first_col(df, ["projection_status"], True),
        "optimizer_eligible": first_col(df, ["optimizer_eligible"]),
        "identity": first_col(df, ["projection_identity", "identity_key", "player_id"]),
        "source_projection": first_col(
            df, ["source_fantasy_projection", "fantasy"]
        ),
    }


def resolve_offense_schema(df: pd.DataFrame):
    return {
        "player": first_col(
            df, ["fd_name", "player_display_name", "player", "player_name"], True
        ),
        "team": first_col(df, ["team", "model_team", "team_internal"], True),
        "position": first_col(
            df, ["fd_position", "model_position", "position", "pos"], True
        ),
        "identity": first_col(df, ["identity_key", "player_id", "gsis_id"]),
    }


def prepare_offense_lookup(df: pd.DataFrame, schema):
    out = df.copy()
    out["_player_norm"] = out[schema["player"]].map(norm_text)
    out["_team_norm"] = out[schema["team"]].map(norm_team)
    out["_position_norm"] = out[schema["position"]].map(norm_position)

    if schema["identity"]:
        out["_identity"] = out[schema["identity"]]
    else:
        out["_identity"] = ""

    # Only exact QB/RB/WR/TE rows can provide offensive position authority.
    out = out[out["_position_norm"].isin(VALID_OFFENSIVE_POSITIONS)].copy()

    counts = (
        out.groupby(["_player_norm", "_team_norm"], dropna=False)
        .size()
        .rename("_match_count")
        .reset_index()
    )
    out = out.merge(counts, on=["_player_norm", "_team_norm"], how="left")

    # Require deterministic uniqueness. If duplicate records resolve to the
    # same exact position and identity, collapse them; conflicting records stay blocked.
    rows = []
    for _, group in out.groupby(["_player_norm", "_team_norm"], dropna=False):
        positions = sorted(set(group["_position_norm"].dropna().astype(str)))
        identities = sorted(
            set(
                str(x)
                for x in group["_identity"].dropna()
                if str(x).strip() and str(x).lower() != "nan"
            )
        )

        if len(positions) != 1:
            continue

        rec = group.iloc[0].copy()
        rec["_position_norm"] = positions[0]
        rec["_identity"] = identities[0] if len(identities) == 1 else (
            rec["_identity"] if len(identities) == 0 else ""
        )
        rec["_source_rows"] = len(group)
        rec["_position_conflict"] = 0
        rows.append(rec)

    if not rows:
        return pd.DataFrame(
            columns=["_player_norm", "_team_norm", "_position_norm", "_identity"]
        )

    return pd.DataFrame(rows)



def load_injury_consensus(conn: sqlite3.Connection) -> pd.DataFrame:
    if not table_exists(conn, INJURY_TABLE):
        raise RuntimeError(
            f"Required injury consensus table not found: {INJURY_TABLE}. "
            "Run injury_consensus.py first."
        )

    cols = {row[1] for row in conn.execute(f'PRAGMA table_info("{INJURY_TABLE}")')}
    required = {
        "gsis_id",
        "injury_gate",
        "consensus_status",
        "injury_gate_reason",
        "injury_risk",
    }
    missing = sorted(required - cols)
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
        raise RuntimeError("Injury consensus table returned zero rows.")

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

    df["injury_gate"] = df["injury_gate"].str.upper()
    df["consensus_status"] = df["consensus_status"].str.upper()
    df["injury_risk"] = df["injury_risk"].str.upper()

    blank_ids = int(df["gsis_id"].eq("").sum())
    if blank_ids:
        raise RuntimeError(
            f"Injury consensus contains {blank_ids} blank GSIS IDs."
        )

    dupes = df.duplicated(["gsis_id"], keep=False)
    if dupes.any():
        bad = sorted(df.loc[dupes, "gsis_id"].unique().tolist())
        raise RuntimeError(
            "Injury consensus contains duplicate GSIS IDs: "
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
        for row in injury_consensus.to_dict("records")
    }

    out["injury_consensus_found"] = 0
    out["injury_gate"] = ""
    out["injury_status"] = ""
    out["injury_gate_reason"] = ""
    out["injury_risk"] = ""

    for idx in out.index:
        if int(out.at[idx, "_is_dst"]) == 1:
            continue

        gsis_id = str(out.at[idx, "position_identity"]).strip()
        if not gsis_id or gsis_id.lower() == "nan":
            continue

        match = injury_map.get(gsis_id)
        if match is None:
            continue

        out.at[idx, "injury_consensus_found"] = 1
        out.at[idx, "injury_gate"] = match["injury_gate"]
        out.at[idx, "injury_status"] = match["consensus_status"]
        out.at[idx, "injury_gate_reason"] = match["injury_gate_reason"]
        out.at[idx, "injury_risk"] = match["injury_risk"]

        if match["injury_gate"] == "BLOCK":
            out.at[idx, "solver_status"] = "BLOCKED_INJURY"

    escapes = out[
        out["injury_gate"].eq("BLOCK")
        & out["solver_eligible"].eq(1)
    ]
    if not escapes.empty:
        raise RuntimeError(
            "Injury gate escape detected: BLOCK row remains solver eligible."
        )

    return out


def build_solver_pool(source: pd.DataFrame, source_schema, offense_lookup: pd.DataFrame):
    out = source.copy()

    out["_player_norm"] = out[source_schema["player"]].map(norm_text)
    out["_team_norm"] = out[source_schema["team"]].map(norm_team)
    out["_salary_num"] = pd.to_numeric(out[source_schema["salary"]], errors="coerce")
    out["_projection_num"] = pd.to_numeric(
        out[source_schema["projection"]], errors="coerce"
    )

    if source_schema["is_dst"]:
        raw_dst = boolish(out[source_schema["is_dst"]]).eq(1)
    else:
        raw_dst = out["_player_norm"].str.contains(
            r"(?:\bdst\b|\bdefense\b|\bdef\b)", regex=True
        )

    # Strong fallback: player name equal to team abbreviation/name-like D/ST label.
    out["_is_dst"] = raw_dst.astype(int)

    position_map = {
        (rec["_player_norm"], rec["_team_norm"]): rec
        for rec in offense_lookup.to_dict("records")
    }

    out["solver_position"] = None
    out["position_match_method"] = ""
    out["position_identity"] = ""
    out["solver_status"] = "BLOCKED"

    for idx in out.index:
        projection_ready = (
            str(out.at[idx, source_schema["projection_status"]]).upper() == "READY"
            and pd.notna(out.at[idx, "_projection_num"])
            and np.isfinite(float(out.at[idx, "_projection_num"]))
        )
        salary_ready = (
            pd.notna(out.at[idx, "_salary_num"])
            and float(out.at[idx, "_salary_num"]) > 0
        )

        if int(out.at[idx, "_is_dst"]) == 1:
            out.at[idx, "solver_position"] = "DST"
            out.at[idx, "position_match_method"] = "dst_exact_slate_row"

            if projection_ready and salary_ready:
                out.at[idx, "solver_status"] = "READY"
            elif not projection_ready:
                out.at[idx, "solver_status"] = "BLOCKED_PROJECTION"
            else:
                out.at[idx, "solver_status"] = "BLOCKED_SALARY"
            continue

        key = (out.at[idx, "_player_norm"], out.at[idx, "_team_norm"])
        match = position_map.get(key)

        if match is None:
            out.at[idx, "solver_status"] = "BLOCKED_POSITION"
            continue

        out.at[idx, "solver_position"] = match["_position_norm"]
        out.at[idx, "position_match_method"] = "exact_name_team_fd_position"
        out.at[idx, "position_identity"] = match.get("_identity", "")

        if not salary_ready:
            out.at[idx, "solver_status"] = "BLOCKED_SALARY"
        elif not projection_ready:
            out.at[idx, "solver_status"] = "BLOCKED_PROJECTION"
        else:
            out.at[idx, "solver_status"] = "READY"

    out["solver_eligible"] = out["solver_status"].eq("READY").astype(int)

    # Canonical solver-facing columns. Preserve source columns as well.
    out["slate_name_solver"] = out[source_schema["slate"]]
    out["slate_slug_solver"] = out[source_schema["slug"]]
    out["player_solver"] = out[source_schema["player"]]
    out["team_solver"] = out["_team_norm"]
    out["salary_solver"] = out["_salary_num"]
    out["game_solver"] = out[source_schema["game"]]
    out["projection_solver"] = out["_projection_num"]

    # Give defenses a stable solver identity even if upstream player identity is blank.
    out["solver_player_key"] = np.where(
        out["_is_dst"].eq(1),
        "DST:" + out["_team_norm"].fillna(""),
        out["position_identity"].astype(str),
    )

    return out


def build_manifest(df: pd.DataFrame):
    rows = []
    for (slate, slug), g in df.groupby(
        ["slate_name_solver", "slate_slug_solver"], dropna=False
    ):
        eligible = g[g["solver_eligible"].eq(1)]
        dst = g[g["_is_dst"].eq(1)]
        offense = g[g["_is_dst"].eq(0)]

        counts = eligible["solver_position"].value_counts().to_dict()

        invalid_position_rows = int(
            eligible["solver_position"].isna().sum()
            + (~eligible["solver_position"].isin(VALID_SOLVER_POSITIONS)).sum()
        )

        rows.append(
            {
                "slate_name": slate,
                "slate_slug": slug,
                "contest_rows": len(g),
                "solver_eligible_rows": len(eligible),
                "solver_blocked_rows": int(g["solver_eligible"].eq(0).sum()),
                "offense_rows": len(offense),
                "dst_rows": len(dst),
                "eligible_qb": int(counts.get("QB", 0)),
                "eligible_rb": int(counts.get("RB", 0)),
                "eligible_wr": int(counts.get("WR", 0)),
                "eligible_te": int(counts.get("TE", 0)),
                "eligible_dst": int(counts.get("DST", 0)),
                "blocked_projection": int(
                    g["solver_status"].eq("BLOCKED_PROJECTION").sum()
                ),
                "blocked_position": int(
                    g["solver_status"].eq("BLOCKED_POSITION").sum()
                ),
                "blocked_salary": int(
                    g["solver_status"].eq("BLOCKED_SALARY").sum()
                ),
                "blocked_injury": int(
                    g["solver_status"].eq("BLOCKED_INJURY").sum()
                ),
                "invalid_eligible_position_rows": invalid_position_rows,
                "status": (
                    "PASS"
                    if (
                        len(eligible) > 0
                        and counts.get("QB", 0) >= 1
                        and counts.get("RB", 0) >= 3
                        and counts.get("WR", 0) >= 4
                        and counts.get("TE", 0) >= 1
                        and counts.get("DST", 0) >= 1
                        and invalid_position_rows == 0
                    )
                    else "BLOCKED"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("slate_name").reset_index(drop=True)


def clean_export(df: pd.DataFrame):
    # Keep source provenance, but remove private normalization helpers.
    drop = [
        "_player_norm", "_team_norm", "_salary_num",
        "_projection_num", "_is_dst",
    ]
    return df.drop(columns=[c for c in drop if c in df.columns])


def main():
    section("FANDUEL SOLVER-READY SLATE POOL")
    print(f"Database: {DATABASE_PATH}")
    print(f"Projection source: {SOURCE_TABLE}")
    print("Position authority: stable offensive fd_position + DST")
    print("Slate salary authority: fanduel_slate_projection_pool")
    print("Fuzzy matching: disabled")
    print("Salary inference: disabled")
    print("Upstream stable files/tables: read only")

    with sqlite3.connect(DATABASE_PATH) as conn:
        if not table_exists(conn, SOURCE_TABLE):
            raise RuntimeError(
                f"Required SQLite table not found: {SOURCE_TABLE}. "
                "Run fanduel_slate_projection_attach_v5.py first."
            )

        source = pd.read_sql_query(f'SELECT * FROM "{SOURCE_TABLE}"', conn)
        offense, offense_source = load_offense_source(conn)

        source_schema = resolve_source_schema(source)
        offense_schema = resolve_offense_schema(offense)

        section("RESOLVED SOURCES")
        print(f"Projection rows: {len(source)}")
        print(f"Offensive position source: {offense_source}")
        print(f"Offensive position rows: {len(offense)}")
        print()
        print("Projection source schema:")
        for k, v in source_schema.items():
            print(f"  {k:<22} -> {v}")
        print()
        print("Offensive position schema:")
        for k, v in offense_schema.items():
            print(f"  {k:<22} -> {v}")

        offense_lookup = prepare_offense_lookup(offense, offense_schema)
        injury_consensus = load_injury_consensus(conn)
        solver = build_solver_pool(source, source_schema, offense_lookup)
        solver = apply_injury_gate(solver, injury_consensus)
        manifest = build_manifest(solver)

        section("SOLVER-READY COVERAGE BY SLATE")
        print(manifest.to_string(index=False))

        section("GLOBAL STRUCTURAL AUDIT")
        ready = solver["solver_eligible"].eq(1)
        blocked = ~ready
        print(f"Source contest rows:             {len(solver)}")
        print(f"Solver eligible rows:           {int(ready.sum())}")
        print(f"Solver blocked rows:            {int(blocked.sum())}")
        print(f"Blocked projection rows:        {int(solver['solver_status'].eq('BLOCKED_PROJECTION').sum())}")
        print(f"Blocked position rows:          {int(solver['solver_status'].eq('BLOCKED_POSITION').sum())}")
        print(f"Blocked salary rows:            {int(solver['solver_status'].eq('BLOCKED_SALARY').sum())}")
        print(f"Blocked injury rows:            {int(solver['solver_status'].eq('BLOCKED_INJURY').sum())}")
        print(f"Eligible rows missing position: {int(solver.loc[ready, 'solver_position'].isna().sum())}")
        print(f"Eligible rows invalid position: {int((~solver.loc[ready, 'solver_position'].isin(VALID_SOLVER_POSITIONS)).sum())}")
        print(f"Duplicate slate/player/team:    {int(solver.duplicated(['slate_slug_solver','player_solver','team_solver','salary_solver']).sum())}")

        section("GLOBAL ELIGIBLE POSITION COUNTS")
        pos_counts = (
            solver.loc[ready, "solver_position"]
            .value_counts()
            .rename_axis("position")
            .reset_index(name="rows")
        )
        print(pos_counts.to_string(index=False))

        gaps = solver[blocked].copy()
        section("BLOCKED ROW BREAKDOWN")
        if gaps.empty:
            print("No blocked rows.")
        else:
            print(
                gaps.groupby(["solver_status"], dropna=False)
                .size()
                .rename("rows")
                .reset_index()
                .to_string(index=False)
            )

        exported = clean_export(solver)

        # Write derived outputs only.
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

        exported.to_csv(OUT_CSV, index=False)
        exported.to_parquet(OUT_PARQUET, index=False)
        manifest.to_csv(OUT_MANIFEST, index=False)

        gap_cols = [
            c for c in [
                "slate_name_solver", "player_solver", "team_solver",
                "salary_solver", "game_solver", "solver_position",
                "projection_solver", "projection_status",
                "solver_status", "position_match_method",
                "injury_consensus_found", "injury_gate",
                "injury_status", "injury_gate_reason", "injury_risk",
            ]
            if c in solver.columns
        ]
        gaps[gap_cols].to_csv(OUT_GAPS, index=False)

        exported.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)
        manifest.to_sql(MANIFEST_TABLE, conn, if_exists="replace", index=False)

        conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_slate '
            f'ON {OUTPUT_TABLE}(slate_slug_solver)'
        )
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_eligible '
            f'ON {OUTPUT_TABLE}(slate_slug_solver, solver_eligible)'
        )
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_position '
            f'ON {OUTPUT_TABLE}(slate_slug_solver, solver_position)'
        )
        conn.commit()

    section("EXPORTS")
    print(f"Pool CSV:     {OUT_CSV}")
    print(f"Pool Parquet: {OUT_PARQUET}")
    print(f"Manifest:     {OUT_MANIFEST}")
    print(f"Gap audit:    {OUT_GAPS}")
    print(f"SQLite pool:  {OUTPUT_TABLE}")
    print(f"SQLite audit: {MANIFEST_TABLE}")

    section("SOLVER-READY POOL COMPLETE")
    if (manifest["status"] == "PASS").all():
        print("ALL SLATES PASS position/projection/salary structural gates.")
    else:
        print("One or more slates are BLOCKED. Review the manifest before solving.")
    print(
        "Only exact-position, positive-salary, internally projected, "
        "injury-allowed rows are marked solver_eligible=1."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 112)
        print("FANDUEL SOLVER-READY POOL FAILED")
        print("=" * 112)
        print(f"{type(exc).__name__}: {exc}")
        raise
