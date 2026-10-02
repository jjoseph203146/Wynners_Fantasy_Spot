#!/usr/bin/env python3
"""Stage and validate exact V3 bytes; never publish production artifacts."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "WFS_V3_IMMUTABLE_PREFLIGHT_V1"
SCHEMA = "WFS_OFFENSIVE_RECONCILIATION_CURRENT_V3_V1"
# Optional, separately versioned equivalence proof; existing bundle schema and
# all conservation/availability checks remain unchanged.
REBIND_NAME = "adaptive_rebind.zip"
REBIND_PATH = "processed/offensive_reconciliation_adaptive_rebind_v2.zip"
LEGACY_VALIDATOR_SHA = "bef80cc4a0f2dcc36bfa90707a91a6adc058dac5582a76378e0c5249bca78c93"

def rebind_module():
    try:
        from scripts import v3_adaptive_rebind
    except ModuleNotFoundError:
        import v3_adaptive_rebind
    return v3_adaptive_rebind

ROLES = {"PRIMARY_QB", "ACTIVE_ROTATION", "CONTINGENCY_QB", "UNAVAILABLE"}
ACTIVE = {"PRIMARY_QB", "ACTIVE_ROTATION"}
STATS = ("attempts", "completions", "passing_yards", "passing_tds",
         "interceptions", "carries", "rushing_yards", "rushing_tds",
         "targets", "receptions", "receiving_yards", "receiving_tds")
INPUTS = {
    "candidate.csv": "processed/offensive_team_reconciliation_shadow_v3.csv",
    "builder_audit.json": "processed/offensive_team_reconciliation_shadow_v3_audit.json",
    "team_audit.csv": "processed/offensive_team_reconciliation_team_audit_v3.csv",
    "baseline.parquet": "data/parquet/nfl_production_projection.parquet",
    "baseline.csv": "data/csv/nfl_production_projection.csv",
    "offense.parquet": "data/parquet/nfl_current_offensive_stat_forecasts.parquet",
    "matrix.parquet": "data/parquet/nfl_current_offensive_model_matrix.parquet",
    "adaptive.csv": "processed/offensive_reconciliation_current_adaptive_team_v1.csv",
    "adaptive_audit.json": "processed/offensive_reconciliation_current_adaptive_team_v1_audit.json",
}


def require(condition, message):
    if not condition:
        raise RuntimeError("FAIL_CLOSED: " + message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_regular(path):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)) and path.is_file(), f"missing/nonregular input: {path}")
    return path.read_bytes()


def keys(df):
    cols = ["game_id", "player_id"]
    require(set(cols).issubset(df), "missing identity columns")
    require(not df[cols].isna().any().any(), "null identity")
    require(all(df[c].astype(str).str.strip().ne("").all() for c in cols), "blank identity")
    require(not df.duplicated(cols).any(), "duplicate game_id/player_id")
    return set(df[cols].itertuples(index=False, name=None))


def key_hash(values):
    return digest(json.dumps(sorted(values), separators=(",", ":")).encode())


def validate_candidate(blobs, season, week):
    """Current-universe equivalents of frozen instance checks, plus binding."""
    v3 = pd.read_csv(io.BytesIO(blobs["candidate.csv"]))
    baseline = pd.read_parquet(io.BytesIO(blobs["baseline.parquet"]))
    offense = pd.read_parquet(io.BytesIO(blobs["offense.parquet"]))
    matrix = pd.read_parquet(io.BytesIO(blobs["matrix.parquet"]))
    team = pd.read_csv(io.BytesIO(blobs["team_audit.csv"]))
    audit = json.loads(blobs["builder_audit.json"])
    context = json.loads(blobs["context.json"])
    adaptive = pd.read_csv(io.BytesIO(blobs["adaptive.csv"]))
    adaptive_audit = json.loads(blobs["adaptive_audit.json"])
    require((context["season"], context["week"]) == (season, week), "context season/week mismatch")
    require(audit["status"] == "PASS", "builder validation failed")
    require(audit["version"] == "WFS_OFFENSIVE_TEAM_RECONCILIATION_SHADOW_V3", "builder contract version mismatch")
    require(adaptive_audit["status"] == "PASS_HARD_CONTRACTS" and isinstance(adaptive_audit["hard_failures"], dict)
            and all(value == 0 for value in adaptive_audit["hard_failures"].values()), "adaptive validation failed")
    require((adaptive_audit["target_season"], adaptive_audit["target_week"]) == (season, week), "adaptive preflight season/week mismatch")
    require(not adaptive.duplicated(["game_id", "team"]).any(), "duplicate adaptive team keys")
    require(adaptive_audit["rows"] == len(adaptive) and adaptive_audit["games"] == adaptive.game_id.nunique(), "adaptive row/game count mismatch")
    require(adaptive.season.eq(season).all() and adaptive.week.eq(week).all(), "adaptive rows season/week mismatch")
    if REBIND_NAME in blobs:
        rebind_module().validate_rebind(blobs[REBIND_NAME], blobs["matrix.parquet"],
                                        blobs["adaptive.csv"], blobs["adaptive_audit.json"], season, week)
    else:
        require(adaptive_audit["source_hashes"].get(str(ROOT / INPUTS["matrix.parquet"])) == digest(blobs["matrix.parquet"]), "adaptive validation matrix revision mismatch")
    require(audit["outputs"]["player_shadow"]["sha256"] == digest(blobs["candidate.csv"]), "builder candidate hash mismatch")
    require(audit["outputs"]["team_audit"]["sha256"] == digest(blobs["team_audit.csv"]), "builder team audit hash mismatch")
    exclusion = audit["target_history_exclusion"]
    require(exclusion["status"] == "PASS" and exclusion["player_overlap"] == 0 and exclusion["team_overlap"] == 0, "target history leakage")
    vk, bk, ok, mk = keys(v3), keys(baseline), keys(offense), keys(matrix)
    require(vk == ok == mk, "candidate/offense/matrix identity coverage mismatch")
    prefix = f"{season}_{week:02d}_"
    for frame in (v3, baseline, offense, matrix):
        require(frame.game_id.astype(str).str.startswith(prefix).all(), "game identity season/week mismatch")
        require(frame.season.eq(season).all() and frame.week.eq(week).all(), "row season/week mismatch")
    required = {"team", "position", "model_group", "reconciliation_role", "active_flag", "injury_flag", "primary_qb_id"}
    required |= {p + s for p in ("original_", "expected_", "reconciled_") for s in STATS}
    require(required.issubset(v3), "candidate schema missing columns")
    require(v3.reconciliation_role.isin(ROLES).all(), "invalid reconciliation role")
    require({"ridge_projection", "production_status"}.issubset(baseline), "production schema missing")
    require("ridge_projection_pre_v3" not in baseline, "ambiguous repeat promotion")
    csv_baseline = pd.read_csv(io.BytesIO(blobs["baseline.csv"]))
    require(keys(csv_baseline) == bk, "baseline CSV/parquet coverage mismatch")
    bp = baseline.set_index(["game_id", "player_id"]).sort_index()
    bc = csv_baseline.set_index(["game_id", "player_id"]).sort_index()
    require(bp.production_status.eq(bc.production_status).all(), "baseline status mismatch")
    require(np.allclose(bp.ridge_projection, bc.ridge_projection, equal_nan=True, atol=1e-8, rtol=0), "baseline projection mismatch")
    # The independently bound matrix/offense universe establishes row counts.
    vi = v3.set_index(["game_id", "player_id"]).sort_index()
    oi = offense.set_index(["game_id", "player_id"]).sort_index()
    mi = matrix.set_index(["game_id", "player_id"]).sort_index()
    for field in ("team", "position", "model_group"):
        require(vi[field].eq(oi[field]).all(), f"offense {field} mismatch")
    for stat in STATS:
        expected = pd.to_numeric(oi["expected_" + stat], errors="coerce").fillna(0)
        require(np.allclose(vi["original_" + stat], expected, atol=1e-8, rtol=0), f"original {stat} input mismatch")
        require(np.allclose(vi["expected_" + stat], expected, atol=1e-8, rtol=0), f"expected {stat} input mismatch")
    injuries = pd.DataFrame(context["injuries"])
    require(not injuries.gsis_id.duplicated().any() and injuries.gsis_id.str.strip().ne("").all(), "ambiguous injury identity")
    require(injuries.injury_gate.isin(["ALLOW", "BLOCK"]).all(), "invalid authority gate")
    blocked = set(injuries.loc[injuries.injury_gate.eq("BLOCK"), "gsis_id"])
    expected_active = mi.active_flag.astype(int).copy()
    expected_active.loc[expected_active.index.get_level_values("player_id").isin(blocked)] = 0
    require(vi.active_flag.eq(expected_active).all(), "current availability mismatch")
    require(vi.injury_flag.eq(mi.injury_flag).all(), "matrix injury flag mismatch")
    require(vi.reconciliation_role.eq("UNAVAILABLE").eq(expected_active.ne(1)).all(), "unavailable role mismatch")
    numeric = v3[["reconciled_" + s for s in STATS]].apply(pd.to_numeric, errors="coerce")
    require(np.isfinite(numeric.to_numpy()).all() and numeric.ge(0).all().all(), "invalid reconciled statistics")
    excluded = ~v3.reconciliation_role.isin(ACTIVE)
    require(numeric.loc[excluded].eq(0).all().all(), "excluded role retains workload")
    game_teams = {(g["game_id"], g[t]) for g in context["games"] for t in ("home_team", "away_team")}
    upcoming_teams = {(g["game_id"], g[t]) for g in context["games"] if g["completed"] != 1 for t in ("home_team", "away_team")}
    adaptive_keys = set(zip(adaptive.game_id, adaptive.team))
    require(adaptive_keys == upcoming_teams, "adaptive upcoming coverage/fallback contract failure")
    require(set(zip(v3.game_id, v3.team)) == upcoming_teams, "upcoming schedule team coverage mismatch")
    require(not team.duplicated(["game_id", "team"]).any(), "duplicate team audit")
    require(set(zip(team.game_id, team.team)) == upcoming_teams, "team audit upcoming coverage mismatch")
    mappings = {"attempts": "team_pass_attempts", "completions": "team_completions",
                "receptions": "team_completions", "passing_yards": "team_passing_yards",
                "receiving_yards": "team_passing_yards", "passing_tds": "team_passing_tds",
                "receiving_tds": "team_passing_tds", "carries": "team_carries",
                "rushing_yards": "team_rushing_yards", "rushing_tds": "team_rushing_tds"}
    max_gap = 0.0
    for (game, club), rows in v3.groupby(["game_id", "team"]):
        qb = rows[rows.reconciliation_role.eq("PRIMARY_QB")]
        require(len(qb) == 1 and qb.position.eq("QB").all(), "primary QB coverage")
        require(rows.primary_qb_id.eq(qb.iloc[0].player_id).all(), "primary QB identity mismatch")
        require(rows.loc[rows.reconciliation_role.eq("CONTINGENCY_QB"), "position"].eq("QB").all(), "contingency role mismatch")
        require(not rows.loc[rows.reconciliation_role.eq("ACTIVE_ROTATION"), "position"].eq("QB").any(), "QB in active rotation")
        tr = team[(team.game_id == game) & (team.team == club)].iloc[0]
        if (game, club) in adaptive_keys:
            ar = adaptive[(adaptive.game_id == game) & (adaptive.team == club)].iloc[0]
            for stat, budget in {"attempts": "team_pass_attempts", "completions": "team_completions",
                                 "passing_yards": "team_passing_yards", "passing_tds": "team_passing_tds",
                                 "carries": "team_carries", "rushing_yards": "team_rushing_yards",
                                 "rushing_tds": "team_rushing_tds"}.items():
                value = float(ar["adaptive_" + stat])
                require(np.isfinite(value) and value >= 0, "invalid adaptive budget")
                require(abs(float(tr[budget]) - value) <= 1e-8, "adaptive budget attribution mismatch")
            require(ar.adaptive_completions <= ar.adaptive_attempts, "adaptive completions exceed attempts")
        require(tr.primary_qb_id == qb.iloc[0].player_id, "team primary QB mismatch")
        for stat, budget in mappings.items():
            require(np.isfinite(float(tr[budget])), "nonfinite team budget")
            max_gap = max(max_gap, abs(rows["reconciled_" + stat].sum() - tr[budget]))
    require(max_gap <= 1e-8, "reconciliation conservation failure")
    pool = v3[v3.reconciliation_role.isin(ACTIVE)]
    pk = keys(pool)
    require(pk.issubset(bk), "missing active production/V3 matches")
    # Import the existing scoring formula; do not introduce another scoring policy.
    try:
        from scripts.promote_offensive_reconciliation_v3 import fanduel_points_from_v3
    except ModuleNotFoundError:
        from promote_offensive_reconciliation_v3 import fanduel_points_from_v3
    points = fanduel_points_from_v3(pool)
    require(np.isfinite(points).all() and points.ge(0).all(), "invalid V3 fantasy points")
    ready = set(baseline.loc[baseline.production_status.eq("MODEL_READY"), ["game_id", "player_id"]].itertuples(index=False, name=None))
    require(bool(ready & pk), "no MODEL_READY promotions")
    roles = {r: int(v3.reconciliation_role.eq(r).sum()) for r in sorted(ROLES)}
    require(audit["rows"] == len(v3) and audit["primary_qbs"] == roles["PRIMARY_QB"] and audit["unavailable_rows"] == roles["UNAVAILABLE"] and audit["contingency_qb_rows"] == roles["CONTINGENCY_QB"], "builder counts mismatch")
    facts = {"row_count": len(v3), "columns": list(v3.columns), "logical_types": {c: str(t) for c, t in v3.dtypes.items()},
             "role_counts": roles, "active_pool_count": len(pool), "identity_set_sha256": key_hash(vk),
             "production_match_count": len(pk), "production_match_identity_sha256": key_hash(pk),
             "max_conservation_gap": float(max_gap), "adaptive_team_games": len(adaptive_keys),
             "fallback_team_games": len(game_teams - adaptive_keys),
             "hard_failures": {"current_contract": 0, "adaptive_coverage": 0, "fallback_scope": 0,
                               "missing_input_matches": 0, "conservation": 0, "duplicate_identity": 0}}
    return v3, baseline, facts


def stage_candidate(root, bundle, db_path, adaptive_rebind=None):
    root, bundle = Path(root), Path(bundle)
    require(not bundle.exists(), "candidate bundle already exists")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from wfs_schedule_context import resolve_schedule_week_context
    ctx = resolve_schedule_week_context(db_path=db_path)
    blobs = {name: read_regular(root / rel) for name, rel in INPUTS.items()}
    proof_path = Path(adaptive_rebind) if adaptive_rebind is not None else root / REBIND_PATH
    if adaptive_rebind is not None or proof_path.exists():
        blobs[REBIND_NAME] = read_regular(proof_path)

    with sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("BEGIN")
        injuries = pd.read_sql_query("SELECT season, week, gsis_id, injury_gate FROM injury_consensus_current", conn)
        # injury_consensus_current is sparse current availability authority.
        # Its source rows may legitimately retain prior-week provenance while
        # current consumers use exact-GSIS BLOCK/ALLOW with healthy-by-absence.
        # Never admit future injury authority into the planning context.
        injury_period = (
            injuries["season"].astype(int) * 100
            + injuries["week"].astype(int)
        )
        target_period = int(ctx.season) * 100 + int(ctx.planning_week)
        require(
            injury_period.le(target_period).all(),
            "future injury authority",
        )
        games = pd.read_sql_query("SELECT game_id, home_team, away_team, completed FROM games WHERE season=? AND week=? AND game_type='REG'", conn, params=(ctx.season, ctx.planning_week))
    context = {"season": ctx.season, "week": ctx.planning_week, "injuries": injuries.to_dict("records"), "games": games.to_dict("records")}
    blobs["context.json"] = json.dumps(context, sort_keys=True).encode()
    _, _, facts = validate_candidate(blobs, ctx.season, ctx.planning_week)
    bundle.mkdir(parents=True, exist_ok=False)
    for name, data in blobs.items():
        with (bundle / name).open("xb") as handle:
            handle.write(data)
        (bundle / name).chmod(0o444)
    metadata = {"contract_version": CONTRACT, "schema_version": SCHEMA,
                "validator_sha256": digest(Path(__file__).read_bytes()), "status": "PASS_HARD_CONTRACTS",
                "candidate_id": bundle.name, "candidate_path": "candidate.csv",
                "candidate_sha256": digest(blobs["candidate.csv"]), "season": ctx.season, "week": ctx.planning_week,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": {name: digest(data) for name, data in blobs.items()}, **facts}
    metadata["adaptive_rebind_validator_sha256"] = digest(Path(rebind_module().__file__).read_bytes())
    if REBIND_NAME in blobs:
        metadata["adaptive_rebind_contract"] = rebind_module().CONTRACT
    temp = bundle / "validation.pending"
    with temp.open("x") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    temp.chmod(0o444)
    temp.replace(bundle / "validation.json")
    return bundle / "validation.json"


def load_validated_candidate(metadata_path, season=None, week=None):
    path = Path(metadata_path)
    require(path.name == "validation.json" and not path.parent.is_symlink(), "candidate metadata path mismatch")
    meta = json.loads(read_regular(path))
    require(meta["contract_version"] == CONTRACT and meta["schema_version"] == SCHEMA, "schema/version mismatch")
    if meta["validator_sha256"] == LEGACY_VALIDATOR_SHA:
        # Exact historical source remains hash-pinned and runs its original
        # validation; existing published bundles are not rewritten.
        return rebind_module().legacy_validator().load_validated_candidate(path, season, week)
    require(meta["validator_sha256"] == digest(Path(__file__).read_bytes()), "validator revision mismatch")
    require(meta.get("adaptive_rebind_validator_sha256") == digest(Path(rebind_module().__file__).read_bytes()), "adaptive rebind validator revision mismatch")
    require(meta["status"] == "PASS_HARD_CONTRACTS", "unsuccessful preflight")
    require(meta["candidate_id"] == path.parent.name and meta["candidate_path"] == "candidate.csv", "candidate identity/path mismatch")
    require(season is None or meta["season"] == season, "season mismatch")
    require(week is None or meta["week"] == week, "week mismatch")
    expected_inputs = set(INPUTS) | {"context.json"}
    if "adaptive_rebind_contract" in meta:
        require(meta["adaptive_rebind_contract"] == rebind_module().CONTRACT, "adaptive rebind contract")
        expected_inputs.add(REBIND_NAME)
    require(set(meta["inputs"]) == expected_inputs, "input binding mismatch")
    blobs = {name: read_regular(path.parent / name) for name in meta["inputs"]}
    for name, data in blobs.items():
        require(digest(data) == meta["inputs"][name], f"hash mismatch: {name}")
    require(digest(blobs["candidate.csv"]) == meta["candidate_sha256"], "candidate hash mismatch")
    v3, baseline, facts = validate_candidate(blobs, meta["season"], meta["week"])
    require(all(meta.get(k) == value for k, value in facts.items()), "validation facts/count/schema mismatch")
    return v3, baseline, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=ROOT / "data/nfl.db")
    parser.add_argument("--adaptive-rebind", type=Path)
    args = parser.parse_args()
    print(stage_candidate(args.root, args.bundle, args.db, args.adaptive_rebind))


if __name__ == "__main__":
    main()
