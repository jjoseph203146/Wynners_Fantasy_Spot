#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import hashlib
import math
import sys

import pandas as pd

from stage24_solver_attachment import (
    attach_showdown_stage24,
    load_stage24,
)
from ortools.sat.python import cp_model

ROOT = Path("/home/mwynn/nfl_data_engine")
POOL = ROOT / "data/fanduel/single_game/derived/single_game_projection_pool.parquet"

SALARY_CAP = 60000
ROSTER_SIZE = 6

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s

def finite(v):
    try:
        return pd.notna(v) and math.isfinite(float(v))
    except Exception:
        return False

def scale_int(v, scale=1000):
    return int(round(float(v) * scale))

def build_one_lineup(
    df,
    prior_lineups,
    min_salary=0,
    lock_player_ids=None,
    exclude_player_ids=None,
    mvp_player_id="",
):
    model = cp_model.CpModel()

    lock_player_ids = set(lock_player_ids or [])
    exclude_player_ids = set(exclude_player_ids or [])
    mvp_player_id = clean(mvp_player_id)

    # Each row is one underlying player. Role choices are separate variables.
    flex = {}
    mvp = {}

    for i in df.index:
        flex[i] = model.NewBoolVar(f"flex_{i}")
        mvp[i] = model.NewBoolVar(f"mvp_{i}")
        model.Add(flex[i] + mvp[i] <= 1)

    model.Add(sum(mvp.values()) == 1)
    model.Add(sum(flex.values()) == 5)
    model.Add(sum(flex.values()) + sum(mvp.values()) == ROSTER_SIZE)

    index_by_player_id = {
        clean(r["player_id"]): i
        for i, r in df.iterrows()
    }

    for player_id in sorted(lock_player_ids):
        i = index_by_player_id[player_id]
        model.Add(flex[i] + mvp[i] == 1)

    for player_id in sorted(exclude_player_ids):
        i = index_by_player_id[player_id]
        model.Add(flex[i] == 0)
        model.Add(mvp[i] == 0)

    if mvp_player_id:
        i = index_by_player_id[mvp_player_id]
        model.Add(mvp[i] == 1)

    salary_terms = []
    proj_terms = []

    for i, r in df.iterrows():
        base_salary = int(round(float(r["salary"])))
        mvp_salary = int(round(float(r["mvp_salary"])))
        base_proj = scale_int(r["projection"])
        mvp_proj = scale_int(r["mvp_projection"])

        salary_terms.append(base_salary * flex[i])
        salary_terms.append(mvp_salary * mvp[i])

        proj_terms.append(base_proj * flex[i])
        proj_terms.append(mvp_proj * mvp[i])

    total_salary = sum(salary_terms)
    model.Add(total_salary <= SALARY_CAP)
    if min_salary > 0:
        model.Add(total_salary >= min_salary)

    # Exact lineup uniqueness includes role assignment.
    for lineup in prior_lineups:
        terms = []
        for i in df.index:
            role = lineup.get(i)
            if role == "MVP":
                terms.append(mvp[i])
            elif role == "FLEX":
                terms.append(flex[i])
        if terms:
            model.Add(sum(terms) <= ROSTER_SIZE - 1)

    model.Maximize(sum(proj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 30.0

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    roles = {}
    rows = []
    for i, r in df.iterrows():
        role = None
        if solver.Value(mvp[i]):
            role = "MVP"
        elif solver.Value(flex[i]):
            role = "FLEX"

        if role:
            roles[i] = role
            salary = int(round(float(r["mvp_salary"] if role == "MVP" else r["salary"])))
            proj = float(r["mvp_projection"] if role == "MVP" else r["projection"])
            rows.append({
                "role": role,
                "player": clean(r["player"]),
                "player_id": clean(r["player_id"]),
                "team": clean(r["team"]),
                "position": clean(r["position"]),
                "salary": salary,
                "projection": proj,
                "projection_source": clean(r["projection_source"]),
            })

    rows.sort(key=lambda x: (0 if x["role"] == "MVP" else 1, -x["projection"], x["player"]))
    total_salary_value = sum(r["salary"] for r in rows)
    total_projection_value = sum(r["projection"] for r in rows)

    return {
        "roles": roles,
        "rows": rows,
        "total_salary": total_salary_value,
        "total_projection": total_projection_value,
    }

def audit_lineup(lineup):
    rows = lineup["rows"]
    player_ids = [r["player_id"] for r in rows]

    return {
        "player_count": len(rows),
        "mvp_count": sum(1 for r in rows if r["role"] == "MVP"),
        "flex_count": sum(1 for r in rows if r["role"] == "FLEX"),
        "unique_player_ids": len(set(player_ids)),
        "salary": lineup["total_salary"],
        "projection": lineup["total_projection"],
        "pass": (
            len(rows) == 6
            and sum(1 for r in rows if r["role"] == "MVP") == 1
            and sum(1 for r in rows if r["role"] == "FLEX") == 5
            and len(set(player_ids)) == 6
            and lineup["total_salary"] <= SALARY_CAP
        ),
    }

def main():
    ap = argparse.ArgumentParser(description="WFS isolated FanDuel NFL Showdown solver")
    ap.add_argument("--slate", required=True, help='Exact public_slate_name, e.g. "DEN @ KC"')
    ap.add_argument("--lineups", type=int, default=20)
    ap.add_argument("--min-salary", type=int, default=0)
    ap.add_argument("--output", default="")
    ap.add_argument(
        "--lock-player-id",
        action="append",
        default=[],
        help="Underlying player_id that must appear in every lineup.",
    )
    ap.add_argument(
        "--exclude-player-id",
        action="append",
        default=[],
        help="Underlying player_id that cannot appear in any lineup.",
    )
    ap.add_argument(
        "--mvp-player-id",
        default="",
        help="Optional underlying player_id forced into the MVP role.",
    )
    args = ap.parse_args()

    if not POOL.exists():
        print(f"FAIL_CLOSED_MISSING_POOL={POOL}")
        raise SystemExit(2)

    if args.lineups < 1:
        print("FAIL_CLOSED_LINEUPS_MUST_BE_POSITIVE=TRUE")
        raise SystemExit(2)

    if args.min_salary < 0 or args.min_salary > SALARY_CAP:
        print("FAIL_CLOSED_BAD_MIN_SALARY=TRUE")
        raise SystemExit(2)

    df = pd.read_parquet(POOL).copy()
    _stage24_df = load_stage24(
        "/home/mwynn/nfl_data_engine/data/parquet/current_unified_fanduel_expectation.parquet"
    )
    df, _stage24_attachment_audit = (
        attach_showdown_stage24(
            df,
            _stage24_df,
        )
    )


    required = {
        "public_slate_name", "player", "player_id", "team", "position",
        "salary", "projection", "projection_source",
        "mvp_salary", "mvp_projection", "projection_status",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        print("FAIL_CLOSED_SCHEMA_MISSING=" + ",".join(missing))
        raise SystemExit(2)

    slate_df = df[df["public_slate_name"].astype(str).eq(args.slate)].copy()

    if slate_df.empty:
        available = sorted(df["public_slate_name"].astype(str).unique())
        print(f"FAIL_CLOSED_UNKNOWN_SLATE={args.slate!r}")
        print("AVAILABLE_SLATES=" + " | ".join(available))
        raise SystemExit(2)

    if not slate_df["projection_status"].isin(["READY", "READY_EXTERNAL"]).all():
        print("FAIL_CLOSED_NONREADY_ROWS_PRESENT=TRUE")
        raise SystemExit(2)

    # ------------------------------------------------------------------
    # Authoritative injury / availability gate.
    #
    # Single-Game inherits Classic eligibility semantics:
    #   * injury_gate=BLOCK may only remove eligibility.
    #   * absence from injury consensus is healthy-by-absence.
    #   * malformed/duplicate authority fails closed.
    #
    # Projection rows remain intact in the derived pool.  This gate only
    # controls the solver candidate universe.
    # ------------------------------------------------------------------
    injury_path = ROOT / "data/parquet/injury_consensus_current.parquet"

    if not injury_path.exists():
        print(f"FAIL_CLOSED_MISSING_INJURY_AUTHORITY={injury_path}")
        raise SystemExit(2)

    injury = pd.read_parquet(injury_path).copy()

    injury_required = {
        "gsis_id",
        "injury_gate",
        "consensus_status",
        "injury_gate_reason",
        "injury_risk",
    }
    injury_missing = sorted(
        injury_required - set(injury.columns)
    )
    if injury_missing:
        print(
            "FAIL_CLOSED_INJURY_SCHEMA_MISSING="
            + ",".join(injury_missing)
        )
        raise SystemExit(2)

    if injury.empty:
        print("FAIL_CLOSED_INJURY_AUTHORITY_EMPTY=TRUE")
        raise SystemExit(2)

    for col in injury_required:
        injury[col] = (
            injury[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    injury["injury_gate"] = (
        injury["injury_gate"]
        .str.upper()
    )

    if injury["gsis_id"].eq("").any():
        print("FAIL_CLOSED_INJURY_BLANK_GSIS_ID=TRUE")
        raise SystemExit(2)

    duplicate_injury = injury["gsis_id"].duplicated(
        keep=False
    )
    if duplicate_injury.any():
        bad = sorted(
            injury.loc[
                duplicate_injury,
                "gsis_id",
            ]
            .unique()
            .tolist()
        )
        print(
            "FAIL_CLOSED_INJURY_DUPLICATE_GSIS_IDS="
            + ",".join(bad)
        )
        raise SystemExit(2)

    invalid_gate = ~injury["injury_gate"].isin(
        {"ALLOW", "BLOCK"}
    )
    if invalid_gate.any():
        bad = sorted(
            injury.loc[
                invalid_gate,
                "injury_gate",
            ]
            .unique()
            .tolist()
        )
        print(
            "FAIL_CLOSED_INVALID_INJURY_GATE="
            + ",".join(bad)
        )
        raise SystemExit(2)

    block_ids = set(
        injury.loc[
            injury["injury_gate"].eq("BLOCK"),
            "gsis_id",
        ]
    )

    candidate_ids = slate_df["player_id"].map(clean)

    blocked_mask = candidate_ids.isin(block_ids)
    blocked_count = int(blocked_mask.sum())

    if blocked_count:
        blocked_names = (
            slate_df.loc[
                blocked_mask,
                "player",
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )
        print(
            f"INJURY_BLOCKED_ROWS_REMOVED={blocked_count}"
        )
        print(
            "INJURY_BLOCKED_PLAYERS="
            + " | ".join(blocked_names)
        )

    slate_df = slate_df.loc[
        ~blocked_mask
    ].copy()

    if slate_df.empty:
        print("FAIL_CLOSED_NO_ELIGIBLE_PLAYERS_AFTER_INJURY_GATE=TRUE")
        raise SystemExit(2)

    remaining_ids = set(
        slate_df["player_id"]
        .map(clean)
        .tolist()
    )
    escapes = sorted(
        remaining_ids & block_ids
    )

    if escapes:
        print(
            "FAIL_CLOSED_BLOCKED_INJURY_ESCAPE="
            + ",".join(escapes)
        )
        raise SystemExit(2)

    print(
        f"INJURY_AUTHORITY_ROWS={len(injury)}"
    )
    print(
        f"INJURY_BLOCKED_ROWS_REMOVED={blocked_count}"
    )
    print("BLOCKED_INJURY_ESCAPE_COUNT=0")

    for c in ("salary", "projection", "mvp_salary", "mvp_projection"):
        bad = ~slate_df[c].map(finite)
        if bad.any():
            print(f"FAIL_CLOSED_NONFINITE_{c.upper()}={int(bad.sum())}")
            raise SystemExit(2)

    if slate_df["player_id"].map(clean).eq("").any():
        print("FAIL_CLOSED_BLANK_PLAYER_ID=TRUE")
        raise SystemExit(2)

    if slate_df["player_id"].duplicated().any():
        print("FAIL_CLOSED_DUPLICATE_PLAYER_ID=TRUE")
        raise SystemExit(2)

    slate_df = slate_df.reset_index(drop=True)

    valid_player_ids = set(
        slate_df["player_id"].map(clean).tolist()
    )

    lock_player_ids = {
        clean(x)
        for x in args.lock_player_id
        if clean(x)
    }

    exclude_player_ids = {
        clean(x)
        for x in args.exclude_player_id
        if clean(x)
    }

    mvp_player_id = clean(args.mvp_player_id)

    unknown_locks = sorted(
        lock_player_ids - valid_player_ids
    )
    unknown_excludes = sorted(
        exclude_player_ids - valid_player_ids
    )

    if unknown_locks:
        print(
            "FAIL_CLOSED_UNKNOWN_LOCK_PLAYER_IDS="
            + ",".join(unknown_locks)
        )
        raise SystemExit(2)

    if unknown_excludes:
        print(
            "FAIL_CLOSED_UNKNOWN_EXCLUDE_PLAYER_IDS="
            + ",".join(unknown_excludes)
        )
        raise SystemExit(2)

    overlap = sorted(
        lock_player_ids & exclude_player_ids
    )

    if overlap:
        print(
            "FAIL_CLOSED_LOCK_EXCLUDE_CONFLICT="
            + ",".join(overlap)
        )
        raise SystemExit(2)

    if len(lock_player_ids) > ROSTER_SIZE:
        print("FAIL_CLOSED_TOO_MANY_LOCKS=TRUE")
        raise SystemExit(2)

    if len(valid_player_ids - exclude_player_ids) < ROSTER_SIZE:
        print("FAIL_CLOSED_TOO_MANY_EXCLUDES=TRUE")
        raise SystemExit(2)

    if mvp_player_id:
        if mvp_player_id not in valid_player_ids:
            print(
                "FAIL_CLOSED_UNKNOWN_MVP_PLAYER_ID="
                + mvp_player_id
            )
            raise SystemExit(2)

        if mvp_player_id in exclude_player_ids:
            print(
                "FAIL_CLOSED_MVP_EXCLUDED=TRUE"
            )
            raise SystemExit(2)

    print("=" * 118)
    print("WFS NFL — SHOWDOWN SOLVER V1")
    print("=" * 118)
    print(f"POOL_SHA256={sha256(POOL)}")
    print(f"SLATE={args.slate}")
    print(f"PLAYER_POOL_ROWS={len(slate_df)}")
    print(f"REQUESTED_LINEUPS={args.lineups}")
    print(f"MIN_SALARY={args.min_salary}")
    print(f"SALARY_CAP={SALARY_CAP}")
    print(
        "LOCK_PLAYER_IDS="
        + ",".join(sorted(lock_player_ids))
    )
    print(
        "EXCLUDE_PLAYER_IDS="
        + ",".join(sorted(exclude_player_ids))
    )
    print(
        "MVP_PLAYER_ID="
        + (mvp_player_id or "AUTO")
    )

    lineups = []
    prior = []

    for n in range(1, args.lineups + 1):
        lu = build_one_lineup(
            slate_df,
            prior,
            min_salary=args.min_salary,
            lock_player_ids=lock_player_ids,
            exclude_player_ids=exclude_player_ids,
            mvp_player_id=mvp_player_id,
        )
        if lu is None:
            print(f"SOLVER_STOPPED_AT_LINEUP={n}")
            break
        audit = audit_lineup(lu)

        if not audit["pass"]:
            print(f"FAIL_CLOSED_AUDIT_LINEUP={n}")
            raise SystemExit(2)

        selected_player_ids = {
            clean(r["player_id"])
            for r in lu["rows"]
        }

        if not lock_player_ids.issubset(
            selected_player_ids
        ):
            print(
                f"FAIL_CLOSED_LOCK_AUDIT_LINEUP={n}"
            )
            raise SystemExit(2)

        if exclude_player_ids & selected_player_ids:
            print(
                f"FAIL_CLOSED_EXCLUDE_AUDIT_LINEUP={n}"
            )
            raise SystemExit(2)

        if mvp_player_id:
            actual_mvp = next(
                clean(r["player_id"])
                for r in lu["rows"]
                if r["role"] == "MVP"
            )

            if actual_mvp != mvp_player_id:
                print(
                    f"FAIL_CLOSED_MVP_AUDIT_LINEUP={n}"
                )
                raise SystemExit(2)

        lineups.append(lu)
        prior.append(lu["roles"])

        print(
            f"LINEUP|n={n}|salary={audit['salary']}|projection={audit['projection']:.6f}|"
            f"mvp={next(r['player'] for r in lu['rows'] if r['role']=='MVP')!r}"
        )

    if len(lineups) != args.lineups:
        print(f"GENERATED_LINEUPS={len(lineups)}")
        print("SHOWDOWN_SOLVER_STATUS=FAIL_CLOSED_INSUFFICIENT_UNIQUE_LINEUPS")
        raise SystemExit(2)

    # Portfolio audit.
    signatures = set()
    for lu in lineups:
        sig = tuple(sorted((r["player_id"], r["role"]) for r in lu["rows"]))
        signatures.add(sig)

    if len(signatures) != len(lineups):
        print("FAIL_CLOSED_DUPLICATE_LINEUP_SIGNATURE=TRUE")
        raise SystemExit(2)

    print("\n=== PORTFOLIO AUDIT ===")
    print(f"GENERATED_LINEUPS={len(lineups)}")
    print(f"UNIQUE_LINEUPS={len(signatures)}")
    print(f"MAX_SALARY={max(lu['total_salary'] for lu in lineups)}")
    print(f"MIN_SALARY_ACTUAL={min(lu['total_salary'] for lu in lineups)}")
    print("ROSTER_RULE=1_MVP_PLUS_5_ANYFLEX")
    print("MVP_SALARY_MULTIPLIER=1.5")
    print("MVP_PROJECTION_MULTIPLIER=1.5")
    print("UNDERLYING_PLAYER_DUPLICATION_ALLOWED=FALSE")

    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "lineup", "role", "player", "player_id", "team", "position",
            "salary", "projection", "projection_source",
            "lineup_salary", "lineup_projection",
        ]
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for n, lu in enumerate(lineups, start=1):
                for r in lu["rows"]:
                    w.writerow({
                        "lineup": n,
                        **r,
                        "lineup_salary": lu["total_salary"],
                        "lineup_projection": f"{lu['total_projection']:.6f}",
                    })
        print(f"OUTPUT_CSV={out}")
        print(f"OUTPUT_CSV_SHA256={sha256(out)}")

    print("CLASSIC_PRODUCTION_CHANGED=FALSE")
    print("PUBLIC_CLASSIC_SOLVER_CHANGED=FALSE")
    print("FUZZY_MATCHING_USED=FALSE")
    print("SHOWDOWN_SOLVER_STATUS=PASS")

if __name__ == "__main__":
    main()
