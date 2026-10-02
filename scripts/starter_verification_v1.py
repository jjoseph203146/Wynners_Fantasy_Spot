#!/usr/bin/env python3

# WFS_STARTER_VERIFICATION_V1_1
#
# ANALYSIS / SHADOW ONLY.
#
# Authority:
#   1. Current depth chart = primary projected football-role authority.
#   2. injury_consensus_current = persisted availability authority.
#   3. injury_gate=BLOCK always wins.
#   4. RotoWire projected lineup = corroborating evidence only.
#   5. player_weekly_usage = PRIOR-WEEK evidence only.
#   6. Exact GSIS identity required for all player evidence joins.
#   7. No fuzzy matching.
#   8. Missing/ambiguous evidence fails closed.
#   9. FB lanes remain football evidence but are excluded from the
#      comparable RotoWire QB/RB/WR/TE disagreement metric.
#  10. No mutation of DB, depth, forecasts, Analyst, Stat Outlook,
#      solver, availability, or production artifacts.

from __future__ import annotations

import argparse
import html as htmlmod
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "nfl.db"
OUT = ROOT / "data" / "audits" / "starter_verification_v1"

ROTOWIRE_URL = "https://www.rotowire.com/football/lineups.php"

TEAM_MAP = {"LAR": "LA"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

OFFENSE = {"QB", "RB", "FB", "WR", "TE"}
RW_COMPARABLE = {"QB", "RB", "WR", "TE"}


def norm_team(v):
    x = str(v or "").strip().upper()
    return TEAM_MAP.get(x, x)


def norm_name(v):
    x = htmlmod.unescape(str(v or "")).lower()
    tokens = re.findall(r"[a-z0-9]+", x)

    if tokens and tokens[-1] in SUFFIXES:
        tokens = tokens[:-1]

    return "".join(tokens)


def clean_id(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def fetch_rotowire():
    r = requests.get(
        ROTOWIRE_URL,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()

    boxes = re.findall(
        r'<div class="lineup__box">(.*?)(?=<div class="lineup__box">|\Z)',
        r.text,
        flags=re.S,
    )

    rows = []

    for box in boxes:
        teams = re.findall(
            r'<div class="lineup__abbr">\s*([^<]+?)\s*</div>',
            box,
            flags=re.S,
        )

        if len(teams) < 2:
            continue

        away, home = [norm_team(x) for x in teams[:2]]

        mains = re.findall(
            r'<div class="lineup__main">(.*?)'
            r'(?=<div class="(?:border-b|lineup__bottom|lineup__main)")',
            box,
            flags=re.S,
        )

        if not mains:
            continue

        # First main block = projected lineup.
        # Do not parse later inactive blocks.
        main = mains[0]

        for side, team in [
            ("is-visit", away),
            ("is-home", home),
        ]:
            m = re.search(
                rf'<ul class="lineup__list {side}">(.*?)</ul>',
                main,
                flags=re.S,
            )

            if not m:
                continue

            players = re.findall(
                r'<li class="lineup__player">.*?'
                r'<div class="lineup__pos">\s*([^<]+?)\s*</div>.*?'
                r'<a[^>]*title="([^"]+)"',
                m.group(1),
                flags=re.S,
            )

            for pos, name in players:
                pos = htmlmod.unescape(pos).strip().upper()
                name = htmlmod.unescape(name).strip()

                if pos not in OFFENSE:
                    continue

                rows.append({
                    "team": team,
                    "position": pos,
                    "rotowire_player_name": name,
                    "name_norm": norm_name(name),
                })

    return pd.DataFrame(rows)


def load_depth(conn):
    df = pd.read_sql_query(
        """
        SELECT
            snapshot_dt,
            team,
            player_name,
            gsis_id,
            pos_grp,
            pos_abb,
            pos_slot,
            pos_rank
        FROM depth_charts
        WHERE snapshot_dt = (
            SELECT MAX(d2.snapshot_dt)
            FROM depth_charts d2
            WHERE d2.team = depth_charts.team
        )
        """,
        conn,
    )

    df["team_norm"] = df["team"].map(norm_team)
    df["name_norm"] = df["player_name"].map(norm_name)
    df["pos_norm"] = (
        df["pos_abb"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df["gsis_id"] = df["gsis_id"].map(clean_id)
    df["pos_rank"] = pd.to_numeric(
        df["pos_rank"],
        errors="coerce",
    )

    return df


def resolve_rw(rw, depth):
    resolved = []
    unresolved = []
    ambiguous = []

    for _, row in rw.iterrows():
        hits = depth[
            (depth["team_norm"] == row["team"])
            & (depth["name_norm"] == row["name_norm"])
            & (depth["pos_norm"] == row["position"])
        ]

        ids = sorted({
            clean_id(x)
            for x in hits["gsis_id"]
            if clean_id(x)
        })

        if len(ids) == 1:
            resolved.append({
                **row.to_dict(),
                "gsis_id": ids[0],
            })
        elif len(ids) == 0:
            unresolved.append(row.to_dict())
        else:
            ambiguous.append({
                **row.to_dict(),
                "candidate_gsis_ids": ids,
            })

    return (
        pd.DataFrame(resolved),
        unresolved,
        ambiguous,
    )


def load_usage(conn, season, prior_week):
    df = pd.read_sql_query(
        """
        SELECT
            season,
            week,
            player_id,
            player_name,
            position,
            team,
            offense_snaps,
            offense_pct,
            snap_pct_3g,
            role_expansion_flag,
            role_decline_flag,
            starter_usage_flag
        FROM player_weekly_usage
        WHERE season = ?
          AND week = ?
        """,
        conn,
        params=(season, prior_week),
    )

    df["player_id"] = df["player_id"].map(clean_id)

    dupes = (
        df[df["player_id"] != ""]
        .groupby("player_id")
        .size()
    )

    bad = dupes[dupes > 1]

    if not bad.empty:
        raise RuntimeError(
            "Duplicate prior-week player_id: "
            + ", ".join(bad.index.tolist()[:20])
        )

    return df


def load_availability(conn, season, week, game_type):
    df = pd.read_sql_query(
        """
        SELECT
            season,
            week,
            game_type,
            team,
            gsis_id,
            player_name,
            position,
            consensus_status,
            consensus_practice_status,
            injury_gate,
            injury_gate_reason,
            availability_risk,
            injury_risk,
            authoritative_source,
            updated_at
        FROM injury_consensus_current
        WHERE season = ?
          AND week = ?
          AND game_type = ?
        """,
        conn,
        params=(season, week, game_type),
    )

    df["gsis_id"] = df["gsis_id"].map(clean_id)

    dupes = (
        df[df["gsis_id"] != ""]
        .groupby("gsis_id")
        .size()
    )

    bad = dupes[dupes > 1]

    if not bad.empty:
        raise RuntimeError(
            "Duplicate injury consensus GSIS IDs: "
            + ", ".join(bad.index.tolist()[:20])
        )

    return df


def latest_depth_starters(depth):
    d = depth[
        depth["pos_norm"].isin(OFFENSE)
        & (depth["gsis_id"] != "")
        & depth["pos_rank"].notna()
    ].copy()

    d = d.sort_values(
        [
            "team_norm",
            "pos_grp",
            "pos_norm",
            "pos_slot",
            "pos_rank",
            "gsis_id",
        ],
        kind="stable",
    )

    # Preserve exact football lane.
    return d.groupby(
        [
            "team_norm",
            "pos_grp",
            "pos_norm",
            "pos_slot",
        ],
        dropna=False,
        as_index=False,
    ).first()


def usage_record(row):
    if row is None:
        return {
            "snaps": None,
            "pct": None,
            "snap_3g": None,
            "starter_flag": None,
            "expansion": None,
            "decline": None,
            "prior_team": None,
        }

    return {
        "snaps": row["offense_snaps"],
        "pct": row["offense_pct"],
        "snap_3g": row["snap_pct_3g"],
        "starter_flag": row["starter_usage_flag"],
        "expansion": row["role_expansion_flag"],
        "decline": row["role_decline_flag"],
        "prior_team": row["team"],
    }


def availability_record(row):
    if row is None:
        return {
            "row_present": False,
            "gate": "NO_CONSENSUS_ROW",
            "status": None,
            "practice": None,
            "reason": None,
            "risk": None,
            "injury_risk": None,
            "source": None,
        }

    return {
        "row_present": True,
        "gate": str(row["injury_gate"] or "").strip().upper(),
        "status": row["consensus_status"],
        "practice": row["consensus_practice_status"],
        "reason": row["injury_gate_reason"],
        "risk": row["availability_risk"],
        "injury_risk": row["injury_risk"],
        "source": row["authoritative_source"],
    }


def role_result(
    comparable,
    depth_id,
    rw_id,
    depth_avail,
    rw_avail,
    depth_usage,
    rw_usage,
):
    if not comparable:
        return "FB_INFORMATIONAL"

    # Persisted hard availability always wins.
    if depth_id and depth_avail["gate"] == "BLOCK":
        if rw_id == depth_id:
            return "BLOCKED_STARTER"
        return "DEPTH_STARTER_BLOCKED"

    if rw_id and rw_avail["gate"] == "BLOCK":
        return "ROTOWIRE_STARTER_BLOCKED"

    if not depth_id and not rw_id:
        return "UNKNOWN"

    if depth_id and rw_id and depth_id == rw_id:
        return "AGREE"

    if depth_id and rw_id and depth_id != rw_id:
        rw_flag = rw_usage["starter_flag"]
        depth_flag = depth_usage["starter_flag"]

        rw_pct = rw_usage["pct"]
        depth_pct = depth_usage["pct"]

        if (
            pd.notna(rw_flag)
            and int(rw_flag) == 1
            and (
                pd.isna(depth_flag)
                or int(depth_flag) == 0
            )
        ):
            return "DISAGREE_RW_PRIOR_USAGE_SUPPORT"

        if (
            pd.notna(rw_pct)
            and pd.notna(depth_pct)
            and float(rw_pct) > float(depth_pct)
        ):
            return "DISAGREE_RW_MORE_PRIOR_SNAPS"

        return "DISAGREE_UNRESOLVED"

    if rw_id and not depth_id:
        return "ROTOWIRE_ONLY"

    if depth_id and not rw_id:
        return "DEPTH_ONLY"

    return "UNKNOWN"


def build_roles(
    depth_starters,
    rw,
    usage,
    availability,
):
    usage_map = {
        clean_id(r["player_id"]): r
        for _, r in usage.iterrows()
        if clean_id(r["player_id"])
    }

    avail_map = {
        clean_id(r["gsis_id"]): r
        for _, r in availability.iterrows()
        if clean_id(r["gsis_id"])
    }

    rw_by_team_pos = {}

    for _, r in rw.iterrows():
        key = (r["team"], r["position"])
        rw_by_team_pos.setdefault(key, []).append(r)

    records = []

    # Depth starter lanes remain the base authority.
    for _, d in depth_starters.iterrows():
        team = d["team_norm"]
        pos = d["pos_norm"]
        depth_id = clean_id(d["gsis_id"])

        candidates = rw_by_team_pos.get(
            (team, pos),
            [],
        )

        # RotoWire has one QB/RB/TE and three WRs.
        # For exact player agreement, consume the matching identity.
        exact = [
            x for x in candidates
            if clean_id(x["gsis_id"]) == depth_id
        ]

        rw_row = exact[0] if exact else None

        # If this depth lane has no exact RotoWire identity, defer
        # alternate pairing until the team-position reconciliation
        # phase below.
        rw_id = clean_id(
            rw_row["gsis_id"]
            if rw_row is not None
            else ""
        )

        du = usage_record(usage_map.get(depth_id))
        da = availability_record(avail_map.get(depth_id))

        ru = usage_record(
            usage_map.get(rw_id)
            if rw_id
            else None
        )
        ra = availability_record(
            avail_map.get(rw_id)
            if rw_id
            else None
        )

        records.append({
            "team": team,
            "position": pos,
            "pos_grp": d["pos_grp"],
            "pos_slot": d["pos_slot"],
            "depth_gsis_id": depth_id,
            "depth_player_name": d["player_name"],
            "rotowire_gsis_id": rw_id or None,
            "rotowire_player_name": (
                rw_row["rotowire_player_name"]
                if rw_row is not None
                else None
            ),
            "comparable_to_rotowire": pos in RW_COMPARABLE,

            "depth_week1_snaps": du["snaps"],
            "depth_week1_pct": du["pct"],
            "depth_week1_starter_usage": du["starter_flag"],
            "depth_week1_role_expansion": du["expansion"],

            "rw_week1_snaps": ru["snaps"],
            "rw_week1_pct": ru["pct"],
            "rw_week1_starter_usage": ru["starter_flag"],
            "rw_week1_role_expansion": ru["expansion"],

            "depth_availability_present": da["row_present"],
            "depth_injury_gate": da["gate"],
            "depth_consensus_status": da["status"],
            "depth_injury_gate_reason": da["reason"],

            "rw_availability_present": ra["row_present"],
            "rw_injury_gate": ra["gate"],
            "rw_consensus_status": ra["status"],
            "rw_injury_gate_reason": ra["reason"],

            "verification_status": role_result(
                pos in RW_COMPARABLE,
                depth_id,
                rw_id,
                da,
                ra,
                du,
                ru,
            ),
        })

    result = pd.DataFrame(records)

    # ----------------------------------------------------------
    # Reconcile unmatched RotoWire starters against unmatched
    # depth starters within the same TEAM + POSITION.
    #
    # This is evidence pairing, NOT identity resolution.
    # Identity remains exact GSIS.
    # ----------------------------------------------------------

    for (team, pos), rw_group in rw.groupby(
        ["team", "position"]
    ):
        if pos not in RW_COMPARABLE:
            continue

        rw_ids = set(
            rw_group["gsis_id"].map(clean_id)
        )

        mask = (
            (result["team"] == team)
            & (result["position"] == pos)
            & result["comparable_to_rotowire"]
        )

        existing_rw_ids = set(
            result.loc[mask, "rotowire_gsis_id"]
            .dropna()
            .map(clean_id)
        )

        unmatched_rw = [
            r
            for _, r in rw_group.iterrows()
            if clean_id(r["gsis_id"])
            not in existing_rw_ids
        ]

        unmatched_depth_idx = [
            idx
            for idx in result.index[mask]
            if clean_id(
                result.at[idx, "depth_gsis_id"]
            ) not in rw_ids
            and not clean_id(
                result.at[idx, "rotowire_gsis_id"]
            )
        ]

        # Pair only when cardinality is deterministic.
        if (
            len(unmatched_rw) == 1
            and len(unmatched_depth_idx) == 1
        ):
            idx = unmatched_depth_idx[0]
            r = unmatched_rw[0]

            rw_id = clean_id(r["gsis_id"])
            depth_id = clean_id(
                result.at[idx, "depth_gsis_id"]
            )

            du = usage_record(usage_map.get(depth_id))
            ru = usage_record(usage_map.get(rw_id))

            da = availability_record(
                avail_map.get(depth_id)
            )
            ra = availability_record(
                avail_map.get(rw_id)
            )

            result.at[idx, "rotowire_gsis_id"] = rw_id
            result.at[
                idx,
                "rotowire_player_name"
            ] = r["rotowire_player_name"]

            result.at[idx, "rw_week1_snaps"] = ru["snaps"]
            result.at[idx, "rw_week1_pct"] = ru["pct"]
            result.at[
                idx,
                "rw_week1_starter_usage"
            ] = ru["starter_flag"]
            result.at[
                idx,
                "rw_week1_role_expansion"
            ] = ru["expansion"]

            result.at[
                idx,
                "rw_availability_present"
            ] = ra["row_present"]
            result.at[
                idx,
                "rw_injury_gate"
            ] = ra["gate"]
            result.at[
                idx,
                "rw_consensus_status"
            ] = ra["status"]
            result.at[
                idx,
                "rw_injury_gate_reason"
            ] = ra["reason"]

            result.at[
                idx,
                "verification_status"
            ] = role_result(
                True,
                depth_id,
                rw_id,
                da,
                ra,
                du,
                ru,
            )

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=2)
    ap.add_argument("--prior-week", type=int, default=1)
    ap.add_argument("--game-type", default="REG")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    rw_raw = fetch_rotowire()

    with sqlite3.connect(DB) as conn:
        depth = load_depth(conn)

        rw, unresolved, ambiguous = resolve_rw(
            rw_raw,
            depth,
        )

        usage = load_usage(
            conn,
            args.season,
            args.prior_week,
        )

        availability = load_availability(
            conn,
            args.season,
            args.week,
            args.game_type,
        )

    identity_gate = (
        len(rw_raw) == 192
        and len(rw) == 192
        and not unresolved
        and not ambiguous
    )

    if not identity_gate:
        print("IDENTITY_GATE = FAIL_CLOSED")
        print("ROTOWIRE_ROWS =", len(rw_raw))
        print("RESOLVED =", len(rw))
        print("UNRESOLVED =", len(unresolved))
        print("AMBIGUOUS =", len(ambiguous))
        raise SystemExit(2)

    starters = latest_depth_starters(depth)

    roles = build_roles(
        starters,
        rw,
        usage,
        availability,
    )

    comparable = roles[
        roles["comparable_to_rotowire"]
    ].copy()

    fb = roles[
        ~roles["comparable_to_rotowire"]
    ].copy()

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")

    csv_path = OUT / (
        f"starter_verification_v1_1_{stamp}.csv"
    )
    json_path = OUT / (
        f"starter_verification_v1_1_{stamp}.json"
    )

    roles.to_csv(csv_path, index=False)

    # WFS_STARTER_VERIFICATION_CURRENT_PUBLICATION_V1
    #
    # Stable, atomic evidence publication for downstream
    # read-only consumers. This does not grant production influence.
    current_dir = ROOT / "data" / "parquet"
    current_dir.mkdir(parents=True, exist_ok=True)

    current_path = (
        current_dir
        / "current_starter_verification.parquet"
    )
    current_manifest_path = (
        current_dir
        / "current_starter_verification_manifest.json"
    )

    current_tmp = (
        current_dir
        / "current_starter_verification.tmp.parquet"
    )
    manifest_tmp = (
        current_dir
        / "current_starter_verification_manifest.tmp.json"
    )

    roles.to_parquet(
        current_tmp,
        index=False,
    )

    current_manifest = {
        "contract": "WFS_STARTER_VERIFICATION_CURRENT_V1",
        "source_contract": "WFS_STARTER_VERIFICATION_V1_1",
        "generated_at_utc": now.isoformat(),
        "season": args.season,
        "week": args.week,
        "prior_usage_week": args.prior_week,
        "game_type": args.game_type,
        "identity_gate": "PASS",
        "rotowire_rows": len(rw_raw),
        "rotowire_resolved": len(rw),
        "identity_unresolved_count": len(unresolved),
        "identity_ambiguous_count": len(ambiguous),
        "role_rows": len(roles),
        "analysis_only": True,
        "production_influence": False,
        "solver_influence": False,
        "forecast_mutation": False,
        "analyst_influence": False,
        "database_mutation": False,
        "depth_chart_mutation": False,
        "current_artifact": str(
            current_path.relative_to(ROOT)
        ),
        "audit_csv": str(
            csv_path.relative_to(ROOT)
        ),
        "audit_json": str(
            json_path.relative_to(ROOT)
        ),
    }

    manifest_tmp.write_text(
        json.dumps(
            current_manifest,
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )

    current_tmp.replace(current_path)
    manifest_tmp.replace(current_manifest_path)

    counts = (
        comparable["verification_status"]
        .value_counts(dropna=False)
        .to_dict()
    )

    fb_counts = (
        fb["verification_status"]
        .value_counts(dropna=False)
        .to_dict()
    )

    payload = {
        "contract": "WFS_STARTER_VERIFICATION_V1_1",
        "generated_at_utc": now.isoformat(),
        "analysis_only": True,
        "production_influence": False,
        "solver_influence": False,
        "forecast_mutation": False,
        "analyst_influence": False,
        "depth_chart_mutation": False,
        "database_mutation": False,

        "season": args.season,
        "week": args.week,
        "prior_usage_week": args.prior_week,
        "game_type": args.game_type,

        "rotowire_rows": len(rw_raw),
        "rotowire_resolved": len(rw),
        "identity_unresolved": unresolved,
        "identity_ambiguous": ambiguous,
        "identity_gate": "PASS",

        "prior_usage_rows": len(usage),
        "availability_rows": len(availability),
        "availability_blocks": int(
            (
                availability["injury_gate"]
                .fillna("")
                .str.upper()
                == "BLOCK"
            ).sum()
        ),

        "role_rows": len(roles),
        "comparable_role_rows": len(comparable),
        "fb_informational_rows": len(fb),
        "comparable_status_counts": counts,
        "fb_status_counts": fb_counts,

        "csv_artifact": str(
            csv_path.relative_to(ROOT)
        ),
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )

    print("STARTER_VERIFICATION_V1_1")
    print("IDENTITY_GATE = PASS")
    print("ROTOWIRE_ROWS =", len(rw_raw))
    print("ROTOWIRE_RESOLVED =", len(rw))
    print("PRIOR_USAGE_ROWS =", len(usage))
    print("AVAILABILITY_ROWS =", len(availability))
    print(
        "AVAILABILITY_BLOCKS =",
        payload["availability_blocks"],
    )
    print("ROLE_ROWS =", len(roles))
    print(
        "COMPARABLE_ROLE_ROWS =",
        len(comparable),
    )
    print(
        "FB_INFORMATIONAL_ROWS =",
        len(fb),
    )
    print(
        "COMPARABLE_STATUS_COUNTS =",
        counts,
    )
    print("FB_STATUS_COUNTS =", fb_counts)
    print("CSV =", csv_path)
    print("JSON =", json_path)
    print("ANALYSIS_ONLY = True")
    print("PRODUCTION_INFLUENCE = False")

    interesting = comparable[
        comparable["verification_status"] != "AGREE"
    ].copy()

    print(
        "NON_AGREE_COMPARABLE_ROWS =",
        len(interesting),
    )

    if not interesting.empty:
        print("\n=== COMPARABLE NON-AGREEMENTS ===")

        cols = [
            "team",
            "position",
            "depth_player_name",
            "rotowire_player_name",
            "depth_gsis_id",
            "rotowire_gsis_id",
            "depth_week1_snaps",
            "depth_week1_pct",
            "depth_week1_starter_usage",
            "rw_week1_snaps",
            "rw_week1_pct",
            "rw_week1_starter_usage",
            "depth_injury_gate",
            "rw_injury_gate",
            "verification_status",
        ]

        print(
            interesting[cols]
            .sort_values(
                [
                    "team",
                    "position",
                    "verification_status",
                ]
            )
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
