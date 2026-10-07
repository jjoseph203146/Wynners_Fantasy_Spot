#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ROSTERS = ROOT / "data/parquet/nfl_weekly_rosters.parquet"

OUTDIR = ROOT / "data/research/single_game_nonparticipant_results_v1"
OUT = OUTDIR / "2026_week_04.parquet"
MANIFEST = OUTDIR / "2026_week_04_manifest.json"

SEASON = 2026
WEEK = 4

# Exact E-19 unresolved universe.
# E-20 proved 75 are nonparticipants and three are ACT.
TARGETS = {
    "00-0033553": "James Conner",
    "00-0039921": "Trey Benson",
    "00-0035250": "Devin Singletary",
    "00-0040225": "Thomas Fidone II",
    "00-0038705": "Emari Demercado",
    "00-0038389": "Israel Abanikanda",
    "00-0037563": "Malik Davis",
    "00-0039603": "British Brooks",
    "00-0041467": "Lewis Bond",
    "00-0038977": "Tank Dell",
    "00-0035406": "Lil'Jordan Humphrey",
    "00-0037112": "Jake Tonges",
    "00-0034775": "Christian Kirk",
    "00-0032775": "Demarcus Robinson",
    "00-0039344": "Jonathon Brooks",
    "00-0040644": "Trevor Etienne",
    "00-0039356": "Ja'Tavion Sanders",
    "00-0039394": "Casey Washington",
    "00-0039491": "Jalen Coker",
    "00-0039342": "Xavier Legette",
    "00-0037197": "Isiah Pacheco",
    "00-0034270": "Tyler Conklin",
    "00-0039146": "Jayden Reed",
    "00-0040009": "Savion Williams",
    "00-0037311": "Ko Kieft",
    "00-0035359": "David Sills V",
    "00-0039855": "Jalen McMillan",
    "00-0040179": "DJ Giddens",
    "00-0038394": "Will Mallory",
    "00-0037664": "Alec Pierce",
    "00-0030279": "Keenan Allen",
    "00-0033955": "Jeremy McNichols",
    "00-0037256": "Rachaad White",
    "00-0039355": "Luke McCaffrey",
    "00-0035659": "Terry McLaurin",
    "00-0039266": "Kendall Milton",
    "00-0038619": "Andrei Iosivas",
    "00-0040375": "Jordan Moore",
    "00-0039824": "Jared Wiley",
    "00-0040581": "Dont'e Thornton Jr.",
    "00-0040729": "Jack Bech",
    "00-0038046": "Charlie Kolar",
    "00-0033885": "David Njoku",
    "00-0035589": "Gary Jennings Jr.",
    "00-0040184": "KeAndre Lambert-Smith",
    "00-0040649": "Robbie Ouzts",
    "00-0038752": "Jake Bobo",
    "00-0040653": "Ricky White III",
    "00-0037557": "Ronnie Rivers",
    "00-0036244": "Colby Parkinson",
    "00-0040737": "Terrance Ferguson",
    "00-0040597": "Brennan Presley",
    "00-0038359": "Xavier Smith",
    "00-0034351": "Dallas Goedert",
    "00-0037086": "Grant Calcaterra",
    "00-0036912": "DeVonta Smith",
    "00-0035662": "Marquise Brown",
    "00-0039040": "De'Von Achane",
    "00-0037525": "Jordan Mason",
    "00-0035249": "Josh Oliver",
    "00-0041342": "Dillon Bell",
    "00-0036322": "Justin Jefferson",
    "00-0038720": "Julian Hill",
    "00-0035676": "A.J. Brown",
    "00-0037329": "Brittain Brown",
    "00-0038905": "Blake Grupe",
    "00-0038120": "Breece Hall",
    "00-0040736": "Mason Taylor",
    "00-0039890": "Adonai Mitchell",
    "00-0033375": "Tim Patrick",
    "00-0040162": "Dylan Sampson",
    "00-0040218": "Jimmy Horn Jr.",
    "00-0036630": "Tylan Wallace",
    "00-0036139": "Rico Dowdle",
    "00-0036876": "Kylen Granson",
}

EXCLUDED_ACTIVE = {
    "00-0039814": "Erick All Jr.",
    "00-0035704": "Drew Lock",
    "00-0037745": "Velus Jones",
}

ALLOWED = {"INA", "RES", "DEV", "CUT"}


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


print("=" * 72)
print("WFS_STAGE5_SG_NONPARTICIPANT_RESULTS_V1")
print("RESEARCH ONLY — NO PRODUCTION WRITES")
print("=" * 72)

if len(TARGETS) != 75:
    raise RuntimeError(
        f"FAIL_CLOSED:TARGET_COUNT={len(TARGETS)}"
    )

if set(TARGETS) & set(EXCLUDED_ACTIVE):
    raise RuntimeError(
        "FAIL_CLOSED:ACTIVE_PLAYER_IN_TARGETS"
    )

roster = pd.read_parquet(ROSTERS)

required = {
    "season",
    "week",
    "gsis_id",
    "full_name",
    "team",
    "position",
    "status",
    "status_description_abbr",
}

missing = required - set(roster.columns)

if missing:
    raise RuntimeError(
        f"FAIL_CLOSED:ROSTER_COLUMNS={sorted(missing)}"
    )

r = roster[
    pd.to_numeric(
        roster["season"],
        errors="coerce",
    ).eq(SEASON)
    & pd.to_numeric(
        roster["week"],
        errors="coerce",
    ).eq(WEEK)
].copy()

r["gsis_id"] = (
    r["gsis_id"]
    .astype("string")
    .str.strip()
)

rows = []

for pid, expected_name in TARGETS.items():

    hit = r[
        r["gsis_id"].eq(pid)
    ].copy()

    if len(hit) != 1:
        raise RuntimeError(
            f"FAIL_CLOSED:ROSTER_ROWS:{pid}:{len(hit)}"
        )

    x = hit.iloc[0]

    status = str(x["status"]).strip().upper()

    if status not in ALLOWED:
        raise RuntimeError(
            f"FAIL_CLOSED:NONPARTICIPANT_STATUS:"
            f"{pid}:{status}"
        )

    rows.append({
        "season": SEASON,
        "week": WEEK,
        "player_id": pid,
        "player_name": expected_name,
        "roster_name": str(x["full_name"]).strip(),
        "team": str(x["team"]).strip(),
        "position": str(x["position"]).strip(),
        "roster_status": status,
        "roster_status_abbr": (
            str(x["status_description_abbr"]).strip()
        ),
        "fanduel_points": 0.0,
        "result_status":
            "AUTHORITATIVE_NONPARTICIPANT_ZERO",
        "authority":
            "NFL_WEEKLY_ROSTER_2026_WEEK4",
        "production_influence": False,
        "solver_influence": False,
    })

out = pd.DataFrame(rows)

if len(out) != 75:
    raise RuntimeError("FAIL_CLOSED:OUTPUT_COUNT")

if out["player_id"].duplicated().any():
    raise RuntimeError("FAIL_CLOSED:DUPLICATE_PLAYER")

if not out["roster_status"].isin(ALLOWED).all():
    raise RuntimeError("FAIL_CLOSED:BAD_STATUS")

if not out["fanduel_points"].eq(0.0).all():
    raise RuntimeError("FAIL_CLOSED:NONZERO_RESULT")

# Reconfirm the three active players are actually ACT and excluded.
for pid, name in EXCLUDED_ACTIVE.items():

    hit = r[r["gsis_id"].eq(pid)]

    if len(hit) != 1:
        raise RuntimeError(
            f"FAIL_CLOSED:ACTIVE_ROSTER_ROWS:{pid}:{len(hit)}"
        )

    status = str(
        hit.iloc[0]["status"]
    ).strip().upper()

    if status != "ACT":
        raise RuntimeError(
            f"FAIL_CLOSED:EXPECTED_ACTIVE:{pid}:{status}"
        )

captured = datetime.now(timezone.utc).isoformat()

out["captured_at_utc"] = captured

OUTDIR.mkdir(parents=True, exist_ok=True)

out.to_parquet(OUT, index=False)

manifest = {
    "contract":
        "WFS_STAGE5_SG_NONPARTICIPANT_RESULTS_V1",
    "season": SEASON,
    "week": WEEK,
    "rows": len(out),
    "allowed_nonparticipant_statuses":
        sorted(ALLOWED),
    "excluded_active_players":
        EXCLUDED_ACTIVE,
    "zero_policy":
        "exact_weekly_roster_nonparticipant_status_only",
    "generic_missing_to_zero":
        False,
    "production_influence":
        False,
    "solver_influence":
        False,
    "created_at_utc":
        captured,
    "artifact":
        str(OUT.relative_to(ROOT)),
}

MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)

print("ROWS=", len(out))
print(
    "STATUS_COUNTS=",
    out["roster_status"].value_counts().to_dict(),
)

print("OUTPUT=", OUT)
print("OUTPUT_SHA256=", sha256(OUT))
print("MANIFEST=", MANIFEST)
print("MANIFEST_SHA256=", sha256(MANIFEST))

print()
print("EXCLUDED_ACTIVE_PLAYERS=")
for pid, name in EXCLUDED_ACTIVE.items():
    print(pid, name)

print()
print("GENERIC_MISSING_TO_ZERO=False")
print("PRODUCTION_WRITES=False")
print("STAGE5_NONPARTICIPANT_SIDECAR=PASS")
