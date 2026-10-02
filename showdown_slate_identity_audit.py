#!/usr/bin/env python3
from pathlib import Path
import re
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
POOL = ROOT / "data/fanduel/single_game/derived/single_game_projection_pool.parquet"

ALIASES = {"LA": "LAR", "WAS": "WSH", "JAC": "JAX"}

def clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan","none","null"} else s

def canon_team(v):
    s = clean(v).upper()
    return ALIASES.get(s, s)

def parse_matchup(label):
    s = clean(label).upper()
    m = re.fullmatch(r"\s*([A-Z]+)\s*@\s*([A-Z]+)\s*", s)
    if not m:
        return None
    return canon_team(m.group(1)), canon_team(m.group(2))

print("=" * 110)
print("WFS NFL — SHOWDOWN SLATE IDENTITY AUDIT")
print("=" * 110)

if not POOL.exists():
    print(f"FAIL_CLOSED_MISSING_POOL={POOL}")
    raise SystemExit(2)

df = pd.read_parquet(POOL)

needed = {"public_slate_name", "game", "team", "player", "player_id"}
missing = sorted(needed - set(df.columns))
if missing:
    print("FAIL_CLOSED_SCHEMA_MISSING=" + ",".join(missing))
    raise SystemExit(2)

slates = sorted(df["public_slate_name"].astype(str).unique())
overall_pass = True

for slate in slates:
    sdf = df[df["public_slate_name"].astype(str).eq(slate)].copy()
    parsed = parse_matchup(slate)

    print(f"\nSLATE={slate!r}")
    print(f"ROWS={len(sdf)}")

    if parsed is None:
        print("STATUS=FAIL_CLOSED_BAD_SELECTOR_LABEL")
        overall_pass = False
        continue

    away, home = parsed
    expected = {away, home}

    teams = sorted({canon_team(x) for x in sdf["team"] if clean(x)})
    games = sorted({clean(x) for x in sdf["game"] if clean(x)})

    print(f"EXPECTED_TEAMS={sorted(expected)}")
    print(f"ACTUAL_TEAMS={teams}")
    print(f"UNIQUE_GAME_VALUES={games}")

    bad_team_rows = sdf[~sdf["team"].map(canon_team).isin(expected)]
    blank_team_rows = sdf[sdf["team"].map(clean).eq("")]
    blank_game_rows = sdf[sdf["game"].map(clean).eq("")]

    print(f"BAD_TEAM_ROWS={len(bad_team_rows)}")
    print(f"BLANK_TEAM_ROWS={len(blank_team_rows)}")
    print(f"BLANK_GAME_ROWS={len(blank_game_rows)}")

    if len(bad_team_rows):
        print("BAD_TEAM_PLAYERS:")
        for _, r in bad_team_rows[["player","player_id","team","game"]].iterrows():
            print(
                f"  player={clean(r['player'])!r}|player_id={clean(r['player_id'])}|"
                f"team={clean(r['team'])}|game={clean(r['game'])}"
            )

    # Game string should also parse to same canonical matchup for every nonblank row.
    bad_game_rows = []
    for idx, r in sdf.iterrows():
        g = parse_matchup(r["game"])
        if g is None or set(g) != expected:
            bad_game_rows.append(idx)

    print(f"BAD_GAME_ROWS={len(bad_game_rows)}")

    slate_pass = (
        teams == sorted(expected)
        and len(bad_team_rows) == 0
        and len(blank_team_rows) == 0
        and len(blank_game_rows) == 0
        and len(bad_game_rows) == 0
    )
    print(f"STATUS={'PASS' if slate_pass else 'FAIL_CLOSED'}")
    overall_pass = overall_pass and slate_pass

print("\nFUZZY_MATCHING_USED=FALSE")
print(f"SHOWDOWN_SLATE_IDENTITY_AUDIT_STATUS={'PASS' if overall_pass else 'FAIL_CLOSED'}")

if not overall_pass:
    raise SystemExit(2)
