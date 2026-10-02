#!/usr/bin/env python3
import json
import math
import re
import sys

try:
    import nflreadpy as nfl
except Exception as e:
    print(f"FAIL_CLOSED_IMPORT_NFLREADPY={type(e).__name__}:{e}")
    raise SystemExit(2)

HISTORICAL_SEASONS = [2023, 2024, 2025]

TARGET_KICKERS = {
    "00-0032569": "Wil Lutz",
    "00-0033303": "Harrison Butker",
    "00-0040200": "Andy Borregales",
    "00-0031492": "Jason Myers",
    "00-0039498": "Harrison Mevis",
    "00-0034173": "Eddy Pineiro",
}

IDENTITY_PATTERNS = (
    "kicker", "player_id", "player_name", "gsis", "fantasy_id",
)
KICK_PATTERNS = (
    "field_goal", "extra_point", "kick_distance", "pat_result",
)

def clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s

def is_true(v):
    s = clean(v).lower()
    return s in {"1", "1.0", "true", "t", "yes"}

print("=" * 118)
print("WFS NFL — NFL-POSTGAME-1F-I-U")
print("NFLVERSE PBP KICKER RAW CONTRACT AUDIT — READ ONLY")
print("=" * 118)

all_exact_rows = 0
all_kick_rows = 0
schema_union = set()
season_summaries = []

for season in HISTORICAL_SEASONS:
    print(f"\n=== NFLVERSE PBP {season} ===")
    try:
        pl_df = nfl.load_pbp(season)
    except Exception as e:
        print(f"FAIL_CLOSED_LOAD_PBP|season={season}|error={type(e).__name__}:{e}")
        raise SystemExit(2)

    cols = list(pl_df.columns)
    schema_union.update(cols)

    identity_cols = [c for c in cols if any(p in c.lower() for p in IDENTITY_PATTERNS)]
    kick_cols = [c for c in cols if any(p in c.lower() for p in KICK_PATTERNS)]

    print(f"PBP_COLUMNS={len(cols)}")
    print(f"IDENTITY_CANDIDATE_COLUMNS={json.dumps(identity_cols)}")
    print(f"KICK_COLUMNS={json.dumps(kick_cols)}")

    required_event = ["game_id", "posteam", "field_goal_attempt", "field_goal_result",
                      "extra_point_attempt", "extra_point_result"]
    missing_required = [c for c in required_event if c not in cols]
    print(f"MISSING_REQUIRED_EVENT_COLUMNS={json.dumps(missing_required)}")

    if missing_required:
        print(f"FAIL_CLOSED_REQUIRED_EVENT_SCHEMA|season={season}")
        raise SystemExit(2)

    select_cols = []
    for c in ["game_id", "season", "week", "posteam",
              "field_goal_attempt", "field_goal_result",
              "extra_point_attempt", "extra_point_result", "kick_distance"]:
        if c in cols and c not in select_cols:
            select_cols.append(c)
    for c in identity_cols:
        if c not in select_cols:
            select_cols.append(c)

    df = pl_df.select(select_cols).to_pandas()

    fg = df["field_goal_attempt"].fillna(0).astype(str).isin(["1", "1.0", "True", "true"])
    xp = df["extra_point_attempt"].fillna(0).astype(str).isin(["1", "1.0", "True", "true"])
    kicks = df[fg | xp].copy()
    all_kick_rows += len(kicks)

    print(f"KICK_EVENT_ROWS={len(kicks)}")
    print(f"FG_ATTEMPT_ROWS={int(fg.sum())}")
    print(f"XP_ATTEMPT_ROWS={int(xp.sum())}")

    # Determine which identity columns actually contain exact target GSIS IDs/names.
    exact_cols = []
    for c in identity_cols:
        vals = kicks[c].astype(str).str.strip()
        if vals.isin(TARGET_KICKERS.keys()).any() or vals.isin(TARGET_KICKERS.values()).any():
            exact_cols.append(c)
    print(f"EXACT_TARGET_IDENTITY_COLUMNS={json.dumps(exact_cols)}")

    season_exact = 0
    for pid, name in TARGET_KICKERS.items():
        mask = None
        matched_cols = []
        for c in identity_cols:
            s = kicks[c].astype(str).str.strip()
            m = s.eq(pid) | s.eq(name)
            if m.any():
                matched_cols.append(c)
                mask = m if mask is None else (mask | m)
        hit = kicks[mask] if mask is not None else kicks.iloc[0:0]
        if len(hit):
            season_exact += len(hit)
            print(
                f"KICKER_EXACT|season={season}|player={name!r}|player_id={pid}|"
                f"rows={len(hit)}|matched_columns={json.dumps(matched_cols)}"
            )
            sample_cols = ["game_id", "week", "posteam",
                           "field_goal_attempt", "field_goal_result",
                           "extra_point_attempt", "extra_point_result"]
            if "kick_distance" in hit.columns:
                sample_cols.append("kick_distance")
            for c in matched_cols:
                if c not in sample_cols:
                    sample_cols.append(c)
            for _, row in hit[sample_cols].head(5).iterrows():
                print("KICKER_SAMPLE|" + json.dumps(
                    {c: row[c] for c in sample_cols}, default=str, sort_keys=True
                ))

    all_exact_rows += season_exact
    season_summaries.append({
        "season": season,
        "kick_event_rows": int(len(kicks)),
        "exact_target_rows": int(season_exact),
        "exact_identity_columns": exact_cols,
    })
    print(f"SEASON_EXACT_TARGET_ROWS={season_exact}")

print("\n=== UNION SCHEMA CLASSIFICATION ===")
union = sorted(schema_union)
for token in ("kicker", "field_goal", "extra_point", "kick_distance"):
    hits = [c for c in union if token in c.lower()]
    print(f"UNION_{token.upper()}_COLUMNS={json.dumps(hits)}")

print("\n=== SUMMARY ===")
print(f"HISTORICAL_SEASONS={json.dumps(HISTORICAL_SEASONS)}")
print(f"TOTAL_KICK_EVENT_ROWS={all_kick_rows}")
print(f"EXACT_TARGET_KICKER_ROWS={all_exact_rows}")
print("SEASON_SUMMARIES=" + json.dumps(season_summaries, sort_keys=True))

print("\n=== SAFETY ===")
print("NFLVERSE_READ_ONLY=TRUE")
print("LOCAL_FILE_WRITES=0")
print("NFL_DB_WRITE_OPERATIONS=0")
print("KICKER_PROJECTION_CREATED=FALSE")
print("HISTORICAL_ACTUAL_USED_AS_PROJECTION=FALSE")
print("FUZZY_MATCHING_USED=FALSE")
print("CLASSIC_PRODUCTION_CHANGED=FALSE")
print("PUBLIC_SOLVER_CHANGED=FALSE")

# PASS means the raw contract audit executed deterministically.
# Whether exact kicker identity exists is reported separately and is not fabricated.
print("NFL_POSTGAME_1F_I_U_STATUS=PASS")
