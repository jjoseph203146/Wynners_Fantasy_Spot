#!/usr/bin/env python3

from pathlib import Path
from itertools import combinations
import sqlite3
import re
import json
import hashlib

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DB = ROOT / "data/nfl.db"
POOL = ROOT / "data/fanduel/single_game/derived/single_game_projection_pool.parquet"
RAW = ROOT / "data/fanduel/single_game/raw"
DST_PATH = ROOT / "data/parquet/nfl_current_dst_postgame_actuals.parquet"

COMPLETION_PATH = (
    ROOT
    / "data/research/single_game_result_completion_v1/2026_week_04.parquet"
)
COMPLETION_MANIFEST = (
    ROOT
    / "data/research/single_game_result_completion_v1/"
      "2026_week_04_manifest.json"
)

NONPARTICIPANT_PATH = (
    ROOT
    / "data/research/single_game_nonparticipant_results_v1/2026_week_04.parquet"
)

OUTDIR = ROOT / "data/research/single_game_perfect_lineups_v1"
OUT = OUTDIR / "2026_week_04_perfect_lineups.parquet"
MANIFEST = OUTDIR / "2026_week_04_perfect_lineups_manifest.json"

SEASON = 2026
WEEK = 4
SALARY_CAP = 60000

EXPECTED_COMPLETION_ROWS = 28

EXPECTED_COMPLETION_SHA256 = (
    "25c8a4e19212bd6c020a9a38c845225153c7b106c256bfc394bc4f1789ce4d22"
)

EXPECTED_COMPLETION_MANIFEST_SHA256 = (
    "ec1120a1d782579a79ef70df98448a60ca9ad8375e2c8b53481a0ae6974fad38"
)

ALIASES = {
    "LA": "LAR",
    "LAR": "LAR",
    "WAS": "WSH",
    "WSH": "WSH",
    "JAC": "JAX",
    "JAX": "JAX",
}

DST_POSITIONS = {"D", "DEF", "DST", "D/ST"}


def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def canon(x):
    x = clean(x).upper()
    return ALIASES.get(x, x)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def bind_source_csv_to_game(pool, games):
    games = games.copy()

    games["away_c"] = games["away_team"].map(canon)
    games["home_c"] = games["home_team"].map(canon)

    mapping = {}

    for src in pool["source_csv"].dropna().astype(str).unique():

        f = Path(src)

        if not f.exists():
            f = RAW / f.name

        if not f.exists():
            raise RuntimeError(
                f"RAW_SOURCE_MISSING: {src}"
            )

        raw = pd.read_csv(f)

        if "gameInfo" not in raw.columns:
            raise RuntimeError(
                f"GAMEINFO_MISSING: {f}"
            )

        infos = (
            raw["gameInfo"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(infos) != 1:
            raise RuntimeError(
                f"GAMEINFO_NOT_UNIQUE: "
                f"{f} values={infos.tolist()}"
            )

        m = re.search(
            r"\b([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\b",
            infos[0].upper(),
        )

        if not m:
            raise RuntimeError(
                f"GAMEINFO_PARSE_FAIL: "
                f"{f} value={infos[0]}"
            )

        away = canon(m.group(1))
        home = canon(m.group(2))

        hit = games[
            games["away_c"].eq(away)
            & games["home_c"].eq(home)
        ]

        if len(hit) != 1:
            raise RuntimeError(
                f"GAME_BIND_FAIL: "
                f"{f} {away}@{home} "
                f"matches={len(hit)}"
            )

        mapping[src] = clean(
            hit.iloc[0]["game_id"]
        )

    out = pool.copy()

    out["game_id"] = (
        out["source_csv"]
        .astype(str)
        .map(mapping)
    )

    if out["game_id"].isna().any():
        raise RuntimeError(
            "UNBOUND_POOL_ROWS"
        )

    return out


def exact_best_lineups(game_pool):
    """
    Exact deterministic FanDuel Single-Game perfect-lineup search.

    Same contest rules as the original exhaustive implementation:
      - exactly 1 MVP
      - exactly 5 FLEX
      - six unique players
      - MVP salary from production mvp_salary
      - salary <= $60,000
      - MVP scores 1.5x actual FD
      - FLEX scores 1.0x actual FD
      - preserve every tied optimal lineup

    Optimization:
      For each MVP, enumerate only 5-player FLEX combinations using
      primitive Python tuples/lists instead of repeated pandas iloc
      and DataFrame construction inside the hot loop.

    This remains an exact exhaustive search. No heuristic pruning.
    """

    gp = game_pool.reset_index(drop=True)

    n = len(gp)

    if n < 6:
        raise RuntimeError("POOL_TOO_SMALL")

    player_ids = [
        clean(x)
        for x in gp["player_id"].tolist()
    ]

    salaries = [
        int(x)
        for x in gp["salary"].tolist()
    ]

    mvp_salaries = [
        int(x)
        for x in gp["mvp_salary"].tolist()
    ]

    actuals = [
        float(x)
        for x in gp["actual_fd"].tolist()
    ]

    if any(pd.isna(x) for x in actuals):
        raise RuntimeError(
            "EXACT_SEARCH_RECEIVED_UNRESOLVED_RESULT"
        )

    best_score = None
    best = []

    for mvp_idx in range(n):

        mvp_salary = mvp_salaries[mvp_idx]

        flex_budget = SALARY_CAP - mvp_salary

        if flex_budget < 0:
            continue

        mvp_score = 1.5 * actuals[mvp_idx]

        others = [
            i
            for i in range(n)
            if i != mvp_idx
        ]

        for flex_idxs in combinations(others, 5):

            a, b, c, d, e = flex_idxs

            flex_salary = (
                salaries[a]
                + salaries[b]
                + salaries[c]
                + salaries[d]
                + salaries[e]
            )

            if flex_salary > flex_budget:
                continue

            total_salary = (
                mvp_salary
                + flex_salary
            )

            score = (
                mvp_score
                + actuals[a]
                + actuals[b]
                + actuals[c]
                + actuals[d]
                + actuals[e]
            )

            if (
                best_score is not None
                and score < best_score - 1e-9
            ):
                continue

            identity_key = tuple(
                sorted(
                    (
                        player_ids[mvp_idx],
                        player_ids[a],
                        player_ids[b],
                        player_ids[c],
                        player_ids[d],
                        player_ids[e],
                    )
                )
            )

            record = {
                "score": round(score, 6),
                "salary": total_salary,
                "mvp_idx": mvp_idx,
                "flex_idxs": flex_idxs,
                "identity_key": identity_key,
            }

            if (
                best_score is None
                or score > best_score + 1e-9
            ):
                best_score = score
                best = [record]

            elif abs(score - best_score) <= 1e-9:
                best.append(record)

    if not best:
        raise RuntimeError("NO_LEGAL_LINEUP")

    best.sort(
        key=lambda x: (
            -x["salary"],
            player_ids[x["mvp_idx"]],
            x["identity_key"],
        )
    )

    return best_score, best


print("=" * 72)
print(
    "WFS STAGE 5.3E-19 — "
    "SG PERFECT-LINEUP RECONSTRUCTION V1"
)
print(
    "RESEARCH ONLY — "
    "ZERO PRODUCTION INFLUENCE"
)
print("=" * 72)


# =====================================================================
# PRODUCTION SG CONTEST POOL
# =====================================================================

pool = pd.read_parquet(POOL)

required_pool = {
    "source_csv",
    "player_id",
    "player",
    "team",
    "position",
    "salary",
    "mvp_salary",
}

missing_cols = (
    required_pool
    - set(pool.columns)
)

if missing_cols:
    raise RuntimeError(
        f"POOL_COLUMNS_MISSING: "
        f"{sorted(missing_cols)}"
    )

pool["player_id"] = (
    pool["player_id"].map(clean)
)

pool["team"] = (
    pool["team"].map(canon)
)

pool["position"] = (
    pool["position"].map(
        lambda x: clean(x).upper()
    )
)

pool["salary"] = pd.to_numeric(
    pool["salary"],
    errors="raise",
).astype(int)

pool["mvp_salary"] = pd.to_numeric(
    pool["mvp_salary"],
    errors="raise",
).astype(int)

if pool["player_id"].eq("").any():
    raise RuntimeError(
        "POOL_EMPTY_PLAYER_ID"
    )

if pool["salary"].le(0).any():
    raise RuntimeError(
        "POOL_NONPOSITIVE_SALARY"
    )

if pool["mvp_salary"].le(0).any():
    raise RuntimeError(
        "POOL_NONPOSITIVE_MVP_SALARY"
    )


# =====================================================================
# GAME + VERIFIED PLAYER ACTUAL AUTHORITY
# =====================================================================

uri = (
    f"file:{DB.resolve()}"
    "?mode=ro&immutable=1"
)

with sqlite3.connect(
    uri,
    uri=True,
) as con:

    games = pd.read_sql_query(
        """
        SELECT
            game_id,
            season,
            week,
            away_team,
            home_team,
            away_score,
            home_score,
            completed
        FROM games
        WHERE season=?
          AND week=?
        ORDER BY game_id
        """,
        con,
        params=(SEASON, WEEK),
    )

    actual = pd.read_sql_query(
        """
        SELECT
            game_id,
            player_id,
            player_name,
            position,
            team,
            fanduel_points,
            fanduel_points_verified
        FROM player_game_stats
        WHERE season=?
          AND week=?
        """,
        con,
        params=(SEASON, WEEK),
    )


pool = bind_source_csv_to_game(
    pool,
    games,
)

completed = games[
    pd.to_numeric(
        games["completed"],
        errors="coerce",
    ).fillna(0).eq(1)
].copy()

completed_ids = set(
    completed["game_id"]
    .astype(str)
)

pool = pool[
    pool["game_id"].isin(
        completed_ids
    )
].copy()

print(
    f"COMPLETED_GAMES="
    f"{len(completed_ids)}"
)

print(
    f"COMPLETED_POOL_ROWS="
    f"{len(pool)}"
)


# =====================================================================
# OFFENSIVE VERIFIED ACTUAL AUTHORITY
# =====================================================================

actual["game_id"] = (
    actual["game_id"].map(clean)
)

actual["player_id"] = (
    actual["player_id"].map(clean)
)

if not (
    actual["fanduel_points_verified"]
    .fillna(0)
    .astype(int)
    .eq(1)
    .all()
):
    raise RuntimeError(
        "UNVERIFIED_PLAYER_ACTUAL"
    )

actual = actual[
    [
        "game_id",
        "player_id",
        "fanduel_points",
    ]
].copy()

actual["actual_fd"] = pd.to_numeric(
    actual["fanduel_points"],
    errors="raise",
)

if actual.duplicated(
    ["game_id", "player_id"],
    keep=False,
).any():
    raise RuntimeError(
        "DUPLICATE_PLAYER_ACTUAL_KEY"
    )


# =====================================================================
# DST VERIFIED ACTUAL AUTHORITY
# =====================================================================

dst = pd.read_parquet(
    DST_PATH
)

dst = dst[
    pd.to_numeric(
        dst["season"],
        errors="coerce",
    ).eq(SEASON)
    & pd.to_numeric(
        dst["week"],
        errors="coerce",
    ).eq(WEEK)
].copy()

dst["game_id"] = (
    dst["game_id"].map(clean)
)

dst["team"] = (
    dst["team"].map(canon)
)

if not (
    dst["fanduel_scoring_verified"]
    .fillna(0)
    .astype(int)
    .eq(1)
    .all()
):
    raise RuntimeError(
        "UNVERIFIED_DST_ACTUAL"
    )

dst["actual_fd"] = pd.to_numeric(
    dst["fanduel_dst_points"],
    errors="raise",
)

if dst.duplicated(
    ["game_id", "team"],
    keep=False,
).any():
    raise RuntimeError(
        "DUPLICATE_DST_ACTUAL_KEY"
    )


# =====================================================================
# FROZEN EVIDENCE-BACKED ZERO COMPLETION SIDECAR
# =====================================================================

if not COMPLETION_PATH.exists():
    raise RuntimeError(
        "COMPLETION_SIDECAR_MISSING"
    )

if not COMPLETION_MANIFEST.exists():
    raise RuntimeError(
        "COMPLETION_MANIFEST_MISSING"
    )

completion_hash = sha256(
    COMPLETION_PATH
)

completion_manifest_hash = sha256(
    COMPLETION_MANIFEST
)

print()
print(
    "COMPLETION_SIDECAR_SHA256="
    + completion_hash
)

print(
    "COMPLETION_MANIFEST_SHA256="
    + completion_manifest_hash
)

if (
    completion_hash
    != EXPECTED_COMPLETION_SHA256
):
    raise RuntimeError(
        "COMPLETION_SIDECAR_HASH_MISMATCH"
    )

if (
    completion_manifest_hash
    != EXPECTED_COMPLETION_MANIFEST_SHA256
):
    raise RuntimeError(
        "COMPLETION_MANIFEST_HASH_MISMATCH"
    )

completion = pd.read_parquet(
    COMPLETION_PATH
)

required_completion = {
    "season",
    "week",
    "player_id",
    "fanduel_points",
    "result_status",
    "pbp_event_count",
    "production_influence",
    "solver_influence",
}

missing_completion = (
    required_completion
    - set(completion.columns)
)

if missing_completion:
    raise RuntimeError(
        "COMPLETION_COLUMNS_MISSING: "
        + str(
            sorted(
                missing_completion
            )
        )
    )

completion = completion[
    pd.to_numeric(
        completion["season"],
        errors="coerce",
    ).eq(SEASON)
    & pd.to_numeric(
        completion["week"],
        errors="coerce",
    ).eq(WEEK)
].copy()

completion["player_id"] = (
    completion["player_id"]
    .map(clean)
)

completion["fanduel_points"] = (
    pd.to_numeric(
        completion[
            "fanduel_points"
        ],
        errors="raise",
    )
)

completion["pbp_event_count"] = (
    pd.to_numeric(
        completion[
            "pbp_event_count"
        ],
        errors="raise",
    )
)

if len(completion) != EXPECTED_COMPLETION_ROWS:
    raise RuntimeError(
        "COMPLETION_ROW_COUNT_FAIL: "
        + str(len(completion))
    )

if completion[
    "player_id"
].duplicated().any():
    raise RuntimeError(
        "COMPLETION_DUPLICATE_PLAYER"
    )

if not completion[
    "fanduel_points"
].eq(0.0).all():
    raise RuntimeError(
        "COMPLETION_NONZERO_RESULT"
    )

if not completion[
    "result_status"
].eq(
    "EVIDENCE_BACKED_PARTICIPATED_ZERO"
).all():
    raise RuntimeError(
        "COMPLETION_STATUS_FAIL"
    )

if not completion[
    "pbp_event_count"
].eq(0).all():
    raise RuntimeError(
        "COMPLETION_PBP_EVENT_FAIL"
    )

if completion[
    "production_influence"
].astype(bool).any():
    raise RuntimeError(
        "COMPLETION_PRODUCTION_INFLUENCE_FAIL"
    )

if completion[
    "solver_influence"
].astype(bool).any():
    raise RuntimeError(
        "COMPLETION_SOLVER_INFLUENCE_FAIL"
    )

# Critical collision protection:
# the sidecar must NEVER override a verified aggregate actual.
aggregate_ids = set(
    actual["player_id"]
    .astype(str)
)

completion_ids = set(
    completion["player_id"]
    .astype(str)
)

collision = sorted(
    aggregate_ids
    & completion_ids
)

if collision:
    raise RuntimeError(
        "COMPLETION_COLLIDES_WITH_VERIFIED_ACTUAL: "
        + str(collision)
    )

completion = completion[
    [
        "player_id",
        "fanduel_points",
    ]
].copy()

completion = completion.rename(
    columns={
        "fanduel_points":
        "completion_actual_fd"
    }
)

print(
    "COMPLETION_ROWS_VALIDATED="
    + str(len(completion))
)


# =====================================================================
# AUTHORITATIVE NONPARTICIPANT ZERO SIDECAR
# =====================================================================

if not NONPARTICIPANT_PATH.exists():
    raise RuntimeError(
        "NONPARTICIPANT_SIDECAR_MISSING"
    )

nonparticipant = pd.read_parquet(
    NONPARTICIPANT_PATH
)

required_np = {
    "season",
    "week",
    "player_id",
    "fanduel_points",
    "result_status",
    "roster_status",
    "production_influence",
    "solver_influence",
}

missing_np = required_np - set(nonparticipant.columns)

if missing_np:
    raise RuntimeError(
        "NONPARTICIPANT_COLUMNS_MISSING: "
        + str(sorted(missing_np))
    )

nonparticipant = nonparticipant[
    pd.to_numeric(
        nonparticipant["season"],
        errors="coerce",
    ).eq(SEASON)
    & pd.to_numeric(
        nonparticipant["week"],
        errors="coerce",
    ).eq(WEEK)
].copy()

nonparticipant["player_id"] = (
    nonparticipant["player_id"].map(clean)
)

nonparticipant["fanduel_points"] = pd.to_numeric(
    nonparticipant["fanduel_points"],
    errors="raise",
)

if len(nonparticipant) != 75:
    raise RuntimeError(
        "NONPARTICIPANT_ROW_COUNT_FAIL"
    )

if nonparticipant["player_id"].duplicated().any():
    raise RuntimeError(
        "NONPARTICIPANT_DUPLICATE_PLAYER"
    )

if not nonparticipant["fanduel_points"].eq(0.0).all():
    raise RuntimeError(
        "NONPARTICIPANT_NONZERO_RESULT"
    )

if not nonparticipant["result_status"].eq(
    "AUTHORITATIVE_NONPARTICIPANT_ZERO"
).all():
    raise RuntimeError(
        "NONPARTICIPANT_STATUS_FAIL"
    )

if not nonparticipant["roster_status"].isin(
    {"INA", "RES", "DEV", "CUT"}
).all():
    raise RuntimeError(
        "NONPARTICIPANT_ROSTER_STATUS_FAIL"
    )

if nonparticipant["production_influence"].astype(bool).any():
    raise RuntimeError(
        "NONPARTICIPANT_PRODUCTION_INFLUENCE_FAIL"
    )

if nonparticipant["solver_influence"].astype(bool).any():
    raise RuntimeError(
        "NONPARTICIPANT_SOLVER_INFLUENCE_FAIL"
    )

np_ids = set(nonparticipant["player_id"])

if np_ids & aggregate_ids:
    raise RuntimeError(
        "NONPARTICIPANT_COLLIDES_WITH_VERIFIED_ACTUAL"
    )

if np_ids & completion_ids:
    raise RuntimeError(
        "NONPARTICIPANT_COLLIDES_WITH_PARTICIPATED_ZERO"
    )

nonparticipant = nonparticipant[
    ["player_id", "fanduel_points"]
].rename(
    columns={
        "fanduel_points":
            "nonparticipant_actual_fd"
    }
)

print(
    "NONPARTICIPANT_ROWS_VALIDATED="
    + str(len(nonparticipant))
)


# =====================================================================
# ATTACH RESULTS
#
# Strict authority order:
#   1. verified player_game_stats
#   2. verified current DST actuals
#   3. frozen evidence-backed completion sidecar
#   4. otherwise unresolved
#
# Generic missing -> zero remains prohibited.
# =====================================================================

non_dst = ~pool[
    "position"
].isin(DST_POSITIONS)

off_pool = pool[
    non_dst
].merge(
    actual,
    on=[
        "game_id",
        "player_id",
    ],
    how="left",
    validate="many_to_one",
)

off_pool = off_pool.merge(
    completion,
    on="player_id",
    how="left",
    validate="many_to_one",
)

off_pool["actual_source"] = pd.NA

aggregate_mask = (
    off_pool["actual_fd"]
    .notna()
)

off_pool.loc[
    aggregate_mask,
    "actual_source",
] = "VERIFIED_PLAYER_GAME_STATS"

completion_mask = (
    off_pool["actual_fd"].isna()
    & off_pool[
        "completion_actual_fd"
    ].notna()
)

off_pool.loc[
    completion_mask,
    "actual_fd",
] = off_pool.loc[
    completion_mask,
    "completion_actual_fd",
]

off_pool.loc[
    completion_mask,
    "actual_source",
] = "EVIDENCE_BACKED_ZERO_SIDECAR"

off_pool = off_pool.merge(
    nonparticipant,
    on="player_id",
    how="left",
    validate="many_to_one",
)

np_mask = (
    off_pool["actual_fd"].isna()
    & off_pool[
        "nonparticipant_actual_fd"
    ].notna()
)

off_pool.loc[
    np_mask,
    "actual_fd",
] = off_pool.loc[
    np_mask,
    "nonparticipant_actual_fd",
]

off_pool.loc[
    np_mask,
    "actual_source",
] = "AUTHORITATIVE_NONPARTICIPANT_ZERO"

off_pool = off_pool.drop(
    columns=[
        "completion_actual_fd",
        "nonparticipant_actual_fd",
    ]
)

dst_pool = pool[
    ~non_dst
].merge(
    dst[
        [
            "game_id",
            "team",
            "actual_fd",
        ]
    ],
    on=[
        "game_id",
        "team",
    ],
    how="left",
    validate="many_to_one",
)

dst_pool["actual_source"] = pd.NA

dst_pool.loc[
    dst_pool["actual_fd"].notna(),
    "actual_source",
] = "VERIFIED_CURRENT_DST_ACTUAL"

scored = pd.concat(
    [
        off_pool,
        dst_pool,
    ],
    ignore_index=True,
    sort=False,
)

print()
print("===== RESULT AUTHORITY =====")

print(
    scored[
        "actual_source"
    ]
    .fillna("UNRESOLVED")
    .value_counts()
    .to_string()
)


# =====================================================================
# PER-GAME RESULT COMPLETENESS
# =====================================================================

print()
print(
    "===== RESULT COMPLETENESS ====="
)

readiness = []

for gid in sorted(
    completed_ids
):

    gp = scored[
        scored["game_id"].eq(gid)
    ].copy()

    unresolved = gp[
        gp["actual_fd"].isna()
    ]

    ready = unresolved.empty

    readiness.append(
        {
            "game_id": gid,
            "pool_rows": len(gp),
            "resolved_rows": int(
                gp["actual_fd"]
                .notna()
                .sum()
            ),
            "unresolved_rows": len(
                unresolved
            ),
            "ready": ready,
        }
    )

    print(
        f"{gid}: "
        f"POOL={len(gp)} "
        f"RESOLVED="
        f"{gp['actual_fd'].notna().sum()} "
        f"UNRESOLVED="
        f"{len(unresolved)} "
        f"READY={ready}"
    )

    if len(unresolved):
        print(
            unresolved[
                [
                    "player_id",
                    "player",
                    "team",
                    "position",
                    "salary",
                ]
            ].to_string(
                index=False
            )
        )


ready_df = pd.DataFrame(
    readiness
)

ready_games = (
    ready_df.loc[
        ready_df["ready"],
        "game_id",
    ]
    .tolist()
)

print()
print(
    "FULLY_RESULT_COMPLETE_GAMES="
    + str(len(ready_games))
)

print(
    "TOTAL_COMPLETED_GAMES="
    + str(len(completed_ids))
)


# =====================================================================
# HINDSIGHT PERFECT-LINEUP RECONSTRUCTION
# =====================================================================

rows = []

for gid in ready_games:

    gp = scored[
        scored["game_id"].eq(gid)
    ].copy()

    best_score, tied = (
        exact_best_lineups(gp)
    )

    tie_count = len(tied)

    print()
    print(
        f"RECONSTRUCT {gid}: "
        f"PERFECT_SCORE="
        f"{best_score:.3f} "
        f"TIED_OPTIMAL_LINEUPS="
        f"{tie_count}"
    )

    for tie_rank, result in enumerate(
        tied,
        start=1,
    ):

        mvp = gp.iloc[
            result["mvp_idx"]
        ]

        flex = gp.iloc[
            list(
                result[
                    "flex_idxs"
                ]
            )
        ]

        lineup = pd.concat(
            [
                mvp.to_frame()
                .T.assign(
                    slot="MVP"
                ),
                flex.assign(
                    slot="FLEX"
                ),
            ],
            ignore_index=True,
        )

        mvp_part = lineup[
            lineup["slot"].eq(
                "MVP"
            )
        ]

        flex_part = lineup[
            lineup["slot"].eq(
                "FLEX"
            )
        ].sort_values(
            [
                "actual_fd",
                "salary",
                "player_id",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )

        lineup = pd.concat(
            [
                mvp_part,
                flex_part,
            ],
            ignore_index=True,
        )

        for slot_order, (_, r) in enumerate(
            lineup.iterrows(),
            start=1,
        ):

            actual_fd = float(
                r["actual_fd"]
            )

            multiplier = (
                1.5
                if r["slot"] == "MVP"
                else 1.0
            )

            rows.append(
                {
                    "season": SEASON,
                    "week": WEEK,
                    "game_id": gid,
                    "tie_rank": tie_rank,
                    "tie_count": tie_count,
                    "slot_order": slot_order,
                    "slot": r["slot"],
                    "player_id": clean(
                        r["player_id"]
                    ),
                    "player": clean(
                        r["player"]
                    ),
                    "team": canon(
                        r["team"]
                    ),
                    "position": clean(
                        r["position"]
                    ).upper(),
                    "base_salary": int(
                        r["salary"]
                    ),
                    "lineup_salary": (
                        int(
                            r[
                                "mvp_salary"
                            ]
                        )
                        if r["slot"]
                        == "MVP"
                        else int(
                            r["salary"]
                        )
                    ),
                    "actual_fd":
                    actual_fd,
                    "actual_source":
                    clean(
                        r[
                            "actual_source"
                        ]
                    ),
                    "multiplier":
                    multiplier,
                    "lineup_fd":
                    actual_fd
                    * multiplier,
                    "total_lineup_salary":
                    int(
                        result[
                            "salary"
                        ]
                    ),
                    "salary_left":
                    SALARY_CAP
                    - int(
                        result[
                            "salary"
                        ]
                    ),
                    "perfect_lineup_score":
                    float(
                        result[
                            "score"
                        ]
                    ),
                }
            )


result_df = pd.DataFrame(
    rows
)

if result_df.empty:
    print()
    print(
        "NO_RESULT_COMPLETE_GAME_"
        "AVAILABLE_FOR_RECONSTRUCTION"
    )
    print(
        "NO_ARTIFACT_WRITTEN=True"
    )
    print(
        "STAGE5_3E19_GATE="
        "FAIL_CLOSED"
    )
    raise SystemExit(2)


# =====================================================================
# STRUCTURAL VALIDATION
# =====================================================================

groups = result_df.groupby(
    [
        "game_id",
        "tie_rank",
    ],
    sort=False,
)

for key, g in groups:

    if len(g) != 6:
        raise RuntimeError(
            f"LINEUP_SIZE_FAIL "
            f"{key}: {len(g)}"
        )

    if (
        g["player_id"]
        .nunique()
        != 6
    ):
        raise RuntimeError(
            f"DUPLICATE_PLAYER_FAIL "
            f"{key}"
        )

    if int(
        g["slot"]
        .eq("MVP")
        .sum()
    ) != 1:
        raise RuntimeError(
            f"MVP_COUNT_FAIL "
            f"{key}"
        )

    salary = int(
        g["lineup_salary"].sum()
    )

    if salary > SALARY_CAP:
        raise RuntimeError(
            f"SALARY_CAP_FAIL "
            f"{key}: {salary}"
        )

    score = float(
        g["lineup_fd"].sum()
    )

    expected = float(
        g[
            "perfect_lineup_score"
        ].iloc[0]
    )

    if (
        abs(
            score
            - expected
        )
        > 1e-6
    ):
        raise RuntimeError(
            f"SCORE_RECONCILIATION_FAIL "
            f"{key}"
        )


# =====================================================================
# WRITE RESEARCH ARTIFACT
# =====================================================================

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

result_df.to_parquet(
    OUT,
    index=False,
)

artifact_hash = sha256(
    OUT
)

unresolved_games = (
    ready_df.loc[
        ~ready_df["ready"],
        "game_id",
    ]
    .astype(str)
    .tolist()
)

manifest = {
    "contract":
    "WFS_SINGLE_GAME_PERFECT_LINEUP_RECONSTRUCTION_V1",
    "stage":
    "5.3E-19",
    "season":
    SEASON,
    "week":
    WEEK,
    "salary_cap":
    SALARY_CAP,
    "contest_rules": {
        "roster_size": 6,
        "mvp_count": 1,
        "flex_count": 5,
        "mvp_points_multiplier": 1.5,
        "mvp_salary_authority":
        "production_pool_mvp_salary",
    },
    "result_authority_order": [
        "verified_player_game_stats",
        "verified_current_dst_actuals",
        "frozen_evidence_backed_zero_sidecar",
        "otherwise_unresolved_fail_closed",
    ],
    "completion_sidecar": {
        "path": str(
            COMPLETION_PATH.relative_to(
                ROOT
            )
        ),
        "rows":
        len(completion),
        "sha256":
        completion_hash,
        "manifest_sha256":
        completion_manifest_hash,
    },
    "production_influence":
    "NONE",
    "generic_missing_to_zero":
    "PROHIBITED",
    "evidence_backed_zero_completion":
    "PERMITTED_ONLY_FROM_FROZEN_SIDECAR",
    "completed_games":
    len(completed_ids),
    "fully_result_complete_games":
    len(ready_games),
    "unresolved_games":
    unresolved_games,
    "reconstructed_games":
    int(
        result_df[
            "game_id"
        ].nunique()
    ),
    "optimal_lineup_instances":
    int(
        result_df[
            [
                "game_id",
                "tie_rank",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    ),
    "output_rows":
    len(result_df),
    "output_sha256":
    artifact_hash,
}

MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(
    "===== RECONSTRUCTION SUMMARY ====="
)

print(
    "RECONSTRUCTED_GAMES="
    + str(
        result_df[
            "game_id"
        ].nunique()
    )
)

print(
    "OPTIMAL_LINEUP_INSTANCES="
    + str(
        result_df[
            [
                "game_id",
                "tie_rank",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )
)

print(
    "OUTPUT_ROWS="
    + str(len(result_df))
)

print(
    "OUTPUT="
    + str(OUT)
)

print(
    "OUTPUT_SHA256="
    + artifact_hash
)

print(
    "MANIFEST="
    + str(MANIFEST)
)

print(
    "MANIFEST_SHA256="
    + sha256(MANIFEST)
)

print()
print("PER_GAME:")

summary = (
    result_df
    .groupby(
        "game_id",
        as_index=False,
    )
    .agg(
        perfect_score=(
            "perfect_lineup_score",
            "first",
        ),
        tie_count=(
            "tie_count",
            "first",
        ),
        max_salary=(
            "total_lineup_salary",
            "max",
        ),
        min_salary_left=(
            "salary_left",
            "min",
        ),
    )
)

print(
    summary.to_string(
        index=False
    )
)

print()
print(
    "STAGE5_3E19_GATE=PASS"
)
print(
    "PRODUCTION_INFLUENCE=NONE"
)
print(
    "GENERIC_MISSING_TO_ZERO=False"
)
