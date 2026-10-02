#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import math
import sqlite3

import nflreadpy as nfl
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

DB = ROOT / "data/nfl.db"

BRIDGE = (
    ROOT
    / "data/fanduel/single_game/derived"
    / "single_game_projection_bridge_r1.parquet"
)

OUT_DIR = (
    ROOT
    / "data/fanduel/single_game/derived"
)

OUT_PARQUET = (
    OUT_DIR
    / "single_game_projection_pool.parquet"
)

OUT_REPORT = (
    OUT_DIR
    / "single_game_projection_pool_report.json"
)

SEASONS = [
    2023,
    2024,
    2025,
]

TEAM_ALIASES = {
    "LA": "LAR",
    "WAS": "WSH",
}

REQ = [
    "game_id",
    "season",
    "week",
    "posteam",
    "kicker_player_id",
    "kicker_player_name",
    "field_goal_attempt",
    "field_goal_result",
    "extra_point_attempt",
    "extra_point_result",
    "kick_distance",
]


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for b in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def clean(v):
    if v is None:
        return ""

    s = str(v).strip()

    return (
        ""
        if s.lower()
        in {
            "nan",
            "none",
            "null",
        }
        else s
    )


def canon_team(v):
    s = clean(v).upper()

    return TEAM_ALIASES.get(
        s,
        s,
    )


def flag(s):
    return (
        s.fillna(0)
        .astype(str)
        .isin(
            [
                "1",
                "1.0",
                "True",
                "true",
            ]
        )
    )


def finite(v):
    try:
        return (
            pd.notna(v)
            and math.isfinite(
                float(v)
            )
        )

    except Exception:
        return False


def load_gav2_block_ids(
    db_path=DB,
):
    """
    WFS Global Player Availability V2.

    Exact identity only:
        Showdown player_id
        ==
        injury_consensus_current.gsis_id

    No fuzzy matching.
    No display-name fallback.
    """

    db_path = Path(
        db_path
    )

    if not db_path.exists():
        raise RuntimeError(
            "GAV2 database missing"
        )

    conn = sqlite3.connect(
        f"file:{db_path.resolve()}?mode=ro",
        uri=True,
    )

    try:
        tables = {
            row[0]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                """
            )
        }

        if (
            "injury_consensus_current"
            not in tables
        ):
            raise RuntimeError(
                "GAV2 authority table missing"
            )

        cols = {
            row[1]
            for row in conn.execute(
                """
                PRAGMA table_info(
                    injury_consensus_current
                )
                """
            ).fetchall()
        }

        required = {
            "gsis_id",
            "injury_gate",
        }

        missing = sorted(
            required - cols
        )

        if missing:
            raise RuntimeError(
                "GAV2 schema missing: "
                + ",".join(
                    missing
                )
            )

        authority = (
            pd.read_sql_query(
                """
                SELECT
                    gsis_id,
                    injury_gate
                FROM injury_consensus_current
                """,
                conn,
            )
        )

    finally:
        conn.close()

    if authority.empty:
        raise RuntimeError(
            "GAV2 authority empty"
        )

    authority[
        "gsis_id"
    ] = (
        authority[
            "gsis_id"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    authority[
        "injury_gate"
    ] = (
        authority[
            "injury_gate"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if (
        authority[
            "gsis_id"
        ]
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "GAV2 blank gsis_id"
        )

    if (
        authority[
            "gsis_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "GAV2 duplicate gsis_id"
        )

    invalid = sorted(
        set(
            authority[
                "injury_gate"
            ]
        )
        - {
            "ALLOW",
            "BLOCK",
        }
    )

    if invalid:
        raise RuntimeError(
            "GAV2 invalid injury_gate: "
            + ",".join(
                invalid
            )
        )

    block_ids = set(
        authority.loc[
            authority[
                "injury_gate"
            ].eq("BLOCK"),
            "gsis_id",
        ]
    )

    return (
        block_ids,
        int(
            len(
                authority
            )
        ),
    )



# ==================================================================
# WFS Single-Game QB Authority Gate V1
# ==================================================================
#
# Publication eligibility only.
#
# Authoritative identity:
#     game_id + team
#         ->
#     games.away_qb_id / games.home_qb_id
#         ->
#     exact player_id
#
# Non-authoritative QBs retain their underlying model projections.
# This gate only prevents those projections from becoming
# Single-Game solver-consumable.
#
# No fuzzy matching.
# No player-specific exceptions.
# No contest-rule changes.
# ==================================================================

def apply_single_game_qb_authority_gate_v1(
    df,
    db_path=DB,
):
    """
    Single-Game QB publication authority.

    Bridge identity:
        game = "AWAY @ HOME"

    Authority:
        games.away_team/home_team
        games.away_qb_id/home_qb_id

    Final validation:
        exact game matchup + team + player_id

    Underlying projections remain untouched.
    Only projection_status is blocked for a
    non-authoritative QB.
    """

    import sqlite3

    out = df.copy()

    required = {
        "game",
        "team",
        "player_id",
        "position",
        "projection_status",
    }

    missing = sorted(
        required - set(out.columns)
    )

    if missing:
        raise RuntimeError(
            "Showdown QB authority fields missing: "
            + ",".join(missing)
        )

    with sqlite3.connect(db_path) as conn:
        games = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                away_team,
                home_team,
                away_qb_id,
                home_qb_id
            FROM games
            """,
            conn,
        )

    for col in [
        "game_id",
        "away_team",
        "home_team",
        "away_qb_id",
        "home_qb_id",
    ]:
        games[col] = (
            games[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    games["away_team"] = games["away_team"].str.upper()
    games["home_team"] = games["home_team"].str.upper()

    games["_showdown_game"] = (
        games["away_team"]
        + " @ "
        + games["home_team"]
    )

    # SINGLE_GAME_TEAM_ALIAS_NORMALIZATION_V1
    #
    # FanDuel/bridge uses WSH while the authoritative games
    # schedule uses WAS. Normalize only this known representation
    # difference before exact matchup/team/QB-ID validation.
    #
    # This does not change player team identity, projections,
    # contest rules, or the games authority.
    showdown_game = (
        out["game"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    showdown_game = showdown_game.str.replace(
        r"^WSH @ ",
        "WAS @ ",
        regex=True,
    )

    showdown_game = showdown_game.str.replace(
        r" @ WSH$",
        " @ WAS",
        regex=True,
    )

    out["_showdown_game"] = showdown_game

    # The current Single-Game bridge contains only the public
    # matchup string, not game_id. A matchup must therefore map
    # to exactly one current games row.
    #
    # Prefer games represented by this bridge's matchup universe.
    # Use the normalized matchup identity here.
    # Important: this must occur after WSH -> WAS normalization.
    bridge_games = set(
        out["_showdown_game"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    relevant_games = games[
        games["_showdown_game"].isin(
            bridge_games
        )
    ].copy()

    # Multiple seasons can contain the same AWAY @ HOME matchup.
    # Resolve to the latest season/week represented in the games
    # authority. The Single-Game derived pool is a current-slate
    # product, so stale historical instances must not compete with
    # the current scheduled game.
    relevant_games["_season_num"] = pd.to_numeric(
        relevant_games["season"],
        errors="coerce",
    )

    relevant_games["_week_num"] = pd.to_numeric(
        relevant_games["week"],
        errors="coerce",
    )

    if relevant_games.empty:
        raise RuntimeError(
            "No games authority rows matched "
            "Single-Game bridge matchups"
        )

    relevant_games = (
        relevant_games
        .sort_values(
            [
                "_showdown_game",
                "_season_num",
                "_week_num",
                "game_id",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )
        .drop_duplicates(
            "_showdown_game",
            keep="first",
        )
        .copy()
    )

    # Every bridge matchup must resolve.
    resolved_matchups = set(
        relevant_games["_showdown_game"]
    )

    unresolved_matchups = sorted(
        bridge_games - resolved_matchups
    )

    if unresolved_matchups:
        raise RuntimeError(
            "Unresolved Single-Game matchups: "
            + ",".join(unresolved_matchups)
        )

    away = relevant_games[
        [
            "game_id",
            "_showdown_game",
            "away_team",
            "away_qb_id",
        ]
    ].rename(
        columns={
            "away_team": "_qb_team",
            "away_qb_id": "authority_qb_id",
        }
    )

    home = relevant_games[
        [
            "game_id",
            "_showdown_game",
            "home_team",
            "home_qb_id",
        ]
    ].rename(
        columns={
            "home_team": "_qb_team",
            "home_qb_id": "authority_qb_id",
        }
    )

    authority = pd.concat(
        [away, home],
        ignore_index=True,
    )

    duplicate_authority = authority.duplicated(
        [
            "_showdown_game",
            "_qb_team",
        ],
        keep=False,
    )

    if duplicate_authority.any():
        raise RuntimeError(
            "Showdown QB authority duplicate "
            "game/team rows"
        )

    out["_qb_team"] = (
        out["team"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(
            {
                "WSH": "WAS",
            }
        )
    )

    out["_qb_player"] = (
        out["player_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    out["_qb_position"] = (
        out["position"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out = out.merge(
        authority,
        on=[
            "_showdown_game",
            "_qb_team",
        ],
        how="left",
        validate="many_to_one",
    )

    qb = out["_qb_position"].eq("QB")

    authority_present = (
        out["authority_qb_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    exact_match = (
        out["_qb_player"]
        .eq(
            out["authority_qb_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    )

    out["qb_authority_gate"] = "NOT_APPLICABLE"

    out.loc[
        qb & authority_present & exact_match,
        "qb_authority_gate",
    ] = "ALLOW"

    out.loc[
        qb & authority_present & ~exact_match,
        "qb_authority_gate",
    ] = "BLOCK_NON_PRIMARY_QB"

    out.loc[
        qb & ~authority_present,
        "qb_authority_gate",
    ] = "BLOCK_MISSING_QB_AUTHORITY"

    blocked = (
        qb
        & ~out[
            "qb_authority_gate"
        ].eq("ALLOW")
    )

    # Publication-layer suppression only.
    # projection and mvp_projection remain intact.
    out.loc[
        blocked,
        "projection_status",
    ] = "BLOCKED_QB_AUTHORITY"

    audit = {
        "contract":
            "EXACT_MATCHUP_TEAM_QB_ID",
        "qb_rows":
            int(qb.sum()),
        "allowed":
            int(
                (
                    qb
                    & out[
                        "qb_authority_gate"
                    ].eq("ALLOW")
                ).sum()
            ),
        "blocked":
            int(blocked.sum()),
        "missing_authority":
            int(
                (
                    qb
                    & ~authority_present
                ).sum()
            ),
        "resolved_games":
            int(
                relevant_games[
                    "_showdown_game"
                ].nunique()
            ),
    }

    out = out.drop(
        columns=[
            "_showdown_game",
            "_qb_team",
            "_qb_player",
            "_qb_position",
            "authority_qb_id",
            "game_id",
        ],
        errors="ignore",
    )

    return out, audit


def apply_gav2_availability_gate(
    df,
    db_path=DB,
):
    """
    Final availability overlay.

    Modeling/source/authority/status metadata
    remain untouched.

    GAV2 BLOCK:
        projection = 0
        mvp_projection = 0
    """

    required = {
        "player_id",
        "projection",
        "mvp_projection",
    }

    missing = sorted(
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Showdown GAV2 fields missing: "
            + ",".join(
                missing
            )
        )

    (
        block_ids,
        authority_rows,
    ) = load_gav2_block_ids(
        db_path=db_path,
    )

    ids = (
        df[
            "player_id"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    target = ids.isin(
        block_ids
    )

    before_projection = (
        pd.to_numeric(
            df.loc[
                target,
                "projection",
            ],
            errors="coerce",
        )
        .fillna(0.0)
    )

    before_mvp = (
        pd.to_numeric(
            df.loc[
                target,
                "mvp_projection",
            ],
            errors="coerce",
        )
        .fillna(0.0)
    )

    positive_before = int(
        (
            before_projection
            .abs()
            .gt(1e-12)
            |
            before_mvp
            .abs()
            .gt(1e-12)
        )
        .sum()
    )

    df.loc[
        target,
        "projection",
    ] = 0.0

    df.loc[
        target,
        "mvp_projection",
    ] = 0.0

    after_projection = (
        pd.to_numeric(
            df.loc[
                target,
                "projection",
            ],
            errors="coerce",
        )
        .fillna(0.0)
    )

    after_mvp = (
        pd.to_numeric(
            df.loc[
                target,
                "mvp_projection",
            ],
            errors="coerce",
        )
        .fillna(0.0)
    )

    if (
        after_projection
        .abs()
        .gt(1e-12)
        .any()
    ):
        raise RuntimeError(
            "GAV2 BLOCK projection survived"
        )

    if (
        after_mvp
        .abs()
        .gt(1e-12)
        .any()
    ):
        raise RuntimeError(
            "GAV2 BLOCK MVP survived"
        )

    audit = {
        "contract":
            "WFS_GLOBAL_PLAYER_AVAILABILITY_V2",

        "identity":
            "EXACT_PLAYER_ID_TO_GSIS_ID",

        "fuzzy_matching_used":
            False,

        "display_name_matching_used":
            False,

        "authority_rows":
            authority_rows,

        "block_ids":
            int(
                len(
                    block_ids
                )
            ),

        "showdown_block_rows":
            int(
                target.sum()
            ),

        "positive_block_rows_before":
            positive_before,

        "block_rows_zero_after":
            int(
                target.sum()
            ),
    }

    return (
        df,
        audit,
    )


print(
    "=" * 118
)

print(
    "WFS NFL — NFL-POSTGAME-1F-I-X"
)

print(
    "SHOWDOWN 96/96 PROJECTION POOL — "
    "ISOLATED KICKER MODEL PROMOTION"
)

print(
    "=" * 118
)


if not BRIDGE.exists():
    print(
        f"FAIL_CLOSED_MISSING={BRIDGE}"
    )
    raise SystemExit(2)


df = pd.read_parquet(
    BRIDGE
).copy()


print(
    f"INPUT_BRIDGE_SHA256={sha256(BRIDGE)}"
)

print(
    f"INPUT_ROWS={len(df)}"
)


k_mask = (
    df[
        "position"
    ]
    .astype(str)
    .str.upper()
    .eq("K")
)


expected_kicker_count = int(
    k_mask.sum()
)


print(
    f"EXPECTED_KICKER_COUNT={expected_kicker_count}"
)


if expected_kicker_count <= 0:
    print(
        "FAIL_CLOSED_KICKER_COUNT="
        f"{expected_kicker_count}"
    )

    raise SystemExit(2)


bad_k_state = (
    df.loc[
        k_mask,
        "projection_status",
    ]
    .astype(str)
    .ne(
        "BLOCKED_NO_KICKER_PROJECTION_AUTHORITY"
    )
)


if bad_k_state.any():
    print(
        "FAIL_CLOSED_UNEXPECTED_KICKER_INPUT_STATE=TRUE"
    )

    raise SystemExit(2)


frames = []


for season in SEASONS:
    pl = nfl.load_pbp(
        season
    )

    missing = [
        c
        for c in REQ
        if c not in pl.columns
    ]

    if missing:
        print(
            "FAIL_CLOSED_PBP_SCHEMA"
            f"|season={season}"
            f"|missing={json.dumps(missing)}"
        )

        raise SystemExit(2)

    p = (
        pl.select(
            REQ
        )
        .to_pandas()
    )

    p = p[
        flag(
            p[
                "field_goal_attempt"
            ]
        )
        |
        flag(
            p[
                "extra_point_attempt"
            ]
        )
    ].copy()

    frames.append(
        p
    )


pbp = pd.concat(
    frames,
    ignore_index=True,
)


pbp[
    "kicker_player_id"
] = (
    pbp[
        "kicker_player_id"
    ]
    .map(clean)
)


pbp[
    "posteam"
] = (
    pbp[
        "posteam"
    ]
    .map(canon_team)
)


pbp[
    "field_goal_result"
] = (
    pbp[
        "field_goal_result"
    ]
    .map(
        lambda x:
        clean(x).lower()
    )
)


pbp[
    "extra_point_result"
] = (
    pbp[
        "extra_point_result"
    ]
    .map(
        lambda x:
        clean(x).lower()
    )
)


pbp[
    "kick_distance"
] = pd.to_numeric(
    pbp[
        "kick_distance"
    ],
    errors="coerce",
)


pbp["fg"] = (
    flag(
        pbp[
            "field_goal_attempt"
        ]
    )
    .astype(int)
)


pbp["xp"] = (
    flag(
        pbp[
            "extra_point_attempt"
        ]
    )
    .astype(int)
)


pbp[
    "fg_made"
] = (
    pbp[
        "fg"
    ].eq(1)
    &
    pbp[
        "field_goal_result"
    ].eq("made")
).astype(int)


pbp[
    "xp_made"
] = (
    pbp[
        "xp"
    ].eq(1)
    &
    pbp[
        "extra_point_result"
    ].isin(
        [
            "good",
            "made",
        ]
    )
).astype(int)


pbp[
    "fd_points"
] = 0.0


pbp.loc[
    pbp[
        "xp_made"
    ].eq(1),
    "fd_points",
] += 1.0


pbp.loc[
    (
        pbp[
            "fg_made"
        ].eq(1)
        &
        pbp[
            "kick_distance"
        ].lt(50)
    ),
    "fd_points",
] += 3.0


pbp.loc[
    (
        pbp[
            "fg_made"
        ].eq(1)
        &
        pbp[
            "kick_distance"
        ].ge(50)
    ),
    "fd_points",
] += 5.0


game = (
    pbp[
        pbp[
            "kicker_player_id"
        ].ne("")
    ]
    .groupby(
        [
            "season",
            "week",
            "game_id",
            "posteam",
            "kicker_player_id",
        ],
        as_index=False,
    )
    .agg(
        fd_points=(
            "fd_points",
            "sum",
        ),
        fg_att=(
            "fg",
            "sum",
        ),
        fg_made=(
            "fg_made",
            "sum",
        ),
        xp_att=(
            "xp",
            "sum",
        ),
        xp_made=(
            "xp_made",
            "sum",
        ),
    )
    .sort_values(
        [
            "season",
            "week",
            "game_id",
        ],
        kind="mergesort",
    )
)


league_mean = float(
    game[
        "fd_points"
    ].mean()
)


promoted = 0


print(
    "\n=== KICKER PROMOTION ==="
)


for idx, r in (
    df[
        k_mask
    ]
    .sort_values(
        [
            "team",
            "player",
        ],
        kind="mergesort",
    )
    .iterrows()
):
    pid = clean(
        r[
            "player_id"
        ]
    )

    name = clean(
        r[
            "player"
        ]
    )

    team = canon_team(
        r[
            "team"
        ]
    )

    hist = game[
        game[
            "kicker_player_id"
        ].eq(
            pid
        )
    ].copy()

    if hist.empty:
        print(
            "FAIL_CLOSED_KICKER"
            f"|player={name!r}"
            "|reason=NO_PERSONAL_HISTORY"
        )

        raise SystemExit(2)

    last16 = hist.tail(
        16
    )

    last8 = hist.tail(
        8
    )

    if (
        len(last16) == 0
        or len(last8) == 0
    ):
        print(
            "FAIL_CLOSED_KICKER"
            f"|player={name!r}"
            "|reason=INSUFFICIENT_HISTORY"
        )

        raise SystemExit(2)

    p8 = float(
        last8[
            "fd_points"
        ].mean()
    )

    p16 = float(
        last16[
            "fd_points"
        ].mean()
    )

    personal_recent = (
        0.60 * p8
        +
        0.40 * p16
    )

    team_hist = (
        game[
            game[
                "posteam"
            ].eq(
                team
            )
        ]
        .tail(
            16
        )
    )

    if len(
        team_hist
    ):
        prior = float(
            team_hist[
                "fd_points"
            ].mean()
        )

        prior_source = (
            "TEAM_LAST16"
        )

    else:
        prior = (
            league_mean
        )

        prior_source = (
            "LEAGUE"
        )

    n16 = len(
        last16
    )

    projection = (
        (
            n16
            * personal_recent
        )
        +
        (
            4.0
            * prior
        )
    ) / (
        n16
        + 4.0
    )

    df.at[
        idx,
        "projection",
    ] = projection

    df.at[
        idx,
        "projection_source",
    ] = (
        "WFS_SHOWDOWN_KICKER_V1"
    )

    df.at[
        idx,
        "projection_authority",
    ] = (
        "WFS_SHOWDOWN_MODEL"
    )

    df.at[
        idx,
        "projection_status",
    ] = "READY"

    df.at[
        idx,
        "mvp_salary",
    ] = (
        float(
            r[
                "salary"
            ]
        )
        * 1.5
    )

    df.at[
        idx,
        "mvp_projection",
    ] = (
        projection
        * 1.5
    )

    promoted += 1

    print(
        "K_PROMOTED"
        f"|player={name!r}"
        f"|team={team}"
        f"|player_id={pid}"
        f"|last8={p8:.6f}"
        f"|last16={p16:.6f}"
        f"|prior_source={prior_source}"
        f"|prior={prior:.6f}"
        f"|projection={projection:.6f}"
        f"|mvp_projection={projection * 1.5:.6f}"
    )


# ==================================================================
# SINGLE-GAME QB AUTHORITY GATE V1
# ==================================================================

try:
    (
        df,
        qb_authority_audit,
    ) = apply_single_game_qb_authority_gate_v1(
        df,
        db_path=DB,
    )

except Exception as exc:
    print(
        "FAIL_CLOSED_QB_AUTHORITY_GATE="
        f"{type(exc).__name__}:{exc}"
    )

    raise SystemExit(2)


print(
    "\n=== SINGLE-GAME QB AUTHORITY GATE ==="
)

print(
    "QB_AUTHORITY_CONTRACT="
    f"{qb_authority_audit['contract']}"
)

print(
    "QB_AUTHORITY_QB_ROWS="
    f"{qb_authority_audit['qb_rows']}"
)

print(
    "QB_AUTHORITY_ALLOWED="
    f"{qb_authority_audit['allowed']}"
)

print(
    "QB_AUTHORITY_BLOCKED="
    f"{qb_authority_audit['blocked']}"
)

print(
    "QB_AUTHORITY_MISSING="
    f"{qb_authority_audit['missing_authority']}"
)


# ==================================================================
# SINGLE_GAME_QB_PUBLICATION_FILTER_V1
# ==================================================================
#
# The QB authority gate intentionally preserves underlying backup-QB
# projections for audit/model use while marking non-authoritative QBs
# BLOCKED_QB_AUTHORITY.
#
# The final Showdown projection pool is solver-consumable only, so
# those intentionally blocked QB rows must not be published into the
# final pool. Contest rules, projections, GAV2, and solver logic are
# unchanged.
#
_qb_publication_block = (
    df["projection_status"]
    .fillna("")
    .astype(str)
    .eq("BLOCKED_QB_AUTHORITY")
)

qb_authority_publication_blocked = int(
    _qb_publication_block.sum()
)

if qb_authority_publication_blocked:
    df = (
        df.loc[
            ~_qb_publication_block
        ]
        .copy()
        .reset_index(drop=True)
    )

print(
    "QB_AUTHORITY_PUBLICATION_BLOCKED="
    f"{qb_authority_publication_blocked}"
)

print(
    "QB_AUTHORITY_FINAL_POOL_ROWS="
    f"{len(df)}"
)


# ==================================================================
# GLOBAL AVAILABILITY V2
# ==================================================================

try:
    (
        df,
        gav2_audit,
    ) = apply_gav2_availability_gate(
        df,
        db_path=DB,
    )

except Exception as exc:
    print(
        "FAIL_CLOSED_GAV2_AVAILABILITY_GATE="
        f"{type(exc).__name__}:{exc}"
    )

    raise SystemExit(2)


print(
    "\n=== GAV2 AVAILABILITY OVERLAY ==="
)

print(
    "GAV2_CONTRACT="
    f"{gav2_audit['contract']}"
)

print(
    "GAV2_IDENTITY="
    f"{gav2_audit['identity']}"
)

print(
    "GAV2_AUTHORITY_ROWS="
    f"{gav2_audit['authority_rows']}"
)

print(
    "GAV2_BLOCK_IDS="
    f"{gav2_audit['block_ids']}"
)

print(
    "GAV2_SHOWDOWN_BLOCK_ROWS="
    f"{gav2_audit['showdown_block_rows']}"
)

print(
    "GAV2_POSITIVE_BLOCK_ROWS_BEFORE="
    f"{gav2_audit['positive_block_rows_before']}"
)

print(
    "GAV2_BLOCK_ROWS_ZERO_AFTER="
    f"{gav2_audit['block_rows_zero_after']}"
)


ready = (
    df[
        "projection_status"
    ]
    .isin(
        [
            "READY",
            "READY_EXTERNAL",
        ]
    )
)


blocked = (
    ~ready
)


bad_projection = (
    ready
    &
    ~df[
        "projection"
    ].map(finite)
)


bad_salary = (
    ready
    &
    ~pd.to_numeric(
        df[
            "salary"
        ],
        errors="coerce",
    ).gt(0)
)


dup = (
    df.duplicated(
        [
            "public_slate_name",
            "game",
            "player_id",
        ],
        keep=False,
    )
)


if (
    promoted
    != expected_kicker_count
    or blocked.any()
    or bad_projection.any()
    or bad_salary.any()
    or dup.any()
):
    print(
        f"PROMOTED_KICKERS={promoted}"
    )

    print(
        f"BLOCKED_ROWS={int(blocked.sum())}"
    )

    print(
        "BAD_READY_PROJECTIONS="
        f"{int(bad_projection.sum())}"
    )

    print(
        "BAD_READY_SALARIES="
        f"{int(bad_salary.sum())}"
    )

    print(
        "DUPLICATE_ROWS="
        f"{int(dup.sum())}"
    )

    print(
        "NFL_POSTGAME_1F_I_X_STATUS=FAIL_CLOSED"
    )

    raise SystemExit(2)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


df.to_parquet(
    OUT_PARQUET,
    index=False,
)


report = {
    "input_bridge_sha256":
        sha256(
            BRIDGE
        ),

    "rows":
        int(
            len(df)
        ),

    "ready_rows":
        int(
            ready.sum()
        ),

    "blocked_rows":
        int(
            blocked.sum()
        ),

    "promoted_kickers":
        int(
            promoted
        ),

    "sources": {
        str(k):
            int(v)
        for k, v
        in (
            df[
                "projection_source"
            ]
            .value_counts(
                dropna=False
            )
            .items()
        )
    },

    "kicker_model": {
        "name":
            "WFS_SHOWDOWN_KICKER_V1",

        "historical_seasons":
            SEASONS,

        "personal_recent":
            "0.60*last8_mean + 0.40*last16_mean",

        "prior":
            "team_last16_mean_else_league_mean",

        "prior_strength_games":
            4,

        "formula":
            "(personal_games*personal_recent + 4*prior)/(personal_games+4)",
    },

    "gav2_availability":
        gav2_audit,

    "fuzzy_matching_used":
        False,

    "classic_production_changed":
        False,

    "public_solver_changed":
        False,
}


OUT_REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)


print(
    "\n=== FINAL 96/96 AUDIT ==="
)

print(
    f"TOTAL_ROWS={len(df)}"
)

print(
    f"READY_ROWS={int(ready.sum())}"
)

print(
    f"BLOCKED_ROWS={int(blocked.sum())}"
)

print(
    "KICKER_MODEL_ROWS="
    f"{int((df['projection_source'] == 'WFS_SHOWDOWN_KICKER_V1').sum())}"
)

print(
    "WFS_RIDGE_ROWS="
    f"{int((df['projection_source'] == 'WFS_RIDGE').sum())}"
)

print(
    "WFS_DST_ROWS="
    f"{int((df['projection_source'] == 'WFS_DST_PRODUCTION').sum())}"
)

print(
    "EXTERNAL_FANDUEL_ROWS="
    f"{int((df['projection_source'] == 'FANDUEL_RAW_FANTASY').sum())}"
)

print(
    f"DUPLICATE_ROWS={int(dup.sum())}"
)

print(
    "BAD_READY_PROJECTIONS="
    f"{int(bad_projection.sum())}"
)

print(
    "BAD_READY_SALARIES="
    f"{int(bad_salary.sum())}"
)

print(
    "GAV2_BLOCK_IDS="
    f"{gav2_audit['block_ids']}"
)

print(
    "GAV2_SHOWDOWN_BLOCK_ROWS="
    f"{gav2_audit['showdown_block_rows']}"
)

print(
    "GAV2_POSITIVE_BLOCK_ROWS_BEFORE="
    f"{gav2_audit['positive_block_rows_before']}"
)

print(
    "GAV2_BLOCK_ROWS_ZERO_AFTER="
    f"{gav2_audit['block_rows_zero_after']}"
)

print(
    f"\nOUT_PARQUET={OUT_PARQUET}"
)

print(
    "OUT_PARQUET_SHA256="
    f"{sha256(OUT_PARQUET)}"
)

print(
    f"OUT_REPORT={OUT_REPORT}"
)

print(
    "OUT_REPORT_SHA256="
    f"{sha256(OUT_REPORT)}"
)

print(
    "FUZZY_MATCHING_USED=FALSE"
)

print(
    "CLASSIC_PRODUCTION_CHANGED=FALSE"
)

print(
    "PUBLIC_SOLVER_CHANGED=FALSE"
)

print(
    "GAV2_SHOWDOWN_REGENERATION_GATE=ACTIVE"
)

print(
    "NFL_POSTGAME_1F_I_X_STATUS=PASS"
)
