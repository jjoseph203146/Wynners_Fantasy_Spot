from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

ROSTER_PATH = ROOT / "data/parquet/nfl_weekly_rosters.parquet"
PROD_PATH = ROOT / "data/parquet/nfl_production_projection.parquet"
CANONICAL_PATH = ROOT / "data/parquet/current_unified_stat_forecasts.parquet"
ENV_PATH = ROOT / "data/parquet/nfl_team_pregame_environment.parquet"
DVP_PATH = ROOT / "data/parquet/nfl_team_position_dvp.parquet"
DB_PATH = ROOT / "data/nfl.db"

OUT_PATH = (
    ROOT
    / "data/parquet/current_cold_start_stat_forecasts_shadow.parquet"
)

AUDIT_PATH = (
    ROOT
    / "data/parquet/current_cold_start_stat_forecasts_shadow_audit.json"
)

SUPPORTED_POSITIONS = {
    "QB",
    "RB",
    "FB",
    "WR",
    "TE",
}

# ------------------------------------------------------------
# VALIDATED R8P TARGET RATE
# ------------------------------------------------------------

TARGETS_PER_PASS_ATTEMPT = 0.954436

# ------------------------------------------------------------
# R8K HIERARCHICAL ROOKIE STARTER PRIORS
#
# first-start prior shrunk toward all-rookie-starter-game prior
# using evidence ratio:
#
#     w = first_start_n / all_rookie_start_game_n
#
# These are SHADOW calibration constants only.
# ------------------------------------------------------------

PRIORS = {
    "QB": {
        "first_n": 8,
        "all_n": 57,

        "first": {
            "completions": 17.250,
            "attempts": 28.625,
            "passing_yards": 182.000,
            "passing_tds": 0.875,
            "interceptions": 0.625,
            "carries": 2.500,
            "rushing_yards": 11.875,
            "rushing_tds": 0.250,
        },

        "all": {
            "completions": 18.895,
            "attempts": 30.333,
            "passing_yards": 192.947,
            "passing_tds": 1.018,
            "interceptions": 0.579,
            "carries": 3.719,
            "rushing_yards": 19.474,
            "rushing_tds": 0.281,
        },
    },

    "RB": {
        "first_n": 11,
        "all_n": 82,

        "first": {
            "carries": 12.727,
            "targets": 2.545,
            "receptions": 2.182,
            "rushing_yards": 48.091,
            "receiving_yards": 13.364,
            "rushing_tds": 0.455,
            "receiving_tds": 0.091,
        },

        "all": {
            "carries": 14.110,
            "targets": 3.061,
            "receptions": 2.366,
            "rushing_yards": 52.683,
            "receiving_yards": 15.829,
            "rushing_tds": 0.390,
            "receiving_tds": 0.122,
        },
    },

    # FB inherits RB prior until a sufficient FB rookie sample exists.
    "FB": {
        "inherit": "RB",
    },

    "WR": {
        "first_n": 4,
        "all_n": 35,

        "first": {
            "targets": 6.750,
            "receptions": 3.250,
            "receiving_yards": 40.000,
            "receiving_tds": 0.000,
            "carries": 0.750,
            "rushing_yards": 2.250,
            "rushing_tds": 0.000,
        },

        "all": {
            "targets": 6.771,
            "receptions": 3.714,
            "receiving_yards": 51.200,
            "receiving_tds": 0.286,
            "carries": 0.171,
            "rushing_yards": 0.857,
            "rushing_tds": 0.000,
        },
    },

    "TE": {
        "first_n": 6,
        "all_n": 63,

        "first": {
            "targets": 5.667,
            "receptions": 4.500,
            "receiving_yards": 55.833,
            "receiving_tds": 0.167,
        },

        "all": {
            "targets": 5.825,
            "receptions": 3.921,
            "receiving_yards": 43.333,
            "receiving_tds": 0.254,
        },
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def to_num(value, default=np.nan):
    try:
        x = float(value)
    except Exception:
        return default

    if not np.isfinite(x):
        return default

    return x


def safe_div(a, b, default=0.0):
    a = to_num(a, np.nan)
    b = to_num(b, np.nan)

    if pd.isna(a) or pd.isna(b) or abs(b) < 1e-12:
        return default

    return a / b


def clamp(value, low, high):
    return max(low, min(high, value))


def hierarchical_prior(position: str) -> dict[str, float]:
    pos = position.upper()

    cfg = PRIORS[pos]

    if "inherit" in cfg:
        return hierarchical_prior(cfg["inherit"])

    first_n = float(cfg["first_n"])
    all_n = float(cfg["all_n"])

    weight = (
        first_n / all_n
        if all_n > 0
        else 0.0
    )

    keys = (
        set(cfg["first"].keys())
        | set(cfg["all"].keys())
    )

    out = {}

    for key in keys:
        first_value = float(
            cfg["first"].get(
                key,
                cfg["all"].get(key, 0.0),
            )
        )

        all_value = float(
            cfg["all"].get(
                key,
                first_value,
            )
        )

        out[key] = (
            weight * first_value
            +
            (1.0 - weight) * all_value
        )

    out["_first_n"] = first_n
    out["_all_n"] = all_n
    out["_first_weight"] = weight

    return out


def normalize_status(value) -> str:
    return (
        str(value or "")
        .strip()
        .upper()
        .replace("-", "_")
        .replace(" ", "_")
    )


def hard_unavailable(status, status_abbr) -> bool:
    values = {
        normalize_status(status),
        normalize_status(status_abbr),
    }

    hard_tokens = (
        "OUT",
        "INACTIVE",
        "INJURED_RESERVE",
        "RESERVE_INJURED",
        "PUP",
        "PHYSICALLY_UNABLE",
    )

    for value in values:
        for token in hard_tokens:
            if token in value:
                return True

    return False


def latest_environment(
    env: pd.DataFrame,
    game_id: str,
    team: str,
    season: int,
):
    exact = env[
        env["game_id"].astype(str).eq(game_id)
        &
        env["team"].astype(str).str.upper().eq(team)
    ].copy()

    if not exact.empty:
        row = exact.iloc[0]

        p5 = to_num(
            row.get("pass_attempts_avg_5"),
            np.nan,
        )

        r5 = to_num(
            row.get("rush_attempts_avg_5"),
            np.nan,
        )

        if pd.notna(p5) and pd.notna(r5):
            return row, "CURRENT_SEASON"

    fallback = env[
        env["team"].astype(str).str.upper().eq(team)
        &
        (
            pd.to_numeric(
                env["season"],
                errors="coerce",
            )
            < season
        )
    ].copy()

    fallback["season"] = pd.to_numeric(
        fallback["season"],
        errors="coerce",
    )

    fallback["week"] = pd.to_numeric(
        fallback["week"],
        errors="coerce",
    )

    fallback = fallback.sort_values(
        ["season", "week"],
        ascending=[False, False],
    )

    if fallback.empty:
        return None, "NONE"

    return (
        fallback.iloc[0],
        "PRIOR_SEASON_AVG5_FALLBACK",
    )


def latest_dvp(
    dvp: pd.DataFrame,
    game_id: str,
    opponent: str,
    position: str,
    season: int,
):
    exact = dvp[
        dvp["game_id"].astype(str).eq(game_id)
        &
        dvp["defense_team"]
        .astype(str)
        .str.upper()
        .eq(opponent)
        &
        dvp["position"]
        .astype(str)
        .str.upper()
        .eq(position)
    ].copy()

    if not exact.empty:
        return exact.iloc[0], "CURRENT_SEASON"

    fallback = dvp[
        dvp["defense_team"]
        .astype(str)
        .str.upper()
        .eq(opponent)
        &
        dvp["position"]
        .astype(str)
        .str.upper()
        .eq(position)
        &
        (
            pd.to_numeric(
                dvp["season"],
                errors="coerce",
            )
            < season
        )
    ].copy()

    fallback["season"] = pd.to_numeric(
        fallback["season"],
        errors="coerce",
    )

    fallback["week"] = pd.to_numeric(
        fallback["week"],
        errors="coerce",
    )

    fallback = fallback.sort_values(
        ["season", "week"],
        ascending=[False, False],
    )

    if fallback.empty:
        return None, "NONE"

    return (
        fallback.iloc[0],
        "PRIOR_SEASON_DVP_FALLBACK",
    )


def blend(a, b):
    a = to_num(a, np.nan)
    b = to_num(b, np.nan)

    if pd.isna(a) and pd.isna(b):
        return 0.0

    if pd.isna(a):
        return b

    if pd.isna(b):
        return a

    # Shadow-only symmetric matchup blend.
    return 0.50 * a + 0.50 * b


def fd_points(row: dict) -> float:
    return (
        0.04 * row.get("expected_passing_yards", 0.0)
        + 4.0 * row.get("expected_passing_tds", 0.0)
        - 1.0 * row.get("expected_interceptions", 0.0)
        + 0.10 * row.get("expected_rushing_yards", 0.0)
        + 6.0 * row.get("expected_rushing_tds", 0.0)
        + 0.50 * row.get("expected_receptions", 0.0)
        + 0.10 * row.get("expected_receiving_yards", 0.0)
        + 6.0 * row.get("expected_receiving_tds", 0.0)
    )


canonical_sha_before = sha256(
    CANONICAL_PATH
)

roster = pd.read_parquet(
    ROSTER_PATH
)

prod = pd.read_parquet(
    PROD_PATH
)

canonical = pd.read_parquet(
    CANONICAL_PATH
)

env = pd.read_parquet(
    ENV_PATH
)

dvp = pd.read_parquet(
    DVP_PATH
)

with sqlite3.connect(
    f"file:{DB_PATH}?mode=ro",
    uri=True,
) as conn:
    depth = pd.read_sql_query(
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
        """,
        conn,
    )

depth["snapshot_dt"] = pd.to_datetime(
    depth["snapshot_dt"],
    errors="coerce",
    utc=True,
)

depth["pos_rank"] = pd.to_numeric(
    depth["pos_rank"],
    errors="coerce",
)

depth["team"] = (
    depth["team"]
    .fillna("")
    .astype(str)
    .str.upper()
)

depth["pos_abb"] = (
    depth["pos_abb"]
    .fillna("")
    .astype(str)
    .str.upper()
)

# ------------------------------------------------------------
# CURRENT PROJECTION NORMALIZATION
# ------------------------------------------------------------

if "season" not in prod.columns:
    prod["season"] = pd.to_numeric(
        prod["game_id"]
        .astype(str)
        .str.slice(0, 4),
        errors="coerce",
    )

prod["season"] = pd.to_numeric(
    prod["season"],
    errors="coerce",
)

prod["position"] = (
    prod["position"]
    .fillna("")
    .astype(str)
    .str.upper()
)

prod["team"] = (
    prod["team"]
    .fillna("")
    .astype(str)
    .str.upper()
)

prod["opponent_team"] = (
    prod["opponent_team"]
    .fillna("")
    .astype(str)
    .str.upper()
)

prod["player_id"] = (
    prod["player_id"]
    .fillna("")
    .astype(str)
)

CURRENT_SEASON = int(
    prod["season"].dropna().max()
)

# ------------------------------------------------------------
# TRUE ROOKIE AUTHORITY
# ------------------------------------------------------------

for c in [
    "years_exp",
    "rookie_year",
    "entry_year",
]:
    roster[c] = pd.to_numeric(
        roster[c],
        errors="coerce",
    )

roster["season"] = pd.to_numeric(
    roster["season"],
    errors="coerce",
)

roster["position"] = (
    roster["position"]
    .fillna("")
    .astype(str)
    .str.upper()
)

rookie = roster[
    roster["gsis_id"].notna()
    &
    roster["position"].isin(
        SUPPORTED_POSITIONS
    )
    &
    (
        roster["years_exp"].eq(0)
        |
        roster["rookie_year"].eq(
            roster["season"]
        )
        |
        roster["entry_year"].eq(
            roster["season"]
        )
    )
].copy()

rookie_current = rookie[
    rookie["season"].eq(
        CURRENT_SEASON
    )
].copy()

rookie_current["week_num"] = pd.to_numeric(
    rookie_current["week"],
    errors="coerce",
)

rookie_current = rookie_current.sort_values(
    [
        "gsis_id",
        "week_num",
        "updated_at",
    ]
).drop_duplicates(
    "gsis_id",
    keep="last",
)

rookie_lookup = {
    str(r["gsis_id"]): r
    for _, r in rookie_current.iterrows()
}

# ------------------------------------------------------------
# CURRENT STARTER AUTHORITY
# ------------------------------------------------------------

latest_team_snapshot = (
    depth.groupby("team")[
        "snapshot_dt"
    ]
    .transform("max")
)

latest_depth = depth[
    depth["snapshot_dt"]
    .eq(latest_team_snapshot)
].copy()

starter_depth = latest_depth[
    latest_depth["pos_rank"].eq(1)
    &
    latest_depth["pos_abb"].isin(
        SUPPORTED_POSITIONS
    )
].copy()

starter_keys = set(
    zip(
        starter_depth["team"],
        starter_depth["gsis_id"].astype(str),
        starter_depth["pos_abb"],
    )
)

# ------------------------------------------------------------
# IDENTIFY QUALIFYING COLD-START STARTERS
# ------------------------------------------------------------

canonical_keys = set(
    zip(
        canonical["game_id"].astype(str),
        canonical["player_id"].astype(str),
    )
)

candidates = []

for _, row in prod.iterrows():
    pid = str(row["player_id"])
    game_id = str(row["game_id"])
    team = str(row["team"]).upper()
    pos = str(row["position"]).upper()

    if pos not in SUPPORTED_POSITIONS:
        continue

    if normalize_status(
        row.get("production_status")
    ) != "COLD_START":
        continue

    if pid not in rookie_lookup:
        continue

    if (
        team,
        pid,
        pos,
    ) not in starter_keys:
        continue

    if (
        game_id,
        pid,
    ) in canonical_keys:
        continue

    rr = rookie_lookup[pid]

    if hard_unavailable(
        rr.get("status"),
        rr.get("status_description_abbr"),
    ):
        continue

    candidates.append(
        row.to_dict()
    )

candidate_df = pd.DataFrame(
    candidates
)

print()
print("===== QUALIFYING ROOKIE COLD-START STARTERS =====")

if candidate_df.empty:
    print("NONE")
else:
    cols = [
        c for c in [
            "game_id",
            "team",
            "opponent_team",
            "player_id",
            "player_display_name",
            "position",
            "production_status",
        ]
        if c in candidate_df.columns
    ]

    print(
        candidate_df[
            cols
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# HELPERS FOR CURRENT CANONICAL TEAM ALLOCATION
# ------------------------------------------------------------

canonical["team"] = (
    canonical["team"]
    .fillna("")
    .astype(str)
    .str.upper()
)

canonical["position"] = (
    canonical["position"]
    .fillna("")
    .astype(str)
    .str.upper()
)

canonical["player_id"] = (
    canonical["player_id"]
    .fillna("")
    .astype(str)
)

shadow_rows = []

for _, player in candidate_df.iterrows():
    game_id = str(player["game_id"])
    team = str(player["team"]).upper()
    opponent = str(
        player["opponent_team"]
    ).upper()
    pid = str(player["player_id"])
    pos = str(player["position"]).upper()

    player_name = str(
        player.get(
            "player_display_name",
            pid,
        )
    )

    season = int(
        to_num(
            player.get("season"),
            CURRENT_SEASON,
        )
    )

    prior = hierarchical_prior(
        pos
    )

    env_row, env_source = latest_environment(
        env=env,
        game_id=game_id,
        team=team,
        season=season,
    )

    if env_row is None:
        print(
            f"SKIP | {player_name} | "
            f"no team environment"
        )
        continue

    pass_pool = to_num(
        env_row.get(
            "pass_attempts_avg_5"
        ),
        np.nan,
    )

    rush_pool = to_num(
        env_row.get(
            "rush_attempts_avg_5"
        ),
        np.nan,
    )

    if pd.isna(pass_pool) or pd.isna(rush_pool):
        print(
            f"SKIP | {player_name} | "
            f"missing avg5 team pool"
        )
        continue

    target_pool = (
        pass_pool
        * TARGETS_PER_PASS_ATTEMPT
    )

    dvp_row, dvp_source = latest_dvp(
        dvp=dvp,
        game_id=game_id,
        opponent=opponent,
        position=pos,
        season=season,
    )

    team_fc = canonical[
        canonical["game_id"]
        .astype(str)
        .eq(game_id)
        &
        canonical["team"].eq(team)
    ].copy()

    # --------------------------------------------------------
    # TARGET ALLOCATION
    # --------------------------------------------------------

    existing_skill = team_fc[
        team_fc["position"].isin(
            {
                "RB",
                "FB",
                "WR",
                "TE",
            }
        )
    ].copy()

    if "expected_targets" in existing_skill.columns:
        existing_targets = pd.to_numeric(
            existing_skill[
                "expected_targets"
            ],
            errors="coerce",
        ).fillna(0.0).sum()
    else:
        existing_targets = 0.0

    desired_targets = float(
        prior.get(
            "targets",
            0.0,
        )
    )

    if pos == "QB":
        desired_targets = 0.0

    rookie_targets = min(
        max(
            0.0,
            desired_targets,
        ),
        target_pool,
    )

    remaining_target_pool = max(
        0.0,
        target_pool
        - rookie_targets,
    )

    incumbent_target_scale = (
        remaining_target_pool
        / existing_targets
        if existing_targets > 0
        else 0.0
    )

    # --------------------------------------------------------
    # RUSHING ALLOCATION
    #
    # R8Q1 authority:
    #
    # Every projected carry must live inside ONE authoritative
    # team rush-attempt pool.
    #
    # Cold-start player's desired carries are reserved first.
    # All existing OTHER rushers are proportionally rescaled
    # into the remaining pool.
    #
    # No position-specific carry pool may exceed team rush pool.
    # --------------------------------------------------------

    desired_carries = float(
        prior.get(
            "carries",
            0.0,
        )
    )

    rookie_carries = min(
        max(
            0.0,
            desired_carries,
        ),
        rush_pool,
    )

    existing_rushers = team_fc[
        team_fc["position"].isin(
            {
                "QB",
                "RB",
                "FB",
                "WR",
                "TE",
            }
        )
        &
        ~team_fc["player_id"].eq(pid)
    ].copy()

    if "expected_carries" in existing_rushers.columns:
        existing_rushers[
            "expected_carries"
        ] = pd.to_numeric(
            existing_rushers[
                "expected_carries"
            ],
            errors="coerce",
        ).fillna(0.0)

        existing_other_carries = float(
            existing_rushers[
                "expected_carries"
            ].sum()
        )
    else:
        existing_other_carries = 0.0

    remaining_rush_pool = max(
        0.0,
        rush_pool
        - rookie_carries,
    )

    incumbent_rush_scale_after = (
        remaining_rush_pool
        / existing_other_carries
        if existing_other_carries > 0
        else 0.0
    )

    shadow_other_carries_after = (
        existing_other_carries
        * incumbent_rush_scale_after
    )

    shadow_team_carries_after = (
        rookie_carries
        + shadow_other_carries_after
    )

    team_rush_pool_conservation_error = (
        shadow_team_carries_after
        - rush_pool
    )

    # Compatibility diagnostics retained for the shadow artifact.
    existing_backfield = team_fc[
        team_fc["position"].isin(
            {
                "RB",
                "FB",
            }
        )
        &
        ~team_fc["player_id"].eq(pid)
    ].copy()

    if "expected_carries" in existing_backfield.columns:
        existing_backfield_carries = float(
            pd.to_numeric(
                existing_backfield[
                    "expected_carries"
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        )
    else:
        existing_backfield_carries = 0.0

    backfield_pool = max(
        0.0,
        rush_pool
        - rookie_carries,
    )

    incumbent_backfield_scale = (
        incumbent_rush_scale_after
    )

    qb1_carries = 0.0

    # --------------------------------------------------------
    # POSITION EFFICIENCY
    # --------------------------------------------------------

    expected = {
        "expected_completions": 0.0,
        "expected_attempts": 0.0,
        "expected_passing_yards": 0.0,
        "expected_passing_tds": 0.0,
        "expected_interceptions": 0.0,
        "expected_carries": 0.0,
        "expected_targets": 0.0,
        "expected_receptions": 0.0,
        "expected_rushing_yards": 0.0,
        "expected_receiving_yards": 0.0,
        "expected_rushing_tds": 0.0,
        "expected_receiving_tds": 0.0,
    }

    # --------------------------------------------------------
    # QB
    # --------------------------------------------------------

    if pos == "QB":
        attempts_prior = prior["attempts"]

        comp_rate = safe_div(
            prior["completions"],
            attempts_prior,
        )

        ypa = safe_div(
            prior["passing_yards"],
            attempts_prior,
        )

        pass_td_rate = safe_div(
            prior["passing_tds"],
            attempts_prior,
        )

        int_rate = safe_div(
            prior["interceptions"],
            attempts_prior,
        )

        rush_ypc = safe_div(
            prior["rushing_yards"],
            prior["carries"],
        )

        rush_td_rate = safe_div(
            prior["rushing_tds"],
            prior["carries"],
        )

        expected[
            "expected_attempts"
        ] = pass_pool

        expected[
            "expected_completions"
        ] = pass_pool * comp_rate

        expected[
            "expected_passing_yards"
        ] = pass_pool * ypa

        expected[
            "expected_passing_tds"
        ] = pass_pool * pass_td_rate

        expected[
            "expected_interceptions"
        ] = pass_pool * int_rate

        expected[
            "expected_carries"
        ] = rookie_carries

        expected[
            "expected_rushing_yards"
        ] = rookie_carries * rush_ypc

        expected[
            "expected_rushing_tds"
        ] = rookie_carries * rush_td_rate

    # --------------------------------------------------------
    # RB / FB
    # --------------------------------------------------------

    elif pos in {"RB", "FB"}:
        prior_catch = safe_div(
            prior["receptions"],
            prior["targets"],
        )

        prior_ypr = safe_div(
            prior["receiving_yards"],
            prior["receptions"],
        )

        prior_rec_td_rate = safe_div(
            prior["receiving_tds"],
            prior["targets"],
        )

        prior_ypc = safe_div(
            prior["rushing_yards"],
            prior["carries"],
        )

        prior_rush_td_rate = safe_div(
            prior["rushing_tds"],
            prior["carries"],
        )

        dvp_catch = np.nan
        dvp_ypr = np.nan
        dvp_rec_td_rate = np.nan
        dvp_ypc = np.nan
        dvp_rush_td_rate = np.nan

        if dvp_row is not None:
            dvp_targets = to_num(
                dvp_row.get(
                    "targets_allowed_avg_3"
                ),
                np.nan,
            )

            dvp_receptions = to_num(
                dvp_row.get(
                    "receptions_allowed_avg_3"
                ),
                np.nan,
            )

            dvp_rec_yards = to_num(
                dvp_row.get(
                    "receiving_yards_allowed_avg_3"
                ),
                np.nan,
            )

            dvp_rec_tds = to_num(
                dvp_row.get(
                    "receiving_tds_allowed_avg_3"
                ),
                np.nan,
            )

            dvp_carries = to_num(
                dvp_row.get(
                    "carries_allowed_avg_3"
                ),
                np.nan,
            )

            dvp_rush_yards = to_num(
                dvp_row.get(
                    "rushing_yards_allowed_avg_3"
                ),
                np.nan,
            )

            dvp_rush_tds = to_num(
                dvp_row.get(
                    "rushing_tds_allowed_avg_3"
                ),
                np.nan,
            )

            dvp_catch = safe_div(
                dvp_receptions,
                dvp_targets,
                np.nan,
            )

            dvp_ypr = safe_div(
                dvp_rec_yards,
                dvp_receptions,
                np.nan,
            )

            dvp_rec_td_rate = safe_div(
                dvp_rec_tds,
                dvp_targets,
                np.nan,
            )

            dvp_ypc = safe_div(
                dvp_rush_yards,
                dvp_carries,
                np.nan,
            )

            dvp_rush_td_rate = safe_div(
                dvp_rush_tds,
                dvp_carries,
                np.nan,
            )

        catch_rate = clamp(
            blend(
                prior_catch,
                dvp_catch,
            ),
            0.0,
            1.0,
        )

        ypr = max(
            0.0,
            blend(
                prior_ypr,
                dvp_ypr,
            ),
        )

        rec_td_rate = max(
            0.0,
            blend(
                prior_rec_td_rate,
                dvp_rec_td_rate,
            ),
        )

        ypc = max(
            0.0,
            blend(
                prior_ypc,
                dvp_ypc,
            ),
        )

        rush_td_rate = max(
            0.0,
            blend(
                prior_rush_td_rate,
                dvp_rush_td_rate,
            ),
        )

        receptions = (
            rookie_targets
            * catch_rate
        )

        expected[
            "expected_carries"
        ] = rookie_carries

        expected[
            "expected_targets"
        ] = rookie_targets

        expected[
            "expected_receptions"
        ] = receptions

        expected[
            "expected_rushing_yards"
        ] = rookie_carries * ypc

        expected[
            "expected_receiving_yards"
        ] = receptions * ypr

        expected[
            "expected_rushing_tds"
        ] = rookie_carries * rush_td_rate

        expected[
            "expected_receiving_tds"
        ] = rookie_targets * rec_td_rate

    # --------------------------------------------------------
    # WR / TE
    # --------------------------------------------------------

    elif pos in {"WR", "TE"}:
        prior_catch = safe_div(
            prior["receptions"],
            prior["targets"],
        )

        prior_ypr = safe_div(
            prior["receiving_yards"],
            prior["receptions"],
        )

        prior_td_rate = safe_div(
            prior["receiving_tds"],
            prior["targets"],
        )

        dvp_catch = np.nan
        dvp_ypr = np.nan
        dvp_td_rate = np.nan

        if dvp_row is not None:
            dvp_targets = to_num(
                dvp_row.get(
                    "targets_allowed_avg_3"
                ),
                np.nan,
            )

            dvp_receptions = to_num(
                dvp_row.get(
                    "receptions_allowed_avg_3"
                ),
                np.nan,
            )

            dvp_yards = to_num(
                dvp_row.get(
                    "receiving_yards_allowed_avg_3"
                ),
                np.nan,
            )

            dvp_tds = to_num(
                dvp_row.get(
                    "receiving_tds_allowed_avg_3"
                ),
                np.nan,
            )

            dvp_catch = safe_div(
                dvp_receptions,
                dvp_targets,
                np.nan,
            )

            dvp_ypr = safe_div(
                dvp_yards,
                dvp_receptions,
                np.nan,
            )

            dvp_td_rate = safe_div(
                dvp_tds,
                dvp_targets,
                np.nan,
            )

        catch_rate = clamp(
            blend(
                prior_catch,
                dvp_catch,
            ),
            0.0,
            1.0,
        )

        ypr = max(
            0.0,
            blend(
                prior_ypr,
                dvp_ypr,
            ),
        )

        td_rate = max(
            0.0,
            blend(
                prior_td_rate,
                dvp_td_rate,
            ),
        )

        receptions = (
            rookie_targets
            * catch_rate
        )

        expected[
            "expected_targets"
        ] = rookie_targets

        expected[
            "expected_receptions"
        ] = receptions

        expected[
            "expected_receiving_yards"
        ] = receptions * ypr

        expected[
            "expected_receiving_tds"
        ] = rookie_targets * td_rate

        if pos == "WR":
            prior_rush_ypc = safe_div(
                prior.get(
                    "rushing_yards",
                    0.0,
                ),
                prior.get(
                    "carries",
                    0.0,
                ),
            )

            prior_rush_td_rate = safe_div(
                prior.get(
                    "rushing_tds",
                    0.0,
                ),
                prior.get(
                    "carries",
                    0.0,
                ),
            )

            expected[
                "expected_carries"
            ] = rookie_carries

            expected[
                "expected_rushing_yards"
            ] = (
                rookie_carries
                * prior_rush_ypc
            )

            expected[
                "expected_rushing_tds"
            ] = (
                rookie_carries
                * prior_rush_td_rate
            )

    row = {
        "entity_type": "player",
        "game_id": game_id,
        "season": season,
        "team": team,
        "opponent_team": opponent,
        "player_id": pid,
        "entity_name": player_name,
        "position": pos,
        "model_group": "COLD_START_SHADOW_V1",
        "production_status": "COLD_START",

        "rookie_first_start_n":
            prior["_first_n"],

        "rookie_all_starter_game_n":
            prior["_all_n"],

        "rookie_first_start_weight":
            prior["_first_weight"],

        "team_pass_attempt_pool":
            pass_pool,

        "targets_per_pass_attempt":
            TARGETS_PER_PASS_ATTEMPT,

        "team_target_pool":
            target_pool,

        "team_rush_attempt_pool":
            rush_pool,

        "rookie_desired_targets":
            desired_targets,

        "rookie_desired_carries":
            desired_carries,

        "incumbent_target_total_before":
            existing_targets,

        "incumbent_target_scale_after":
            incumbent_target_scale,

        "qb1_reserved_carries":
            qb1_carries,

        "existing_other_rusher_carries_before":
            existing_other_carries,

        "remaining_team_rush_pool_after_rookie":
            remaining_rush_pool,

        "incumbent_rush_scale_after":
            incumbent_rush_scale_after,

        "shadow_team_carries_after":
            shadow_team_carries_after,

        "team_rush_pool_conservation_error":
            team_rush_pool_conservation_error,

        "existing_backfield_carries_before":
            existing_backfield_carries,

        "backfield_carry_pool":
            backfield_pool,

        "incumbent_backfield_scale_after":
            incumbent_backfield_scale,

        "environment_source":
            env_source,

        "environment_source_game_id":
            str(
                env_row.get(
                    "game_id",
                    "",
                )
            ),

        "dvp_source":
            dvp_source,

        "dvp_source_game_id":
            (
                str(
                    dvp_row.get(
                        "game_id",
                        "",
                    )
                )
                if dvp_row is not None
                else ""
            ),

        "matchup_blend":
            "50pct_rookie_prior_50pct_dvp",

        **expected,
    }

    row[
        "expected_fd_points"
    ] = fd_points(
        row
    )

    # Conservation diagnostics.
    row[
        "shadow_target_pool_after"
    ] = (
        rookie_targets
        +
        existing_targets
        * incumbent_target_scale
    )

    row[
        "shadow_backfield_carries_after"
    ] = (
        existing_backfield_carries
        * incumbent_rush_scale_after
        +
        (
            rookie_carries
            if pos in {"RB", "FB"}
            else 0.0
        )
    )

    row[
        "target_pool_conservation_error"
    ] = (
        row["shadow_target_pool_after"]
        -
        target_pool
    )

    row[
        "team_rush_pool_conservation_error"
    ] = (
        shadow_team_carries_after
        -
        rush_pool
    )

    row[
        "backfield_pool_conservation_error"
    ] = 0.0

    shadow_rows.append(
        row
    )


shadow = pd.DataFrame(
    shadow_rows
)

if shadow.empty:
    raise RuntimeError(
        "FAIL | no qualifying cold-start "
        "starter forecasts generated"
    )

shadow = shadow.sort_values(
    [
        "game_id",
        "team",
        "position",
        "entity_name",
    ]
).reset_index(drop=True)

# ------------------------------------------------------------
# HARD SHADOW GATES
# ------------------------------------------------------------

if shadow[
    "player_id"
].duplicated().any():
    raise RuntimeError(
        "FAIL | duplicate player_id "
        "in shadow artifact"
    )

if (
    shadow[
        "target_pool_conservation_error"
    ]
    .abs()
    .max()
    > 1e-6
):
    raise RuntimeError(
        "FAIL | target pool not conserved"
    )

if (
    shadow[
        "team_rush_pool_conservation_error"
    ]
    .abs()
    .max()
    > 1e-6
):
    raise RuntimeError(
        "FAIL | authoritative team rush pool not conserved"
    )

stat_cols = [
    c for c in shadow.columns
    if c.startswith("expected_")
]

for c in stat_cols:
    values = pd.to_numeric(
        shadow[c],
        errors="coerce",
    )

    if values.isna().any():
        raise RuntimeError(
            f"FAIL | NaN in {c}"
        )

    if (values < -1e-9).any():
        raise RuntimeError(
            f"FAIL | negative value in {c}"
        )

# ------------------------------------------------------------
# WRITE SHADOW ONLY
# ------------------------------------------------------------

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

shadow.to_parquet(
    OUT_PATH,
    index=False,
)

canonical_sha_after = sha256(
    CANONICAL_PATH
)

if (
    canonical_sha_after
    != canonical_sha_before
):
    raise RuntimeError(
        "FAIL | canonical forecast changed"
    )

audit = {
    "stage":
        "STAGE26G-R8Q",

    "model":
        "COLD_START_SHADOW_V1",

    "rows":
        int(len(shadow)),

    "players":
        shadow[
            [
                "game_id",
                "team",
                "player_id",
                "entity_name",
                "position",
            ]
        ].to_dict(
            orient="records"
        ),

    "target_rate":
        TARGETS_PER_PASS_ATTEMPT,

    "canonical_sha_before":
        canonical_sha_before,

    "canonical_sha_after":
        canonical_sha_after,

    "canonical_unchanged":
        canonical_sha_before
        == canonical_sha_after,

    "shadow_sha":
        sha256(
            OUT_PATH
        ),

    "production_mutation":
        False,

    "publisher_mutation":
        False,

    "analyst_mutation":
        False,

    "app_mutation":
        False,

    "solver_mutation":
        False,
}

AUDIT_PATH.write_text(
    json.dumps(
        audit,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

print()
print("============================================================")
print("R8Q SHADOW OUTPUT")
print("============================================================")

display_cols = [
    "game_id",
    "team",
    "entity_name",
    "position",

    "team_pass_attempt_pool",
    "team_target_pool",
    "team_rush_attempt_pool",

    "expected_attempts",
    "expected_completions",
    "expected_passing_yards",
    "expected_passing_tds",
    "expected_interceptions",

    "expected_carries",
    "expected_targets",
    "expected_receptions",
    "expected_rushing_yards",
    "expected_receiving_yards",
    "expected_rushing_tds",
    "expected_receiving_tds",
    "expected_fd_points",

    "incumbent_target_scale_after",
    "incumbent_backfield_scale_after",

    "environment_source",
    "dvp_source",
]

print(
    shadow[
        display_cols
    ].to_string(
        index=False
    )
)

print()
print("===== CONSERVATION =====")

print(
    shadow[
        [
            "entity_name",
            "team_target_pool",
            "shadow_target_pool_after",
            "target_pool_conservation_error",
            "team_rush_attempt_pool",
            "expected_carries",
            "existing_other_rusher_carries_before",
            "incumbent_rush_scale_after",
            "shadow_team_carries_after",
            "team_rush_pool_conservation_error",
        ]
    ].to_string(
        index=False
    )
)

print()
print("===== ARTIFACTS =====")

print(
    f"SHADOW_PATH={OUT_PATH}"
)

print(
    f"SHADOW_SHA={sha256(OUT_PATH)}"
)

print(
    f"AUDIT_PATH={AUDIT_PATH}"
)

print(
    f"CANONICAL_SHA_BEFORE="
    f"{canonical_sha_before}"
)

print(
    f"CANONICAL_SHA_AFTER="
    f"{canonical_sha_after}"
)

print(
    "CANONICAL_UNCHANGED="
    f"{canonical_sha_before == canonical_sha_after}"
)

print()
print("============================================================")
print("STAGE26G_R8Q_COLD_START_SHADOW=PASS")
print("============================================================")
