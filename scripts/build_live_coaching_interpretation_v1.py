#!/usr/bin/env python3

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NFL_DB = ROOT / "data" / "nfl.db"

LIVE_DIR = (
    ROOT
    / "processed"
    / "live_game_intelligence"
)

COACH_IDENTITY = (
    ROOT
    / "processed"
    / "forecast_v1_coach_identity.csv"
)

ATTACHED_EVIDENCE = (
    ROOT
    / "processed"
    / "coaching_intelligence"
    / "postgame_coach_attached_evidence_v1.csv"
)

OUT_DIR = (
    ROOT
    / "processed"
    / "coaching_intelligence"
    / "live_interpretation"
)

VERSION = "WFS_LIVE_COACHING_INTERPRETATION_V1"


TEAM_IDENTITY_ALIAS = {
    "LAR": "LA",
    "WSH": "WAS",
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def as_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    if np.isnan(x):
        return None

    return x


def pct(value):
    value = as_float(value)

    if value is None:
        return "—"

    return f"{value * 100:.1f}%"


def pp(value):
    value = as_float(value)

    if value is None:
        return "—"

    return f"{value * 100:+.1f} pp"


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
):
    v = pd.to_numeric(
        values,
        errors="coerce",
    )

    w = pd.to_numeric(
        weights,
        errors="coerce",
    )

    mask = (
        v.notna()
        & w.notna()
        & (w > 0)
    )

    if not mask.any():
        return None

    return float(
        np.average(
            v[mask],
            weights=w[mask],
        )
    )


def live_sample_confidence(
    plays: int,
) -> str:
    if plays >= 40:
        return "HIGH"

    if plays >= 20:
        return "MEDIUM"

    return "LOW"


def historical_confidence(
    games: int,
    plays: int,
) -> str:
    if games >= 6 and plays >= 300:
        return "HIGH"

    if games >= 3 and plays >= 140:
        return "MEDIUM"

    return "LOW"


def deviation_label(delta):
    delta = as_float(delta)

    if delta is None:
        return "NO_BASELINE"

    magnitude = abs(delta)

    if magnitude >= 0.10:
        return (
            "MATERIAL_PASS_HEAVY"
            if delta > 0
            else "MATERIAL_RUN_HEAVY"
        )

    if magnitude >= 0.05:
        return (
            "MODERATE_PASS_HEAVY"
            if delta > 0
            else "MODERATE_RUN_HEAVY"
        )

    return "NEAR_EXPECTATION"


def paper_interpretation(
    team: str,
    live_rate,
    expected_rate,
    live_plays: int,
):
    live_rate = as_float(live_rate)
    expected_rate = as_float(expected_rate)

    if live_rate is None:
        return (
            f"{team} does not yet have enough "
            "live pass/run evidence for interpretation."
        )

    if expected_rate is None:
        return (
            f"{team} has no matching pregame Expected Pass Rate "
            "baseline for this game. Current behavior is descriptive "
            "only and should not be labeled above or below paper."
        )

    delta = live_rate - expected_rate
    label = deviation_label(delta)

    sample_confidence = live_sample_confidence(
        live_plays
    )

    if sample_confidence == "LOW":
        return (
            f"Early sample: {team} currently has a "
            f"{pct(live_rate)} pass rate versus "
            f"{pct(expected_rate)} expected "
            f"({pp(delta)}), based on only "
            f"{live_plays} scrimmage plays. "
            "Treat the directional difference as LOW-confidence "
            "live evidence, not an established coaching tendency."
        )

    if label == "MATERIAL_PASS_HEAVY":
        phrase = (
            "operating materially more pass-heavy "
            "than the pregame game-plan expectation"
        )

    elif label == "MODERATE_PASS_HEAVY":
        phrase = (
            "operating moderately more pass-heavy "
            "than the pregame game-plan expectation"
        )

    elif label == "MATERIAL_RUN_HEAVY":
        phrase = (
            "operating materially more run-heavy "
            "than the pregame game-plan expectation"
        )

    elif label == "MODERATE_RUN_HEAVY":
        phrase = (
            "operating moderately more run-heavy "
            "than the pregame game-plan expectation"
        )

    else:
        phrase = (
            "operating near the pregame "
            "game-plan expectation"
        )

    return (
        f"{team} is {phrase}. "
        f"Live pass rate is {pct(live_rate)} versus "
        f"{pct(expected_rate)} expected "
        f"({pp(delta)})."
    )


def regime_interpretation(
    coach: str,
    identity_team: str,
    live_rate,
    regime_rate,
    regime_games: int,
    regime_plays: int,
):
    live_rate = as_float(live_rate)
    regime_rate = as_float(regime_rate)

    if regime_games <= 0:
        return (
            f"No prior completed game-level evidence exists for "
            f"the {coach} {identity_team} regime. Treat today's "
            "behavior as new game-level evidence, not an established "
            "coaching tendency."
        )

    confidence = historical_confidence(
        regime_games,
        regime_plays,
    )

    if live_rate is None or regime_rate is None:
        return (
            f"The {coach} {identity_team} regime has "
            f"{regime_games} prior completed evidence game(s), "
            f"but a comparable pass-rate signal is unavailable."
        )

    delta = live_rate - regime_rate
    label = deviation_label(delta)

    if label == "MATERIAL_PASS_HEAVY":
        relation = (
            "materially more pass-heavy than "
            "the prior regime evidence"
        )

    elif label == "MODERATE_PASS_HEAVY":
        relation = (
            "moderately more pass-heavy than "
            "the prior regime evidence"
        )

    elif label == "MATERIAL_RUN_HEAVY":
        relation = (
            "materially more run-heavy than "
            "the prior regime evidence"
        )

    elif label == "MODERATE_RUN_HEAVY":
        relation = (
            "moderately more run-heavy than "
            "the prior regime evidence"
        )

    else:
        relation = (
            "near the prior regime evidence"
        )

    return (
        f"Today's behavior is {relation}: "
        f"{pct(live_rate)} live versus "
        f"{pct(regime_rate)} across "
        f"{regime_games} prior completed game(s) "
        f"and {regime_plays} scrimmage plays. "
        f"Historical evidence confidence is {confidence}."
    )


def load_live_paper(
    event_id: str,
):
    path = (
        LIVE_DIR
        / f"{event_id}.json"
    )

    if not path.exists():
        raise RuntimeError(
            f"missing game intelligence JSON: {path}"
        )

    data = json.loads(
        path.read_text()
    )

    return data


def find_team_rows(data):
    # Support the current Game Intelligence V1 JSON
    # without inventing missing values.
    for key in (
        "teams",
        "team_intelligence",
        "team_metrics",
        "team_results",
    ):
        value = data.get(key)

        if isinstance(value, list):
            return value

    # Some versions may store team objects in a dict.
    for key in (
        "team_data",
        "team_evidence",
    ):
        value = data.get(key)

        if isinstance(value, dict):
            rows = []

            for team, payload in value.items():
                if not isinstance(payload, dict):
                    continue

                row = dict(payload)
                row.setdefault(
                    "team",
                    team,
                )
                rows.append(row)

            if rows:
                return rows

    raise RuntimeError(
        "unable to locate team rows in "
        "Game Intelligence JSON"
    )


def pick_value(
    row: dict,
    names,
):
    for name in names:
        if name in row:
            return row.get(name)

    return None


def resolve_live_score_context(
    data: dict,
    game_id: str,
) -> dict:
    """
    Resolve the latest authoritative ESPN per-play score state already
    persisted by Game Intelligence V1.

    Analysis-only. No score reconstruction and no production influence.
    """
    teams = data.get("teams") or []

    latest_play = None
    latest_play_id = -1

    for team_row in teams:
        for play in team_row.get("evidence_plays") or []:
            try:
                play_id = int(
                    str(play.get("play_id") or "0")
                )
            except (TypeError, ValueError):
                continue

            if play_id > latest_play_id:
                latest_play_id = play_id
                latest_play = play

    if latest_play is None:
        return {
            "score_available": False,
            "away_team": None,
            "home_team": None,
            "away_score": None,
            "home_score": None,
        }

    try:
        away_score = int(
            latest_play.get("away_score")
        )
    except (TypeError, ValueError):
        away_score = None

    try:
        home_score = int(
            latest_play.get("home_score")
        )
    except (TypeError, ValueError):
        home_score = None

    away_team = None
    home_team = None

    if NFL_DB.exists():
        try:
            with sqlite3.connect(
                f"file:{NFL_DB}?mode=ro",
                uri=True,
            ) as conn:
                row = conn.execute(
                    """
                    SELECT
                        away_team,
                        home_team
                    FROM games
                    WHERE game_id = ?
                    LIMIT 1
                    """,
                    (game_id,),
                ).fetchone()

            if row:
                away_team = (
                    str(row[0]).strip().upper()
                    if row[0] is not None
                    else None
                )
                home_team = (
                    str(row[1]).strip().upper()
                    if row[1] is not None
                    else None
                )

        except Exception:
            away_team = None
            home_team = None

    return {
        "score_available": (
            away_score is not None
            and home_score is not None
        ),
        "away_team": away_team,
        "home_team": home_team,
        "away_score": away_score,
        "home_score": home_score,
        "latest_play_id": (
            str(latest_play.get("play_id") or "")
            if latest_play is not None
            else None
        ),
    }


def main() -> int:
    print("=" * 72)
    print(VERSION)
    print("=" * 72)

    if len(sys.argv) != 2:
        print(
            "USAGE: "
            "build_live_coaching_interpretation_v1.py "
            "<event_id>"
        )
        return 2

    event_id = str(
        sys.argv[1]
    ).strip()

    if not COACH_IDENTITY.exists():
        raise RuntimeError(
            f"missing coach identity: {COACH_IDENTITY}"
        )

    data = load_live_paper(
        event_id
    )

    game_id = str(
        data.get("game_id", "")
    ).strip()

    if not game_id:
        raise RuntimeError(
            "Game Intelligence JSON missing game_id"
        )

    # WFS_LIVE_SCORE_CONTEXT_V1
    # Consume the authoritative per-play score already persisted by
    # Game Intelligence V1. Analysis-only; no production influence.
    score_context = resolve_live_score_context(
        data,
        game_id,
    )

    team_rows = find_team_rows(
        data
    )

    identity = pd.read_csv(
        COACH_IDENTITY
    )

    attached = (
        pd.read_csv(
            ATTACHED_EVIDENCE
        )
        if ATTACHED_EVIDENCE.exists()
        else pd.DataFrame()
    )

    outputs = []

    for source in team_rows:
        team = str(
            source.get("team", "")
        ).strip().upper()

        if not team:
            continue

        identity_team = (
            TEAM_IDENTITY_ALIAS.get(
                team,
                team,
            )
        )

        match = identity[
            (
                identity["game_id"]
                .astype(str)
                == game_id
            )
            &
            (
                identity["team"]
                .astype(str)
                == identity_team
            )
        ]

        if len(match) != 1:
            raise RuntimeError(
                f"coach identity resolution failed "
                f"for {game_id} {team}: "
                f"matches={len(match)}"
            )

        ident = match.iloc[0]

        coach = str(
            ident["coach_identity"]
        ).strip()

        live_rate = pick_value(
            source,
            [
                "live_pass_rate",
                "pass_rate",
                "final_pass_rate",
            ],
        )

        expected_rate = pick_value(
            source,
            [
                "expected_pass_rate",
                "paper_pass_rate",
                "pregame_pass_rate",
            ],
        )

        scrimmage_plays = pick_value(
            source,
            [
                "scrimmage_plays",
                "plays",
            ],
        )

        live_rate = as_float(
            live_rate
        )

        expected_rate = as_float(
            expected_rate
        )

        try:
            live_plays = int(
                float(scrimmage_plays)
            )
        except (TypeError, ValueError):
            live_plays = 0

        history = pd.DataFrame()

        if not attached.empty:
            history = attached[
                (
                    attached[
                        "coach_identity"
                    ].astype(str)
                    == coach
                )
                &
                (
                    attached[
                        "identity_lookup_team"
                    ].astype(str)
                    == identity_team
                )
                &
                (
                    attached[
                        "event_id"
                    ].astype(str)
                    != event_id
                )
            ].copy()

        history_games = int(
            len(history)
        )

        if history_games:
            history_weights = pd.to_numeric(
                history[
                    "scrimmage_plays"
                ],
                errors="coerce",
            ).fillna(0)

            history_plays = int(
                history_weights.sum()
            )

            history_pass_rate = weighted_mean(
                history[
                    "final_pass_rate"
                ],
                history_weights,
            )

        else:
            history_plays = 0
            history_pass_rate = None

        paper_delta = (
            live_rate - expected_rate
            if (
                live_rate is not None
                and expected_rate is not None
            )
            else None
        )

        regime_delta = (
            live_rate - history_pass_rate
            if (
                live_rate is not None
                and history_pass_rate is not None
            )
            else None
        )

        # WFS_LIVE_SCORE_CONTEXT_V1
        team_score = None
        opponent_score = None
        score_differential = None
        game_state = "UNKNOWN"

        away_team = score_context.get("away_team")
        home_team = score_context.get("home_team")
        away_score = score_context.get("away_score")
        home_score = score_context.get("home_score")

        if (
            score_context.get("score_available")
            and away_team
            and home_team
        ):
            if team == away_team:
                team_score = away_score
                opponent_score = home_score

            elif team == home_team:
                team_score = home_score
                opponent_score = away_score

            if (
                team_score is not None
                and opponent_score is not None
            ):
                score_differential = (
                    team_score - opponent_score
                )

                if score_differential > 0:
                    game_state = "LEADING"
                elif score_differential < 0:
                    game_state = "TRAILING"
                else:
                    game_state = "TIED"

        row = {
            "event_id":
                event_id,

            "game_id":
                game_id,

            "team":
                team,

            "score_available":
                bool(
                    score_context.get(
                        "score_available",
                        False,
                    )
                    and team_score is not None
                    and opponent_score is not None
                ),

            "team_score":
                team_score,

            "opponent_score":
                opponent_score,

            "score_differential":
                score_differential,

            "game_state":
                game_state,

            "identity_lookup_team":
                identity_team,

            "coach_identity":
                coach,

            "coach_change_flag":
                int(
                    pd.to_numeric(
                        ident[
                            "coach_change_flag"
                        ],
                        errors="coerce",
                    )
                    if pd.notna(
                        ident[
                            "coach_change_flag"
                        ]
                    )
                    else 0
                ),

            "coach_team_era_start_flag":
                int(
                    pd.to_numeric(
                        ident[
                            "coach_team_era_start_flag"
                        ],
                        errors="coerce",
                    )
                    if pd.notna(
                        ident[
                            "coach_team_era_start_flag"
                        ]
                    )
                    else 0
                ),

            "live_scrimmage_plays":
                live_plays,

            "live_sample_confidence":
                live_sample_confidence(
                    live_plays
                ),

            "live_pass_rate":
                live_rate,

            "expected_pass_rate":
                expected_rate,

            "paper_delta":
                paper_delta,

            "paper_deviation":
                deviation_label(
                    paper_delta
                ),

            "prior_regime_games":
                history_games,

            "prior_regime_scrimmage_plays":
                history_plays,

            "prior_regime_pass_rate":
                history_pass_rate,

            "regime_delta":
                regime_delta,

            "regime_deviation":
                deviation_label(
                    regime_delta
                ),

            "regime_evidence_confidence":
                historical_confidence(
                    history_games,
                    history_plays,
                ),

            "paper_interpretation":
                paper_interpretation(
                    team,
                    live_rate,
                    expected_rate,
                    live_plays,
                ),

            "regime_interpretation":
                regime_interpretation(
                    coach,
                    identity_team,
                    live_rate,
                    history_pass_rate,
                    history_games,
                    history_plays,
                ),
        }

        outputs.append(
            row
        )

    if len(outputs) != 2:
        raise RuntimeError(
            "expected exactly two team interpretations; "
            f"found {len(outputs)}"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_json = (
        OUT_DIR
        / f"{event_id}.json"
    )

    out_md = (
        OUT_DIR
        / f"{event_id}.md"
    )

    payload = {
        "version":
            VERSION,

        "generated_at_utc":
            utc_now(),

        "event_id":
            event_id,

        "game_id":
            game_id,

        "policy": {
            "analysis_only":
                True,

            "production_influence":
                False,

            "solver_influence":
                False,

            "forecast_mutation":
                False,

            "persistent_coach_prior_mutation":
                False,

            "current_event_excluded_from_regime_history":
                True,

            "fuzzy_matching":
                False,
        },

        "teams":
            outputs,
    }

    out_json.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )

    md = [
        f"# Live Coaching Interpretation — {game_id}",
        "",
        (
            "**Analysis only — no production forecast "
            "or solver influence.**"
        ),
        "",
    ]

    for row in outputs:
        md.extend(
            [
                (
                    f"## {row['team']} — "
                    f"{row['coach_identity']}"
                ),
                "",
                (
                    f"- Live Pass Rate: "
                    f"{pct(row['live_pass_rate'])}"
                ),
                (
                    f"- Live Scrimmage Plays: "
                    f"{row['live_scrimmage_plays']}"
                ),
                (
                    f"- Live Sample Confidence: "
                    f"{row['live_sample_confidence']}"
                ),
                (
                    f"- Score State: "
                    f"{row['team_score']} - "
                    f"{row['opponent_score']} "
                    f"({row['game_state']})"
                    if row["score_available"]
                    else "- Score State: unavailable"
                ),
                (
                    f"- Score Differential: "
                    f"{row['score_differential']:+d}"
                    if row["score_available"]
                    else "- Score Differential: unavailable"
                ),
                (
                    f"- Pregame Expected Pass Rate: "
                    f"{pct(row['expected_pass_rate'])}"
                ),
                (
                    f"- Difference vs Paper: "
                    f"{pp(row['paper_delta'])}"
                ),
                (
                    f"- Prior Same-Regime Games: "
                    f"{row['prior_regime_games']}"
                ),
                (
                    f"- Prior Same-Regime Plays: "
                    f"{row['prior_regime_scrimmage_plays']}"
                ),
                (
                    f"- Prior Regime Pass Rate: "
                    f"{pct(row['prior_regime_pass_rate'])}"
                ),
                (
                    f"- Regime Evidence Confidence: "
                    f"{row['regime_evidence_confidence']}"
                ),
                "",
                (
                    "**Game vs Paper:** "
                    + row[
                        "paper_interpretation"
                    ]
                ),
                "",
                (
                    "**Game vs Regime:** "
                    + row[
                        "regime_interpretation"
                    ]
                ),
                "",
            ]
        )

    out_md.write_text(
        "\n".join(md)
        + "\n"
    )

    for row in outputs:
        print()
        print(
            f"{row['team']} | "
            f"{row['coach_identity']}"
        )

        print(
            f"  LIVE   : "
            f"{pct(row['live_pass_rate'])}"
        )

        print(
            f"  SAMPLE : "
            f"{row['live_scrimmage_plays']} plays | "
            f"{row['live_sample_confidence']}"
        )

        print(
            f"  PAPER  : "
            f"{pct(row['expected_pass_rate'])}"
        )

        print(
            f"  Δ PAPER: "
            f"{pp(row['paper_delta'])}"
        )

        print(
            f"  REGIME : "
            f"{row['prior_regime_games']} prior games | "
            f"{pct(row['prior_regime_pass_rate'])} | "
            f"{row['regime_evidence_confidence']}"
        )

        print(
            "  READ   : "
            + row[
                "paper_interpretation"
            ]
        )

        print(
            "  HISTORY: "
            + row[
                "regime_interpretation"
            ]
        )

    print()
    print(
        "JSON:",
        out_json,
    )

    print(
        "MD  :",
        out_md,
    )

    print("PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
