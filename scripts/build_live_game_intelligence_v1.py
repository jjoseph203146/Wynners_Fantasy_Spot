#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from live_ingest import fetch_json


ROOT = Path(__file__).resolve().parents[1]
NFL_DB = ROOT / "data" / "nfl.db"
EXPECTED_PASS = (
    ROOT
    / "processed"
    / "pregame_context_pass_expectation_v1.csv"
)
OUT_DIR = ROOT / "processed" / "live_game_intelligence"

VERSION = "WFS_GAME_INTELLIGENCE_V1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_summary(event_id: str) -> dict:
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/"
        f"football/nfl/summary?event={event_id}"
    )

    result = fetch_json(url)

    if isinstance(result, dict):
        return result

    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, dict):
                return item

    raise RuntimeError("ESPN summary payload not found")


def internal_game_id(event_id: str) -> str | None:
    if not NFL_DB.exists():
        return None

    try:
        with sqlite3.connect(
            f"file:{NFL_DB}?mode=ro",
            uri=True,
        ) as conn:
            row = conn.execute(
                """
                SELECT game_id
                FROM games
                WHERE CAST(espn AS TEXT) = ?
                LIMIT 1
                """,
                (str(event_id),),
            ).fetchone()

        return str(row[0]) if row else None

    except Exception:
        return None


def game_teams(data: dict) -> dict[str, dict]:
    out = {}

    header = data.get("header") or {}
    comps = header.get("competitions") or []

    if not comps:
        return out

    for c in comps[0].get("competitors") or []:
        team = c.get("team") or {}
        tid = str(team.get("id") or c.get("id") or "").strip()
        abbr = str(team.get("abbreviation") or "").strip().upper()

        if not tid or not abbr:
            continue

        out[tid] = {
            "team_id": tid,
            "abbr": abbr,
            "display_name": (
                team.get("displayName")
                or team.get("name")
                or abbr
            ),
            "home_away": c.get("homeAway"),
        }

    return out


def expected_pass_rates(
    game_id: str | None,
) -> dict[str, float]:
    out = {}

    if not game_id or not EXPECTED_PASS.exists():
        return out

    try:
        df = pd.read_csv(EXPECTED_PASS)

        needed = {
            "game_id",
            "team",
            "pregame_context_pass_expectation",
        }

        if not needed.issubset(df.columns):
            return out

        x = df[
            df["game_id"].astype(str).eq(str(game_id))
        ].copy()

        for _, row in x.iterrows():
            team = str(row["team"]).strip().upper()

            try:
                val = float(
                    row["pregame_context_pass_expectation"]
                )
            except Exception:
                continue

            # Artifact may store rate as 0-1 or 0-100.
            if val > 1.0:
                val /= 100.0

            out[team] = val

    except Exception:
        pass

    return out


def collect_drives(data: dict) -> list[dict]:
    drives = data.get("drives") or {}

    if not isinstance(drives, dict):
        return []

    items = []

    previous = drives.get("previous") or []
    if isinstance(previous, list):
        items.extend(previous)

    current = drives.get("current")
    if isinstance(current, dict):
        items.append(current)

    seen = set()
    clean = []

    for drive in items:
        did = str(drive.get("id") or "")
        if did and did in seen:
            continue

        if did:
            seen.add(did)

        clean.append(drive)

    return clean


def offense_team_id(play: dict) -> str | None:
    for tp in play.get("teamParticipants") or []:
        if str(tp.get("type") or "").lower() == "offense":
            tid = str(tp.get("id") or "").strip()
            if tid:
                return tid

    start = play.get("start") or {}
    team = start.get("team") or {}
    tid = str(team.get("id") or "").strip()

    return tid or None


def participant_name(
    play: dict,
    role: str,
) -> str | None:
    role = role.lower()

    for p in play.get("participants") or []:
        if str(p.get("type") or "").lower() != role:
            continue

        athlete = p.get("athlete") or {}

        return (
            athlete.get("displayName")
            or athlete.get("fullName")
            or athlete.get("shortName")
        )

    return None



def pbp_usage_name(
    play: dict,
    kind: str,
) -> str | None:
    """
    Deterministic fallback for ESPN plays whose participant
    arrays are absent/incomplete. No fuzzy identity matching.
    """
    text = str(play.get("text") or "").strip()

    if not text:
        return None

    if kind == "PASS":
        # Examples:
        # "pass short left to E.Egbuka ..."
        # "pass incomplete deep left to E.Egbuka ..."
        m = re.search(
            r"\bpass\b.*?\bto\s+"
            r"([A-Z][A-Za-z.'-]+"
            r"(?:\s+(?:Jr\.?|Sr\.?|II|III|IV))?)",
            text,
            flags=re.IGNORECASE,
        )

        if m:
            return m.group(1).strip()

        return None

    if kind == "RUSH":
        # Normal ESPN rush PBP begins with the ball carrier,
        # optionally after a formation marker.
        clean = re.sub(
            r"^\s*\([^)]*\)\s*",
            "",
            text,
        )

        # ESPN may prefix a rushing play with an eligible-lineman
        # announcement, e.g.:
        # "W.Milum reported in as eligible. Direct snap to C.Rodriguez..."
        # That player is not the ball carrier.
        while True:
            stripped = re.sub(
                r"^\s*[A-Z][A-Za-z.'-]+"
                r"(?:\s+(?:Jr\.?|Sr\.?|II|III|IV))?"
                r"\s+reported in as eligible\.\s*",
                "",
                clean,
                count=1,
                flags=re.IGNORECASE,
            )

            if stripped == clean:
                break

            clean = stripped

        # A direct-snap announcement may precede the actual rushing
        # sentence. Remove it so the next leading player is the carrier.
        clean = re.sub(
            r"^\s*Direct snap to\s+"
            r"[A-Z][A-Za-z.'-]+"
            r"(?:\s+(?:Jr\.?|Sr\.?|II|III|IV))?"
            r"\.\s*",
            "",
            clean,
            count=1,
            flags=re.IGNORECASE,
        )

        m = re.match(
            r"([A-Z][A-Za-z.'-]+"
            r"(?:\s+(?:Jr\.?|Sr\.?|II|III|IV))?)\s+",
            clean,
        )

        if m:
            name = m.group(1).strip()

            blocked = {
                "penalty",
                "timeout",
                "official",
                "end",
                "two-minute",
            }

            if name.lower() not in blocked:
                return name

    return None


def classify_play(play: dict) -> str:
    ptype = str(
        (play.get("type") or {}).get("text") or ""
    ).lower()

    text = str(play.get("text") or "")
    low = text.lower()

    if "no play" in low:
        return "NO_PLAY"

    if "timeout" in ptype or "timeout" in low:
        return "OTHER"

    if "kickoff" in ptype:
        return "SPECIAL"

    if "punt" in ptype:
        return "SPECIAL"

    if "field goal" in ptype:
        return "SPECIAL"

    if "extra point" in ptype:
        return "SPECIAL"

    if "two-point" in ptype or "two point" in ptype:
        return "SPECIAL"

    roles = {
        str(p.get("type") or "").lower()
        for p in play.get("participants") or []
    }

    if "passer" in roles:
        return "PASS"

    if "receiver" in roles:
        return "PASS"

    if "sack" in ptype:
        return "PASS"

    if "pass" in ptype:
        return "PASS"

    if re.search(r"\bpass\b", low):
        return "PASS"

    if "rusher" in roles:
        return "RUSH"

    if "rush" in ptype:
        return "RUSH"

    if (
        "left guard" in low
        or "right guard" in low
        or "left tackle" in low
        or "right tackle" in low
        or "left end" in low
        or "right end" in low
        or "up the middle" in low
    ):
        return "RUSH"

    return "OTHER"


def confidence(n: int) -> str:
    if n >= 40:
        return "HIGH"
    if n >= 20:
        return "MEDIUM"
    return "LOW"


def pct(n, d):
    if not d:
        return None
    return n / d


def fmt_pct(value):
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def deviation_label(delta):
    if delta is None:
        return "No paper baseline"

    if delta >= 0.10:
        return "Materially more pass-heavy than paper"

    if delta <= -0.10:
        return "Materially more run-heavy than paper"

    if delta >= 0.05:
        return "Moderately more pass-heavy than paper"

    if delta <= -0.05:
        return "Moderately more run-heavy than paper"

    return "Near pregame expectation"


def analyze_team(
    team_abbr: str,
    team_id: str,
    drives: list[dict],
    expected_pass_rate: float | None,
) -> dict:

    m = defaultdict(int)
    targets = Counter()
    carries = Counter()
    evidence_plays = []

    for drive in drives:
        drive_team = str(
            ((drive.get("team") or {}).get("id")) or ""
        ).strip()

        for play in drive.get("plays") or []:
            off_id = offense_team_id(play)

            if off_id != team_id:
                continue

            kind = classify_play(play)

            start = play.get("start") or {}
            down = int(start.get("down") or 0)
            distance = int(start.get("distance") or 0)
            yte = start.get("yardsToEndzone")

            try:
                yte = int(yte)
            except Exception:
                yte = None

            text = str(play.get("text") or "")

            if kind in {"PASS", "RUSH"}:
                m["scrimmage_plays"] += 1

                if kind == "PASS":
                    m["pass_calls"] += 1

                    receiver = participant_name(
                        play,
                        "receiver",
                    )
                    if not receiver:
                        receiver = pbp_usage_name(
                            play,
                            "PASS",
                        )
                    if receiver:
                        targets[receiver] += 1

                else:
                    m["rush_calls"] += 1

                    rusher = participant_name(
                        play,
                        "rusher",
                    )
                    if not rusher:
                        rusher = pbp_usage_name(
                            play,
                            "RUSH",
                        )
                    if rusher:
                        carries[rusher] += 1

                if down in {1, 2}:
                    m["early_down_plays"] += 1

                    if kind == "PASS":
                        m["early_down_pass"] += 1

                if yte is not None and yte <= 20:
                    m["red_zone_plays"] += 1

                    if kind == "PASS":
                        m["red_zone_pass"] += 1

                if down == 3:
                    m["third_down_plays"] += 1

                    if kind == "PASS":
                        m["third_down_pass"] += 1

                if down == 4:
                    m["fourth_down_scrimmage"] += 1

                if down in {3, 4} and 0 < distance <= 2:
                    m["short_yardage_plays"] += 1

                    if kind == "PASS":
                        m["short_yardage_pass"] += 1

                low = text.lower()

                if "(shotgun)" in low:
                    m["shotgun_plays"] += 1

                if "no huddle" in low:
                    m["no_huddle_plays"] += 1

                # WFS_LIVE_SCORE_CONTEXT_V1
                # Preserve ESPN's authoritative per-play score state.
                # Analysis-only: no score reconstruction or production influence.
                try:
                    away_score = int(
                        play.get("awayScore") or 0
                    )
                except (TypeError, ValueError):
                    away_score = None

                try:
                    home_score = int(
                        play.get("homeScore") or 0
                    )
                except (TypeError, ValueError):
                    home_score = None

                evidence_plays.append({
                    "play_id": str(
                        play.get("id") or ""
                    ),
                    "period": int(
                        (play.get("period") or {}).get(
                            "number"
                        )
                        or 0
                    ),
                    "clock": str(
                        (play.get("clock") or {}).get(
                            "displayValue"
                        )
                        or ""
                    ),
                    "down": down,
                    "distance": distance,
                    "yards_to_endzone": yte,
                    "away_score": away_score,
                    "home_score": home_score,
                    "call": kind,
                    "text": text,
                })

    live_pass_rate = pct(
        m["pass_calls"],
        m["scrimmage_plays"],
    )

    early_down_pass_rate = pct(
        m["early_down_pass"],
        m["early_down_plays"],
    )

    red_zone_pass_rate = pct(
        m["red_zone_pass"],
        m["red_zone_plays"],
    )

    third_down_pass_rate = pct(
        m["third_down_pass"],
        m["third_down_plays"],
    )

    short_yardage_pass_rate = pct(
        m["short_yardage_pass"],
        m["short_yardage_plays"],
    )

    shotgun_rate = pct(
        m["shotgun_plays"],
        m["scrimmage_plays"],
    )

    no_huddle_rate = pct(
        m["no_huddle_plays"],
        m["scrimmage_plays"],
    )

    delta = None

    if (
        live_pass_rate is not None
        and expected_pass_rate is not None
    ):
        delta = live_pass_rate - expected_pass_rate

    signals = []

    if live_pass_rate is not None:
        signals.append({
            "signal": "overall_pass_rate",
            "live_value": live_pass_rate,
            "paper_value": expected_pass_rate,
            "delta": delta,
            "sample_size": m["scrimmage_plays"],
            "confidence": confidence(
                m["scrimmage_plays"]
            ),
            "interpretation": deviation_label(delta),
        })

    if early_down_pass_rate is not None:
        signals.append({
            "signal": "early_down_pass_rate",
            "live_value": early_down_pass_rate,
            "sample_size": m["early_down_plays"],
            "confidence": confidence(
                m["early_down_plays"]
            ),
        })

    if red_zone_pass_rate is not None:
        signals.append({
            "signal": "red_zone_pass_rate",
            "live_value": red_zone_pass_rate,
            "sample_size": m["red_zone_plays"],
            "confidence": confidence(
                m["red_zone_plays"]
            ),
        })

    return {
        "team": team_abbr,
        "expected_pass_rate": expected_pass_rate,
        "live_pass_rate": live_pass_rate,
        "pass_rate_delta": delta,
        "interpretation": deviation_label(delta),

        "scrimmage_plays": m["scrimmage_plays"],
        "pass_calls": m["pass_calls"],
        "rush_calls": m["rush_calls"],

        "early_down_pass_rate": early_down_pass_rate,
        "red_zone_pass_rate": red_zone_pass_rate,
        "third_down_pass_rate": third_down_pass_rate,
        "short_yardage_pass_rate":
            short_yardage_pass_rate,

        "shotgun_rate": shotgun_rate,
        "no_huddle_rate": no_huddle_rate,

        "fourth_down_scrimmage":
            m["fourth_down_scrimmage"],

        "top_targets": targets.most_common(5),
        "top_carries": carries.most_common(5),

        "coaching_evidence": signals,
        "evidence_plays": evidence_plays,
    }


def markdown_report(
    event_id: str,
    game_id: str | None,
    team_results: list[dict],
) -> str:

    lines = [
        f"# Live Game Intelligence — {event_id}",
        "",
        f"- Version: `{VERSION}`",
        f"- Internal game: `{game_id or 'UNKNOWN'}`",
        f"- Generated UTC: `{utc_now()}`",
        "",
        "## Coaching / Game-Plan Read",
        "",
    ]

    for r in team_results:
        lines += [
            f"### {r['team']}",
            "",
            (
                f"**Live pass rate:** "
                f"{fmt_pct(r['live_pass_rate'])}"
            ),
            (
                f"**Pregame paper:** "
                f"{fmt_pct(r['expected_pass_rate'])}"
            ),
            (
                f"**Read:** {r['interpretation']}"
            ),
            "",
            (
                f"- Plays: {r['scrimmage_plays']} "
                f"({r['pass_calls']} pass / "
                f"{r['rush_calls']} rush)"
            ),
            (
                f"- Early-down pass rate: "
                f"{fmt_pct(r['early_down_pass_rate'])}"
            ),
            (
                f"- Red-zone pass rate: "
                f"{fmt_pct(r['red_zone_pass_rate'])}"
            ),
            (
                f"- Third-down pass rate: "
                f"{fmt_pct(r['third_down_pass_rate'])}"
            ),
            (
                f"- Short-yardage pass rate: "
                f"{fmt_pct(r['short_yardage_pass_rate'])}"
            ),
            (
                f"- Shotgun: "
                f"{fmt_pct(r['shotgun_rate'])}"
            ),
            (
                f"- No-huddle: "
                f"{fmt_pct(r['no_huddle_rate'])}"
            ),
            (
                f"- Fourth-down scrimmage calls: "
                f"{r['fourth_down_scrimmage']}"
            ),
            "",
        ]

        if r["top_targets"]:
            target_text = ", ".join(
                f"{name} ({n})"
                for name, n in r["top_targets"]
            )
            lines.append(
                f"**Target concentration:** {target_text}"
            )
            lines.append("")

        if r["top_carries"]:
            carry_text = ", ".join(
                f"{name} ({n})"
                for name, n in r["top_carries"]
            )
            lines.append(
                f"**Carry concentration:** {carry_text}"
            )
            lines.append("")

    lines += [
        "## Model Policy",
        "",
        (
            "This report is analysis-only. Live PBP evidence "
            "does not directly mutate production forecasts, "
            "the DFS solver, or the persistent coaching prior."
        ),
        "",
        (
            "Postgame coaching updates should consume this "
            "evidence with sample-size and historical "
            "confidence weighting."
        ),
        "",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event-id",
        required=True,
    )
    args = parser.parse_args()

    event_id = str(args.event_id).strip()

    data = fetch_summary(event_id)
    teams = game_teams(data)
    drives = collect_drives(data)

    if len(teams) != 2:
        raise SystemExit(
            f"FAIL: expected 2 teams, got {teams}"
        )

    if not drives:
        raise SystemExit("FAIL: no ESPN drives")

    game_id = internal_game_id(event_id)

    paper_rates = expected_pass_rates(game_id)

    results = []

    for tid, info in teams.items():
        abbr = info["abbr"]

        results.append(
            analyze_team(
                team_abbr=abbr,
                team_id=tid,
                drives=drives,
                expected_pass_rate=paper_rates.get(abbr),
            )
        )

    payload = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "event_id": event_id,
        "game_id": game_id,
        "teams": results,
    }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = OUT_DIR / f"{event_id}.json"
    md_path = OUT_DIR / f"{event_id}.md"

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    md_path.write_text(
        markdown_report(
            event_id,
            game_id,
            results,
        )
        + "\n"
    )

    print("=" * 72)
    print("WFS GAME INTELLIGENCE V1")
    print("=" * 72)
    print("EVENT:", event_id)
    print("GAME :", game_id)
    print()

    for r in results:
        print(
            r["team"],
            "| paper",
            fmt_pct(r["expected_pass_rate"]),
            "| live",
            fmt_pct(r["live_pass_rate"]),
            "|",
            r["interpretation"],
        )

    print()
    print("JSON:", json_path)
    print("PAPER:", md_path)
    print("PASS")


if __name__ == "__main__":
    main()
