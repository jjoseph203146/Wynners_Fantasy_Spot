#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1E-A
Data Center postgame integration architecture audit — READ ONLY.

Reads app.py and the three postgame databases only.
Writes nothing. Makes no service/cron changes.
"""

from pathlib import Path
import hashlib
import sqlite3
import re

ROOT = Path("/home/mwynn/nfl_data_engine")
APP = ROOT / "app.py"
DBS = {
    "NFL_DB": ROOT / "data" / "nfl.db",
    "FORECAST_LEDGER_DB": ROOT / "data" / "forecast_ledger.db",
    "PLAYER_PROJECTION_LEDGER_DB": ROOT / "data" / "player_projection_ledger.db",
}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ro(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c

def show_range(lines, start, end, label):
    print(f"\n=== {label}: app.py {start}-{end} ===")
    for n in range(start, min(end, len(lines)) + 1):
        print(f"{n:05d}: {lines[n-1]}")

def main():
    print("=" * 76)
    print("WFS NFL — NFL-POSTGAME-1E-A")
    print("DATA CENTER POSTGAME INTEGRATION ARCHITECTURE AUDIT — READ ONLY")
    print("=" * 76)

    if not APP.exists():
        raise RuntimeError("app.py missing")

    print(f"APP_SHA256={sha256(APP)}")
    lines = APP.read_text(errors="replace").splitlines()
    print(f"APP_LINES={len(lines)}")

    # Exact regions needed to design the integration without guessing.
    show_range(lines, 2640, 2990, "DATA CENTER LOADERS / HELPERS")
    show_range(lines, 3035, 3335, "DATA CENTER ENTRY / HEADER / CONTROLS")
    show_range(lines, 3336, 3700, "DATA CENTER BODY CONTINUATION")
    show_range(lines, 10190, 10275, "ADMIN DIAGNOSTICS BOUNDARY")
    show_range(lines, 11060, 11140, "SIDEBAR / ADMIN MODE AUTHORITY")
    show_range(lines, 11570, 11610, "DATA CENTER ROUTING")

    print("\n=== RELEVANT FUNCTION DEFINITIONS ===")
    rx = re.compile(
        r"^def\s+([A-Za-z0-9_]*(?:data_center|forecast|admin|postgame|grade|accuracy)[A-Za-z0-9_]*)\s*\("
    )
    for i, line in enumerate(lines, 1):
        m = rx.match(line)
        if m:
            print(f"{i}:{m.group(1)}")

    print("\n=== DATABASE READ-ONLY PROOF / REQUIRED TABLES ===")
    required = {
        "NFL_DB": ["games", "player_game_stats"],
        "FORECAST_LEDGER_DB": ["forecast_predictions"],
        "PLAYER_PROJECTION_LEDGER_DB": [
            "player_projection_snapshots",
            "player_projection_predictions",
        ],
    }

    for label, path in DBS.items():
        if not path.exists():
            raise RuntimeError(f"{label} missing: {path}")
        with ro(path) as conn:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            missing = [t for t in required[label] if t not in tables]
            print(f"{label}_INTEGRITY={integrity}")
            print(f"{label}_MISSING_REQUIRED_TABLES={len(missing)}")
            if missing:
                print(f"{label}_MISSING={','.join(missing)}")
            if integrity != "ok" or missing:
                raise RuntimeError(f"{label} contract failure")

    print("\n=== INTEGRATION CONTRACT ===")
    print("TARGET_PAGE=NFL Data Center")
    print("TARGET_VISIBILITY=Admin only")
    print("GAME_GRADER=nfl_postgame_grader_r2.py")
    print("PLAYER_GRADER=nfl_postgame_player_grader.py")
    print("GAME_JOIN=exact game_id + latest eligible pregame forecast")
    print("PLAYER_JOIN=exact game_id + exact GSIS player_id + latest eligible pregame snapshot")
    print("DST_GRADING=EXCLUDED")
    print("PUBLIC_SOLVER_PATH_CHANGE=FORBIDDEN")
    print("UPDATER_CHANGE=FORBIDDEN")
    print("LIVE_CHANGE=FORBIDDEN")
    print("INJURY_PIPELINE_CHANGE=FORBIDDEN")
    print("CRON_CHANGE=FORBIDDEN")

    print("\nNFL_POSTGAME_1E_A_STATUS=PASS")
    print("READ_ONLY_AUDIT=TRUE")
    print("DATABASE_WRITES=0")
    print("APP_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")

if __name__ == "__main__":
    main()
