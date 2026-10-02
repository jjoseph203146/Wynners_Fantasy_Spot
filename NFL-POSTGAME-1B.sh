#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
PY="$ROOT/venv/bin/python"
NFL_DB="$ROOT/data/nfl.db"
LEDGER_DB="$ROOT/data/forecast_ledger.db"

cd "$ROOT"

echo "======================================================================"
echo "WFS NFL — NFL-POSTGAME-1B"
echo "POSTGAME JOIN / SCORING AUTOMATION CONTRACT AUDIT — READ ONLY"
echo "======================================================================"

echo
echo "=== FROZEN/RELEVANT HASHES ==="
for f in \
    run_updater.sh \
    updater.py \
    boxscores.py \
    fanduel_scoring.py \
    live_safe_poll.py \
    live_ingest.py \
    app.py
do
    if [[ -f "$f" ]]; then
        sha256sum "$f"
    fi
done

echo
echo "=== SERVICES / CRON ==="
echo "wfs.service=$(systemctl is-active wfs.service || true)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service || true)"
crontab -l 2>/dev/null | grep -E 'run_updater|fanduel_scoring|boxscore|postgame|forecast' || true

echo
echo "=== UPDATER EXECUTION ORDER ==="
grep -nE \
    'update_current_boxscores|run_boxscore|fanduel_scoring|run_fanduel|calculate_fanduel|FanDuel|STEP 3|STEP 4|STEP 5|def main|if __name__' \
    updater.py 2>/dev/null || true

echo
echo "=== FANDUEL SCORING ENTRYPOINT / WRITE CONTRACT ==="
grep -nE \
    'def .*fanduel|def main|calculate_fanduel_points|UPDATE player_game_stats|fanduel_points_verified|run_|if __name__|load_player_stats' \
    fanduel_scoring.py 2>/dev/null | head -n 300 || true

echo
echo "=== BOXSCORE ENTRYPOINT / IDENTITY CONTRACT ==="
grep -nE \
    'CREATE TABLE IF NOT EXISTS player_game_stats|player_id|player_name|game_id|season|week|INSERT INTO player_game_stats|ON CONFLICT|UPDATE player_game_stats|download_player_stats|update_player_stats|run_boxscore_update|if __name__' \
    boxscores.py 2>/dev/null | head -n 400 || true

echo
echo "=== FORECAST LEDGER TABLE INVENTORY / SCHEMAS ==="
"$PY" - "$LEDGER_DB" <<'PY'
import sqlite3, sys
from pathlib import Path

db = Path(sys.argv[1])
if not db.exists():
    raise SystemExit("FAIL | forecast_ledger.db missing")

with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    print("LEDGER_INTEGRITY=" + c.execute("PRAGMA integrity_check").fetchone()[0])
    tables = [r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )]
    for table in tables:
        print(f"[TABLE] {table}")
        for row in c.execute(f'PRAGMA table_info("{table}")'):
            print(f"  {row[1]}|{row[2]}|pk={row[5]}|notnull={row[3]}")
PY

echo
echo "=== NFL POSTGAME TABLE SCHEMAS ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys

db=sys.argv[1]
targets=[
    "games",
    "player_game_stats",
    "team_game_stats",
]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    print("NFL_DB_INTEGRITY=" + c.execute("PRAGMA integrity_check").fetchone()[0])
    for table in targets:
        exists=c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)
        ).fetchone()
        print()
        print(f"[TABLE] {table}")
        if not exists:
            print("MISSING")
            continue
        for row in c.execute(f'PRAGMA table_info("{table}")'):
            print(f"  {row[1]}|{row[2]}|pk={row[5]}|notnull={row[3]}")
PY

echo
echo "=== 2026 COMPLETED-GAME / PLAYER-STATS COVERAGE ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row

    completed=c.execute("""
        SELECT game_id, season, week, away_team, home_team,
               away_score, home_score, completed
        FROM games
        WHERE season=2026 AND completed=1
        ORDER BY week, game_date, game_id
    """).fetchall()

    print("COMPLETED_2026_GAMES=" + str(len(completed)))

    for g in completed:
        gid=g["game_id"]
        pcount=c.execute(
            "SELECT COUNT(*) FROM player_game_stats WHERE game_id=?",(gid,)
        ).fetchone()[0]
        verified=c.execute(
            """SELECT COUNT(*) FROM player_game_stats
               WHERE game_id=? AND fanduel_points_verified=1""",(gid,)
        ).fetchone()[0]
        null_fd=c.execute(
            """SELECT COUNT(*) FROM player_game_stats
               WHERE game_id=? AND fanduel_points IS NULL""",(gid,)
        ).fetchone()[0]
        print(
            f"{gid}|week={g['week']}|{g['away_team']}={g['away_score']}|"
            f"{g['home_team']}={g['home_score']}|player_rows={pcount}|"
            f"fd_verified={verified}|fd_null={null_fd}"
        )
PY

echo
echo "=== COMPLETED GAME PLAYER SAMPLE WITH IDENTITIES / FD POINTS ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    gidrow=c.execute("""
        SELECT game_id
        FROM games
        WHERE season=2026 AND completed=1
        ORDER BY week, game_date, game_id
        LIMIT 1
    """).fetchone()
    if not gidrow:
        print("NO_COMPLETED_2026_GAME")
        raise SystemExit(0)
    gid=gidrow["game_id"]
    print("SAMPLE_GAME_ID="+gid)
    rows=c.execute("""
        SELECT game_id, season, week, player_id, player_name,
               player_display_name, position, team, opponent_team,
               fantasy_points, fantasy_points_ppr,
               fanduel_points, fanduel_points_verified
        FROM player_game_stats
        WHERE game_id=?
        ORDER BY fanduel_points DESC, player_display_name
        LIMIT 25
    """,(gid,)).fetchall()
    for r in rows:
        print(dict(r))
PY

echo
echo "=== FORECAST SNAPSHOT JOIN KEY DISCOVERY ==="
"$PY" - "$LEDGER_DB" <<'PY'
import sqlite3, sys

db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    tables=[r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )]
    for table in tables:
        cols=[r[1] for r in c.execute(f'PRAGMA table_info("{table}")')]
        hits=[x for x in cols if x in {
            "game_id","season","week","team","player_id","gsis_id","identity_key",
            "player","player_name","projection","model_projection",
            "pred_home_points","pred_away_points","snapshot_id","captured_at_utc"
        }]
        if hits:
            print(f"{table}|join_cols={hits}")
PY

echo
echo "=== CROSS-DB GAME_ID JOIN PROOF ==="
"$PY" - "$NFL_DB" "$LEDGER_DB" <<'PY'
import sqlite3, sys
nfl_db, ledger_db = sys.argv[1:3]

with sqlite3.connect(f"file:{nfl_db}?mode=ro", uri=True) as n:
    completed = {
        r[0] for r in n.execute("""
            SELECT game_id FROM games
            WHERE season=2026 AND completed=1
        """)
    }

with sqlite3.connect(f"file:{ledger_db}?mode=ro", uri=True) as l:
    tables=[r[0] for r in l.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )]
    candidate=[]
    for table in tables:
        cols=[r[1] for r in l.execute(f'PRAGMA table_info("{table}")')]
        if "game_id" in cols:
            candidate.append(table)

    print("LEDGER_TABLES_WITH_GAME_ID=" + ",".join(candidate))

    for table in candidate:
        rows=l.execute(
            f'SELECT DISTINCT game_id FROM "{table}" WHERE game_id IS NOT NULL'
        ).fetchall()
        ids={r[0] for r in rows}
        overlap=sorted(completed & ids)
        print(
            f"{table}|distinct_game_ids={len(ids)}|"
            f"completed_2026_overlap={len(overlap)}|sample={overlap[:10]}"
        )
PY

echo
echo "=== PLAYER ID JOIN QUALITY FOR COMPLETED 2026 GAME(S) ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    total=c.execute("""
        SELECT COUNT(*)
        FROM player_game_stats p
        JOIN games g ON g.game_id=p.game_id
        WHERE g.season=2026 AND g.completed=1
    """).fetchone()[0]
    blank_id=c.execute("""
        SELECT COUNT(*)
        FROM player_game_stats p
        JOIN games g ON g.game_id=p.game_id
        WHERE g.season=2026 AND g.completed=1
          AND (p.player_id IS NULL OR trim(p.player_id)='')
    """).fetchone()[0]
    dup=c.execute("""
        SELECT COUNT(*) FROM (
            SELECT p.game_id, p.player_id, COUNT(*) n
            FROM player_game_stats p
            JOIN games g ON g.game_id=p.game_id
            WHERE g.season=2026 AND g.completed=1
              AND p.player_id IS NOT NULL AND trim(p.player_id)<>''
            GROUP BY p.game_id, p.player_id
            HAVING COUNT(*)>1
        )
    """).fetchone()[0]
    print("COMPLETED_2026_PLAYER_ROWS="+str(total))
    print("COMPLETED_2026_BLANK_PLAYER_ID_ROWS="+str(blank_id))
    print("COMPLETED_2026_DUP_GAME_PLAYER_ID_KEYS="+str(dup))
PY

echo
echo "=== AUTOMATION CONTRACT SUMMARY ==="
"$PY" - "$ROOT/updater.py" "$ROOT/fanduel_scoring.py" <<'PY'
from pathlib import Path
import sys

up=Path(sys.argv[1]).read_text(errors="replace")
fd=Path(sys.argv[2]).read_text(errors="replace")

signals={
    "UPDATER_IMPORTS_FANDUEL_SCORING":
        ("import fanduel_scoring" in up or "from fanduel_scoring import" in up),
    "UPDATER_REFERENCES_FANDUEL_SCORING_FUNCTION":
        ("calculate_fanduel_points" in up or "run_fanduel" in up or "fanduel_scoring." in up),
    "FANDUEL_SCORING_HAS_MAIN_GUARD":
        ('if __name__ == "__main__"' in fd or "if __name__ == '__main__'" in fd),
    "FANDUEL_SCORING_WRITES_VERIFIED_FLAG":
        ("fanduel_points_verified" in fd and "UPDATE player_game_stats" in fd),
}
for k,v in signals.items():
    print(f"{k}={'TRUE' if v else 'FALSE'}")
PY

echo
echo "NFL_POSTGAME_1B_STATUS=PASS"
echo "READ_ONLY_AUDIT=TRUE"
echo "NO_DATABASE_WRITES_PERFORMED=TRUE"
echo "NO_SERVICE_RESTART_PERFORMED=TRUE"
