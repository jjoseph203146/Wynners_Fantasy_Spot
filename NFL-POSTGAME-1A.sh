#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
DB="$ROOT/data/nfl.db"
LIVE_DB="$ROOT/data/wfs_live.db"
LEDGER_DB="$ROOT/data/forecast_ledger.db"
PY="$ROOT/venv/bin/python"

cd "$ROOT"

echo "======================================================================"
echo "WFS NFL — NFL-POSTGAME-1A"
echo "COMPLETED-GAME / POSTGAME ARCHITECTURE AUDIT — READ ONLY"
echo "======================================================================"

echo
echo "=== SERVICES / CRON ==="
echo "wfs.service=$(systemctl is-active wfs.service || true)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service || true)"
crontab -l 2>/dev/null | grep -E 'nfl_data_engine|wfs|updater' || true

echo
echo "=== RELEVANT FILE HASHES ==="
for f in \
    live_ingest.py \
    live_safe_poll.py \
    live_orchestrator.py \
    live_concurrency_gate.py \
    updater.py \
    database.py \
    context_data.py \
    fanduel_scoring.py \
    app.py
do
    if [[ -f "$f" ]]; then
        sha256sum "$f"
    fi
done

echo
echo "=== POSTGAME / FINAL / SCORE / STATS CODE DISCOVERY ==="
grep -RniE \
    --include='*.py' \
    --exclude-dir='venv' \
    --exclude-dir='backups' \
    --exclude-dir='__pycache__' \
    'postgame|post_game|game.*final|final.*game|game_status|status.*final|final_score|home_score|away_score|player.*stat|boxscore|box_score|fantasy.*point|fanduel.*point|actual.*point|projection.*actual|actual.*projection|completed.*game' \
    "$ROOT" 2>/dev/null | head -n 500 || true

echo
echo "=== LIVE SOURCE TARGETS / TABLE NAMES ==="
grep -nE \
    'nfl\.db|wfs_live\.db|forecast_ledger|CREATE TABLE|INSERT INTO|UPDATE |DELETE FROM|to_sql|game_status|status|score|stat|final|complete|gsis|player_id|game_id' \
    live_ingest.py live_orchestrator.py live_concurrency_gate.py live_safe_poll.py \
    2>/dev/null | head -n 700 || true

echo
echo "=== DATABASE TABLE INVENTORY ==="
"$PY" - "$DB" "$LIVE_DB" "$LEDGER_DB" <<'PY'
import sqlite3
import sys
from pathlib import Path

for db in map(Path, sys.argv[1:]):
    print()
    print(f"--- {db} ---")
    if not db.exists():
        print("MISSING")
        continue
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        print("integrity_check=" + str(c.execute("PRAGMA integrity_check").fetchone()[0]))
        tables = [
            r[0] for r in c.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' ORDER BY name"
            )
        ]
        print("TABLE_COUNT=" + str(len(tables)))
        for table in tables:
            try:
                n = c.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            except Exception as e:
                n = f"ERROR:{e}"
            print(f"{table}|rows={n}")
PY

echo
echo "=== RELEVANT TABLE SCHEMAS ==="
"$PY" - "$DB" "$LIVE_DB" "$LEDGER_DB" <<'PY'
import sqlite3
import sys
from pathlib import Path

keywords = (
    "game", "live", "player", "stat", "score", "box",
    "fantasy", "projection", "forecast", "result",
    "snap", "weekly", "play"
)

for db in map(Path, sys.argv[1:]):
    print()
    print(f"--- {db} ---")
    if not db.exists():
        continue
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        tables = [
            r[0] for r in c.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' ORDER BY name"
            )
        ]
        for table in tables:
            low = table.lower()
            if not any(k in low for k in keywords):
                continue
            print()
            print(f"[TABLE] {table}")
            for row in c.execute(f'PRAGMA table_info("{table}")'):
                print(
                    f"  cid={row[0]} name={row[1]} type={row[2]} "
                    f"notnull={row[3]} pk={row[5]}"
                )
PY

echo
echo "=== FINAL/COMPLETED GAME VALUE DISCOVERY ==="
"$PY" - "$DB" "$LIVE_DB" <<'PY'
import sqlite3
import sys
from pathlib import Path

candidate_cols = {
    "status", "game_status", "state", "game_state",
    "phase", "game_phase", "completed", "is_final",
    "home_score", "away_score", "score_home", "score_away"
}

for db in map(Path, sys.argv[1:]):
    print()
    print(f"--- {db} ---")
    if not db.exists():
        continue
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        tables = [
            r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        for table in tables:
            cols = [
                r[1] for r in c.execute(f'PRAGMA table_info("{table}")')
            ]
            hits = [x for x in cols if x.lower() in candidate_cols]
            if not hits:
                continue
            print(f"[TABLE] {table} candidate_columns={hits}")
            for col in hits:
                try:
                    vals = c.execute(
                        f'SELECT "{col}", COUNT(*) FROM "{table}" '
                        f'GROUP BY "{col}" ORDER BY COUNT(*) DESC LIMIT 20'
                    ).fetchall()
                    print(f"  {col}={vals}")
                except Exception as e:
                    print(f"  {col}=ERROR:{e}")
PY

echo
echo "=== RECENT GAME / LIVE ROW SAMPLES ==="
"$PY" - "$DB" "$LIVE_DB" <<'PY'
import sqlite3
import sys
from pathlib import Path

preferred = (
    "games",
    "schedules",
    "schedule",
    "live_games",
    "live_game_state",
    "game_state",
    "player_game_stats",
    "player_stats",
    "weekly_stats",
    "player_weekly_stats",
    "pbp",
    "play_by_play",
)

for db in map(Path, sys.argv[1:]):
    print()
    print(f"--- {db} ---")
    if not db.exists():
        continue
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        c.row_factory = sqlite3.Row
        tables = {
            r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        selected = [t for t in preferred if t in tables]
        if not selected:
            selected = [
                t for t in sorted(tables)
                if any(k in t.lower() for k in ("game", "live", "stat", "play"))
            ][:12]

        for table in selected:
            print()
            print(f"[SAMPLE] {table}")
            cols = [
                r[1] for r in c.execute(f'PRAGMA table_info("{table}")')
            ]
            order_col = next(
                (
                    x for x in (
                        "updated_at", "source_timestamp", "timestamp",
                        "game_date", "date", "season", "week", "game_id"
                    )
                    if x in cols
                ),
                None,
            )
            sql = f'SELECT * FROM "{table}"'
            if order_col:
                sql += f' ORDER BY "{order_col}" DESC'
            sql += " LIMIT 5"
            try:
                for row in c.execute(sql):
                    d = dict(row)
                    print(d)
            except Exception as e:
                print("ERROR:", e)
PY

echo
echo "=== IDENTITY CONTRACT DISCOVERY ==="
grep -RniE \
    --include='*.py' \
    --exclude-dir='venv' \
    --exclude-dir='backups' \
    --exclude-dir='__pycache__' \
    'gsis_id|player_id|identity_key|game_id|normalize.*name|name.*normalize' \
    live_ingest.py live_orchestrator.py updater.py context_data.py database.py \
    2>/dev/null | head -n 500 || true

echo
echo "=== DATABASE FILE STATE ==="
for db in "$DB" "$LIVE_DB" "$LEDGER_DB"; do
    [[ -f "$db" ]] || continue
    ls -lh "$db"
    ls -l "${db}-wal" "${db}-shm" "${db}-journal" 2>/dev/null || true
done

echo
echo "=== OPEN DATABASE HANDLES ==="
lsof "$DB" "$LIVE_DB" "$LEDGER_DB" 2>/dev/null || true

echo
echo "NFL_POSTGAME_1A_STATUS=PASS"
echo "READ_ONLY_AUDIT=TRUE"
echo "NO_DATABASE_WRITES_PERFORMED=TRUE"
echo "NO_SERVICE_RESTART_PERFORMED=TRUE"
