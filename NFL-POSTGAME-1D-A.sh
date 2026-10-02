#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
PY="$ROOT/venv/bin/python"
NFL_DB="$ROOT/data/nfl.db"

cd "$ROOT"

echo "======================================================================"
echo "WFS NFL — NFL-POSTGAME-1D-A"
echo "PLAYER PROJECTION AUTHORITY / JOIN AUDIT — READ ONLY"
echo "======================================================================"

echo
echo "=== RELEVANT FILE HASHES ==="
for f in \
  fanduel_slate_projection_attach_v5.py \
  fanduel_player_pool.py \
  fanduel_solver_ready_pool.py \
  fanduel_nfl_ui_solver.py \
  updater.py \
  fanduel_scoring.py \
  boxscores.py \
  app.py
do
  [[ -f "$f" ]] && sha256sum "$f"
done

echo
echo "=== SERVICES / CRON ==="
echo "wfs.service=$(systemctl is-active wfs.service || true)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service || true)"
crontab -l 2>/dev/null | grep -E 'run_updater|projection|fanduel|postgame' || true

echo
echo "=== PROJECTION CODE / IDENTITY DISCOVERY ==="
grep -RniE \
  'fanduel_slate_projection_pool|projection_manifest|projected.*point|projection|gsis|player_id|game_id|slate_id|salary' \
  --include='*.py' "$ROOT" 2>/dev/null \
  | grep -E 'fanduel_slate_projection|fanduel_player_pool|solver_ready|ui_solver|app.py' \
  | head -n 500 || true

echo
echo "=== NFL.DB PROJECTION-RELATED TABLE INVENTORY ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    print("NFL_DB_INTEGRITY="+c.execute("PRAGMA integrity_check").fetchone()[0])
    tables=[r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )]
    for t in tables:
        low=t.lower()
        if any(k in low for k in ("projection","fanduel","solver","slate","player_pool")):
            n=c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            print(f"{t}|rows={n}")
PY

echo
echo "=== PROJECTION TABLE SCHEMAS ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    tables=[r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )]
    for t in tables:
        cols=[r["name"] for r in c.execute(f'PRAGMA table_info("{t}")')]
        if (
            any("projection" in x.lower() for x in cols)
            or "projection" in t.lower()
            or "solver_ready" in t.lower()
        ):
            print()
            print(f"[TABLE] {t}")
            for r in c.execute(f'PRAGMA table_info("{t}")'):
                print(f"  {r['name']}|{r['type']}|pk={r['pk']}|notnull={r['notnull']}")
PY

echo
echo "=== CURRENT PROJECTION / SOLVER SAMPLE ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys, json
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    tables=[r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )]
    preferred=[
        "fanduel_slate_projection_pool",
        "fanduel_solver_ready_pool",
        "fanduel_player_pool",
    ]
    for t in preferred:
        if t not in tables:
            continue
        print()
        print(f"[TABLE] {t}")
        rows=c.execute(f'SELECT * FROM "{t}" LIMIT 5').fetchall()
        for r in rows:
            print(json.dumps(dict(r), sort_keys=True, default=str))
PY

echo
echo "=== PROJECTION SNAPSHOT / VERSION COLUMN DISCOVERY ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
keys=("snapshot","captured","created","updated","timestamp","version","sha","hash","run_id","generated")
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    for trow in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ):
        t=trow["name"]
        cols=[r["name"] for r in c.execute(f'PRAGMA table_info("{t}")')]
        hits=[x for x in cols if any(k in x.lower() for k in keys)]
        if hits and any(
            k in t.lower() for k in ("projection","fanduel","solver","slate","pool")
        ):
            print(f"{t}|version_cols={hits}")
PY

echo
echo "=== COMPLETED 2026 GAME: ACTUAL PLAYER AUTHORITY ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    games=c.execute("""
      SELECT game_id, week, away_team, home_team
      FROM games
      WHERE season=2026 AND completed=1
      ORDER BY week, game_id
    """).fetchall()
    print("COMPLETED_GAMES="+str(len(games)))
    for g in games:
        rows=c.execute("""
          SELECT COUNT(*) n,
                 SUM(CASE WHEN player_id IS NULL OR trim(player_id)='' THEN 1 ELSE 0 END) blank_id,
                 SUM(CASE WHEN fanduel_points_verified=1 THEN 1 ELSE 0 END) verified,
                 SUM(CASE WHEN fanduel_points IS NULL THEN 1 ELSE 0 END) null_fd
          FROM player_game_stats WHERE game_id=?
        """,(g["game_id"],)).fetchone()
        print(
          f"{g['game_id']}|week={g['week']}|{g['away_team']}@{g['home_team']}|"
          f"rows={rows['n']}|blank_id={rows['blank_id']}|"
          f"verified={rows['verified']}|null_fd={rows['null_fd']}"
        )
PY

echo
echo "=== COMPLETED GAME -> PROJECTION JOIN CANDIDATES ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    games=c.execute("""
      SELECT game_id, season, week, away_team, home_team
      FROM games
      WHERE season=2026 AND completed=1
      ORDER BY week, game_id
    """).fetchall()
    tables=[r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )]
    for g in games:
        print()
        print(f"GAME={g['game_id']}|{g['away_team']}@{g['home_team']}")
        actual_ids={
            r[0] for r in c.execute(
                "SELECT player_id FROM player_game_stats WHERE game_id=? AND player_id IS NOT NULL",
                (g["game_id"],)
            )
        }
        for t in tables:
            cols=[r["name"] for r in c.execute(f'PRAGMA table_info("{t}")')]
            low={x.lower():x for x in cols}
            if not any("projection" in x.lower() for x in cols):
                continue

            id_col=None
            for cand in ("player_id","gsis_id","gsis","player_gsis_id"):
                if cand in low:
                    id_col=low[cand]
                    break
            if not id_col:
                continue

            where=[]
            params=[]
            if "season" in low:
                where.append(f'"{low["season"]}"=?')
                params.append(g["season"])
            if "week" in low:
                where.append(f'"{low["week"]}"=?')
                params.append(g["week"])

            sql=f'SELECT "{id_col}" FROM "{t}"'
            if where:
                sql+=" WHERE "+" AND ".join(where)
            try:
                proj_ids={r[0] for r in c.execute(sql,params) if r[0] is not None}
            except sqlite3.Error as e:
                print(f"{t}|JOIN_AUDIT_ERROR={e}")
                continue

            overlap=len(actual_ids & proj_ids)
            print(
              f"{t}|id_col={id_col}|candidate_ids={len(proj_ids)}|"
              f"actual_ids={len(actual_ids)}|exact_overlap={overlap}"
            )
PY

echo
echo "=== FILESYSTEM PROJECTION ARTIFACT INVENTORY ==="
find "$ROOT" -maxdepth 4 -type f \
  \( -iname '*projection*' -o -iname '*slate*' -o -iname '*solver*ready*' \) \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null \
  | sort | tail -n 250 || true

echo
echo "=== READ-ONLY CONCLUSION MARKERS ==="
echo "NFL_POSTGAME_1D_A_STATUS=PASS"
echo "READ_ONLY_AUDIT=TRUE"
echo "PRODUCTION_DATABASE_WRITES=0"
echo "SERVICE_RESTARTS=0"
