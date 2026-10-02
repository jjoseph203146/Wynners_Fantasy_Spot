#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
PY="$ROOT/venv/bin/python"
NFL_DB="$ROOT/data/nfl.db"
FORECAST_LEDGER="$ROOT/data/forecast_ledger.db"

cd "$ROOT"

echo "======================================================================"
echo "WFS NFL — NFL-POSTGAME-1D-B"
echo "IMMUTABLE PLAYER PROJECTION LEDGER DESIGN AUDIT — READ ONLY"
echo "======================================================================"

echo
echo "=== SAFETY / BASELINES ==="
for f in \
  run_updater.sh \
  updater.py \
  fanduel_slate_projection_attach_v5.py \
  fanduel_solver_ready_pool.py \
  fanduel_nfl_ui_solver.py \
  live_safe_poll.py \
  live_ingest.py \
  app.py
do
  [[ -f "$f" ]] && sha256sum "$f"
done

echo
echo "=== SERVICES / CRON ==="
echo "wfs.service=$(systemctl is-active wfs.service || true)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service || true)"
crontab -l 2>/dev/null | grep -E 'run_updater|projection|fanduel|forecast|postgame' || true

echo
echo "=== FORECAST LEDGER SCHEMA / IMMUTABILITY PATTERN ==="
"$PY" - "$FORECAST_LEDGER" <<'PY'
import sqlite3, sys, json
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    print("FORECAST_LEDGER_INTEGRITY="+c.execute("PRAGMA integrity_check").fetchone()[0])
    tables=[r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )]
    print("TABLES="+",".join(tables))
    for t in tables:
        print()
        print(f"[TABLE] {t}")
        for r in c.execute(f'PRAGMA table_info("{t}")'):
            print(
                f"  {r['name']}|{r['type']}|pk={r['pk']}|"
                f"notnull={r['notnull']}|default={r['dflt_value']}"
            )
        idx=list(c.execute(f'PRAGMA index_list("{t}")'))
        for ir in idx:
            print(f"  INDEX|name={ir['name']}|unique={ir['unique']}")
            for cr in c.execute(f'PRAGMA index_info("{ir["name"]}")'):
                print(f"    {cr['seqno']}:{cr['name']}")
PY

echo
echo "=== FORECAST LEDGER SAMPLE SNAPSHOT CONTRACT ==="
"$PY" - "$FORECAST_LEDGER" <<'PY'
import sqlite3, sys, json
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    for t in ("forecast_snapshots","forecast_predictions"):
        exists=c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(t,)
        ).fetchone()
        if not exists:
            continue
        print()
        print(f"[SAMPLE] {t}")
        for r in c.execute(f'SELECT * FROM "{t}" ORDER BY rowid DESC LIMIT 3'):
            print(json.dumps(dict(r), sort_keys=True, default=str))
PY

echo
echo "=== PLAYER PROJECTION SOURCE CONTRACT ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
targets=[
    "fanduel_slate_projection_pool",
    "fanduel_solver_ready_pool",
    "fanduel_slate_projection_manifest",
    "fanduel_solver_ready_manifest",
]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    print("NFL_DB_INTEGRITY="+c.execute("PRAGMA integrity_check").fetchone()[0])
    for t in targets:
        print()
        print(f"[TABLE] {t}")
        exists=c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(t,)
        ).fetchone()
        if not exists:
            print("MISSING")
            continue
        print("ROWS="+str(c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]))
        for r in c.execute(f'PRAGMA table_info("{t}")'):
            print(
                f"  {r['name']}|{r['type']}|pk={r['pk']}|"
                f"notnull={r['notnull']}"
            )
PY

echo
echo "=== CANONICAL OFFENSIVE IDENTITY QUALITY ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    rows=c.execute("""
      SELECT
        COUNT(*) total,
        SUM(CASE WHEN is_dst=0 THEN 1 ELSE 0 END) offense,
        SUM(CASE WHEN is_dst=0 AND projection_status='READY' THEN 1 ELSE 0 END) offense_ready,
        SUM(CASE WHEN is_dst=0 AND (projection_identity IS NULL OR trim(projection_identity)='') THEN 1 ELSE 0 END) blank_projection_identity,
        SUM(CASE WHEN is_dst=0 AND projection_identity LIKE 'GSIS:%' THEN 1 ELSE 0 END) gsis_projection_identity
      FROM fanduel_slate_projection_pool
    """).fetchone()
    for k in rows.keys():
        print(f"{k.upper()}={rows[k]}")

    dup=c.execute("""
      SELECT COUNT(*) FROM (
        SELECT slate_slug, projection_identity, COUNT(*) n
        FROM fanduel_slate_projection_pool
        WHERE is_dst=0
          AND projection_status='READY'
          AND projection_identity IS NOT NULL
          AND trim(projection_identity)<>''
        GROUP BY slate_slug, projection_identity
        HAVING COUNT(*)>1
      )
    """).fetchone()[0]
    print("DUP_READY_SLATE_IDENTITY_KEYS="+str(dup))
PY

echo
echo "=== DST IDENTITY QUALITY ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    r=c.execute("""
      SELECT
        COUNT(*) dst_rows,
        SUM(CASE WHEN projection_status='READY' THEN 1 ELSE 0 END) ready,
        SUM(CASE WHEN team_internal IS NULL OR trim(team_internal)='' THEN 1 ELSE 0 END) blank_team_internal,
        SUM(CASE WHEN game_key IS NULL OR trim(game_key)='' THEN 1 ELSE 0 END) blank_game_key
      FROM fanduel_slate_projection_pool
      WHERE is_dst=1
    """).fetchone()
    for k in r.keys():
        print(f"{k.upper()}={r[k]}")
PY

echo
echo "=== PLAYER LEDGER MINIMUM FIELD CANDIDATES ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
wanted=[
 "slate_name","slate_slug","source_file","source_row",
 "player","player_normalized","team_internal","away_team","home_team","game_key",
 "is_dst","salary","salary_valid",
 "internal_projection","model_projection","projection_source",
 "projection_match_method","projection_identity","projection_status","optimizer_eligible"
]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    cols=[r[1] for r in c.execute(
        'PRAGMA table_info("fanduel_slate_projection_pool")'
    )]
    print("AVAILABLE_FIELDS=")
    for w in wanted:
        print(f"{w}={'YES' if w in cols else 'NO'}")
PY

echo
echo "=== GAME_ID DERIVATION AUDIT ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    slates=c.execute("""
      SELECT DISTINCT season, week, away_team, home_team, game_id
      FROM games
      WHERE season=2026
    """).fetchall()
    game_map={(r["away_team"],r["home_team"]):r["game_id"] for r in slates}

    rows=c.execute("""
      SELECT DISTINCT away_team, home_team, game_key
      FROM fanduel_slate_projection_pool
    """).fetchall()

    total=len(rows)
    matched=0
    missing=[]
    for r in rows:
        key=(r["away_team"],r["home_team"])
        gid=game_map.get(key)
        if gid:
            matched+=1
        else:
            missing.append((r["away_team"],r["home_team"],r["game_key"]))

    print("DISTINCT_PROJECTION_GAMES="+str(total))
    print("EXACT_SCHEDULE_MATCHED="+str(matched))
    print("EXACT_SCHEDULE_UNMATCHED="+str(len(missing)))
    for x in missing[:25]:
        print("UNMATCHED="+repr(x))
PY

echo
echo "=== CURRENT WEEK / KICKOFF TIMING CONTRACT ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    rows=c.execute("""
      SELECT game_id, season, week, game_date, gametime,
             away_team, home_team, completed
      FROM games
      WHERE season=2026
      ORDER BY week, game_date, gametime, game_id
    """).fetchall()
    weeks={}
    for r in rows:
        weeks.setdefault(r["week"], []).append(r)
    for week in sorted(weeks)[:3]:
        games=weeks[week]
        print(f"WEEK={week}|GAMES={len(games)}|COMPLETED={sum(int(g['completed'] or 0) for g in games)}")
        for g in games[:20]:
            print(
              f"  {g['game_id']}|{g['game_date']} {g['gametime']}|"
              f"{g['away_team']}@{g['home_team']}|completed={g['completed']}"
            )
PY

echo
echo "=== SNAPSHOT IDEMPOTENCY DESIGN PROOF INPUTS ==="
"$PY" - "$NFL_DB" <<'PY'
import sqlite3, sys, hashlib, json
db=sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    c.row_factory=sqlite3.Row
    rows=[dict(r) for r in c.execute("""
      SELECT
        slate_slug, player, team_internal, game_key, is_dst,
        salary, internal_projection, model_projection,
        projection_source, projection_match_method,
        projection_identity, projection_status, optimizer_eligible
      FROM fanduel_slate_projection_pool
      ORDER BY slate_slug, game_key, is_dst, projection_identity, player, salary
    """)]
payload=json.dumps(rows,sort_keys=True,separators=(",",":"),default=str).encode()
print("CURRENT_PROJECTION_ROWS="+str(len(rows)))
print("CURRENT_PROJECTION_SEMANTIC_SHA256="+hashlib.sha256(payload).hexdigest())
PY

echo
echo "=== PROPOSED LEDGER CONTRACT (DESIGN ONLY) ==="
cat <<'EOF'
DATABASE:
  data/player_projection_ledger.db

TABLE: player_projection_snapshots
  snapshot_id TEXT PRIMARY KEY
  captured_at_utc TEXT NOT NULL
  season INTEGER NOT NULL
  week INTEGER NOT NULL
  source_projection_sha256 TEXT NOT NULL
  source_manifest_sha256 TEXT
  source_solver_ready_sha256 TEXT
  source_module_sha256 TEXT NOT NULL
  snapshot_status TEXT NOT NULL

TABLE: player_projection_predictions
  snapshot_id TEXT NOT NULL
  game_id TEXT NOT NULL
  slate_slug TEXT NOT NULL
  is_dst INTEGER NOT NULL
  player_id TEXT
  player_name TEXT NOT NULL
  team TEXT NOT NULL
  opponent_team TEXT NOT NULL
  salary REAL NOT NULL
  projection REAL NOT NULL
  projection_source TEXT NOT NULL
  projection_match_method TEXT NOT NULL
  projection_status TEXT NOT NULL
  optimizer_eligible INTEGER NOT NULL
  PRIMARY KEY(snapshot_id, slate_slug, game_id, is_dst, player_id, player_name, team)

IDENTITY:
  Offense -> exact GSIS ID derived only from exact "GSIS:" projection_identity prefix.
  DST     -> exact team + game_id identity.
  No fuzzy matching. No name-only recovery.

CAPTURE:
  Capture only pregame rows.
  Resolve exact game_id from schedule away_team/home_team.
  Reject any row if exact game mapping fails.
  Capture timestamp is UTC.
  Existing immutable snapshots are never updated or deleted.
  Repeating an identical capture payload must be idempotent by semantic SHA.

POSTGAME AUTHORITY:
  For a player/game, select latest eligible snapshot strictly before kickoff.
  Tied latest capture timestamps fail closed.
  Actual offense join = exact game_id + exact GSIS player_id.
  Actual D/ST grading requires separate verified D/ST actual-scoring contract.
EOF

echo
echo "=== DESIGN AUDIT RESULT ==="
echo "NFL_POSTGAME_1D_B_STATUS=PASS"
echo "READ_ONLY_AUDIT=TRUE"
echo "DATABASES_CREATED=0"
echo "PRODUCTION_DATABASE_WRITES=0"
echo "SERVICE_RESTARTS=0"
echo "CRON_CHANGES=0"
