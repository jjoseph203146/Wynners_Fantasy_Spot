"""Allowlisted read adapters. Never import a builder, publisher or solver."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from urllib.parse import quote, unquote, urlsplit
from urllib.request import ProxyHandler, build_opener
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
from .core import CURRENT, require, forecast_frame, schema, age

ET = ZoneInfo('America/New_York')
CONNECT = sqlite3.connect


class ReadOnlyConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


def readonly_connect(database, *args, **kwargs):
    """Also constrain legacy consumer connections that omit mode=ro."""
    raw = os.fspath(database)
    path = Path(unquote(urlsplit(raw).path) if raw.startswith('file:') else raw).resolve()
    if not path.is_file():
        raise FileNotFoundError('Required database unavailable')
    kwargs['uri'] = True
    kwargs['timeout'] = 2
    kwargs['factory'] = ReadOnlyConnection
    connection = CONNECT('file:' + quote(str(path), safe='/') + '?mode=ro', *args, **kwargs)
    connection.execute('PRAGMA query_only=ON')
    deadline = time.monotonic() + 5
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
    return connection


def install_write_guard():
    """CLI-only process guard: deny Python filesystem mutations, including imported code."""
    def audit(event, args):
        if event == 'open':
            mode, flags = args[1], args[2]
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                raise PermissionError('Monitor filesystem writes are forbidden')
        if event in {'os.remove', 'os.rename', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.chown',
                     'os.link', 'os.symlink', 'os.truncate', 'os.utime', 'shutil.copyfile'}:
            raise PermissionError('Monitor filesystem mutations are forbidden')
    sys.addaudithook(audit)


class Production:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.signatures = {}
        self.cache = {}
        for name in ['data/nfl.db', 'data/nfl.db-wal', 'data/wfs_live.db', 'data/wfs_live.db-wal',
                     'wfs_ai_analyst.py', 'forecast_publication_selector.py']:
            self.watch(self.root / name)

    @staticmethod
    def signature(path):
        try:
            s = path.stat()
            return s.st_ino, s.st_size, s.st_mtime_ns
        except FileNotFoundError:
            return None

    def watch(self, path):
        self.signatures.setdefault(path, self.signature(path))

    def changed(self):
        return any(self.signature(p) != sig for p, sig in self.signatures.items())

    def read(self, name):
        path = self.root / name
        self.watch(path)
        if name not in self.cache:
            self.cache[name] = path.read_bytes()
        return self.cache[name]

    def frame(self, name):
        return pd.read_parquet(io.BytesIO(self.read(name)))

    def json(self, name):
        return json.loads(self.read(name))

    def digest(self, name):
        return hashlib.sha256(self.read(name)).hexdigest()

    def mtime(self, name):
        return (self.root / name).stat().st_mtime

    def service(self, name):
        require(name in {'wfs.service', 'wfs-nfl-live.service'}, 'SERVICE_NOT_ALLOWLISTED')
        p = subprocess.run(['systemctl', 'show', name, '-p', 'ActiveState', '-p', 'SubState'],
                           capture_output=True, text=True, timeout=5, check=False)
        require(p.returncode == 0, 'SERVICE_STATUS_UNAVAILABLE')
        props = dict(line.split('=', 1) for line in p.stdout.splitlines() if '=' in line)
        return props.get('ActiveState') == 'active' and props.get('SubState') == 'running'

    def http_health(self):
        try:
            with build_opener(ProxyHandler({})).open('http://127.0.0.1:8502/_stcore/health', timeout=3) as response:
                return response.status == 200 and response.read(128).strip() == b'ok'
        except Exception:
            return False

    def storage(self):
        required = {
            'nfl.db': {'games', 'player_identity', 'weekly_rosters', 'injuries', 'injury_consensus_current', 'depth_charts'},
            'forecast_ledger.db': {'forecast_predictions', 'forecast_snapshots'},
            'player_projection_ledger.db': {'player_projection_predictions', 'player_projection_snapshots'},
            'fanduel_player_role_performance.db': {'dim_game', 'dim_player', 'fact_player_game_fanduel'},
            'wfs_live.db': {'live_events', 'live_plays', 'live_ingest_audit'},
        }
        for name, tables in required.items():
            with readonly_connect(self.root / 'data' / name) as conn:
                actual = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                require(tables <= actual, 'REQUIRED_DATABASE_TABLE_MISSING', database=name, missing=sorted(tables-actual))
                for table in sorted(tables):
                    conn.execute('SELECT * FROM "' + table + '" LIMIT 1').fetchone()
        disk = shutil.disk_usage(self.root)
        fraction = disk.free / disk.total
        status = 'CRITICAL' if disk.free < 1024**3 or fraction < .02 else 'WARNING' if disk.free < 5*1024**3 or fraction < .05 else 'HEALTHY'
        return dict(status=status, reason='STORAGE_READABLE' if status == 'HEALTHY' else 'LOW_DISK_SPACE',
                    free_bytes=disk.free, databases_checked=len(required))

    def schedule(self, now):
        from wfs_schedule_context import resolve_schedule_week_context
        ctx = resolve_schedule_week_context(db_path=self.root/'data/nfl.db', as_of=now.astimezone(ET).date())
        with readonly_connect(self.root/'data/nfl.db') as conn:
            conn.row_factory = sqlite3.Row
            games = [dict(r) for r in conn.execute(
                'SELECT game_id,season,week,game_type,home_team,away_team,game_date,gametime,completed,espn FROM games WHERE season=? AND week=? AND game_type=? ORDER BY game_id',
                (ctx.season, ctx.planning_week, 'REG'))]
        require(bool(games), 'SCHEDULE_EMPTY')
        with readonly_connect(self.root/'data/wfs_live.db') as conn:
            events = dict(conn.execute('SELECT event_id,state FROM live_events'))
        teams = set()
        for game in games:
            require(game['completed'] in (0, 1), 'SCHEDULE_COMPLETION_UNKNOWN')
            event = str(game['espn'] or '')
            game['event_id'] = event
            start = datetime.fromisoformat(game['game_date']+'T'+game['gametime']).replace(tzinfo=ET)
            game['kickoff'] = start
            state = events.get(event)
            game['mode'] = 'POSTGAME' if game['completed'] or state == 'post' else 'LIVE' if state in {'in','live'} else 'PREGAME'
            # After kickoff an unfinished game remains expected, even if feed is absent/stale.
            game['live_expected'] = not game['completed'] and now >= start
            if game['mode'] == 'PREGAME':
                teams.update((game['home_team'], game['away_team']))
        return dict(season=ctx.season, week=ctx.planning_week, games=games, pregame_teams=sorted(teams))

    def live(self, context, now):
        running = self.service('wfs-nfl-live.service')
        require(running, 'LIVE_SERVICE_UNAVAILABLE')
        expected = [g for g in context['games'] if g['live_expected']]
        if not expected:
            with readonly_connect(self.root/'data/wfs_live.db') as conn:
                row = conn.execute('SELECT captured_at_utc,ingest_status,ambiguous_occurrences,unresolved_occurrences FROM live_ingest_audit ORDER BY captured_at_utc DESC,ingest_id DESC LIMIT 1').fetchone()
            require(row is not None, 'LIVE_AUDIT_HISTORY_MISSING')
            clean = row[1] == 'PASS' and row[2] == row[3] == 0
            return dict(status='HEALTHY' if clean else 'WARNING',
                        reason='NO_ACTIVE_GAME_EXPECTED' if clean else 'IDLE_LIVE_LAST_AUDIT_FAILED',
                        service_running=True, last_capture=row[0], last_ingest_status=row[1],
                        evidence_policy='historical audit inspected; ingest age not applicable outside scheduled games')
        ages = []
        with readonly_connect(self.root/'data/wfs_live.db') as conn:
            conn.row_factory = sqlite3.Row
            for game in expected:
                r = conn.execute('SELECT * FROM live_ingest_audit WHERE event_id=? ORDER BY captured_at_utc DESC,ingest_id DESC LIMIT 1', (game['event_id'],)).fetchone()
                require(r is not None, 'LIVE_AUDIT_MISSING', game_id=game['game_id'])
                require(r['unresolved_occurrences'] == r['ambiguous_occurrences'] == 0, 'LIVE_IDENTITY_UNRESOLVED', game_id=game['game_id'])
                require(r['ingest_status'] == 'PASS', 'LIVE_INGEST_FAILED', game_id=game['game_id'])
                ages.append(age(now, r['captured_at_utc']))
        oldest = max(ages)
        return dict(status='CRITICAL' if oldest > 300 else 'WARNING' if oldest > 180 else 'HEALTHY',
                    reason='LIVE_INGEST_STALE' if oldest > 180 else 'LIVE_INGEST_CURRENT', oldest_age_seconds=int(oldest), games=len(expected))

    def updater(self, now):
        # Bounded tail, no assumptions based on log mtime.
        path = self.root/'logs/cron.log'
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size - 2_000_000))
            text = handle.read().decode('utf-8', errors='replace')
        starts = list(re.finditer(r'^NFL hourly pipeline launcher started: (.+)$', text, re.M))
        require(bool(starts), 'UPDATER_RUN_EVIDENCE_MISSING')
        def parse(s):
            # Shell date uses local EST/EDT; validate the abbreviation against the date.
            match = re.fullmatch(r'(\w{3} \w{3}\s+\d+ \d\d:\d\d:\d\d) (EST|EDT) (\d{4})', s)
            require(match is not None, 'UPDATER_TIMESTAMP_FORMAT')
            d = datetime.strptime(match[1]+' '+match[3], '%a %b %d %H:%M:%S %Y').replace(tzinfo=ET)
            require(d.tzname() == match[2], 'UPDATER_TIMEZONE_MISMATCH')
            return d
        latest = starts[-1]
        started = parse(latest[1])
        seconds = (now-started).total_seconds()
        require(seconds >= 0, 'FUTURE_TIMESTAMP')
        segment = text[latest.end():]
        terminal = re.search(r'^NFL hourly pipeline (?:completed successfully\.|exited with code: \d+|deferred because LIVE lock remained busy\.)$', segment, re.M)
        finished = re.search(r'^Finished: (.+)$', segment[terminal.end():], re.M) if terminal else None
        success = 'NFL hourly pipeline completed successfully.' in segment
        failure = re.search(r'NFL hourly pipeline exited with code: (\d+)', segment)
        evidence = dict(started_at=started.isoformat(), age_seconds=int(seconds))
        due = now.astimezone(ET).replace(minute=7, second=0, microsecond=0)
        if due > now:
            due -= timedelta(hours=1)
        missed = started < due and (now-due).total_seconds() > 900
        if missed:
            return dict(status='CRITICAL', reason='UPDATER_MISSED_EXPECTED_RUN', **evidence)
        if finished:
            ended = parse(finished[1])
            require(started <= ended <= now, 'UPDATER_TIMESTAMP_ORDER')
            evidence['finished_at'] = ended.isoformat()
            lock_busy_deferred = (
                'NFL hourly pipeline deferred because LIVE lock remained busy.'
                in segment
            )
            if lock_busy_deferred:
                return dict(
                    status='WARNING',
                    reason='UPDATER_DEFERRED_FOR_LIVE',
                    **evidence,
                )
            deferred_for_live = (
                'NFL hourly pipeline authority completed; downstream analytics deferred for LIVE.'
                in segment
            )
            if deferred_for_live:
                return dict(
                    status='WARNING',
                    reason='UPDATER_DOWNSTREAM_DEFERRED_FOR_LIVE',
                    exit_code=int(failure[1]) if failure else None,
                    **evidence,
                )
            if failure or not success:
                return dict(status='CRITICAL', reason='UPDATER_FAILED', exit_code=int(failure[1]) if failure else None, **evidence)
            return dict(status='HEALTHY', reason='UPDATER_COMPLETED', **evidence)
        previous = text[starts[-2].end():latest.start()] if len(starts)>1 else ''
        previous_failed = 'NFL hourly pipeline exited with code:' in previous
        return dict(status='CRITICAL' if seconds > 2700 or previous_failed else 'WARNING',
                    reason='UPDATER_EXCESSIVE_RUNTIME' if seconds > 2700 else 'UPDATER_RUNNING_AFTER_FAILURE' if previous_failed else 'UPDATER_RUNNING', **evidence)

    def observe(self, game_id):
        import wfs_ai_analyst as consumer
        require(consumer.STAT_FORECAST_CURRENT.resolve() == (self.root/CURRENT).resolve(), 'CONSUMER_AUTHORITY_AMBIGUOUS')
        used = []
        original = consumer._format_stat_outlook_row
        def capture(row):
            used.append(dict(row))
            return original(row)
        # Actual consumer loading/selection/rendering, without replacing its chosen artifact.
        with patch.object(sqlite3, 'connect', readonly_connect), patch.object(consumer, '_format_stat_outlook_row', capture):
            packet = consumer.build_game_evidence(game_id)
            outlook = consumer._build_stat_outlook(packet)
        require(outlook.get('available') is True, 'STAT_OUTLOOK_UNAVAILABLE')
        return dict(context=packet['stat_forecast_context'], selected=outlook.get('selected_players', []), used=used)

    def validate_snapshot(self, game_id, obs):
        prefix = f'data/forecast_snapshots/{game_id}/stat_forecast_kickoff'
        meta = self.json(prefix+'_manifest.json')
        frame = self.frame(prefix+'.parquet')
        require(meta.get('contract') == 'WFS_STAT_FORECAST_KICKOFF_SNAPSHOT_V1' and meta.get('immutable') is True
                and meta.get('postgame_regeneration_allowed') is False and meta.get('game_id') == game_id, 'KICKOFF_MANIFEST_CONTRACT')
        require(meta.get('snapshot_sha256') == self.digest(prefix+'.parquet'), 'KICKOFF_HASH_MISMATCH')
        forecast_frame(frame)
        require(len(frame) == meta.get('rows') and frame.game_id.eq(game_id).all(), 'KICKOFF_ROW_CONTEXT')
        actual = pd.DataFrame(obs['context'].get('rows', []))
        schema(actual, frame.columns)
        # Compare all raw context fields, permitting pandas null representation only.
        def canonical(df):
            return df[frame.columns].sort_values(['entity_type','team','player_id']).reset_index(drop=True).to_json(orient='records', double_precision=15)
        require(canonical(frame) == canonical(actual), 'KICKOFF_CONSUMER_VALUES_MISMATCH')
