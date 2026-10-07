"""Immutable prospective lineup evidence. No solver/scoring/forecast changes."""
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
from contextlib import closing
import uuid

ALIASES = {'LA': 'LAR', 'WAS': 'WSH', 'JAC': 'JAX'}

def canon(value):
    value = str(value).strip().upper()
    return ALIASES.get(value, value)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, 'item'):
        return safe(value.item())
    return str(value)

def write_json(path, value):
    with Path(path).open('w', encoding='utf-8') as handle:
        handle.write(json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False))
        handle.flush()
        os.fsync(handle.fileno())

def records(frame):
    return {'columns': list(frame.columns), 'rows': safe(frame.to_dict('records'))}

def kickoff(row):
    raw = str(row['game_date']).split(' ')[0] + 'T' + str(row['gametime'])
    return datetime.fromisoformat(raw).replace(tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc)

def primary_id(row, format):
    team = canon(row['team_solver'] if format == 'CLASSIC' else row['team'])
    position = str(row['solver_position'] if format == 'CLASSIC' else row['position']).upper()
    if position in {'DST', 'D/ST', 'DEF', 'D'}:
        return 'DST:' + team
    value = str(row.get('injury_gsis_id', '') if format == 'CLASSIC' else row.get('player_id', '')).strip()
    if not re.fullmatch(r'00-\d{7}', value):
        raise RuntimeError('LINEUP_CAPTURE_NONCANONICAL_SELECTED_ID: ' + value)
    return value

def begin(root, format, slate, contest, eligible, args, solver):
    """Called inside solver, after final eligibility and before optimization."""
    import pandas as pd
    root = Path(root).resolve()
    if format not in {'CLASSIC', 'SINGLE_GAME'} or eligible.empty or contest.empty:
        raise RuntimeError('LINEUP_CAPTURE_INVALID_SCOPE')
    now = datetime.now(timezone.utc)
    with closing(sqlite3.connect(f'file:{root / "data/nfl.db"}?mode=ro', uri=True)) as con:
        con.row_factory = sqlite3.Row
        if format == 'CLASSIC':
            gids = sorted(set(contest['game_id'].dropna().astype(str)))
            if not gids or contest['game_id'].isna().any():
                raise RuntimeError('LINEUP_CAPTURE_UNRESOLVED_CONTEST_GAME')
            games = [dict(x) for x in con.execute(
                'SELECT * FROM games WHERE game_id IN (' + ','.join('?' for _ in gids) + ')', gids)]
            if len(games) != len(gids):
                raise RuntimeError('LINEUP_CAPTURE_MISSING_SCHEDULE')
        else:
            scope = json.loads((root / 'data/parquet/current_unified_stat_forecasts_manifest.json').read_text())['validated_v3_parent']
            games = [dict(x) for x in con.execute('SELECT * FROM games WHERE season=? AND week=?', (scope['season'], scope['week']))]
            pairs = set()
            for value in contest['game'].astype(str):
                parts = re.split(r'\s*@\s*', value.strip())
                if len(parts) != 2:
                    raise RuntimeError('LINEUP_CAPTURE_BAD_GAME_PAIR')
                pairs.add(tuple(map(canon, parts)))
            games = [g for g in games if (canon(g['away_team']), canon(g['home_team'])) in pairs]
            if len(games) != 1 or len(pairs) != 1:
                raise RuntimeError('LINEUP_CAPTURE_AMBIGUOUS_GAME')
            gids = [games[0]['game_id']]
        if any(int(g['completed'] or 0) != 0 or kickoff(g) <= now for g in games):
            raise RuntimeError('LINEUP_CAPTURE_REQUIRES_PREGAME')
        if len({(g['season'], g['week']) for g in games}) != 1:
            raise RuntimeError('LINEUP_CAPTURE_MIXED_WEEK')
    rows = []
    for row in eligible.to_dict('records'):
        pid = primary_id(row, format)
        team = canon(row['team_solver'] if format == 'CLASSIC' else row['team'])
        gid = str(row['game_id']) if format == 'CLASSIC' else gids[0]
        game = next((g for g in games if g['game_id'] == gid), None)
        if game is None or team not in {canon(game['away_team']), canon(game['home_team'])}:
            raise RuntimeError('LINEUP_CAPTURE_TEAM_GAME_CONFLICT')
        rows.append({'player_id': pid, 'game_id': gid, 'team': team,
                     'opponent': canon(game['home_team']) if team == canon(game['away_team']) else canon(game['away_team']),
                     'player': str(row['player_solver'] if format == 'CLASSIC' else row['player']),
                     'position': str(row['solver_position'] if format == 'CLASSIC' else row['position']),
                     'salary': float(row['salary_solver'] if format == 'CLASSIC' else row['salary']),
                     'projection': float(row['projection_solver'] if format == 'CLASSIC' else row['projection']),
                     'mvp_salary': None if format == 'CLASSIC' else float(row['mvp_salary'])})
    if len({(r['game_id'], r['player_id']) for r in rows}) != len(rows):
        raise RuntimeError('LINEUP_CAPTURE_DUPLICATE_ID')
    if any(not math.isfinite(r['salary']) or r['salary'] <= 0 or not math.isfinite(r['projection']) for r in rows):
        raise RuntimeError('LINEUP_CAPTURE_BAD_NUMERIC')
    base = root / 'data/lineup_learning_v1/captures'
    base.mkdir(parents=True, exist_ok=True)
    path = base / ('.pending_' + uuid.uuid4().hex)
    path.mkdir()
    try:
        write_json(path / 'contest_pool.json', records(contest))
        write_json(path / 'solver_pool.json', records(eligible))
        write_json(path / 'canonical_solver_pool.json', rows)
        write_json(path / 'schedule.json', games)
        sources = []
        def preserve(original, label, mandatory=True, expected=None):
            original = Path(original).resolve()
            if not original.is_relative_to(root):
                raise RuntimeError('LINEUP_CAPTURE_SOURCE_OUTSIDE_PROJECT')
            if not original.exists():
                if mandatory:
                    raise RuntimeError('LINEUP_CAPTURE_SOURCE_MISSING: ' + str(original))
                return
            dest = path / label
            shutil.copyfile(original, dest)
            sha = digest(dest)
            if expected and sha != expected:
                raise RuntimeError('LINEUP_CAPTURE_RAW_SHA_MISMATCH')
            sources.append({'path': str(original.relative_to(root)), 'saved_as': label, 'sha256': sha})
        preserve(solver, 'solver.py')
        preserve(root / 'data/parquet/injury_consensus_current.parquet', 'injury_asof.parquet')
        preserve(root / 'data/parquet/current_unified_stat_forecasts_manifest.json', 'forecast_asof_manifest.json')
        preserve(root / 'data/parquet/current_unified_stat_forecasts.parquet', 'forecast_asof.parquet')
        preserve(root / 'data/parquet/current_unified_fanduel_expectation.parquet', 'stage24_asof.parquet')
        if format == 'SINGLE_GAME':
            identity_path = root / 'data/fanduel/single_game/derived/single_game_identity_pool.parquet'
            raw = pd.read_parquet(identity_path)
            raw = raw.loc[raw['public_slate_name'].astype(str).eq(str(slate))]
            if raw.empty or not set(eligible['player_id']).issubset(set(raw['player_id'])):
                raise RuntimeError('LINEUP_CAPTURE_IDENTITY_POOL_MISMATCH')
            write_json(path / 'full_identity_pool.json', records(raw))
            for n, (src, block) in enumerate(raw.groupby('source_csv')):
                hashes = set(block['source_csv_sha256'].astype(str))
                if len(hashes) != 1:
                    raise RuntimeError('LINEUP_CAPTURE_MULTIPLE_RAW_SHA')
                preserve(root / 'data/fanduel/single_game/raw' / src, f'raw_{n}.csv', expected=next(iter(hashes)))
        else:
            with closing(sqlite3.connect(f'file:{root / "data/nfl.db"}?mode=ro', uri=True)) as con:
                raw = pd.read_sql_query('SELECT * FROM fanduel_slate_pool WHERE slate_key IN (' + ','.join('?' for _ in set(contest['slate_key'])) + ')', con, params=sorted(set(contest['slate_key'])))
            if raw.empty or set(raw['game_id']) != set(gids):
                raise RuntimeError('LINEUP_CAPTURE_RAW_SLATE_SCOPE_MISMATCH')
            write_json(path / 'full_contest_pool.json', records(raw))
            for n, src in enumerate(sorted(set(raw['source_file'].astype(str)))):
                preserve(root / 'data/fanduel/slates' / src, f'raw_{n}.csv')
        resume = str(getattr(args, 'resume_players', '') or '')
        if resume:
            preserve(resume, 'resume_players.csv')
        # Capture weather files if present; absence remains explicit, never inferred.
        weather = sorted((root / 'data/parquet').glob('*weather*.parquet'))
        for n, original in enumerate(weather):
            preserve(original, f'weather_asof_{n}.parquet')
        metadata = {'contract': 'WFS_LINEUP_PROSPECTIVE_CAPTURE_V1', 'format': format,
                    'slate': str(slate), 'captured_at_utc': now.isoformat(),
                    'season': int(games[0]['season']), 'week': int(games[0]['week']),
                    'game_ids': gids, 'settings': safe(vars(args)), 'sources': sources,
                    'weather_files_captured': len(weather),
                    'contest_rules': {'salary_cap': 60000, 'roster_size': 9 if format == 'CLASSIC' else 6,
                                      'classic_max_players_per_team': 4 if format == 'CLASSIC' else None,
                                      'mvp_points_multiplier': None if format == 'CLASSIC' else 1.5},
                    'full_contest_identity_validation': 'PENDING_RECONSTRUCTION',
                    'benchmark_status': 'NOT_EVALUATED', 'production_model_changes': False}
        write_json(path / 'capture_metadata.json', metadata)
        return path
    except BaseException:
        shutil.rmtree(path)
        raise

def finish(path, players):
    """Validate generated selections, then publish one immutable capture."""
    path = Path(path)
    metadata = json.loads((path / 'capture_metadata.json').read_text())
    pool = json.loads((path / 'canonical_solver_pool.json').read_text())
    format = metadata['format']
    long = []
    for r in players.to_dict('records'):
        matches = [p for p in pool if p['player'] == str(r['player']) and p['team'] == canon(r['team']) and p['position'] == str(r['position'])]
        if len(matches) != 1:
            raise RuntimeError('LINEUP_CAPTURE_SELECTION_IDENTITY_AMBIGUOUS')
        p = matches[0]
        if format == 'SINGLE_GAME' and str(r['player_id']) != p['player_id']:
            raise RuntimeError('LINEUP_CAPTURE_SELECTION_ID_MISMATCH')
        role = str(r['slot'] if format == 'CLASSIC' else r['role'])
        salary = p['mvp_salary'] if role == 'MVP' else p['salary']
        if float(r['salary']) != salary:
            raise RuntimeError('LINEUP_CAPTURE_SELECTION_SALARY_MISMATCH')
        if not math.isclose(float(r['projection']), p['projection'] * (1.5 if role == 'MVP' else 1), abs_tol=0.00001):
            raise RuntimeError('LINEUP_CAPTURE_SELECTION_PROJECTION_MISMATCH')
        long.append({**p, 'lineup': int(r['lineup']), 'slot': role, 'slot_salary': salary,
                     'slot_projection': float(r['projection'])})
    if not long:
        raise RuntimeError('LINEUP_CAPTURE_EMPTY_LINEUPS')
    for no in {r['lineup'] for r in long}:
        block = [r for r in long if r['lineup'] == no]
        if len(block) != metadata['contest_rules']['roster_size'] or len({r['player_id'] for r in block}) != len(block):
            raise RuntimeError('LINEUP_CAPTURE_BAD_ROSTER')
        if sum(r['slot_salary'] for r in block) > 60000:
            raise RuntimeError('LINEUP_CAPTURE_SALARY_CAP')
        if format == 'CLASSIC':
            if sorted(r['slot'] for r in block) != sorted(['QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX','DST']):
                raise RuntimeError('LINEUP_CAPTURE_BAD_SLOTS')
            for r in block:
                allowed = {'RB','WR','TE'} if r['slot'] == 'FLEX' else {re.sub(r'\d+$', '', r['slot'])}
                if r['position'] not in allowed:
                    raise RuntimeError('LINEUP_CAPTURE_POSITION_ILLEGAL')
            if any(sum(r['team'] == team for r in block) > 4 for team in {r['team'] for r in block}):
                raise RuntimeError('LINEUP_CAPTURE_TEAM_LIMIT')
        elif sum(r['slot'] == 'MVP' for r in block) != 1 or sum(r['slot'] == 'FLEX' for r in block) != 5:
            raise RuntimeError('LINEUP_CAPTURE_BAD_ROLES')
    write_json(path / 'generated_players.json', long)
    metadata['files'] = {p.name: digest(p) for p in path.iterdir() if p.is_file() and p.name != 'capture_metadata.json'}
    metadata['status'] = 'PROSPECTIVE_CAPTURE_COMPLETE'
    write_json(path / 'manifest.json', metadata)
    # No capture can be committed after one of its games has started.
    games = json.loads((path / 'schedule.json').read_text())
    if any(kickoff(g) <= datetime.now(timezone.utc) for g in games):
        raise RuntimeError('LINEUP_CAPTURE_SOLVE_CROSSED_KICKOFF')
    final = path.parent / (metadata['captured_at_utc'].replace(':', '').replace('+', '_') + '_' + uuid.uuid4().hex[:12])
    for p in path.iterdir():
        os.chmod(p, 0o444)
    for p in path.iterdir():
        with p.open('rb') as handle:
            os.fsync(handle.fileno())
    path.rename(final)
    fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print('LINEUP_LEARNING_CAPTURE=' + str(final))
    print('LINEUP_LEARNING_CAPTURE_STATUS=PASS')
    return final
