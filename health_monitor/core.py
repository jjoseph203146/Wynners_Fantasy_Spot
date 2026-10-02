"""Deterministic checks; only adapters perform reads. No production actions."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from datetime import datetime, timezone

import pandas as pd

STATUSES = ('HEALTHY', 'WARNING', 'CRITICAL', 'OWNER_ACTION_REQUIRED')
EXIT_CODES = dict(zip(STATUSES, (0, 10, 20, 30)))
STATS = tuple('expected_' + x for x in (
    'attempts', 'completions', 'passing_yards', 'passing_tds', 'interceptions',
    'carries', 'rushing_yards', 'rushing_tds', 'targets', 'receptions',
    'receiving_yards', 'receiving_tds'))
CURRENT = 'data/parquet/nfl_current_unified_stat_forecasts.parquet'
STARTER = 'data/parquet/current_starter_verification.parquet'
MANIFEST = 'data/parquet/current_starter_verification_manifest.json'
AUDIT = 'processed/offensive_team_reconciliation_shadow_v3_audit.json'
CANDIDATE = 'processed/offensive_team_reconciliation_shadow_v3.csv'


class Invalid(Exception):
    def __init__(self, reason, **evidence):
        self.reason, self.evidence = reason, evidence


def require(ok, reason, **evidence):
    if not ok:
        raise Invalid(reason, **evidence)


def age(now, value):
    stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    require(stamp.tzinfo is not None, 'TIMESTAMP_WITHOUT_TIMEZONE')
    seconds = (now - stamp).total_seconds()
    require(seconds >= 0, 'FUTURE_TIMESTAMP')
    return seconds


def schema(frame, columns):
    require(set(columns) <= set(frame.columns), 'REQUIRED_SCHEMA_MISSING',
            missing=sorted(set(columns) - set(frame.columns)))


def gsis(value):
    return bool(re.fullmatch(r'00-\d{7}', str(value)))


def forecast_frame(frame):
    schema(frame, ('entity_type', 'game_id', 'team', 'player_id', 'position', *STATS))
    require(not frame.empty, 'EMPTY_FORECAST')
    require(frame.entity_type.isin(['OFFENSE_PLAYER', 'KICKER', 'DST']).all(), 'ENTITY_CONTRACT')
    require(frame[['game_id', 'team', 'position']].notna().all().all(), 'NULL_IDENTITY')
    require(all(frame[c].astype(str).str.strip().ne('').all() for c in ['game_id', 'team', 'position']), 'BLANK_IDENTITY')
    for entity, rows in frame.groupby('entity_type'):
        keys = ['game_id', 'team'] if entity == 'DST' else ['game_id', 'player_id']
        require(not rows.duplicated(keys).any(), 'DUPLICATE_FORECAST_IDENTITY', entity=entity)
        if entity != 'DST':
            require(rows.player_id.map(gsis).all(), 'UNRESOLVED_FORECAST_IDENTITY')
    qb = frame.loc[frame.position.eq('QB'), list(STATS[:8])]
    require(not qb.empty and qb.map(lambda v: pd.notna(v) and math.isfinite(float(v))).all().all(), 'INVALID_QB_VALUES')


def qualified_starters(frame, meta, context, now):
    require(meta.get('contract') == 'WFS_STARTER_VERIFICATION_CURRENT_V1'
            and meta.get('source_contract') == 'WFS_STARTER_VERIFICATION_V1_1'
            and meta.get('analysis_only') is True and meta.get('production_influence') is False,
            'STARTER_CONTRACT')
    require((meta.get('season'), meta.get('week'), meta.get('game_type')) ==
            (context['season'], context['week'], 'REG'), 'STARTER_CONTEXT_MISMATCH')
    require(meta.get('identity_gate') == 'PASS' and meta.get('identity_unresolved_count') == 0
            and meta.get('identity_ambiguous_count') == 0, 'UNRESOLVED_STARTER_IDENTITY')
    seconds = age(now, meta.get('generated_at_utc'))
    require(seconds <= 7200, 'STALE_STARTER_EVIDENCE', age_seconds=int(seconds))
    schema(frame, ['team', 'position', 'depth_gsis_id', 'rotowire_gsis_id',
                   'verification_status', 'comparable_to_rotowire', 'depth_availability_present',
                   'rw_availability_present', 'depth_injury_gate', 'rw_injury_gate'])
    require(len(frame) == meta.get('role_rows'), 'STARTER_ROW_COUNT')
    qualified = {}
    for team in context['pregame_teams']:
        rows = frame[frame.team.eq(team) & frame.position.eq('QB')]
        require(len(rows) == 1, 'STARTER_QB_COVERAGE_OR_DUPLICATE', team=team, rows=len(rows))
        r = rows.iloc[0]
        # RotoWire is the primary starter-identity authority.
        # The depth chart is corroboration/fallback evidence and may disagree
        # without vetoing an otherwise eligible RotoWire starter.
        #
        # Availability/injury truth remains a hard veto: RotoWire may identify
        # the starter, but it may never override a blocked/inactive player.
        require(gsis(r.rotowire_gsis_id),
                'UNRESOLVED_STARTER_IDENTITY', team=team)
        # Established current-injury contract:
        # absence from injury_consensus_current is healthy-by-absence.
        # Preserve rw_availability_present as provenance; do not fabricate
        # an availability row.  An explicit BLOCK remains a hard veto,
        # while an explicit ALLOW or exact-GSIS absence is eligible.
        rw_gate = str(r.rw_injury_gate or '').strip().upper()
        rw_present = bool(r.rw_availability_present)

        require(
            (rw_present and rw_gate == 'ALLOW')
            or (not rw_present and rw_gate == 'NO_CONSENSUS_ROW'),
            'ROTOWIRE_STARTER_BLOCKED',
            team=team,
            rotowire_id=r.rotowire_gsis_id,
            verification_status=r.verification_status,
        )
        qualified[team] = r.rotowire_gsis_id
    return qualified, seconds


def qb_invariant(game, team, starter, primary, authoritative, observation, root):
    context, selected, used = observation['context'], observation['selected'], observation['used']
    mode = context.get('mode')
    require(mode == game['mode'], 'CONSUMER_MODE_MISMATCH')
    require(context.get('available') is True, 'CONSUMER_EVIDENCE_MISSING')
    expected_path = root / (CURRENT if mode == 'PREGAME' else
                           f"data/forecast_snapshots/{game['game_id']}/stat_forecast_kickoff.parquet")
    from pathlib import Path
    require(Path(context.get('source_path', '')).resolve() == expected_path.resolve(), 'CONSUMER_SOURCE_MISMATCH')
    if mode != 'PREGAME':
        require(context.get('frozen_at_kickoff') is True and context.get('source_kind') == 'KICKOFF_SNAPSHOT', 'FROZEN_SEMANTICS_VIOLATION')
        # Today's starter and PRIMARY_QB are deliberately irrelevant here.
        return 'FROZEN_SNAPSHOT', {}
    require(starter is not None, 'QUALIFIED_STARTER_UNAVAILABLE')
    require(primary == starter, 'STARTER_PRIMARY_QB_MISMATCH', starter=starter, primary=primary)
    public = [r for r in selected if r.get('team') == team and r.get('position') == 'QB']
    require(len(public) == 1, 'PUBLIC_QB_COVERAGE_OR_DUPLICATE', rows=len(public))
    require(public[0].get('player_id') == primary, 'PRIMARY_PUBLIC_QB_MISMATCH',
            primary=primary, public=public[0].get('player_id'))
    rows = [r for r in authoritative if r.get('game_id') == game['game_id'] and r.get('team') == team
            and r.get('player_id') == primary and r.get('position') == 'QB']
    consumed = [r for r in used if r.get('team') == team and r.get('position') == 'QB']
    require(len(rows) == len(consumed) == 1 and consumed[0].get('player_id') == primary,
            'QB_ROW_COVERAGE_OR_DUPLICATE')
    mismatches = []
    for field in STATS:
        a, b = rows[0].get(field), consumed[0].get(field)
        if field in STATS[8:] and pd.isna(a) and pd.isna(b):
            continue
        if a is None or b is None or not math.isfinite(float(a)) or not math.isfinite(float(b)) or abs(float(a)-float(b)) > 1e-8:
            mismatches.append(field)
    require(not mismatches, 'QB_NUMERICAL_FORECAST_MISMATCH', fields=mismatches)
    return 'QB_IDENTITIES_AND_VALUES_AGREE', {'player_id': primary}


def run(adapter, now):
    checks = []
    def add(check, status, reason, scope='', **evidence):
        checks.append(dict(check_id=check, status=status, reason_code=reason, scope=scope,
                           summary=reason.replace('_', ' ').lower(), evidence=evidence))
    def checked(check, fn, scope='', failure='OWNER_ACTION_REQUIRED'):
        try:
            return fn()
        except Invalid as exc:
            add(check, 'OWNER_ACTION_REQUIRED' if exc.reason == 'LIVE_IDENTITY_UNRESOLVED' else failure, exc.reason, scope, **exc.evidence)
        except Exception as exc:
            # Never serialize exception messages: upstream messages can contain sensitive data.
            add(check, failure, 'REQUIRED_CHECK_EXCEPTION', scope, exception_type=type(exc).__name__)
        return None
    def app():
        service, http = adapter.service('wfs.service'), adapter.http_health()
        add('app', 'HEALTHY' if service and http else 'CRITICAL',
            'APP_AVAILABLE' if service and http else 'APP_UNAVAILABLE', service_running=service, http_ok=http)
    checked('app', app, failure='CRITICAL')
    checked('storage', lambda: add('storage', **adapter.storage()))
    checked('updater', lambda: add('updater', **adapter.updater(now)), failure='CRITICAL')
    context = checked('schedule', lambda: adapter.schedule(now))
    if context is None:
        add('live', 'CRITICAL', 'LIVE_EXPECTATION_UNKNOWN')
        add('forecast', 'OWNER_ACTION_REQUIRED', 'SCHEDULE_DEPENDENCY_UNAVAILABLE')
        add('starter', 'OWNER_ACTION_REQUIRED', 'SCHEDULE_DEPENDENCY_UNAVAILABLE')
        add('qb_invariant', 'OWNER_ACTION_REQUIRED', 'SCHEDULE_DEPENDENCY_UNAVAILABLE')
    else:
        add('schedule', 'HEALTHY', 'SCHEDULE_RESOLVED', season=context['season'], week=context['week'])
        checked('live', lambda: add('live', **adapter.live(context, now)), failure='CRITICAL')
        def forecast():
            df = adapter.frame(CURRENT)
            forecast_frame(df)
            allowed = {g['game_id']: {g['home_team'], g['away_team']} for g in context['games']}
            require(all(g in allowed and t in allowed[g] for g, t in zip(df.game_id, df.team)), 'FORECAST_SCHEDULE_MISMATCH')
            expected = {(g['game_id'], t) for g in context['games'] if g['mode'] == 'PREGAME' for t in [g['home_team'], g['away_team']]}
            actual = set(zip(df.game_id, df.team))
            require(expected <= actual, 'FORECAST_GAME_COVERAGE', missing=sorted(expected-actual))
            seconds = (now.timestamp() - adapter.mtime(CURRENT))
            require(seconds >= 0, 'FUTURE_TIMESTAMP')
            require(not expected or seconds <= 7200, 'STALE_FORECAST_EVIDENCE', age_seconds=int(seconds))
            add('forecast', 'WARNING' if expected and seconds > 5400 else 'HEALTHY', 'CURRENT_CONSUMER_FORECAST_VALID',
                rows=len(df), sha256=adapter.digest(CURRENT), age_seconds=int(seconds),
                freshness_basis='mtime plus exact schedule coverage; no standalone publication manifest assumed')
            return df
        df = checked('forecast', forecast)
        def starters():
            frame, meta = adapter.frame(STARTER), adapter.json(MANIFEST)
            _, seconds = qualified_starters(frame, meta, dict(context, pregame_teams=[]), now)
            qualified = {}
            for team in context['pregame_teams']:
                def one():
                    q, _ = qualified_starters(frame, meta, dict(context, pregame_teams=[team]), now)
                    qualified.update(q)
                    add('starter', 'WARNING' if seconds > 5400 else 'HEALTHY', 'STARTER_EVIDENCE_VALID', team,
                        age_seconds=int(seconds))
                checked('starter', one, team)
            return qualified
        if context['pregame_teams']:
            qualified = checked('starter', starters)
        else:
            qualified = {}
            add('starter', 'HEALTHY', 'NO_PREGAME_STARTER_REQUIRED')
        def primary_rows():
            audit, raw = adapter.json(AUDIT), adapter.read(CANDIDATE)
            require(audit.get('version') == 'WFS_OFFENSIVE_TEAM_RECONCILIATION_SHADOW_V3' and audit.get('status') == 'PASS', 'RECONCILIATION_CONTRACT')
            require(audit['outputs']['player_shadow']['sha256'] == hashlib.sha256(raw).hexdigest(), 'RECONCILIATION_HASH_MISMATCH')
            require(age(now, audit['generated_at_utc']) <= 7200, 'STALE_RECONCILIATION_EVIDENCE')
            r = pd.read_csv(io.BytesIO(raw))
            schema(r, ['game_id', 'team', 'player_id', 'position', 'reconciliation_role', 'primary_qb_id'])
            require(not r.duplicated(['game_id', 'player_id']).any(), 'DUPLICATE_RECONCILIATION_IDENTITY')
            require(len(r) == audit['rows'], 'RECONCILIATION_ROW_COUNT')
            result = {}
            for (game, team), group in r.groupby(['game_id', 'team']):
                q = group[group.reconciliation_role.eq('PRIMARY_QB')]
                require(len(q) == 1 and q.iloc[0].position == 'QB' and gsis(q.iloc[0].player_id), 'PRIMARY_QB_COVERAGE')
                pid = q.iloc[0].player_id
                require(group.primary_qb_id.eq(pid).all(), 'PRIMARY_QB_INTERNAL_CONFLICT')
                result[game, team] = pid
            add('reconciliation', 'HEALTHY', 'HASH_BOUND_PRIMARY_QB_EVIDENCE', source=CANDIDATE)
            return result
        primaries = checked('reconciliation', primary_rows) if context['pregame_teams'] else {}
        for game in context['games']:
            gid = game['game_id']
            def observe():
                obs = adapter.observe(gid)
                if game['mode'] != 'PREGAME':
                    adapter.validate_snapshot(gid, obs)
                return obs
            obs = checked('consumer', observe, gid)
            for team in [game['home_team'], game['away_team']]:
                def invariant():
                    require(obs is not None, 'CONSUMER_DEPENDENCY_UNAVAILABLE')
                    if game['mode'] == 'PREGAME':
                        require(df is not None and qualified is not None and primaries is not None, 'QB_DEPENDENCY_UNAVAILABLE')
                    reason, evidence = qb_invariant(game, team, (qualified or {}).get(team),
                        (primaries or {}).get((gid, team)), [] if df is None else df.to_dict('records'), obs, adapter.root)
                    add('qb_invariant', 'HEALTHY', reason, gid + '/' + team, **evidence)
                checked('qb_invariant', invariant, gid + '/' + team)
    if adapter.changed():
        # Discard cross-layer conclusions from an incoherent observation, never call it healthy.
        checks = [c for c in checks if c['check_id'] in {'app', 'storage', 'updater'}]
        add('snapshot', 'WARNING', 'PRODUCTION_CHANGED_DURING_OBSERVATION')
    checks.sort(key=lambda c: (c['check_id'], c['scope'], c['reason_code']))
    counts = {s: sum(c['status'] == s for c in checks) for s in STATUSES}
    overall = max((c['status'] for c in checks), key=STATUSES.index)
    return dict(contract='NFL_APP_HEALTH_MONITOR_V1', generated_at_utc=now.isoformat(),
                overall_status=overall, checks=checks, summary_counts=counts)
