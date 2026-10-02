"""M5B candidate: retrospective, read-only evaluation of frozen V1.3 claims.

No claim construction or evidence interpretation occurs here. The frozen M4A
structural validator is reused, with mandatory strict pregame game binding.
Callers supply already-built claims; their provenance is not re-adjudicated.

ROLE_INCREASE/DECREASE use the existing player_usage.py role definitions:
expansion = usage_delta >= 4 OR snap_delta >= .15; decline is the opposite
threshold. Both flags set is indeterminate. Deltas and lagged baselines must
agree with player_pregame_features before either flag can score a claim.
These are descriptive role classifications, not fantasy-production grades.

TARGET_SHARE uses target_share_avg_3, without fallback or new thresholds, only
for an explicitly INCREASE/DECREASE claim accepted by the frozen contract.
The frozen builder currently emits UNSPECIFIED: those claims are INDETERMINATE,
even when the quote says 'larger' or 'majority'. Text is never inspected.

Pregame features are historical shift(1) features (pregame_features.py), not
an assertion that this database row was captured at the claim cutoff. Their
updated_at may be postgame because tables are rebuilt. Timestamps are audited,
not used to rewrite pregame artifacts. No historical as-of snapshot is made.
Exactly one games, stats and usage row is required. No team/name tiebreaking.
Only integer completed=1 is final, matching the authoritative games schema.
All reads share one SQLite transaction. URI mode=ro plus an authorizer that
allows only SELECT/READ/transaction operations prevent database mutations.
"""

import hashlib
import json
import math
from pathlib import Path
import sqlite3

from .corroboration import _validate as _validate_frozen_claim, _time


ANALYSIS_ONLY = True
PRODUCTION_INFLUENCE = False
SOLVER_INFLUENCE = False
PROJECTION_MUTATION = False
FORECAST_MUTATION = False
AVAILABILITY_AUTHORITY = False
INJURY_AUTHORITY = False
DEPTH_CHART_AUTHORITY = False
AI_ANALYST_INFLUENCE = False
DATABASE_MUTATION = False
CONSENSUS_ENABLED = False

SAFETY = {
    "ANALYSIS_ONLY": True, "PRODUCTION_INFLUENCE": False,
    "SOLVER_INFLUENCE": False, "PROJECTION_MUTATION": False,
    "FORECAST_MUTATION": False, "AVAILABILITY_AUTHORITY": False,
    "INJURY_AUTHORITY": False, "DEPTH_CHART_AUTHORITY": False,
    "AI_ANALYST_INFLUENCE": False, "DATABASE_MUTATION": False,
    "CONSENSUS_ENABLED": False,
}
DEFAULT_DB = Path('/home/mwynn/nfl_data_engine/data/nfl.db')
SUPPORTED_SIGNALS = frozenset(('ROLE_INCREASE', 'ROLE_DECREASE', 'TARGET_SHARE'))
UNSUPPORTED_SIGNALS = frozenset(('STARTER_CHANGE', 'COACH_INTENT',
    'ROUTE_PARTICIPATION', 'BACKFIELD_SHARE', 'RED_ZONE_ROLE', 'DEEP_TARGET_ROLE'))
_IDENTITY = ('game_id', 'player_id', 'season', 'week', 'team', 'updated_at')
_ROLE_ACTUAL = ('opportunities', 'offense_pct', 'targets', 'carries',
    'usage_3g', 'snap_pct_3g', 'target_3g', 'carry_3g',
    'usage_delta', 'snap_delta', 'target_delta', 'carry_delta',
    'role_expansion_flag', 'role_decline_flag')
_ROLE_BASELINE = ('history_games', 'opportunities_avg_3', 'snap_pct_avg_3',
                  'targets_avg_3', 'carries_avg_3')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _copy(value):
    return json.loads(_canonical(value))


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _fields(row, names):
    # Nonfinite database values cannot enter canonical audit JSON.
    return {key: (None if isinstance(row.get(key), float)
                       and not math.isfinite(row[key]) else row.get(key))
            for key in names}


def _row(claim, status, reason, *, state=None, postgame=None, baseline=None,
         game=None):
    value = {key: claim.get(key) if isinstance(claim, dict) else None
             for key in ('claim_id', 'gsis_id', 'game_id', 'signal_type', 'direction')}
    value.update(evaluation_status=status, outcome_state=state,
                 pregame_claim=claim, pregame_claim_sha256=_digest(claim),
                 postgame_outcome=postgame or {}, pregame_baseline=baseline or {},
                 completed_game=game or {}, reason_codes=[reason],
                 safety=dict(SAFETY), evaluation_version='V1_3_M5B_CANDIDATE_1')
    value['evaluation_id'] = _digest(value)
    return value


def _read_authorizer(action, arg1, arg2, database, trigger):
    allowed = (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_TRANSACTION)
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def _one(connection, table, claim):
    # Table identifiers are internal constants; every external value is bound.
    if table == 'games':
        query = 'SELECT * FROM games WHERE game_id COLLATE BINARY = ?'
        params = (claim['game_id'],)
    else:
        if table not in ('player_game_stats', 'player_weekly_usage',
                         'player_pregame_features'):
            raise ValueError('INVALID_INTERNAL_TABLE')
        query = ('SELECT * FROM ' + table + ' WHERE game_id COLLATE BINARY = ?'
                 ' AND player_id COLLATE BINARY = ?')
        params = (claim['game_id'], claim['gsis_id'])
    rows = connection.execute(query, params).fetchmany(2)
    return [dict(row) for row in rows]


def _role_state(usage, baseline, signal):
    numeric = _ROLE_ACTUAL[:-2]
    if (any(not _number(usage.get(k)) for k in numeric)
            or any(not _number(baseline.get(k)) for k in _ROLE_BASELINE)):
        return 'INDETERMINATE', 'MISSING_OR_MALFORMED_ROLE_FIELDS'
    if (type(baseline['history_games']) is not int or baseline['history_games'] < 1):
        return 'INDETERMINATE', 'NO_PRIOR_HISTORY'
    flags = [usage.get(k) for k in _ROLE_ACTUAL[-2:]]
    if any(type(v) is not int or v not in (0, 1) for v in flags):
        return 'INDETERMINATE', 'MALFORMED_ROLE_FLAGS'
    if flags == [1, 1]:
        return 'INDETERMINATE', 'CONTRADICTORY_ROLE_FLAGS'
    pairs = (('opportunities', 'usage_3g', 'usage_delta', 'opportunities_avg_3'),
             ('offense_pct', 'snap_pct_3g', 'snap_delta', 'snap_pct_avg_3'),
             ('targets', 'target_3g', 'target_delta', 'targets_avg_3'),
             ('carries', 'carry_3g', 'carry_delta', 'carries_avg_3'))
    for actual, prior, delta, feature in pairs:
        a, p, d, b = usage[actual], usage[prior], usage[delta], baseline[feature]
        if min(a, p, b) < 0 or (actual == 'offense_pct' and max(a, p, b) > 1):
            return 'INDETERMINATE', 'OUT_OF_RANGE_ROLE_FIELDS'
        # Tolerance accommodates storage arithmetic only, never role thresholds.
        if not (math.isclose(p, b, rel_tol=0, abs_tol=1e-9)
                and math.isclose(a - b, d, rel_tol=0, abs_tol=1e-9)):
            return 'INDETERMINATE', 'ROLE_BASELINE_DELTA_MISMATCH'
    if not math.isclose(usage['opportunities'], usage['targets'] + usage['carries'],
                        rel_tol=0, abs_tol=1e-9):
        return 'INDETERMINATE', 'OPPORTUNITIES_MISMATCH'
    expected = [int(usage['usage_delta'] >= 4 or usage['snap_delta'] >= .15),
                int(usage['usage_delta'] <= -4 or usage['snap_delta'] <= -.15)]
    if flags != expected:
        return 'INDETERMINATE', 'ROLE_FLAG_DELTA_MISMATCH'
    flag = flags[0 if signal == 'ROLE_INCREASE' else 1]
    return ('SUPPORTED' if flag else 'NOT_SUPPORTED'), 'EXISTING_ROLE_FLAG'


def _evaluate(connection, claim):
    game_rows = _one(connection, 'games', claim)
    if len(game_rows) != 1:
        return _row(claim, 'AMBIGUOUS_OUTCOME' if game_rows else 'NO_OUTCOME',
                    'DUPLICATE_GAME' if game_rows else 'MISSING_GAME')
    game = game_rows[0]
    game_audit = _fields(game, ('game_id', 'season', 'week', 'game_type',
                                'completed', 'updated_at'))

    def result(status, reason, state=None, postgame=None, baseline=None):
        return _row(claim, status, reason, state=state, postgame=postgame,
                    baseline=baseline, game=game_audit)

    if type(game.get('completed')) is not int or game['completed'] != 1:
        return result('NOT_FINAL', 'GAME_NOT_FINAL')
    postgame = {}
    sources = {}
    for table in ('player_game_stats', 'player_weekly_usage'):
        rows = _one(connection, table, claim)
        if len(rows) != 1:
            return result('AMBIGUOUS_OUTCOME' if rows else 'NO_OUTCOME',
                          ('DUPLICATE_' if rows else 'MISSING_') + table.upper(),
                          postgame=postgame)
        sources[table] = rows[0]
        postgame[table] = _fields(rows[0], _IDENTITY)
    for row in [game, *sources.values()]:
        if (row.get('game_id') != claim['game_id']
                or any(type(row.get(k)) is not int or row[k] != game.get(k)
                       for k in ('season', 'week'))):
            return result('AMBIGUOUS_OUTCOME', 'OUTCOME_GAME_METADATA_MISMATCH',
                          postgame=postgame)
    if any(row.get('player_id') != claim['gsis_id'] for row in sources.values()):
        return result('NO_OUTCOME', 'IDENTITY_MISMATCH', postgame=postgame)
    teams = [row.get('team') for row in sources.values()]
    if not all(isinstance(t, str) and t.strip() for t in teams) or len(set(teams)) != 1:
        return result('AMBIGUOUS_OUTCOME', 'OUTCOME_TEAM_MISMATCH', postgame=postgame)
    for row in [game, *sources.values()]:
        try:
            if row.get('updated_at') is not None:
                _time(row['updated_at'])
        except (TypeError, ValueError, OverflowError):
            return result('NO_OUTCOME', 'MALFORMED_OUTCOME_TIMESTAMP', postgame=postgame)
    signal = claim['signal_type']
    if signal not in SUPPORTED_SIGNALS:
        return result('UNSUPPORTED_SIGNAL', 'NO_VALIDATED_DIRECT_OUTCOME', postgame=postgame)
    # The frozen builder emits none of these extra comparison constraints.
    # Never silently score a conditional/alternate-denominator proposition.
    if any(claim.get(k) not in (None, '', [], {}) for k in
           ('metric', 'unit', 'denominator', 'comparison_basis', 'conditions')):
        return result('UNSUPPORTED_SIGNAL', 'UNSUPPORTED_COMPARISON_CONSTRAINT',
                      postgame=postgame)
    baseline_rows = _one(connection, 'player_pregame_features', claim)
    if len(baseline_rows) != 1:
        return result('AMBIGUOUS_OUTCOME' if baseline_rows else 'NO_OUTCOME',
                      'DUPLICATE_BASELINE' if baseline_rows else 'MISSING_BASELINE',
                      postgame=postgame)
    baseline = baseline_rows[0]
    usage = sources['player_weekly_usage']
    actual_fields = ('target_share',) if signal == 'TARGET_SHARE' else _ROLE_ACTUAL
    baseline_fields = (('history_games', 'target_share_avg_3')
                       if signal == 'TARGET_SHARE' else _ROLE_BASELINE)
    postgame['player_weekly_usage'].update(_fields(usage, actual_fields))
    baseline_audit = {'player_pregame_features': _fields(baseline, _IDENTITY + baseline_fields)}

    def score(state, reason):
        return result('EVALUATED', reason, state, postgame, baseline_audit)

    if (baseline.get('game_id') != claim['game_id']
            or baseline.get('player_id') != claim['gsis_id']
            or baseline.get('team') != teams[0]
            or any(type(baseline.get(k)) is not int or baseline[k] != game[k]
                   for k in ('season', 'week'))):
        return result('AMBIGUOUS_OUTCOME', 'BASELINE_IDENTITY_METADATA_MISMATCH',
                      postgame=postgame, baseline=baseline_audit)
    try:
        if baseline.get('updated_at') is not None:
            _time(baseline['updated_at'])
    except (TypeError, ValueError, OverflowError):
        return score('INDETERMINATE', 'MALFORMED_BASELINE_TIMESTAMP')
    if signal != 'TARGET_SHARE':
        return score(*_role_state(usage, baseline, signal))
    if claim['direction'] not in ('INCREASE', 'DECREASE'):
        return score('INDETERMINATE', 'NO_EXPLICIT_COMPARABLE_DIRECTION')
    actual, prior = usage.get('target_share'), baseline.get('target_share_avg_3')
    if (type(baseline.get('history_games')) is not int or baseline['history_games'] < 1
            or any(not _number(v) or not 0 <= v <= 1 for v in (actual, prior))):
        return score('INDETERMINATE', 'MISSING_OR_MALFORMED_TARGET_BASELINE_OR_ACTUAL')
    supported = actual > prior if claim['direction'] == 'INCREASE' else actual < prior
    return score('SUPPORTED' if supported else 'NOT_SUPPORTED', 'TARGET_SHARE_VS_PRIOR_3')


def evaluate_outcomes(*, claims, db_path=DEFAULT_DB):
    """Return detached, stable audit rows; no publishing, networking or writes.

    Identical claims collapse. Every distinct variant of a conflicting claim ID
    is rejected, including when one variant is malformed. Non-JSON observations
    become a deterministic invalid marker, retaining a usable ID to poison any
    competing valid variant. A malformed collection raises ValueError before I/O.
    Database/path/schema errors return NO_OUTCOME, never guessed outcomes.
    """
    if not isinstance(claims, (list, tuple)):
        raise ValueError('CLAIMS_MUST_BE_A_SEQUENCE')
    unique = {}
    for claim in claims:
        try:
            detached = _copy(claim)
        except (TypeError, ValueError, OverflowError, RecursionError):
            cid = claim.get('claim_id') if isinstance(claim, dict) else None
            detached = {'claim_id': cid if isinstance(cid, str) else None,
                        'invalid_observation': 'NON_JSON_CLAIM'}
        unique[_canonical(detached)] = detached
    groups = {}
    for key, claim in unique.items():
        cid = claim.get('claim_id') if isinstance(claim, dict) else None
        if isinstance(cid, str) and cid:
            groups.setdefault(cid, []).append(key)
    pending, rows = [], []
    for key in sorted(unique):
        claim = unique[key]
        cid = claim.get('claim_id') if isinstance(claim, dict) else None
        if isinstance(cid, str) and len(groups.get(cid, [])) > 1:
            rows.append(_row(claim, 'INVALID_CLAIM', 'CONFLICTING_CLAIM_ID'))
            continue
        try:
            _, reasons = _validate_frozen_claim(claim, None)
            if not claim.get('game_id'):
                reasons.append('STRICT_PREGAME_GAME_BINDING_REQUIRED')
        except (AttributeError, TypeError, ValueError, OverflowError):
            reasons = ['MALFORMED_CLAIM']
        if reasons:
            rows.append(_row(claim, 'INVALID_CLAIM', sorted(reasons)[0]))
        else:
            pending.append(claim)
    connection = None
    try:
        if pending:
            # Treat input as a filesystem path, never as a caller-controlled URI.
            path = Path(db_path)
            if str(db_path).startswith('file:') or str(db_path) == ':memory:':
                raise ValueError('DB_PATH_MUST_BE_A_FILE_PATH')
            uri = path.absolute().as_uri() + '?mode=ro'
            connection = sqlite3.connect(uri, uri=True)
            connection.row_factory = sqlite3.Row
            connection.set_authorizer(_read_authorizer)
            connection.execute('BEGIN')
            for claim in pending:
                rows.append(_evaluate(connection, claim))
    except (sqlite3.Error, OSError, TypeError, ValueError):
        # A partial database/schema failure invalidates this entire read batch.
        rows = [r for r in rows if r['evaluation_status'] == 'INVALID_CLAIM']
        rows.extend(_row(c, 'NO_OUTCOME', 'DATABASE_READ_UNAVAILABLE') for c in pending)
    finally:
        if connection is not None:
            connection.close()
    rows.sort(key=lambda r: (str(r['claim_id']), r['evaluation_id']))
    evaluations = [r for r in rows if r['evaluation_status'] == 'EVALUATED']
    unevaluated = [r for r in rows if r['evaluation_status'] != 'EVALUATED']
    summary = {'input_count': len(claims), 'unique_count': len(unique),
               'duplicate_count': len(claims) - len(unique),
               'evaluated_count': len(evaluations), 'unevaluated_count': len(unevaluated),
               'status_counts': {}, 'outcome_counts': {}}
    for row in rows:
        for field, counts in [('evaluation_status', 'status_counts'),
                              ('outcome_state', 'outcome_counts')]:
            if row[field] is not None:
                key = row[field]
                summary[counts][key] = summary[counts].get(key, 0) + 1
    return _copy({'evaluations': evaluations, 'unevaluated': unevaluated,
                  'summary': summary, 'safety': dict(SAFETY)})
