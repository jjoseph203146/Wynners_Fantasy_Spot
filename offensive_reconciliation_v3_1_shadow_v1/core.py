"""Pure reconciliation of explicit, authorized snapshot inputs; never allocates."""
from copy import deepcopy
import math
from shadow_player_analysis_v1.core import digest, resolve, timestamp

CONTRACT = 'offensive_reconciliation_v3_1_shadow_v1'
METRICS = ('PASS_ATTEMPTS', 'DROPBACKS', 'TARGETS', 'ROUTES', 'NON_QB_CARRIES', 'QB_CARRIES', 'PLAYER_SNAPS', 'TEAM_PLAYS')
SAFETY = dict(PRODUCTION_PROJECTION_INFLUENCE='NONE', FANDUEL_ATTACHMENT_INFLUENCE='NONE',
              SOLVER_INFLUENCE='NONE', LINEUP_INFLUENCE='NONE', AVAILABILITY_AUTHORITY_CHANGED='NO',
              LIVE_CHANGED='NO', CANONICAL_FORECAST_CHANGED='NO', COLD_START_V1_CHANGED='NO',
              PRODUCTION_BOUNDARY='SHADOW_ONLY')
EPS = 1e-8


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0


def build(original):
    data = deepcopy(original)
    c = data['context']
    for key in ('season', 'week', 'game_id', 'team', 'scenario_version', 'evidence_cutoff',
                'baseline_version', 'qb_context_version', 'opportunity_envelope_version', 'kickoff'):
        require(c.get(key) is not None, 'MISSING_CONTEXT_' + key)
    require(data['input_mode'] in {'SYNTHETIC_FIXTURE', 'EXPLICIT_PREGAME_SNAPSHOT'}, 'INPUT_MODE')
    cutoff = timestamp(c['evidence_cutoff'])
    require(cutoff < timestamp(c['kickoff']), 'PREGAME_ONLY')
    checks, residuals, signals = [], [], set()

    def check(name, status, reason, **details):
        checks.append(dict(check=name, status=status, reason=reason, **details))

    def timely(r):
        try:
            return bool(r.get('source_ref')) and timestamp(r['available_at']) <= cutoff
        except (ValueError, KeyError, TypeError):
            return False

    def authorized(r):
        return (r is not None and timely(r) and r.get('authorized') is True
                and r.get('confidence') == 'SUPPORTED_INFERENCE' and numeric(r.get('value'))
                and bool(r.get('definition')) and bool(r.get('source_version'))
                and r.get('representation') == 'ABSOLUTE'
                and r.get('scope') == {k: c[k] for k in ('season', 'week', 'game_id', 'team')})

    parent = data.get('parent')
    ancestors = []
    if parent:
        require(parent.get('manifest', {}).get('contract') == CONTRACT, 'PARENT_CONTRACT')
        body = {k: v for k, v in parent.items() if k != 'manifest'}
        require(digest(body) == parent['manifest']['result_sha256'], 'PARENT_HASH_MISMATCH')
        pc = parent['scenario_lineage']['context']
        require(all(pc[k] == c[k] for k in ('season', 'week', 'game_id', 'team')), 'PARENT_SCOPE')
        require(c.get('parent_scenario_version') == pc['scenario_version'], 'PARENT_VERSION_MISMATCH')
        ancestors = parent['scenario_lineage']['ancestor_scenarios'] + [pc['scenario_version']]
        require(c['scenario_version'] not in ancestors, 'DUPLICATE_SCENARIO_REVISION')
        require(timestamp(pc['evidence_cutoff']) <= cutoff, 'CUTOFF_REGRESSION')
    else:
        require(c.get('parent_scenario_version') is None, 'MISSING_PARENT')

    qb = deepcopy(data['qb_state'])
    qb['canonical_starter_gsis_id'] = None
    try:
        claim = dict(qb['starter'], season=c['season'], week=c['week'])
        result = resolve(claim, data['identity'], c['evidence_cutoff'])
        pid = result['gsis_id']
        valid = pid and claim['team'] == c['team'] and claim['position'] == 'QB'
        valid = valid and (not claim.get('gsis_id') or pid == claim['gsis_id'])
        require(valid, 'AMBIGUOUS_OR_INVALID_QB_IDENTITY')
        qb['canonical_starter_gsis_id'] = pid
        check('QB_IDENTITY', 'PASS', 'EXACT_CANONICAL_RESOLVER')
    except (ValueError, KeyError):
        check('QB_IDENTITY', 'BLOCKED', 'AMBIGUOUS_OR_INVALID_QB_IDENTITY')
    availability = qb.get('availability', {})
    availability_ok = (timely(availability) and availability.get('authority') == 'WFS_STAT_FORECAST_AVAILABILITY_GATE_V1'
                       and availability.get('gsis_id') == qb['canonical_starter_gsis_id'] and bool(availability.get('version')))
    check('QB_AVAILABILITY', 'PASS' if availability_ok else 'NOT_EVALUABLE',
          'EXISTING_AUTHORITY' if availability_ok else 'MISSING_OR_UNTIMELY_AUTHORITY')
    participation = qb.get('expected_participation')
    participation_ok = (authorized(participation) and participation['value'] <= 1
                        and participation.get('denominator') in {'attempts', 'dropbacks', 'snaps'})
    if not participation_ok:
        qb['expected_participation'] = None
        signals.add('UNRESOLVED_QB_CONTEXT')
    check('QB_PARTICIPATION', 'PASS' if participation_ok else 'NOT_EVALUABLE',
          'EXPLICIT_DENOMINATOR' if participation_ok else 'UNKNOWN_PARTICIPATION')
    for field in ('designed_carries', 'scrambles'):
        r = qb.get(field)
        if r is not None and not authorized(r):
            qb[field] = None
            check('QB_' + field.upper(), 'NOT_EVALUABLE', 'UNSUPPORTED_COMPONENT')

    envelopes = {}
    inputs = data.get('envelopes', {})
    for metric in METRICS:
        r = inputs.get(metric)
        if authorized(r):
            envelopes[metric] = dict(r, metric=metric, state='AUTHORITATIVE_INPUT')
        else:
            envelopes[metric] = dict(metric=metric, value=None, definition=None, confidence='UNRESOLVED',
                                     state='CONSTRAINT_ONLY' if metric in {'ROUTES', 'PLAYER_SNAPS'} else 'UNRESOLVED')
            if r is not None:
                check('INPUT_' + metric, 'BLOCKED' if not timely(r) else 'NOT_EVALUABLE', 'UNSUPPORTED_OR_FUTURE_INPUT')
                signals.add('UNRESOLVED_QB_CONTEXT')
    # Only the existing explicit V3 target convention is supported; never a QB haircut.
    policy = data.get('target_policy')
    if 'TARGETS' not in inputs and policy:
        require(policy == 'V3_ATTEMPTS_X_0_94', 'UNKNOWN_TARGET_POLICY')
        a = envelopes['PASS_ATTEMPTS']
        if a['value'] is not None:
            envelopes['TARGETS'] = dict(a, metric='TARGETS', value=a['value'] * .94,
                                        definition='player_targets', state='SUPPORTED_DERIVATION', policy=policy)
    rush = data.get('team_carries')
    q = envelopes['QB_CARRIES']
    rush_ok = authorized(rush)
    partition_ok = rush_ok and q['value'] is not None and rush['definition'] == q['definition']
    if partition_ok:
        capacity = rush['value'] - q['value']
        if capacity < -EPS:
            check('RUSH_PARTITION', 'BLOCKED', 'QB_EXCEEDS_TEAM_RUSHING', difference=capacity)
        elif 'NON_QB_CARRIES' not in inputs:
            envelopes['NON_QB_CARRIES'] = dict(rush, metric='NON_QB_CARRIES', value=max(0, capacity),
                                              state='SUPPORTED_DERIVATION', policy='TEAM_MINUS_QB')
        elif envelopes['NON_QB_CARRIES']['value'] is not None:
            require(envelopes['NON_QB_CARRIES']['definition'] == rush['definition'], 'INCOMPATIBLE_RUSH_DEFINITION')
            total = q['value'] + envelopes['NON_QB_CARRIES']['value']
            check('RUSH_PARTITION', 'PASS' if abs(total-rush['value']) <= EPS else 'BLOCKED', 'DISJOINT_QB_NON_QB_ACCOUNTING')
    else:
        check('RUSH_PARTITION', 'NOT_EVALUABLE', 'INCOMPLETE_OR_INCOMPATIBLE_RUSH_INPUTS')

    qb.update(pass_attempts=envelopes['PASS_ATTEMPTS']['value'], dropbacks=envelopes['DROPBACKS']['value'],
              total_carries=envelopes['QB_CARRIES']['value'], team_offensive_plays=envelopes['TEAM_PLAYS']['value'],
              team_carry_context=rush if rush_ok else None)
    if availability_ok and (availability.get('status') in {'OUT', 'INACTIVE'} or availability.get('hard_block') is True):
        signals.add('UNRESOLVED_QB_CONTEXT')
        check('UNAVAILABLE_STARTER', 'BLOCKED' if participation_ok and participation['value'] > 0 else 'NOT_EVALUABLE',
              'UNAVAILABLE_STARTER_NEEDS_SUPPORTED_SUCCESSOR')

    changed = []
    if parent:
        old = parent['team_opportunity_envelope']
        for m in METRICS:
            new, prior = envelopes[m], old[m]
            if (new['value'] != prior['value'] and new['state'] == 'AUTHORITATIVE_INPUT'
                    and prior['state'] == 'AUTHORITATIVE_INPUT'
                    and new.get('source_ref') == prior.get('source_ref')
                    and new.get('source_version') == prior.get('source_version')):
                require(False, 'CONFLICTING_SOURCE_VERSION')
            if new['value'] != prior['value'] or new.get('definition') != prior.get('definition'):
                changed.append(m)
            if new['value'] is not None and prior['value'] is not None and new.get('definition') == prior.get('definition'):
                delta = new['value'] - prior['value']
                if abs(delta) > EPS:
                    if m == 'PASS_ATTEMPTS':
                        signals.add('TEAM_PASS_VOLUME_UP' if delta > 0 else 'TEAM_PASS_VOLUME_DOWN')
                    if m == 'NON_QB_CARRIES':
                        signals.add('NON_QB_RUSH_CAPACITY_UP' if delta > 0 else 'NON_QB_RUSH_CAPACITY_DOWN')
        parent_rush = parent['qb_context'].get('team_carry_context')
        if partition_ok and rush['value'] > 0 and parent_rush and parent_rush['value'] > 0 and old['QB_CARRIES']['value'] is not None:
            if abs(q['value']/rush['value'] - old['QB_CARRIES']['value']/parent_rush['value']) > EPS:
                signals.add('QB_RUSH_SHARE_CHANGED')
        qb_changed = qb != parent['qb_context']
        if qb_changed or changed:
            require(c['qb_context_version'] != pc['qb_context_version'], 'REUSED_QB_CONTEXT_VERSION')
        if changed:
            require(c['opportunity_envelope_version'] != pc['opportunity_envelope_version'], 'REUSED_ENVELOPE_VERSION')
            signals.add('REALLOCATION_REQUIRED')
        elif qb_changed:
            signals.add('UNCHANGED_UNSUPPORTED_CAUSAL_CHANGE')
    else:
        qb_changed = False

    snapshot = data['player_snapshot']
    expected = {k: c[k] for k in ('scenario_version', 'baseline_version', 'qb_context_version', 'evidence_cutoff')}
    version_ok = all(snapshot.get(k) == v for k, v in expected.items())
    # Exactly one absolute snapshot. No ledger application or additive volume delta API.
    version_ok = version_ok and snapshot.get('representation') in {'BASELINE', 'POST_TRANSFER_ABSOLUTE'}
    if snapshot.get('representation') == 'POST_TRANSFER_ABSOLUTE':
        version_ok = version_ok and bool(snapshot.get('input_sha256')) and bool(snapshot.get('ledger_ref'))
    check('PLAYER_SNAPSHOT_LINEAGE', 'PASS' if version_ok else 'BLOCKED',
          'VERSION_MATCH' if version_ok else 'BLOCKED_VERSION_MISMATCH')
    cold = data.get('cold_start_reference')
    if cold:
        ok = all(cold.get(k) == v for k, v in expected.items()) and bool(cold.get('input_sha256'))
        check('COLD_START_LINEAGE', 'PASS' if ok else 'BLOCKED', 'VERSION_MATCH' if ok else 'BLOCKED_VERSION_MISMATCH')
    rows = snapshot['players']
    require(len({r['gsis_id'] for r in rows}) == len(rows), 'DUPLICATE_PLAYER')
    require(all(r.get('position') in {'QB','RB','FB','WR','TE'} and r.get('gsis_id') for r in rows), 'PLAYER_IDENTITY_FIELDS')
    roster_ids = {p['gsis_id']: p for p in data['identity']['players'] if p.get('team') == c['team']}
    for row in rows:
        identity_row = roster_ids.get(row['gsis_id'])
        valid = identity_row is not None and identity_row.get('position') == row['position']
        if valid:
            resolved = resolve(dict(identity_row, season=c['season'], week=c['week']), data['identity'], c['evidence_cutoff'])
            valid = resolved['gsis_id'] == row['gsis_id']
        if not valid:
            version_ok = False
            check('PLAYER_IDENTITY_' + row['gsis_id'], 'BLOCKED', 'UNRESOLVED_PLAYER_IDENTITY')
    totals = {}
    for metric in ('TARGETS', 'ROUTES', 'NON_QB_CARRIES', 'QB_CARRIES', 'PLAYER_SNAPS'):
        field = 'carries' if metric.endswith('CARRIES') else {'TARGETS':'targets','ROUTES':'routes','PLAYER_SNAPS':'snaps'}[metric]
        selected = [r for r in rows if (r['position'] == 'QB' if metric == 'QB_CARRIES' else r['position'] != 'QB' if metric == 'NON_QB_CARRIES' else True)]
        vals = [r.get(field) for r in selected]
        require(all(v is None or numeric(v) for v in vals), 'INVALID_PLAYER_QUANTITY')
        known = sum(v for v in vals if v is not None)
        complete = snapshot.get('coverage_complete') is True and all(v is not None for v in vals)
        env = envelopes[metric]
        compatible = snapshot.get('definitions', {}).get(metric) == env.get('definition') and env['value'] is not None
        residual = env['value'] - known if compatible and version_ok else None
        status = ('BLOCKED' if residual is not None and residual < -EPS else
                  'NOT_EVALUABLE' if not compatible or not complete or not version_ok else
                  'WARNING' if residual > EPS else 'PASS')
        check('PLAYER_' + metric, status, 'ENVELOPE_COMPARISON', known_total=known, coverage_complete=complete)
        residuals.append(dict(metric=metric, residual=residual if complete else None,
                              known_excess=max(0, -residual) if residual is not None else None,
                              allocated=known, coverage_complete=complete, status=status))
        totals[metric] = (known, complete)
        if status == 'BLOCKED':
            signals.update({'PLAYER_OPPORTUNITY_INCONSISTENT', 'REALLOCATION_REQUIRED'})
        elif status in {'PASS', 'WARNING'}:
            signals.add('PLAYER_OPPORTUNITY_WITHIN_ENVELOPE')
    if (partition_ok and version_ok and all(totals[m][1] for m in ('QB_CARRIES','NON_QB_CARRIES'))
            and all(snapshot.get('definitions', {}).get(m) == rush['definition'] for m in ('QB_CARRIES','NON_QB_CARRIES'))):
        combined = totals['QB_CARRIES'][0] + totals['NON_QB_CARRIES'][0]
        check('COMBINED_RUSHING', 'BLOCKED' if combined > rush['value'] + EPS else 'WARNING' if combined < rush['value'] - EPS else 'PASS',
              'TEAM_RUSH_COMPARISON', residual=rush['value']-combined)
    else:
        check('COMBINED_RUSHING', 'NOT_EVALUABLE', 'INCOMPLETE_RUSH_ACCOUNTING')
    if data.get('relations', {}).get('QB_COMPONENTS_DISJOINT_SUBSET') is True:
        components = [qb.get('designed_carries'), qb.get('scrambles')]
        if all(r is not None for r in components) and q['value'] is not None:
            check('QB_COMPONENTS', 'PASS' if sum(r['value'] for r in components) <= q['value'] + EPS else 'BLOCKED', 'DISJOINT_SUBSET_NOT_EXHAUSTIVE')
        else:
            check('QB_COMPONENTS', 'NOT_EVALUABLE', 'MISSING_COMPONENTS')
    # Cross-metric relations must be explicitly declared by the authorized export.
    relations = data.get('relations', {})
    for name, left, right in [('ATTEMPTS_WITHIN_DROPBACKS','PASS_ATTEMPTS','DROPBACKS'),
                              ('DROPBACKS_WITHIN_PLAYS','DROPBACKS','TEAM_PLAYS')]:
        a,b = envelopes[left], envelopes[right]
        supported = relations.get(name) is True and a['value'] is not None and b['value'] is not None
        check(name, 'NOT_EVALUABLE' if not supported else 'PASS' if a['value'] <= b['value'] + EPS else 'BLOCKED', 'DECLARED_SUBSET_ONLY')
    for name, field, metric in [('INDIVIDUAL_SNAPS_WITHIN_PLAYS','snaps','TEAM_PLAYS'),
                                ('INDIVIDUAL_ROUTES_WITHIN_DROPBACKS','routes','DROPBACKS')]:
        env = envelopes[metric]
        supported = relations.get(name) is True and env['value'] is not None and version_ok
        vals = [r.get(field) for r in rows]
        status = 'NOT_EVALUABLE'
        if supported and any(v is not None and v > env['value'] + EPS for v in vals):
            status = 'BLOCKED'
        elif supported and snapshot.get('coverage_complete') and all(v is not None for v in vals):
            status = 'PASS'
        check(name, status, 'INDIVIDUAL_NOT_TEAM_SUM')
    qb.update(pass_attempts=envelopes['PASS_ATTEMPTS']['value'], dropbacks=envelopes['DROPBACKS']['value'],
              total_carries=envelopes['QB_CARRIES']['value'], team_offensive_plays=envelopes['TEAM_PLAYS']['value'])
    priority = {'PASS':0, 'WARNING':1, 'NOT_EVALUABLE':2, 'BLOCKED':3}
    overall = max((r['status'] for r in checks), key=priority.get)
    request = None
    if 'REALLOCATION_REQUIRED' in signals:
        request = dict(status='REQUEST_ONLY_NO_ALLOCATION', parent_scenario_version=c.get('parent_scenario_version'),
                       scenario_version=c['scenario_version'], baseline_version=c['baseline_version'],
                       qb_context_version=c['qb_context_version'], opportunity_envelope_version=c['opportunity_envelope_version'],
                       evidence_cutoff=c['evidence_cutoff'], changed_metrics=changed,
                       reason='SUPPORTED_ENVELOPE_CHANGE_OR_PLAYER_INCONSISTENCY',
                       evidence_references=sorted({r['source_ref'] for r in inputs.values() if r.get('source_ref')}),
                       requires_authorized_pretransfer_baseline=True)
    output = dict(qb_context=qb, team_opportunity_envelope=envelopes, reconciliation_checks=checks,
                  residuals=residuals, scenario_lineage=dict(context=c, ancestor_scenarios=ancestors,
                  parent_result_sha256=parent['manifest']['result_sha256'] if parent else None,
                  player_snapshot_sha256=digest(snapshot), changed_metrics=changed,
                  cold_start_reference=deepcopy(cold), qb_state_changed=qb_changed,
                  evidence_references=sorted({r['source_ref'] for r in inputs.values() if r.get('source_ref')}),
                  signals=sorted(signals), reallocation_request=request))
    output['manifest'] = dict(contract=CONTRACT, input_mode=data['input_mode'], input_sha256=digest(original),
                              result_sha256=digest(output), overall_status=overall, safety=SAFETY.copy())
    return output
