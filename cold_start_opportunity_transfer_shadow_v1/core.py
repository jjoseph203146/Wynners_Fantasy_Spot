"""Pure snapshot allocator: no database, network, clock, or filesystem I/O.

Inputs are explicit, versioned pregame exports, not new availability authority.
Evidence quantities must already be established upstream; this module does not
extract quantities from text or estimate missing football metrics.
"""
from collections import defaultdict
import math

from shadow_player_analysis_v1.core import digest, resolve, timestamp

CONTRACT = 'cold_start_opportunity_transfer_shadow_v1'
METRICS = ('targets', 'routes', 'carries', 'snaps')
POSITIONS = {'QB', 'RB', 'FB', 'WR', 'TE'}
UNITS = dict(targets='targets', routes='player_route_participations',
             carries='carries', snaps='offensive_player_snaps')
SAFETY = dict(PRODUCTION_PROJECTION_INFLUENCE='NONE', FANDUEL_ATTACHMENT_INFLUENCE='NONE',
              SOLVER_INFLUENCE='NONE', LINEUP_INFLUENCE='NONE',
              AVAILABILITY_AUTHORITY_CHANGED='NO', LIVE_CHANGED='NO',
              CANONICAL_FORECAST_CHANGED='NO', PRODUCTION_BOUNDARY='SHADOW_ONLY')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def unique(rows, key):
    result = {}
    for row in rows:
        k = key(row)
        require(k not in result or result[k] == row, 'CONFLICTING_DUPLICATE')
        result[k] = row
    return [result[k] for k in sorted(result)]


def build(data):
    """One team/game/scenario per call. Return JSON-serializable shadow tables."""
    context = data['context']
    required = ('season', 'week', 'game_id', 'team', 'scenario_version', 'baseline_version',
                'evidence_cutoff', 'kickoff', 'qb_context_version', 'roster_version',
                'availability_version', 'depth_version')
    require(all(context.get(k) is not None and context[k] != '' for k in required), 'MISSING_CONTEXT')
    cutoff = timestamp(context['evidence_cutoff'])
    require(cutoff < timestamp(context['kickoff']), 'PREGAME_ONLY')
    require(data['input_mode'] in {'SYNTHETIC_FIXTURE', 'EXPLICIT_PREGAME_SNAPSHOT'}, 'INPUT_MODE')
    lineage = {k: context[k] for k in required}
    lineage['contract'] = CONTRACT
    identity = data['identity']

    def timed(record):
        return bool(record.get('source_ref')) and timestamp(record['available_at']) <= cutoff

    def exact(claim):
        try:
            result = resolve(dict(claim, season=context['season'], week=context['week']), identity,
                             context['evidence_cutoff'])
            pid = result['gsis_id']
            if claim.get('gsis_id') and claim['gsis_id'] != pid:
                return None
            return pid if claim.get('team') == context['team'] else None
        except (ValueError, KeyError):
            return None

    roster = {}
    quarantine = []
    for p in unique(data['roster'], lambda p: p['gsis_id']):
        require(p['team'] == context['team'] and p['position'] in POSITIONS, 'ROSTER_CONTEXT')
        require(timed(p), 'ROSTER_LINEAGE')
        pid = exact(p)
        if pid != p['gsis_id']:
            quarantine.append(dict(kind='ROSTER', claim=p, reason='UNRESOLVED_IDENTITY'))
        else:
            roster[pid] = p

    availability = {}
    for a in unique(data['availability'], lambda a: a['gsis_id']):
        require(a['version'] == context['availability_version'] and timed(a), 'AVAILABILITY_LINEAGE')
        require(a['authority'] == 'WFS_STAT_FORECAST_AVAILABILITY_GATE_V1', 'AVAILABILITY_AUTHORITY')
        availability[a['gsis_id']] = a

    def blocked(pid):
        a = availability.get(pid)
        return a is not None and (a['status'] in {'OUT', 'INACTIVE'} or a.get('hard_block') is True)

    def available(pid):
        a = availability.get(pid)
        return a is not None and a['status'] in {'ACTIVE', 'AVAILABLE', 'QUESTIONABLE', 'DOUBTFUL'} and not blocked(pid)

    baseline = {}
    capacities = {}
    for table, target in [('baselines', baseline), ('capacities', capacities)]:
        for r in unique(data[table], lambda r: (r['gsis_id'], r['metric'])):
            require(r['metric'] in METRICS, 'METRIC')
            require(r['version'] == context['baseline_version'] and timed(r), 'BASELINE_LINEAGE')
            require(r['unit'] == UNITS[r['metric']] and bool(r.get('definition')), 'METRIC_DEFINITION')
            require(r.get('authorized_pregame') is True, 'UNAUTHORIZED_BASELINE')
            require(r['value'] is None or number(r['value']), 'INVALID_QUANTITY')
            target[r['gsis_id'], r['metric']] = r

    for metric in METRICS:
        definitions = {r['definition'] for (pid, m), r in baseline.items() if m == metric and pid in roster}
        require(len(definitions) <= 1, 'INCOMPATIBLE_TEAM_METRIC_DEFINITIONS')

    depth = unique(data['depth'], lambda r: (r['gsis_id'], r['pos_grp'], r['pos_abb'], str(r['pos_slot'])))
    for r in depth:
        require(r['version'] == context['depth_version'] and timed(r), 'DEPTH_LINEAGE')
        require(number(r['pos_rank']) and r['pos_rank'] > 0, 'DEPTH_RANK')
        require(r['pos_grp'] and r['pos_abb'] in POSITIONS and r['pos_slot'] is not None, 'DEPTH_LANE')

    donors = set()
    for donor in unique(data['donors'], lambda r: digest(r)):
        pid = exact(donor)
        if pid not in roster:
            quarantine.append(dict(kind='DONOR', claim=donor, reason='BLOCKED_DONOR_IDENTITY'))
        elif roster[pid]['position'] == 'QB':
            quarantine.append(dict(kind='DONOR', claim=donor, reason='QB_OUTSIDE_SCOPE'))
        elif not blocked(pid):
            quarantine.append(dict(kind='DONOR', claim=donor, reason='NO_AUTHORITATIVE_VACANCY'))
        else:
            donors.add(pid)

    # Lane membership is not numerical support. Tied successors fail closed.
    direct = defaultdict(set)
    lane_key = lambda r: (r['pos_grp'], r['pos_abb'], r['pos_slot'])
    for donor in sorted(donors):
        lanes = {lane_key(r) for r in depth if r['gsis_id'] == donor}
        for lane in sorted(lanes, key=str):
            candidates = [r for r in depth if lane_key(r) == lane and r['pos_abb'] != 'QB'
                          and not blocked(r['gsis_id'])]
            if candidates:
                rank = min(r['pos_rank'] for r in candidates)
                ids = {r['gsis_id'] for r in candidates if r['pos_rank'] == rank}
                if len(ids) == 1 and next(iter(ids)) in roster and available(next(iter(ids))):
                    direct[donor].update(ids)
                else:
                    quarantine.append(dict(kind='DEPTH', donor_gsis_id=donor, lane=lane,
                                           reason='AMBIGUOUS_SUCCESSOR'))

    evidence = {r['evidence_id']: r for r in unique(data['evidence'], lambda r: r['evidence_id'])}
    claims = unique(data['claims'], lambda r: r['claim_id'])
    ledger = []
    seen_edges = set()
    for c in claims:
        require(c['metric'] in METRICS, 'METRIC')
        donor, recipient, metric = exact(c['donor']), exact(c['beneficiary']), c['metric']
        key = (donor, recipient, metric)
        # A second claim for an edge is not extra workload, even with a new ID.
        edge = (donor or digest(c['donor']), recipient or digest(c['beneficiary']), metric)
        require(edge not in seen_edges, 'DUPLICATE_TRANSFER_EDGE')
        seen_edges.add(edge)
        relation = 'DIRECT' if recipient in direct.get(donor, set()) else 'SECONDARY'
        row = dict(lineage, claim_id=c['claim_id'], donor_gsis_id=donor,
                   beneficiary_gsis_id=recipient, metric=metric, unit=UNITS[metric],
                   relationship=relation, requested_quantity=c.get('requested_quantity'),
                   supported_quantity=0.0, allocated_quantity=0.0, confidence='UNRESOLVED',
                   reason='UNSUPPORTED_EVIDENCE', evidence_id=c.get('evidence_id'),
                   capacity_rule='TOTAL_POST_TRANSFER_CEILING', candidate=True)
        ledger.append(row)
        if donor not in donors:
            row['reason'] = 'BLOCKED_DONOR'
            continue
        if recipient not in roster:
            row['reason'] = 'UNRESOLVED_BENEFICIARY_IDENTITY'
            quarantine.append(dict(kind='BENEFICIARY', claim=c, reason=row['reason']))
            continue
        if roster[recipient]['position'] == 'QB' or recipient == donor:
            row['reason'] = 'INELIGIBLE_BENEFICIARY'
            continue
        if not available(recipient):
            row['reason'] = 'UNAVAILABLE_OR_UNKNOWN_BENEFICIARY'
            continue
        source = baseline.get((donor, metric))
        if source is None or source['value'] is None:
            row.update(allocated_quantity=None, reason='MISSING_METRIC_SOURCE')
            continue
        e = evidence.get(c.get('evidence_id'))
        if c.get('confidence') == 'COLD_START_INFERENCE':
            row.update(confidence='COLD_START_INFERENCE', reason='COLD_START_NO_QUANTIFIED_SUPPORT')
            continue
        if not e:
            continue
        row['evidence_source'] = e.get('source_ref')
        valid = (timed(e) and e.get('baseline_version') == context['baseline_version']
                 and e.get('game_id') == context['game_id'] and e.get('team') == context['team']
                 and e.get('donor_gsis_id') == donor and e.get('beneficiary_gsis_id') == recipient
                 and e.get('metric') == metric and e.get('unit') == UNITS[metric]
                 and e.get('definition') == source['definition']
                 and e.get('donor_absence_or_role_change') is True
                 and bool(e.get('comparable_context')) and bool(e.get('quantity_basis'))
                 and e.get('evidence_class') == 'OBSERVED' and number(e.get('increment')))
        if not valid:
            row['reason'] = 'INCOMPATIBLE_OR_UNTIMELY_EVIDENCE'
            continue
        requested = c.get('requested_quantity')
        if not number(requested):
            row['reason'] = 'UNKNOWN_REQUESTED_QUANTITY'
            continue
        recipient_base = baseline.get((recipient, metric))
        cap = capacities.get((recipient, metric))
        if (recipient_base is None or recipient_base['value'] is None or cap is None or cap['value'] is None
                or cap['definition'] != source['definition'] or recipient_base['definition'] != source['definition']):
            row['reason'] = 'UNSUPPORTED_RECIPIENT_CAPACITY'
            continue
        row.update(supported_quantity=min(requested, e['increment']), confidence='SUPPORTED_INFERENCE',
                   reason='SUPPORTED', capacity_source=cap['source_ref'])

    # Add zero-quantity direct candidates even when no claim was supplied.
    for donor in sorted(donors):
        for recipient in sorted(direct[donor]):
            for metric in METRICS:
                if any(r['donor_gsis_id'] == donor and r['beneficiary_gsis_id'] == recipient
                       and r['metric'] == metric for r in ledger):
                    continue
                source = baseline.get((donor, metric))
                known = source is not None and source['value'] is not None
                ledger.append(dict(lineage, claim_id=None, donor_gsis_id=donor, beneficiary_gsis_id=recipient,
                                   metric=metric, unit=UNITS[metric], relationship='DIRECT', candidate=True,
                                   requested_quantity=None, supported_quantity=0.0,
                                   allocated_quantity=0.0 if known else None,
                                   confidence='COLD_START_INFERENCE' if known else 'UNRESOLVED',
                                   reason='LANE_ONLY_NO_QUANTIFIED_SUPPORT' if known else 'MISSING_METRIC_SOURCE',
                                   evidence_id=None, capacity_rule='TOTAL_POST_TRANSFER_CEILING'))

    remaining = {(d, m): baseline.get((d, m), {}).get('value') for d in donors for m in METRICS}
    headroom = {k: max(0.0, v['value'] - baseline[k]['value']) for k, v in capacities.items()
                if v['value'] is not None and k in baseline and baseline[k]['value'] is not None}
    for tier in ('DIRECT', 'SECONDARY'):
        groups = defaultdict(list)
        for r in ledger:
            if r['relationship'] == tier and r['supported_quantity'] > 0:
                groups[r['donor_gsis_id'], r['metric']].append(r)
        tentative = []
        for key, rows in sorted(groups.items()):
            total = sum(r['supported_quantity'] for r in rows)
            scale = min(1.0, remaining[key] / total)
            for r in rows:
                tentative.append((r, r['supported_quantity'] * scale))
        by_recipient = defaultdict(float)
        for r, amount in tentative:
            by_recipient[r['beneficiary_gsis_id'], r['metric']] += amount
        for r, amount in tentative:
            key = (r['beneficiary_gsis_id'], r['metric'])
            scale = min(1.0, headroom[key] / by_recipient[key]) if by_recipient[key] else 0.0
            r['allocated_quantity'] = amount * scale
            r['reason'] = 'ALLOCATED' if abs(r['allocated_quantity'] - r['supported_quantity']) < 1e-9 else 'SUPPLY_OR_CAPACITY_CLIPPED'
        # Commit all allocations simultaneously; input ordering cannot choose winners.
        for r, reserved in tentative:
            key = (r['beneficiary_gsis_id'], r['metric'])
            remaining[r['donor_gsis_id'], r['metric']] -= reserved
            headroom[key] -= r['allocated_quantity']

    balances = []
    for donor in sorted(donors):
        for metric in METRICS:
            rows = [r for r in ledger if r['donor_gsis_id'] == donor and r['metric'] == metric]
            value = baseline.get((donor, metric), {}).get('value')
            allocated = sum(r['allocated_quantity'] or 0.0 for r in rows)
            require(value is None or -1e-9 <= allocated <= value + 1e-9, 'CONSERVATION_FAILURE')
            balances.append(dict(lineage, donor_gsis_id=donor, metric=metric, vacated=value,
                                 allocated=allocated if value is not None else None,
                                 residual=max(0.0, value - allocated) if value is not None else None,
                                 conservation_status='PASS' if value is not None else 'NOT_EVALUABLE',
                                 reasons=sorted({r['reason'] for r in rows}) or ['NO_SUPPORTED_CLAIMS']))
    players = []
    for pid, p in sorted(roster.items()):
        for metric in METRICS:
            base = baseline.get((pid, metric), {})
            value = base.get('value')
            rows = [r for r in ledger if r['beneficiary_gsis_id'] == pid and r['metric'] == metric] if p['position'] != 'QB' else []
            credit = sum(r['allocated_quantity'] or 0.0 for r in rows)
            unknown = any(r['allocated_quantity'] is None for r in rows)
            debit = value if pid in donors else 0.0
            classification = ('UNAVAILABLE_DONOR' if pid in donors else
                              'DIRECT_REPLACEMENT' if credit and any(r['relationship'] == 'DIRECT' and (r['allocated_quantity'] or 0) > 0 for r in rows) else
                              'SECONDARY_BENEFICIARY' if credit else 'UNCERTAIN_BENEFICIARY' if rows else 'UNCHANGED_TEAMMATE')
            players.append(dict(lineage, gsis_id=pid, player_name=p['player_name'], position=p['position'],
                                depth_lanes=[{k: r[k] for k in ('pos_grp', 'pos_abb', 'pos_slot', 'pos_rank')}
                                             for r in depth if r['gsis_id'] == pid],
                                availability=availability.get(pid, {}).get('status', 'UNKNOWN'),
                                metric=metric, unit=UNITS[metric], baseline=value,
                                baseline_source=base.get('source_ref'), debit=debit,
                                allocated_credit=None if unknown else credit, known_allocated_credit=credit,
                                hypothetical_post_transfer=None if value is None or debit is None or unknown else value - debit + credit,
                                candidate=bool(rows), classification=classification,
                                relationship=sorted({r['relationship'] for r in rows}),
                                confidence='SUPPORTED_INFERENCE' if credit else 'UNRESOLVED' if unknown or value is None else
                                           'UNRESOLVED' if any(r['confidence'] == 'UNRESOLVED' for r in rows) else
                                           'COLD_START_INFERENCE' if rows else 'OBSERVED',
                                reason=sorted({r['reason'] for r in rows}) or
                                       (['MISSING_BASELINE'] if value is None else ['QB_OUTSIDE_SCOPE'] if p['position'] == 'QB' else ['NO_TRANSFER']),
                                completeness='INCOMPLETE' if value is None or unknown else 'COMPLETE'))
    team_checks = []
    for metric in METRICS:
        rows = [r for r in players if r['metric'] == metric]
        complete = bool(rows) and not quarantine and all(r['completeness'] == 'COMPLETE' for r in rows)
        residual = sum(b['residual'] or 0 for b in balances if b['metric'] == metric)
        gap = sum(r['hypothetical_post_transfer'] for r in rows) + residual - sum(r['baseline'] for r in rows) if complete else None
        require(gap is None or abs(gap) < 1e-8, 'TEAM_CONSERVATION_FAILURE')
        team_checks.append(dict(metric=metric, conservation_status='PASS' if complete else 'NOT_EVALUABLE', gap=gap))
    ledger.sort(key=lambda r: (r['donor_gsis_id'] or '', r['beneficiary_gsis_id'] or '', r['metric'], r['claim_id'] or ''))
    return dict(players=players, ledger=ledger, balances=balances,
                manifest=dict(lineage, input_mode=data['input_mode'], safety=SAFETY.copy(),
                              input_sha256=digest(data), identity_method='FANDUEL_EXACT_RESOLVER_V1',
                              identity_sha256=digest(identity), quarantine=quarantine, team_conservation=team_checks,
                              roster_count=len(roster), player_metric_rows=len(players),
                              coverage_gaps=[dict(gsis_id=r['gsis_id'], metric=r['metric'], reason=r['reason'])
                                             for r in players if r['completeness'] == 'INCOMPLETE']))
