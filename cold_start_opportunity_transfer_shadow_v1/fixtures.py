"""Fictional deterministic input. Never represents current NFL evidence."""
from .core import METRICS, UNITS


def sample():
    time = '2025-09-01T10:00:00Z'
    context = dict(season=2025, week=1, game_id='SYNTHETIC_GAME', team='BUF',
                   scenario_version='fixture-v1', baseline_version='baseline-v1',
                   evidence_cutoff='2025-09-01T12:00:00Z', kickoff='2025-09-01T17:00:00Z',
                   qb_context_version='qb-v1', roster_version='roster-v1',
                   availability_version='availability-v1', depth_version='depth-v1')
    roster = [dict(gsis_id=pid, player_name=name, team='BUF', position=pos,
                   source_ref='SYNTHETIC_ROSTER', available_at=time)
              for pid, name, pos in [('D', 'Donor Wide', 'WR'), ('R', 'Replacement Wide', 'WR'),
                                     ('S', 'Secondary Tight', 'TE'), ('U', 'Unchanged Runner', 'RB'),
                                     ('Q', 'Protected Quarter', 'QB'), ('F', 'Unchanged Full', 'FB')]]
    data = dict(input_mode='SYNTHETIC_FIXTURE', context=context, roster=roster,
                identity=dict(authority='WFS_IDENTITY_SNAPSHOT', season=2025, week=1,
                              available_at_utc=time, players=roster.copy()),
                availability=[dict(gsis_id=p['gsis_id'], status='OUT' if p['gsis_id'] == 'D' else 'ACTIVE',
                                   authority='WFS_STAT_FORECAST_AVAILABILITY_GATE_V1', version='availability-v1',
                                   source_ref='SYNTHETIC_AVAILABILITY', available_at=time) for p in roster],
                depth=[dict(gsis_id=pid, pos_grp='Offense', pos_abb=pos, pos_slot=slot, pos_rank=rank,
                            version='depth-v1', source_ref='SYNTHETIC_DEPTH', available_at=time)
                       for pid, pos, slot, rank in [('D', 'WR', 1, 1), ('R', 'WR', 1, 2), ('S', 'TE', 1, 1),
                                                   ('U', 'RB', 1, 1), ('Q', 'QB', 1, 1), ('F', 'FB', 1, 1)]],
                baselines=[], capacities=[], donors=[roster[0]], claims=[], evidence=[])
    for p in roster:
        for metric in METRICS:
            value = dict(targets=8, routes=30, carries=16, snaps=50)[metric] if p['gsis_id'] == 'D' else 2
            record = dict(gsis_id=p['gsis_id'], metric=metric, unit=UNITS[metric], definition='synthetic-' + metric,
                          version='baseline-v1', source_ref='SYNTHETIC_BASELINE', available_at=time,
                          authorized_pregame=True, value=value)
            data['baselines'].append(record)
            data['capacities'].append(dict(record, value=value + 100, source_ref='SYNTHETIC_CAPACITY'))
    add_claim(data, 'D', 'R', 'targets', 5)
    add_claim(data, 'D', 'S', 'targets', 1)
    return data


def add_claim(data, donor, recipient, metric, quantity, suffix=''):
    players = {p['gsis_id']: p for p in data['roster']}
    key = donor + recipient + metric + suffix
    data['claims'].append(dict(claim_id=key, donor=players[donor], beneficiary=players[recipient],
                               metric=metric, requested_quantity=quantity, evidence_id=key))
    data['evidence'].append(dict(evidence_id=key, donor_gsis_id=donor, beneficiary_gsis_id=recipient,
                                 metric=metric, increment=quantity, unit=UNITS[metric], definition='synthetic-' + metric,
                                 evidence_class='OBSERVED', source_ref='SYNTHETIC_HISTORY/' + key,
                                 available_at='2025-09-01T10:00:00Z', baseline_version='baseline-v1',
                                 game_id='SYNTHETIC_GAME', team='BUF', donor_absence_or_role_change=True,
                                 comparable_context='Fictional equivalent absence and role',
                                 quantity_basis='Fictional metric-specific incremental count'))
