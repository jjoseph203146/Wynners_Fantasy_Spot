"""Fictional authorized-export fixtures; never current NFL evidence."""
from copy import deepcopy

def quantity(value, definition):
    return dict(value=value, definition=definition, confidence='SUPPORTED_INFERENCE', authorized=True,
                source_ref='SYNTHETIC_MODEL', source_version='model-v1-' + str(value), representation='ABSOLUTE', available_at='2025-09-01T10:00:00Z',
                scope=dict(season=2025, week=1, game_id='SYNTHETIC_GAME', team='BUF'))

def sample():
    c=dict(season=2025, week=1, game_id='SYNTHETIC_GAME', team='BUF', scenario_version='s1',
           parent_scenario_version=None, baseline_version='b1', qb_context_version='q1',
           opportunity_envelope_version='e1', evidence_cutoff='2025-09-01T12:00:00Z', kickoff='2025-09-01T17:00:00Z')
    qb=dict(player_name='Fictional Quarter', gsis_id='Q', team='BUF', position='QB')
    return dict(input_mode='SYNTHETIC_FIXTURE', context=c,
                identity=dict(authority='WFS_IDENTITY_SNAPSHOT', season=2025, week=1,
                              available_at_utc='2025-09-01T09:00:00Z', players=[qb, dict(gsis_id='R', player_name='Fictional Runner', team='BUF', position='RB'),
                                       dict(gsis_id='W', player_name='Fictional Receiver', team='BUF', position='WR')]),
                qb_state=dict(starter=qb, availability=dict(gsis_id='Q', status='ACTIVE',
                     authority='WFS_STAT_FORECAST_AVAILABILITY_GATE_V1', version='a1',
                     source_ref='SYNTHETIC_AVAILABILITY', available_at='2025-09-01T10:00:00Z'),
                     expected_participation=dict(quantity(1, 'attempt_share'), denominator='attempts'),
                     designed_carries=None, scrambles=None),
                envelopes=dict(PASS_ATTEMPTS=quantity(30,'pass_attempts'), DROPBACKS=quantity(35,'dropbacks'),
                               QB_CARRIES=quantity(5,'all_rush_attempts'), TEAM_PLAYS=quantity(65,'offensive_plays')),
                target_policy='V3_ATTEMPTS_X_0_94', team_carries=quantity(28,'all_rush_attempts'),
                relations=dict(ATTEMPTS_WITHIN_DROPBACKS=True, DROPBACKS_WITHIN_PLAYS=True,
                               INDIVIDUAL_SNAPS_WITHIN_PLAYS=True, INDIVIDUAL_ROUTES_WITHIN_DROPBACKS=True),
                player_snapshot=dict(scenario_version='s1', baseline_version='b1', qb_context_version='q1',
                     evidence_cutoff=c['evidence_cutoff'], representation='BASELINE', coverage_complete=True,
                     definitions=dict(TARGETS='player_targets', ROUTES='player_routes', NON_QB_CARRIES='all_rush_attempts',
                                      QB_CARRIES='all_rush_attempts', PLAYER_SNAPS='offensive_player_snaps'),
                     players=[dict(gsis_id='Q', position='QB', targets=0, carries=5, routes=0, snaps=60),
                              dict(gsis_id='R', position='RB', targets=8.2, carries=23, routes=20, snaps=40),
                              dict(gsis_id='W', position='WR', targets=20, carries=0, routes=30, snaps=55)]))

def child(data, parent):
    d=deepcopy(data)
    d['parent']=parent
    d['context'].update(scenario_version='s2', parent_scenario_version='s1', qb_context_version='q2', opportunity_envelope_version='e2')
    d['player_snapshot'].update(scenario_version='s2', qb_context_version='q2')
    return d
