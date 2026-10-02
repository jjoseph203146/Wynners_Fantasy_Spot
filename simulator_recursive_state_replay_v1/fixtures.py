"""Synthetic replay fixtures, not historical NFL availability evidence."""
from copy import deepcopy

def sample(c):
    schedule=[]; events=[]
    for i,date in enumerate(('2024-12-22','2024-12-29','2025-01-05')):
        schedule.append(dict(game_id='G'+str(i),home='A',away='B',season=2024 if i<2 else 2025,week=16+i if i<2 else 1,
                             kickoff=date+'T17:00:00Z',game_date=date,available_at='2024-01-01T00:00:00Z'))
        if i<2:
            events.append(dict(game_id='G'+str(i),season=2024,week=16+i,completed_at=date+'T20:00:00Z',
                               observation_time=date+'T21:00:00Z',source_ref='SYNTHETIC',source_version='raw-v1',source_hash='synthetic',
                               provenance='OBSERVED',teams=[dict(team='A',coach_id='CA',points=20+2*i,attempts=30+i,carries=20,
                                   passing_yards=200+10*i,passing_tds=2,rushing_yards=100,rushing_tds=1),
                                   dict(team='B',coach_id='CB',points=10,attempts=25,carries=25,passing_yards=150,passing_tds=1,rushing_yards=120,rushing_tds=0)]))
    ctx=dict(season=2025,week=1,game_id='G2',team='A',opponent='B',kickoff=schedule[2]['kickoff'],
             simulation_step=0,evidence_cutoff='2025-01-05T16:00:00Z',scenario_version='synthetic-v1',
             provenance='OBSERVED',feature_contract_version=c['version'],source_version='snapshot-v1',source_hashes={'raw':'synthetic'})
    return dict(input_mode='SYNTHETIC_FIXTURE',context=ctx,canonical_teams=['A','B'],schedule=schedule,bootstrap=events,
                history_coverage_complete=True,coach_coverage_complete=True,coach_history_start_season=2023,
                coach_assignments={t:dict(coach_id='C'+t,available_at='2024-01-01T00:00:00Z') for t in ('A','B')})

def child(d,parent,score_only=False,simulated=False):
    x=deepcopy(d); x.pop('reference',None); x['bootstrap']=[]; x['parent']=parent
    x['context'].update(game_id='G3',week=2,simulation_step=1,parent_state_version=parent['state_manifest']['state_version'],
                        kickoff='2025-01-12T17:00:00Z',evidence_cutoff='2025-01-12T16:00:00Z',
                        provenance='SIMULATED' if simulated else 'OBSERVED')
    x['schedule'].append(dict(game_id='G3',home='A',away='B',season=2025,week=2,kickoff='2025-01-12T17:00:00Z',game_date='2025-01-12',available_at='2024-01-01T00:00:00Z'))
    e=deepcopy(d['bootstrap'][-1]);e.update(game_id='G2',season=2025,week=1,completed_at='2025-01-05T20:00:00Z',observation_time='2025-01-05T21:00:00Z')
    if score_only:
        for r in e['teams']:
            for k in list(r):
                if k not in {'team','coach_id','points'}: del r[k]
    if simulated: e.update(provenance='SIMULATED',scenario_version=x['context']['scenario_version'])
    x['transitions']=[e]
    return x

def golden_reference(c):
    """Independent hand-calculated 72-field oracle for sample(), not build output."""
    values={}
    for prefix,points,against,attempts,carries,py,ry,pt,rt,rate in [
        ('team_',21,10,30.5,20,205,100,2,1,(30/50+31/51)/2),
        ('opp_',10,21,25,25,150,120,1,0,.5)]:
        own=dict(history_games=2,points_for_last=22 if prefix=='team_' else 10,points_against_last=10 if prefix=='team_' else 22,
                 passing_yards_avg_3=py,rushing_yards_avg_3=ry,passing_tds_avg_3=pt,rushing_tds_avg_3=rt,
                 opponent_pass_yards_allowed_avg_3=py,opponent_rush_yards_allowed_avg_3=ry,
                 opponent_pass_tds_allowed_avg_3=pt,opponent_rush_tds_allowed_avg_3=rt,
                 team_scoring_trend=0,opponent_scoring_trend=0,pace_trend=0,pass_rate_trend=0)
        for n in (3,5):
            for name,value in dict(points_for=points,points_against=against,offensive_plays=attempts+carries,
                                   pass_attempts=attempts,rush_attempts=carries,pass_rate=rate,rush_rate=1-rate,
                                   opponent_points_allowed=points).items(): own[name+'_avg_'+str(n)]=value
        values.update({prefix+k:v for k,v in own.items()})
    values.update(coach_prior_games=2,coach_prior_win_pct=1,coach_prior_points_for_avg=21,coach_prior_points_against_avg=10,
                  opponent_coach_prior_games=2,opponent_coach_prior_win_pct=0,opponent_coach_prior_points_for_avg=10,
                  opponent_coach_prior_points_against_avg=21,team_rest=7,opponent_rest=7)
    assert set(values)==set(c['margin'])
    return dict(game_id='G2',team='A',opponent='B',source_hash=c['training_source']['sha256'],feature_contract_version=c['version'],features=values)
