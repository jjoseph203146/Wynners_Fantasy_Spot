#!/usr/bin/env python3
"""Score prospective captures and prove contest-pool hindsight optimum.
Read-only on NFL sources. Writes only data/lineup_learning_v1/reviews.
Never treats missing outcomes as zero or changes production model weights.
"""
from pathlib import Path
from contextlib import closing
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile

ALIASES = {'LA':'LAR', 'WAS':'WSH', 'JAC':'JAX'}
SLOTS = ['QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX','DST']
SCALE = 1000000

class Pending(RuntimeError):
    pass

def canon(v):
    v=str(v).strip().upper()
    return ALIASES.get(v,v)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def atomic_json(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=p.parent,delete=False) as f:
        tmp=Path(f.name)
        try:
            json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(p)

def flag(v):
    return v is not None and str(v) in {'1','1.0','True','true'}

def numeric(v, label):
    try:v=float(v)
    except (ValueError,TypeError):raise Pending('INVALID_'+label)
    if not math.isfinite(v):raise Pending('NONFINITE_'+label)
    return v

def pid(v):
    s=str(v or '').strip()
    if s.startswith('GSIS:'):s=s[5:]
    if not re.fullmatch(r'00-\d{7}',s):raise Pending('UNRESOLVED_CONTEST_PLAYER_ID')
    return s

def validate_capture(path):
    m=read(path/'manifest.json')
    if m.get('contract')!='WFS_LINEUP_PROSPECTIVE_CAPTURE_V1' or m.get('status')!='PROSPECTIVE_CAPTURE_COMPLETE':
        raise RuntimeError('INVALID_CAPTURE_CONTRACT')
    for name,h in m['files'].items():
        if Path(name).name!=name or sha(path/name)!=h:
            raise RuntimeError('CAPTURE_HASH_MISMATCH:'+name)
    return m

def contest_pool(path,m):
    """Full contest rows, independent of model, injury, lock/exclude filters."""
    games=read(path/'schedule.json');by_id={str(g['game_id']):g for g in games}
    rows=[]
    if m['format']=='CLASSIC':
        raw=read(path/'full_contest_pool.json')['rows']
        authority=read(path/'contest_pool.json')['rows']
        for r in raw:
            # Unsupported source rows block the full-contest benchmark rather
            # than shrinking its universe based on model eligibility.
            if not flag(r.get('contest_eligible')):continue
            key=(str(r['source_file']),str(r['source_row']),str(r['slate_key']))
            matches=[a for a in authority if (str(a['source_file']),str(a['source_row']),str(a['slate_key']))==key]
            if len(matches)!=1:raise Pending('FULL_CONTEST_IDENTITY_AUTHORITY_MISSING')
            a=matches[0]
            position=str(a.get('solver_position') or a.get('position') or '').upper()
            if position in {'D','DEF','D/ST'}:position='DST'
            if position not in {'QB','RB','WR','TE','DST'}:raise Pending('UNRESOLVED_CONTEST_POSITION')
            team=canon(r['team_internal']);gid=str(r['game_id'])
            if canon(a['team_solver'])!=team or str(a['game_id'])!=gid:
                raise Pending('CONTEST_IDENTITY_TEAM_GAME_CONFLICT')
            if position=='DST':identity='DST:'+team
            else:
                candidates=[]
                for name in ('injury_gsis_id','position_identity','projection_identity'):
                    v=a.get(name)
                    if v and (str(v).startswith('GSIS:') or re.fullmatch(r'00-\d{7}',str(v))):candidates.append(pid(v))
                if len(set(candidates))!=1:raise Pending('CONTEST_IDENTITY_MISSING_OR_CONFLICTING')
                identity=candidates[0]
            salary=numeric(r['salary'],'CONTEST_SALARY')
            rows.append({'player_id':identity,'game_id':gid,'team':team,'player':str(r['player']),
                         'position':position,'salary':salary,'mvp_salary':None})
    elif m['format']=='SINGLE_GAME':
        if len(by_id)!=1:raise Pending('SINGLE_GAME_SCOPE_AMBIGUOUS')
        gid=next(iter(by_id))
        for r in read(path/'full_identity_pool.json')['rows']:
            position=str(r['position']).upper();team=canon(r['team'])
            if position in {'D','DEF','D/ST'}:position='DST'
            identity='DST:'+team if position=='DST' else pid(r['player_id'])
            salary=numeric(r['salary'],'CONTEST_SALARY')
            rows.append({'player_id':identity,'game_id':gid,'team':team,'player':str(r['player']),
                         'position':position,'salary':salary,'mvp_salary':salary*1.5})
    else:raise RuntimeError('UNKNOWN_FORMAT')
    if not rows:raise Pending('EMPTY_CONTEST_POOL')
    if len({(r['game_id'],r['player_id']) for r in rows})!=len(rows):raise Pending('DUPLICATE_CONTEST_IDENTITY')
    for r in rows:
        g=by_id.get(r['game_id'])
        if g is None or r['team'] not in {canon(g['away_team']),canon(g['home_team'])}:raise Pending('CONTEST_GAME_TEAM_MISMATCH')
        if r['salary']<=0 or r['salary']!=int(r['salary']):raise Pending('INVALID_CONTEST_SALARY')
        r['opponent']=canon(g['home_team']) if r['team']==canon(g['away_team']) else canon(g['away_team'])
    return rows

def source_outcomes(root,gids):
    """One SQLite read transaction; every requested slate game must be final."""
    marks=','.join('?' for _ in gids)
    with closing(sqlite3.connect(f'file:{root / "data/nfl.db"}?mode=ro',uri=True)) as c:
        c.row_factory=sqlite3.Row;c.execute('BEGIN')
        games=[dict(r) for r in c.execute('SELECT * FROM games WHERE game_id IN ('+marks+')',gids)]
        if len(games)!=len(gids) or any(not flag(g['completed']) or g['away_score'] is None or g['home_score'] is None for g in games):
            raise Pending('SLATE_GAMES_NOT_ALL_FINAL')
        actual=[dict(r) for r in c.execute('SELECT * FROM player_game_stats WHERE game_id IN ('+marks+')',gids)]
    dstpath=root/'data/parquet/nfl_current_dst_postgame_actuals.parquet'
    if dstpath.exists():
        import pandas as pd
        df=pd.read_parquet(dstpath)
        dst=df.loc[df['game_id'].astype(str).isin(gids)].to_dict('records')
    else:dst=[]
    return games,actual,dst

def attach_actuals(pool,actual,dst):
    output=[];unresolved=[]
    for p in pool:
        if p['player_id'].startswith('DST:'):
            matches=[r for r in dst if str(r['game_id'])==p['game_id'] and canon(r['team'])==p['team']]
            valid=len(matches)==1 and flag(matches[0].get('fanduel_scoring_verified')) and flag(matches[0].get('completed'))
            key='fanduel_dst_points';source='VERIFIED_DST_ACTUALS'
        else:
            matches=[r for r in actual if str(r['game_id'])==p['game_id'] and str(r.get('player_id'))==p['player_id']]
            valid=len(matches)==1 and flag(matches[0].get('fanduel_points_verified')) and canon(matches[0]['team'])==p['team']
            key='fanduel_points';source=matches[0].get('lineup_actual_source','VERIFIED_PLAYER_GAME_STATS') if len(matches)==1 else 'VERIFIED_PLAYER_GAME_STATS'
        if not valid:
            unresolved.append({'player_id':p['player_id'],'game_id':p['game_id'],'team':p['team'],'reason':'MISSING_AMBIGUOUS_OR_UNVERIFIED_ACTUAL'})
            continue
        r=matches[0]
        try:points=numeric(r.get(key),'ACTUAL_POINTS')
        except Pending:
            unresolved.append({'player_id':p['player_id'],'game_id':p['game_id'],'team':p['team'],'reason':'INVALID_ACTUAL_POINTS'})
            continue
        # Use 2*SCALE objective units so MVP 1.5x is exact in integer arithmetic.
        units=round(points*SCALE)
        if not math.isclose(points,units/SCALE,abs_tol=1e-9):raise Pending('UNSUPPORTED_ACTUAL_POINTS_PRECISION')
        # Retain actual usage/context fields; unavailable evidence remains absent.
        usage={k:v for k,v in r.items() if k not in {'player_id','game_id'} and (v is None or isinstance(v,(str,int,float,bool)))}
        for k,v in list(usage.items()):
            if isinstance(v,float) and not math.isfinite(v):usage[k]=None
        output.append({**p,'actual_fd':points,'actual_units':units,'actual_source':source,'actual_evidence':usage})
    return output,unresolved

def optimize(pool,format,seconds=60):
    from ortools.sat.python import cp_model
    model=cp_model.CpModel();variables={};terms=[];salary=[]
    slots=SLOTS if format=='CLASSIC' else ['MVP','FLEX1','FLEX2','FLEX3','FLEX4','FLEX5']
    for i,r in enumerate(pool):
        for slot in slots:
            allowed=format=='SINGLE_GAME' or (r['position'] in {'RB','WR','TE'} if slot=='FLEX' else r['position']==re.sub(r'\d+$','',slot))
            if not allowed:continue
            v=model.NewBoolVar(f'p{i}_{slot}');variables[i,slot]=v
            terms.append(v*r['actual_units']*(3 if slot=='MVP' else 2))
            # Doubled salary units support exact half-dollar MVP salaries.
            cost=r['mvp_salary'] if slot=='MVP' else r['salary']
            if cost*2!=round(cost*2):raise Pending('UNSUPPORTED_SALARY_PRECISION')
            salary.append(v*int(round(cost*2)))
    for slot in slots:model.Add(sum(v for (i,s),v in variables.items() if s==slot)==1)
    for i in range(len(pool)):model.Add(sum(v for (j,s),v in variables.items() if i==j)<=1)
    model.Add(sum(salary)<=120000)
    if format=='CLASSIC':
        for team in {r['team'] for r in pool}:
            model.Add(sum(v for (i,s),v in variables.items() if pool[i]['team']==team)<=4)
        # At least two games (FanDuel Classic roster scope); no WFS stacking,
        # salary-floor, locks, exclusions, exposure or projection constraints.
        gamevars=[]
        for gid in {r['game_id'] for r in pool}:
            used=model.NewBoolVar('game_'+gid)
            selected=sum(v for (i,s),v in variables.items() if pool[i]['game_id']==gid)
            model.Add(selected>=used);model.Add(selected<=9*used);gamevars.append(used)
        model.Add(sum(gamevars)>=2)
    model.Maximize(sum(terms))
    solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=seconds;solver.parameters.num_search_workers=1
    result=solver.Solve(model)
    if result!=cp_model.OPTIMAL:raise Pending('HINDSIGHT_OPTIMALITY_NOT_PROVEN:'+solver.StatusName(result))
    chosen=[{**pool[i],'slot':'FLEX' if s.startswith('FLEX') else s,'slot_salary':pool[i]['mvp_salary'] if s=='MVP' else pool[i]['salary'],
             'slot_actual_fd':pool[i]['actual_fd']*(1.5 if s=='MVP' else 1)} for (i,s),v in variables.items() if solver.Value(v)]
    if len(chosen)!=len(slots):raise RuntimeError('OPTIMIZER_ROSTER_AUDIT_FAILED')
    score=sum(r['actual_units']*(3 if r['slot']=='MVP' else 2) for r in chosen)/(2*SCALE)
    return {'score':score,'salary':sum(r['slot_salary'] for r in chosen),'players':chosen,'solver_status':'OPTIMAL','tied_optima':'ONE_PROVEN_OPTIMAL_REPRESENTATIVE'}

def score_generated(selected,actuals):
    index={(r['game_id'],r['player_id']):r for r in actuals};scores=[]
    for no in sorted({r['lineup'] for r in selected}):
        rows=[r for r in selected if r['lineup']==no];enriched=[];missing=[]
        for r in rows:
            a=index.get((r['game_id'],r['player_id']))
            if a is None:missing.append(r['player_id']);continue
            mult=1.5 if r['slot']=='MVP' else 1
            enriched.append({**r,'actual_fd':a['actual_fd'],'slot_actual_fd':a['actual_fd']*mult,
                             'projection_error':r['slot_projection']-a['actual_fd']*mult,
                             'actual_source':a['actual_source'],'actual_evidence':a['actual_evidence']})
        scores.append({'lineup':no,'status':'PENDING_ACTUALS' if missing else 'SCORED',
                       'missing_player_ids':missing,'actual_score':None if missing else sum(r['slot_actual_fd'] for r in enriched),
                       'players':enriched})
    return scores

def review(root,path,pbp_cache=None):
    m=validate_capture(path)
    with (root/'nfl_updater.lock').open('rb') as source_lock:
        try:fcntl.flock(source_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:raise Pending('UPDATER_ACTIVE_RETRY')
        games,actual,dst=source_outcomes(root,m['game_ids'])
    from wfs_lineup_zero_evidence_v1 import complete
    try:frozen_pool=contest_pool(path,m)
    except Pending:frozen_pool=None
    actual,zero_notes=complete(root,frozen_pool if frozen_pool is not None else read(path/'canonical_solver_pool.json'),games,actual,pbp_cache if pbp_cache is not None else {})
    selected=read(path/'generated_players.json')
    # Score available generated lineups even if full-contest optimum is pending.
    scored,missing_selected=attach_actuals(read(path/'canonical_solver_pool.json'),actual,dst)
    generated=score_generated(selected,scored)
    report={'contract':'WFS_LINEUP_POSTGAME_REVIEW_V1','capture':path.name,'capture_manifest_sha256':sha(path/'manifest.json'),
            'format':m['format'],'season':m['season'],'week':m['week'],'slate':m['slate'],
            'generated_lineups':generated,'context_sources':m['sources'],
            'settings':m['settings'],'model_weights_changed':False,'learning_consumer_connected':(root/'wfs_lineup_learning_reader_v1.py').exists(),
            'zero_evidence_notes':zero_notes,
            'benchmark_scope':'FULL_CAPTURED_CONTEST_POOL_CONTEST_RULES_ONLY'}
    try:
        pool=frozen_pool if frozen_pool is not None else contest_pool(path,m)
        scored_pool,unresolved=attach_actuals(pool,actual,dst)
        report['unresolved_contest_actuals']=unresolved
        if unresolved:raise Pending('FULL_CONTEST_ACTUALS_NOT_COMPLETE')
        optimum=optimize(scored_pool,m['format'])
        report['optimal_lineup']=optimum
        optimal_ids={r['player_id'] for r in optimum['players']}
        for g in generated:
            if g['status']!='SCORED':continue
            gap=optimum['score']-g['actual_score']
            if gap < -1e-6:raise RuntimeError('GENERATED_SCORE_EXCEEDS_PROVEN_OPTIMUM')
            ids={r['player_id'] for r in g['players']}
            g.update({'point_gap':max(0,gap),'overlap_with_optimal_representative':len(ids&optimal_ids),
                      'missed_optimal_player_ids':sorted(optimal_ids-ids),'selected_nonoptimal_player_ids':sorted(ids-optimal_ids)})
        report['player_outcomes']=scored_pool
        report['status']='COMPLETE' if all(g['status']=='SCORED' for g in generated) else 'PENDING_SELECTED_ACTUALS'
    except Pending as exc:
        report['status']='PENDING';report['reason']=str(exc)
    # Revision signature includes source results, so corrections create new
    # immutable versions instead of silently changing an earlier comparison.
    report['actual_revision_sha256']=hashlib.sha256(json.dumps({'games':games,'actual':actual,'dst':dst},sort_keys=True,default=str).encode()).hexdigest()
    return report

def run(root):
    root=Path(root).resolve();base=root/'data/lineup_learning_v1';reviews=base/'reviews';reviews.mkdir(parents=True,exist_ok=True)
    with (base/'postgame_worker.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('LINEUP_REVIEW_WORKER=ALREADY_RUNNING');return
        count=0;pbp_cache={}
        for path in sorted((base/'captures').glob('*')):
            if not path.is_dir() or path.name.startswith('.'):continue
            count+=1;target=reviews/path.name
            try:
                report=review(root,path,pbp_cache)
                signature=hashlib.sha256(json.dumps(report,sort_keys=True,allow_nan=False).encode()).hexdigest()
                revision=target/'versions'/f'{signature}.json'
                if not revision.exists():
                    atomic_json(revision,report);revision.chmod(0o444)
                elif read(revision)!=report:
                    raise RuntimeError('REVIEW_REVISION_CORRUPTED')
                atomic_json(target/'latest.json',{'status':report['status'],'reason':report.get('reason'),'version':str(revision.relative_to(base)),'sha256':sha(revision)})
                print(f"LINEUP_REVIEW|capture={path.name}|status={report['status']}|reason={report.get('reason','')}")
            except Pending as exc:
                atomic_json(target/'latest.json',{'status':'PENDING','reason':str(exc)})
                print(f'LINEUP_REVIEW|capture={path.name}|status=PENDING|reason={exc}')
            except Exception as exc:
                atomic_json(target/'latest.json',{'status':'ERROR','reason':f'{type(exc).__name__}: {exc}'})
                print(f'LINEUP_REVIEW|capture={path.name}|status=ERROR|reason={exc}')
        print(f'LINEUP_REVIEW_CAPTURES={count}')
        print('PRODUCTION_MODEL_WEIGHTS_CHANGED=NO')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/home/mwynn/nfl_data_engine'))
    run(ap.parse_args().root)
