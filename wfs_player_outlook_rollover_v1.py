"""Automatic exact-game descriptive Outlook publication. No forecast mutation."""
from pathlib import Path
from datetime import datetime,timezone
from contextlib import redirect_stdout
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE_REL='data/research/player_outlook_rollover_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=p.parent,delete=False) as f:
        json.dump(value,f,sort_keys=True,indent=2,default=str);f.flush();os.fsync(f.fileno());tmp=Path(f.name)
    try:tmp.replace(p)
    finally:tmp.unlink(missing_ok=True)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def scoped_frames(frames,gid):
    """Narrow only AFTER the complete source publication validates.
    Hash metadata continues to identify full upstream files; in-memory MI row
    counts describe the exact subset supplied to the unchanged composer.
    """
    result=dict(frames)
    for key in ['matrix','slate','mi']:
        result[key]=frames[key].loc[frames[key].game_id.astype(str)==gid].copy()
        if result[key].empty:raise RuntimeError('GAME_NOT_IN_CURRENT_SOURCE:'+gid)
    result['mi_manifest']=copy.deepcopy(frames['mi_manifest'])
    result['mi_manifest']['matrix_rows']=len(result['matrix'])
    result['mi_manifest']['rows']=len(result['mi'])
    return result

def eligible_games(matrix,schedule,now):
    from player_outlook_capture_reference_v1 import build_schedule_kickoff
    s=build_schedule_kickoff(schedule)
    if matrix.empty or len(matrix[['season','week']].drop_duplicates())!=1:raise RuntimeError('CURRENT_SOURCE_PERIOD_AMBIGUOUS')
    season,week=map(int,matrix[['season','week']].iloc[0])
    scope=s.loc[(s.season==season)&(s.week==week)]
    games=scope.loc[scope.game_id.astype(str).isin(matrix.game_id.astype(str).unique())]
    if len(games)!=matrix.game_id.nunique():raise RuntimeError('CURRENT_SOURCE_SCHEDULE_MISMATCH')
    if games.completed.isna().any() or not games.completed.isin([0,1]).all():raise RuntimeError('UNKNOWN_COMPLETION_STATE')
    return sorted(games.loc[(games.completed==0)&(games.schedule_kickoff_utc>now),'game_id'].astype(str))

def load_current_player_outlook(*,game_context,capture_dir,now_utc):
    from player_outlook_data import load_player_outlook,ValidatedOutlook
    if not re.fullmatch(r'\d{4}_\d{2}_[A-Z]+_[A-Z]+',game_context.game_id):return ValidatedOutlook(False,reason='GAME_ID')
    index=Path(capture_dir)/game_context.game_id/'latest.json'
    if index.is_file():
        try:
            info=read(index);directory=(index.parent/info['version']).resolve()
            if not directory.is_relative_to((index.parent/'versions').resolve()):raise ValueError('VERSION_PATH')
            if sha(directory/'capture_manifest.json')!=info['manifest_sha256']:raise ValueError('MANIFEST_HASH')
            return load_player_outlook(game_context=game_context,artifact_path=directory/'capture.parquet',manifest_path=directory/'capture_manifest.json',now_utc=now_utc)
        except Exception:return ValidatedOutlook(False,reason='INVALID_CURRENT_CAPTURE_INDEX')
    # Exact-game legacy capture only. Never use the prior week as a fallback.
    legacy=ROOT/'data/research/prospective/player_form_matchup_v1'
    key=f'{game_context.season}_week_{game_context.week:02d}_pregame'
    return load_player_outlook(game_context=game_context,artifact_path=legacy/(key+'.parquet'),manifest_path=legacy/(key+'_manifest.json'),now_utc=now_utc)

def capture_game(root,base,form,frames,metadata,gid):
    from player_outlook_data import read_game_context,load_player_outlook
    scoped=scoped_frames(frames,gid)
    token=hashlib.sha256(json.dumps({k:v.get('sha256') for k,v in metadata.items()},sort_keys=True).encode()).hexdigest()
    gamebase=base/gid;versions=gamebase/'versions';target=versions/token
    if target.exists():
        original=read(target/'capture_manifest.json')
        if sha(target/'source.parquet')!=original['source']['stage1_sha256'] or sha(target/'source_manifest.json')!=original['source']['stage1_manifest_sha256']:raise RuntimeError('IMMUTABLE_SOURCE_BINDING_INVALID')
        context=read_game_context(database_path=root/'data/nfl.db',game_id=gid)
        existing=load_player_outlook(game_context=context,artifact_path=target/'capture.parquet',manifest_path=target/'capture_manifest.json',now_utc=pd.Timestamp.now(tz='UTC'))
        if not existing.available:raise RuntimeError('IMMUTABLE_CAPTURE_INVALID:'+existing.reason)
    else:
        versions.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.building_',dir=versions) as temporary:
            tmp=Path(temporary);now=pd.Timestamp.now(tz='UTC')
            output,report=form.compose(scoped,metadata,now)
            form.check_unchanged(metadata)
            stage=tmp/'source.parquet';output.to_parquet(stage,index=False)
            report['output']={'path':str(stage),'sha256':sha(stage)}
            source_manifest=tmp/'source_manifest.json';atomic(source_manifest,report)
            cap=module(root/'player_outlook_capture_reference_v1.py','capture_'+token)
            cap.STAGE1=stage;cap.STAGE1_MANIFEST=source_manifest;cap.SCHEDULE=root/'data/parquet/nfl_schedule.parquet';cap.CAPTURE_DIR=tmp/'result'
            with redirect_stdout(io.StringIO()):cap.capture()
            generated=list(cap.CAPTURE_DIR.glob('*_pregame.parquet'))
            if len(generated)!=1:raise RuntimeError('GAME_CAPTURE_PUBLICATION_COUNT')
            captured=generated[0];manifest=captured.with_name(captured.stem+'_manifest.json')
            shutil.copyfile(captured,tmp/'capture.parquet');shutil.copyfile(manifest,tmp/'capture_manifest.json')
            context=read_game_context(database_path=root/'data/nfl.db',game_id=gid)
            final_now=pd.Timestamp.now(tz='UTC')
            if context.completed or final_now>=context.kickoff_utc:raise RuntimeError('CROSSED_KICKOFF_REFUSED')
            validated=load_player_outlook(game_context=context,artifact_path=tmp/'capture.parquet',manifest_path=tmp/'capture_manifest.json',now_utc=final_now)
            if not validated.available:raise RuntimeError('NEW_CAPTURE_INVALID:'+validated.reason)
            form.check_unchanged(metadata)
            # Retain composed source bytes and their original hash-bound manifest.
            staged=tmp/'committed';staged.mkdir()
            shutil.move(tmp/'capture.parquet',staged/'capture.parquet');shutil.move(tmp/'capture_manifest.json',staged/'capture_manifest.json')
            shutil.copyfile(stage,staged/'source.parquet');shutil.copyfile(source_manifest,staged/'source_manifest.json')
            bound_manifest=read(staged/'capture_manifest.json')
            bound_manifest['source']['stage1_path']=str(target/'source.parquet')
            bound_manifest['source']['stage1_manifest_path']=str(target/'source_manifest.json')
            bound_manifest['output']['path']=str(target/'capture.parquet')
            atomic(staged/'capture_manifest.json',bound_manifest)
            for p in staged.iterdir():p.chmod(0o444)
            staged.rename(target)
    atomic(gamebase/'latest.json',{'game_id':gid,'version':'versions/'+token,'manifest_sha256':sha(target/'capture_manifest.json')})
    return token

def verify_display(root,base,gid):
    from player_outlook_data import read_game_context
    from player_outlook_relevance import apply_player_outlook_relevance
    context=read_game_context(database_path=root/'data/nfl.db',game_id=gid)
    now=pd.Timestamp.now(tz='UTC')
    validated=load_current_player_outlook(game_context=context,capture_dir=base,now_utc=now)
    relevant=apply_player_outlook_relevance(validated_outlook=validated,game_context=context,
        starter_path=root/'data/parquet/current_starter_verification.parquet',
        starter_manifest_path=root/'data/parquet/current_starter_verification_manifest.json',
        v3_path=root/'processed/offensive_team_reconciliation_shadow_v3.csv',
        v3_audit_path=root/'processed/offensive_team_reconciliation_shadow_v3_audit.json',now_utc=now)
    if not relevant.available:raise RuntimeError('DISPLAY_RELEVANCE_NOT_READY:'+relevant.reason)
    if not relevant.players:raise RuntimeError('NO_RELEVANT_DISPLAY_PLAYERS')
    return len(relevant.players)

def run(root=ROOT):
    root=Path(root).resolve();base=root/BASE_REL;base.mkdir(parents=True,exist_ok=True)
    with (base/'worker.lock').open('a+') as worker:
        try:fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('PLAYER_OUTLOOK=WORKER_ACTIVE_RETRY');return
        with (root/'nfl_updater.lock').open('a+') as authority:
            try:fcntl.flock(authority,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                atomic(base/'health.json',{'status':'RETRY','reason':'UPDATER_ACTIVE','at':datetime.now(timezone.utc).isoformat()});print('PLAYER_OUTLOOK=UPDATER_ACTIVE_RETRY');return
            try:
                if str(root) not in sys.path:sys.path.insert(0,str(root))
                form=module(root/'research/build_player_form_matchup_shadow_v1.py','outlook_form_builder')
                inputs={**form.INPUTS,**form.CODE_INPUTS,'team_environment':'data/parquet/nfl_current_team_environment.parquet','mi_refresh_code':'scripts/run_current_mi_shadow_refresh_v1.py'}
                signature=hashlib.sha256(json.dumps({k:sha(root/v) if (root/v).exists() else None for k,v in inputs.items() if k not in {'mi','mi_manifest'}},sort_keys=True).encode()).hexdigest()
                stamp=base/'mi_refresh.json'
                refresh=not stamp.exists() or read(stamp).get('source_signature')!=signature
                if not refresh:
                    try:frames,metadata=form.load_inputs()
                    except Exception:refresh=True
                if refresh:
                    result=subprocess.run([sys.executable,'-B',str(root/'scripts/run_current_mi_shadow_refresh_v1.py')],cwd=root,capture_output=True,text=True,timeout=180)
                    if result.returncode:raise RuntimeError('MI_REFRESH_FAILED:'+result.stdout[-2000:]+result.stderr[-2000:])
                    frames,metadata=form.load_inputs()
                    atomic(stamp,{'source_signature':signature})
                gids=eligible_games(frames['matrix'],frames['schedule'],pd.Timestamp.now(tz='UTC'))
                outcomes=[]
                for gid in gids:
                    try:
                        version=capture_game(root,base,form,frames,metadata,gid);count=verify_display(root,base,gid);outcomes.append({'game_id':gid,'status':'PASS','version':version,'relevant_players':count});print('PLAYER_OUTLOOK|game='+gid+'|status=PASS|players='+str(count))
                    except Exception as exc:
                        outcomes.append({'game_id':gid,'status':'RETRY','reason':str(exc)});print('PLAYER_OUTLOOK|game='+gid+'|status=RETRY|reason='+str(exc))
                status='PASS' if outcomes and all(x['status']=='PASS' for x in outcomes) else 'RETRY' if outcomes else 'NO_FUTURE_CURRENT_SOURCE_GAMES'
                atomic(base/'health.json',{'status':status,'games':outcomes,'at':datetime.now(timezone.utc).isoformat(),'production_models_changed':False})
            except Exception as exc:
                atomic(base/'health.json',{'status':'RETRY','reason':str(exc),'at':datetime.now(timezone.utc).isoformat()});print('PLAYER_OUTLOOK=RETRY|reason='+str(exc))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);a=ap.parse_args();run(a.root)
