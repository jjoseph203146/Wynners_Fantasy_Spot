"""Read-only prior-week DFS review evidence for current pregame analysis.
No player boosts, solver changes, or causal claims from hindsight outcomes.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from wfs_lineup_postgame_v1 import validate_capture

ALIASES={'LA':'LAR','WAS':'WSH','JAC':'JAX'}
def canon(v):
    v=str(v or '').strip().upper();return ALIASES.get(v,v)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def load_lineup_learning(root,game,mode):
    empty={'contract':'WFS_LINEUP_LEARNING_READER_V1','status':'NO_VERIFIED_PRIOR_REVIEWS',
           'consumer_connected':True,'players':[],'review_count':0,'unique_player_games':0,
           'scope':'PRIOR_WEEKS_SAME_SEASON','model_weights_changed':False}
    if mode!='PREGAME':return {**empty,'status':'FROZEN_MODE_EXCLUDED'}
    root=Path(root).resolve();base=root/'data/lineup_learning_v1'
    try:season=int(game['season']);week=int(game['week']);teams={canon(game['away_team']),canon(game['home_team'])}
    except (KeyError,ValueError,TypeError):return {**empty,'status':'INVALID_TARGET_SCOPE'}
    records={};conflicts=set();reviews=set();ignored=0
    for latest in sorted((base/'reviews').glob('*/latest.json')):
        try:
            index=read(latest)
            if index.get('status')!='COMPLETE':continue
            version=(base/index['version']).resolve()
            if not version.is_relative_to(latest.parent.resolve()/'versions') or version.suffix!='.json':raise ValueError('VERSION_PATH')
            if sha(version)!=index['sha256']:raise ValueError('REVIEW_SHA')
            report=read(version)
            if report.get('contract')!='WFS_LINEUP_POSTGAME_REVIEW_V1' or report.get('status')!='COMPLETE':raise ValueError('REVIEW_CONTRACT')
            if int(report['season'])!=season or not 0<int(report['week'])<week:continue
            capture=(base/'captures'/str(report['capture'])).resolve()
            if not capture.is_relative_to((base/'captures').resolve()) or capture.name!=latest.parent.name:raise ValueError('CAPTURE_PATH')
            if sha(capture/'manifest.json')!=report['capture_manifest_sha256']:raise ValueError('CAPTURE_BINDING')
            m=validate_capture(capture)
            if int(m['season'])!=season or int(m['week'])!=int(report['week']):raise ValueError('CAPTURE_SCOPE')
            if report['optimal_lineup'].get('solver_status')!='OPTIMAL':raise ValueError('OPTIMALITY')
            if report.get('unresolved_contest_actuals'):raise ValueError('INCOMPLETE_ACTUALS')
            if m['format']!=report['format']:raise ValueError('FORMAT_MISMATCH')
            gids=set(m['game_ids']);matched=False;pending_rows=[];seen_keys=set()
            for r in report['player_outcomes']:
                team=canon(r['team'])
                if team not in teams:continue
                gid=str(r['game_id']);identity=str(r['player_id'])
                if gid not in gids or not gid.startswith(f'{season}_{int(report["week"]):02d}_'):raise ValueError('PLAYER_GAME_SCOPE')
                if not (re.fullmatch(r'00-\d{7}',identity) or identity=='DST:'+team):raise ValueError('PLAYER_ID')
                if r.get('actual_source') not in {'VERIFIED_PLAYER_GAME_STATS','VERIFIED_DST_ACTUALS','VERIFIED_ABSENT_RESULT_ZERO_V1'}:raise ValueError('UNVERIFIED_ACTUAL')
                points=float(r['actual_fd'])
                if r.get('actual_source')=='VERIFIED_ABSENT_RESULT_ZERO_V1':
                    z=json.loads(r.get('actual_evidence',{}).get('zero_evidence_json','{}'))
                    if identity.startswith('DST:') or points!=0 or z.get('zero_policy')!='EXACT_ROSTER_FINAL_PBP_NO_PLAYER_EVENTS_V1':raise ValueError('INVALID_ZERO_AUTHORITY')
                    if z.get('zero_basis') not in {'EXPLICIT_WEEKLY_INACTIVE','EXACT_SNAP_PARTICIPATION'} or z.get('no_player_pbp_events') is not True or z.get('final_scores_matched') is not True:raise ValueError('INVALID_ZERO_EVIDENCE')
                    if any(not re.fullmatch(r'[0-9a-f]{64}',str(z.get(k,''))) for k in ['pbp_game_sha256','roster_sha256','snap_sha256']):raise ValueError('ZERO_SOURCE_HASH')
                if not math.isfinite(points):raise ValueError('ACTUAL_NONFINITE')
                key=(gid,identity)
                outcome={'game_id':gid,'player_id':identity,'player':r['player'],'team':team,
                         'opponent':canon(r['opponent']),'position':r['position'],'actual_fd':points,
                         'week':int(report['week']),'actual_evidence':r.get('actual_evidence',{}),
                         'review_paths':[], 'context_captures':[]}
                sig={k:v for k,v in outcome.items() if k not in {'review_paths','context_captures'}}
                if key in seen_keys:raise ValueError('DUPLICATE_REVIEW_PLAYER_GAME')
                seen_keys.add(key)
                rel=str(version.relative_to(base))
                context={'capture':m['captured_at_utc'],'directory':str(capture.relative_to(base)),
                         'weather_files_captured':int(m.get('weather_files_captured',0)),
                         'injury_evidence_saved':'injury_asof.parquet' in m['files']}
                pending_rows.append((key,sig,outcome,rel,context))
            # Commit a review only after every relevant player row validates.
            for key,sig,outcome,rel,context in pending_rows:
                if key in records and records[key]['signature']!=sig:conflicts.add(key)
                if key not in records:records[key]={'signature':sig,'row':outcome}
                record=records[key]['row']
                if rel not in record['review_paths']:record['review_paths'].append(rel)
                if context not in record['context_captures']:record['context_captures'].append(context)
                matched=True
            if matched:reviews.add(str(version.relative_to(base)))
        except (OSError,ValueError,TypeError,KeyError,RuntimeError):ignored+=1
    players=[v['row'] for k,v in records.items() if k not in conflicts]
    players.sort(key=lambda r:(-r['week'],r['team'],r['game_id'],r['player_id']))
    return {**empty,'status':'AVAILABLE' if players else 'NO_VERIFIED_PRIOR_REVIEWS',
            'players':players,'review_count':len(reviews),'unique_player_games':len(players),
            'conflicting_player_games_excluded':len(conflicts),'invalid_reviews_excluded':ignored,
            'interpretation':'Historical actual outcomes are evidence, not current forecasts or automatic player recommendations. Injury/weather context must be considered alongside current opportunity and opponent evidence.'}

def attach_learning_response(response,evidence):
    response['lineup_learning']=evidence
    if response.get('mode')!='PREGAME' or evidence.get('status')!='AVAILABLE':return
    observation=(f"Prior DFS review evidence is available for {evidence['unique_player_games']} player-game outcomes "
                 f"across {evidence['review_count']} verified reviews involving these teams. "
                 "These are historical results; compare current roles, opponents, injuries and weather before applying them.")
    response['observations']=list(response.get('observations') or [])+[observation]
