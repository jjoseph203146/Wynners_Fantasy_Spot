"""Conservative zero evidence for absent results; never updates source tables.
Any PBP identity occurrence prevents zero completion, even a non-scoring event.
"""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd

SOURCE='VERIFIED_ABSENT_RESULT_ZERO_V1'
POLICY='EXACT_ROSTER_FINAL_PBP_NO_PLAYER_EVENTS_V1'
ALIASES={'LA':'LAR','WAS':'WSH','JAC':'JAX'}
def canon(v):return ALIASES.get(str(v).strip().upper(),str(v).strip().upper())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# All identity columns observed in the full source sample are required. A
# narrowed DST loader cannot establish absence of offensive scoring events.
ID_FIELDS='td_player_id passer_player_id receiver_player_id rusher_player_id lateral_receiver_player_id lateral_rusher_player_id lateral_sack_player_id interception_player_id lateral_interception_player_id punt_returner_player_id lateral_punt_returner_player_id kickoff_returner_player_id lateral_kickoff_returner_player_id punter_player_id kicker_player_id own_kickoff_recovery_player_id blocked_player_id tackle_for_loss_1_player_id tackle_for_loss_2_player_id qb_hit_1_player_id qb_hit_2_player_id forced_fumble_player_1_player_id forced_fumble_player_2_player_id solo_tackle_1_player_id solo_tackle_2_player_id assist_tackle_1_player_id assist_tackle_2_player_id assist_tackle_3_player_id assist_tackle_4_player_id tackle_with_assist_1_player_id tackle_with_assist_2_player_id pass_defense_1_player_id pass_defense_2_player_id fumbled_1_player_id fumbled_2_player_id fumble_recovery_1_player_id fumble_recovery_2_player_id sack_player_id half_sack_1_player_id half_sack_2_player_id penalty_player_id safety_player_id fantasy_player_id'.split()
STRUCTURAL='game_id season week season_type play_id desc home_team away_team home_score away_score total_home_score total_away_score game_seconds_remaining'.split()

def game_evidence(pbp,game):
    if str(game.get('completed')) not in {'1','True'}:return None
    if not set(ID_FIELDS+STRUCTURAL).issubset(pbp.columns):return None
    df=pbp.loc[pbp.game_id.astype(str)==str(game['game_id'])].copy()
    if df.empty or df.play_id.isna().any() or df.play_id.duplicated().any():return None
    for col in ID_FIELDS:
        name=col[:-3]+'_name'
        if name in df:
            named=df[name].notna()
            identified=df[col].astype(str).str.contains(r'00-\d{7}',regex=True)
            if (named & ~identified).any():return None
    for col,expected in [('season',int(game['season'])),('week',int(game['week'])),('home_score',int(game['home_score'])),('away_score',int(game['away_score']))]:
        vals=pd.to_numeric(df[col],errors='coerce')
        if vals.isna().any() or not vals.eq(expected).all():return None
    for col in ['home_team','away_team']:
        if not df[col].map(canon).eq(canon(game[col])).all():return None
    expected_type=str(game.get('game_type','REG'))
    if not df.season_type.astype(str).eq(expected_type).all():return None
    end=df.loc[df.desc.astype(str).str.strip().eq('END GAME')]
    if len(end)!=1:return None
    terminal=end.iloc[0]
    if float(terminal.game_seconds_remaining)!=0:return None
    if float(terminal.total_home_score)!=float(game['home_score']) or float(terminal.total_away_score)!=float(game['away_score']):return None
    # Source order need not match play_id order (timeouts can be interleaved).
    # Scan every identity-like column, including composite/list values.
    cols=[c for c in df if c.endswith('_player_id') or c in {'player_id','fantasy_id','passer_id','rusher_id','receiver_id','id'}]
    seen=set()
    for col in cols:
        for value in df[col]:seen.update(re.findall(r'00-\d{7}',str(value)))
    normalized=df.sort_values('play_id').to_json(orient='split',date_format='iso',double_precision=15)
    return {'ids':seen,'pbp_game_sha256':hashlib.sha256(normalized.encode()).hexdigest(),'pbp_rows':len(df),'pbp_terminal_play_id':float(terminal.play_id),'pbp_identity_columns':len(cols)}

def prove_zero(player,game,actual,rosters,snaps,pbp_evidence,source_hashes):
    identity=player['player_id'];gid=player['game_id'];team=canon(player['team'])
    if not re.fullmatch(r'00-\d{7}',identity) or str(game['game_id'])!=gid:return None
    # Even wrong-team/unverified/duplicate rows block fallback; never override.
    if any(str(r.get('game_id'))==gid and str(r.get('player_id'))==identity for r in actual):return None
    if not pbp_evidence or identity in pbp_evidence['ids']:return None
    required={'season','week','game_type','team','gsis_id','pfr_id','status'}
    if not required.issubset(rosters.columns):return None
    roster=rosters.loc[(pd.to_numeric(rosters.season,errors='coerce')==int(game['season'])) & (pd.to_numeric(rosters.week,errors='coerce')==int(game['week'])) & (rosters.gsis_id.astype(str)==identity)]
    if len(roster)!=1:return None
    r=roster.iloc[0]
    if canon(r.team)!=team or str(r.game_type)!=str(game.get('game_type','REG')):return None
    if team not in {canon(game['away_team']),canon(game['home_team'])}:return None
    required_snap={'game_id','season','week','team','pfr_player_id','offense_snaps','defense_snaps','st_snaps'}
    if not required_snap.issubset(snaps.columns):return None
    pfr=str(r.pfr_id).strip()
    if not pfr or pfr in {'None','nan','<NA>'}:return None
    # Reject collisions in the week's crosswalk, including another team.
    cross=rosters.loc[(pd.to_numeric(rosters.season,errors='coerce')==int(game['season'])) & (pd.to_numeric(rosters.week,errors='coerce')==int(game['week'])) & (rosters.pfr_id.astype(str)==pfr)]
    if len(cross)!=1:return None
    found=snaps.loc[(snaps.game_id.astype(str)==gid)&(snaps.pfr_player_id.astype(str)==pfr)]
    if len(found)>1:return None
    status=str(r.status).upper();snap_total=None
    if len(found)==1:
        s=found.iloc[0]
        if canon(s.team)!=team or int(s.season)!=int(game['season']) or int(s.week)!=int(game['week']):return None
        vals=pd.to_numeric(s[['offense_snaps','defense_snaps','st_snaps']],errors='coerce')
        if vals.isna().any() or (vals<0).any() or (vals%1!=0).any():return None
        snap_total=int(vals.sum())
    # Explicit inactive + no positive participation OR exact positive snaps.
    # CUT/RES/DEV and missing active snap rows deliberately remain unresolved.
    if status=='INA' and (snap_total is None or snap_total==0):basis='EXPLICIT_WEEKLY_INACTIVE'
    elif snap_total is not None and snap_total>0 and status=='ACT':basis='EXACT_SNAP_PARTICIPATION'
    else:return None
    evidence={'zero_policy':POLICY,'zero_basis':basis,'roster_status':status,'pfr_id':pfr,'snap_total':snap_total,
              'pbp_game_sha256':pbp_evidence['pbp_game_sha256'],'pbp_rows':pbp_evidence['pbp_rows'],
              'pbp_terminal_play_id':pbp_evidence['pbp_terminal_play_id'],'pbp_identity_columns':pbp_evidence['pbp_identity_columns'],
              'roster_sha256':source_hashes['roster'],'snap_sha256':source_hashes['snap'],
              'no_player_pbp_events':True,'final_scores_matched':True}
    return {'game_id':gid,'player_id':identity,'team':team,'fanduel_points':0.0,'fanduel_points_verified':1,
            'zero_evidence_json':json.dumps(evidence,sort_keys=True),'lineup_actual_source':SOURCE}

def complete(root,pool,games,actual,pbp_cache):
    """Download full PBP once per season per worker run, only for absent results.
    Cache stays in memory so future cycles see source corrections. Failures
    preserve pending outcomes; no partial or unverified zero is emitted.
    """
    candidates=[p for p in pool if not p['player_id'].startswith('DST:') and not any(str(r.get('game_id'))==p['game_id'] and str(r.get('player_id'))==p['player_id'] for r in actual)]
    if not candidates:return actual,[]
    roster_path=Path(root)/'data/parquet/nfl_weekly_rosters.parquet';snap_path=Path(root)/'data/parquet/nfl_snap_counts.parquet'
    try:
        before={'roster':digest(roster_path),'snap':digest(snap_path)}
        rosters=pd.read_parquet(roster_path);snaps=pd.read_parquet(snap_path)
        if before!={'roster':digest(roster_path),'snap':digest(snap_path)}:return actual,['ZERO_EVIDENCE_SOURCE_CHANGED_RETRY']
        bygame={str(g['game_id']):g for g in games};proofs={};additions=[]
        for p in candidates:
            g=bygame[p['game_id']];season=int(g['season'])
            if season not in pbp_cache:
                import nflreadpy as nfl
                pbp_cache[season]=nfl.load_pbp(season).to_pandas()
            if p['game_id'] not in proofs:proofs[p['game_id']]=game_evidence(pbp_cache[season],g)
            result=prove_zero(p,g,actual,rosters,snaps,proofs[p['game_id']],before)
            if result:additions.append(result)
        return actual+additions,[]
    except Exception as exc:
        return actual,[f'ZERO_EVIDENCE_UNAVAILABLE:{type(exc).__name__}:{exc}']
