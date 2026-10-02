"""Pure deterministic replay from explicitly supplied, versioned observations.

No DB, filesystem, network, model fitting, imputation, or stochastic generator.
Legacy zero encoding is recorded separately and never repairs semantic unknowns.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math

VERSION = 'SIMULATOR_RECURSIVE_STATE_REPLAY_V1'
PRIMITIVES = ('points', 'attempts', 'carries', 'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds')
SAFETY = dict(PRODUCTION_PROJECTION_INFLUENCE='NONE', FANDUEL_ATTACHMENT_INFLUENCE='NONE',
              SOLVER_INFLUENCE='NONE', LINEUP_INFLUENCE='NONE', AVAILABILITY_AUTHORITY_CHANGED='NO',
              LIVE_CHANGED='NO', COLD_START_V1_CHANGED='NO', V3_1_CHANGED='NO',
              FROZEN_SIMULATOR_EXPERIMENTS_CHANGED='NO')


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()


def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()


def time(x):
    t=datetime.fromisoformat(x.replace('Z','+00:00'))
    if t.tzinfo is None: raise ValueError('TIMESTAMP_TIMEZONE_REQUIRED')
    return t


def require(ok,reason):
    if not ok: raise ValueError(reason)


def number(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def validate_contract(contract):
    require(len(contract['margin'])==len(set(contract['margin']))==72,'MARGIN_MAPPING')
    require(len(contract['total'])==len(set(contract['total']))==64,'TOTAL_MAPPING')
    require(len(contract['excluded'])==len(set(contract['excluded']))==20,'EXCLUDED_MAPPING')
    require(set(contract['total']) <= set(contract['margin']),'TOTAL_NOT_SUBSET')
    require(not set(contract['excluded']) & set(contract['margin']),'EXCLUDED_OVERLAP')


def build(data,contract):
    """Reconstruct one team-oriented pregame state, optionally advancing a parent.

Every result event contains both teams. Parent histories are explicit and immutable.
The caller must attest complete bootstrap history and first-known season/coach scope.
"""
    validate_contract(contract)
    d=deepcopy(data); c=d['context']; issues=[]
    for k in ('season','week','game_id','team','opponent','kickoff','simulation_step','evidence_cutoff',
              'scenario_version','provenance','feature_contract_version','source_version','source_hashes'):
        require(k in c,'MISSING_CONTEXT_'+k)
    require(c['feature_contract_version']==contract['version'],'SOURCE_VERSION_MISMATCH')
    require(c['provenance'] in {'OBSERVED','SIMULATED'},'PROVENANCE')
    cutoff=time(c['evidence_cutoff']); kickoff=time(c['kickoff'])
    require(cutoff < kickoff,'LEAKAGE_CUTOFF_NOT_PREGAME')
    require(c['team']!=c['opponent'],'IDENTITY_MISMATCH')
    require(c['source_hashes'] and c['source_version'],'SOURCE_VERSION_MISMATCH')
    registry=d['schedule']
    require(len({g['game_id'] for g in registry})==len(registry),'DUPLICATE_GAME')
    games={g['game_id']:g for g in registry}
    teams=set(d['canonical_teams'])
    require(len(teams)==len(d['canonical_teams']),'AMBIGUOUS_TEAM_IDENTITY')
    for g in registry:
        require(g['home'] in teams and g['away'] in teams and g['home']!=g['away'],'IDENTITY_MISMATCH')
    g=games.get(c['game_id'])
    require(g and {g['home'],g['away']}=={c['team'],c['opponent']} and g['kickoff']==c['kickoff']
            and g['season']==c['season'] and g['week']==c['week'],'IDENTITY_MISMATCH')
    schedule_eligible = time(g['available_at'])<=cutoff
    if not schedule_eligible: issues.append(dict(classification='LEAKAGE',reason='FUTURE_SCHEDULE'))
    parent=d.get('parent'); history=[]; applied=[]; root_cutoff=c['evidence_cutoff']
    if parent:
        body={k:v for k,v in parent.items() if k!='audit'}
        require(digest(body)==parent['audit']['result_hash'],'PARENT_MUTATION')
        pm=parent['state_manifest']; pc=pm['context']
        require(c.get('parent_state_version')==pm['state_version'],'PARENT_VERSION_MISMATCH')
        require(c['scenario_version']==pc['scenario_version'],'BRANCH_MISMATCH')
        require(c['team']==pc['team'],'IDENTITY_MISMATCH')
        require(c['simulation_step']==pc['simulation_step']+1,'DUPLICATE_TRANSITION')
        require(cutoff>time(pc['evidence_cutoff']),'NONCHRONOLOGICAL_TRANSITION')
        require(not (pc['provenance']=='SIMULATED' and c['provenance']=='OBSERVED'),'PROVENANCE_MISMATCH')
        require(c['source_version']==pc['source_version'] and c['source_hashes']==pc['source_hashes'],'SOURCE_VERSION_MISMATCH')
        history=deepcopy(parent['lineage']['history'])
        applied=list(parent['lineage']['applied_event_ids'])
        root_cutoff=parent['lineage']['observed_root_cutoff']
        require(not d.get('bootstrap'),'CHILD_BOOTSTRAP_FORBIDDEN')
    else:
        require(c.get('parent_state_version') is None and c['simulation_step']==0,'MISSING_PARENT')
    new_events=d.get('bootstrap',[])+d.get('transitions',[])
    require(len({e['game_id'] for e in new_events})==len(new_events),'DUPLICATE_GAME_TEAM')
    for e in sorted(new_events,key=lambda x:(x.get('completed_at') or '',x['game_id'])):
        require(e['game_id'] not in applied,'DUPLICATE_TRANSITION')
        eg=games.get(e['game_id'])
        require(eg and len(e['teams'])==2 and {r['team'] for r in e['teams']}=={eg['home'],eg['away']},'IDENTITY_MISMATCH')
        require(e['season']==eg['season'] and e['week']==eg['week'],'IDENTITY_MISMATCH')
        require(e.get('source_ref') and e.get('source_version') and e.get('source_hash'),'SOURCE_VERSION_MISMATCH')
        if not e.get('completed_at') or not e.get('observation_time'):
            issues.append(dict(classification='EXPECTED_UNAVAILABLE',reason='MISSING_HISTORICAL_AVAILABILITY_TIMESTAMP',game_id=e['game_id']))
            continue
        completed=time(e['completed_at']); observed=time(e['observation_time'])
        if completed<=time(eg['kickoff']) or observed<completed or observed>cutoff or completed>=cutoff or e['game_id']==c['game_id']:
            issues.append(dict(classification='LEAKAGE',reason='INELIGIBLE_RESULT_TIME',game_id=e['game_id']))
            continue
        require(e['provenance'] in {'OBSERVED','SIMULATED'},'PROVENANCE_MISMATCH')
        if e['provenance']=='SIMULATED':
            require(c['provenance']=='SIMULATED' and e.get('scenario_version')==c['scenario_version'],'BRANCH_MISMATCH')
        elif c['provenance']=='SIMULATED' and observed>time(root_cutoff):
            issues.append(dict(classification='LEAKAGE',reason='ACTUAL_FUTURE_IN_SIMULATED_BRANCH',game_id=e['game_id']))
            continue
        for r in e['teams']:
            for metric in PRIMITIVES:
                require(r.get(metric) is None or number(r[metric]),'INVALID_PRIMITIVE')
        history.append(e); applied.append(e['game_id'])
    history.sort(key=lambda e:(time(e['completed_at']),e['game_id']))
    if any(e['game_id']==c['game_id'] for e in history):
        issues.append(dict(classification='LEAKAGE',reason='CURRENT_GAME_ALREADY_IN_PARENT_HISTORY'))
    # Legacy ordering must not be substituted for chronological availability silently.
    for team in (c['team'],c['opponent']):
        seq=[e for e in history if any(r['team']==team for r in e['teams'])]
        if [e['game_id'] for e in seq]!=[e['game_id'] for e in sorted(seq,key=lambda e:(e['season'],e['week'],e['game_id']))]:
            issues.append(dict(classification='STATE_TRANSITION_GAP',reason='LEGACY_ORDER_DIFFERS',team=team))
    complete=d.get('history_coverage_complete') is True and not any(i['classification']=='EXPECTED_UNAVAILABLE' for i in issues)
    features={}
    def record(name,value,status=None,why=None):
        semantic=status or ('UNKNOWN' if value is None else 'KNOWN_ZERO' if value==0 else 'KNOWN_VALUE')
        features[name]=dict(name=name,value=value,semantic_state=semantic,
                            source_version=c['source_version'],source_hashes=c['source_hashes'],as_of=c['evidence_cutoff'],
                            update_class='SCHEDULE_DERIVABLE' if name in {'team_rest','opponent_rest'} else 'RECURSIVE_PRIOR_STATE',
                            reconstruction_status='PASS' if value is not None else 'EXPECTED_UNAVAILABLE',reason=why)
    def metric_rows(team,metric):
        out=[]
        for e in history:
            own=next((r for r in e['teams'] if r['team']==team),None)
            if own is None: continue
            other=next(r for r in e['teams'] if r['team']!=team)
            if metric=='points_against': value=other.get('points')
            elif metric.startswith('allowed_'): value=other.get(metric[8:])
            elif metric in {'offensive_plays','pass_rate','rush_rate'}:
                a,b=own.get('attempts'),own.get('carries')
                value=None if a is None or b is None else a+b if metric=='offensive_plays' else (a if metric=='pass_rate' else b)/(a+b) if a+b else 0.0
            else: value=own.get(metric)
            out.append(value)
        return out
    def mean(team,metric,n):
        rows=metric_rows(team,metric)[-n:]
        return sum(rows)/len(rows) if complete and rows and all(v is not None for v in rows) else None
    def last(team,metric):
        rows=metric_rows(team,metric)
        return rows[-1] if complete and rows else None
    for prefix,team,opponent in [('team_',c['team'],c['opponent']),('opp_',c['opponent'],c['team'])]:
        cache={}
        for name in contract['margin']:
            if not name.startswith(prefix) or name in {'team_rest','opp_player_rows'}: continue
            suffix=name[len(prefix):]
            if suffix=='history_games': value=len(metric_rows(team,'points')) if complete else None
            elif suffix.endswith('_last'):
                value=last(team,{'points_for':'points','points_against':'points_against'}[suffix[:-5]])
            elif '_avg_' in suffix:
                metric,n=suffix.rsplit('_avg_',1); n=int(n)
                mappings={'points_for':'points','pass_attempts':'attempts','rush_attempts':'carries'}
                if metric.startswith('opponent_'):
                    metric={'opponent_points_allowed':'points_against','opponent_pass_yards_allowed':'allowed_passing_yards',
                            'opponent_rush_yards_allowed':'allowed_rushing_yards','opponent_pass_tds_allowed':'allowed_passing_tds',
                            'opponent_rush_tds_allowed':'allowed_rushing_tds'}[metric]
                    value=mean(opponent,metric,n)
                else: value=mean(team,mappings.get(metric,metric),n)
            elif suffix.endswith('_trend'): continue
            else: raise ValueError('UNMAPPED_FEATURE_'+name)
            cache[suffix]=value; record(name,value,why='NO_HISTORY_OR_MISSING_PRIMITIVE' if value is None else None)
        for trend,base in [('team_scoring_trend','points_for'),('opponent_scoring_trend','opponent_points_allowed'),
                           ('pace_trend','offensive_plays'),('pass_rate_trend','pass_rate')]:
            a,b=cache[base+'_avg_3'],cache[base+'_avg_5']
            record(prefix+trend,None if a is None or b is None else a-b)
    # Coach priors preserve the builder's finite historical scope and tie convention.
    for prefix,team in [('coach_',c['team']),('opponent_coach_',c['opponent'])]:
        assignment=d.get('coach_assignments',{}).get(team)
        valid=assignment and assignment.get('coach_id') and time(assignment['available_at'])<=cutoff
        prior=[]
        if valid:
            for e in history:
                if e['season']<d['coach_history_start_season']: continue
                for r in e['teams']:
                    if r.get('coach_id')==assignment['coach_id']:
                        other=next(o for o in e['teams'] if o['team']!=r['team'])
                        prior.append((r.get('points'),other.get('points')))
        known=valid and d.get('coach_coverage_complete') is True
        count=len(prior) if known else None
        vals=bool(prior) and all(a is not None and b is not None for a,b in prior)
        record(prefix+'prior_games',count)
        record(prefix+'prior_win_pct',sum(a>b for a,b in prior)/len(prior) if known and vals else None)
        record(prefix+'prior_points_for_avg',sum(a for a,b in prior)/len(prior) if known and vals else None)
        record(prefix+'prior_points_against_avg',sum(b for a,b in prior)/len(prior) if known and vals else None)
        if assignment and not valid: issues.append(dict(classification='LEAKAGE',reason='FUTURE_OR_UNRESOLVED_COACH_ASSIGNMENT'))
    for name,team in [('team_rest',c['team']),('opponent_rest',c['opponent'])]:
        previous=[s for s in registry if team in {s['home'],s['away']} and time(s['kickoff'])<kickoff and time(s['available_at'])<=cutoff]
        value=None
        if schedule_eligible and previous:
            prev=max(previous,key=lambda s:time(s['kickoff']))
            # Calendar game-date difference, not elapsed hours or filesystem mtime.
            value=(datetime.fromisoformat(g['game_date']).date()-datetime.fromisoformat(prev['game_date']).date()).days
        record(name,value,why='NO_PRIOR_SCHEDULE_OR_FIRST_GAME_REST_POLICY' if value is None else None)
    require(set(features)==set(contract['margin']),'FEATURE_MAPPING_INCOMPLETE')
    comparison=[]
    reference=d.get('reference')
    if reference:
        version_ok=(reference.get('source_hash')==contract['training_source']['sha256'] and
                    reference.get('feature_contract_version')==contract['version'])
        identity_ok=all(reference.get(k)==c[k] for k in ('game_id','team','opponent'))
        for name in contract['margin']:
            actual=features[name]; expected=reference.get('features',{}).get(name)
            if any(i['classification']=='LEAKAGE' for i in issues): status='LEAKAGE'
            elif not identity_ok: status='IDENTITY_MISMATCH'
            elif not version_ok: status='SOURCE_VERSION_MISMATCH'
            elif expected is None or actual['value'] is None: status='EXPECTED_UNAVAILABLE'
            elif isinstance(expected,dict):
                if expected['semantic_state']!=actual['semantic_state']: status='STATE_TRANSITION_GAP'
                elif math.isclose(actual['value'],expected['value'],rel_tol=1e-10,abs_tol=1e-8): status='PASS'
                else: status='NUMERIC_TOLERANCE'
            elif number(expected) and math.isclose(actual['value'],expected,rel_tol=1e-10,abs_tol=1e-8): status='PASS'
            else: status='NUMERIC_TOLERANCE'
            comparison.append(dict(feature=name,classification=status,actual=actual['value'],expected=expected))
    missing=[dict(name=n,status='UNRESOLVED',value=None,reason='EXCLUDED_FROM_EXP004_NOT_RECONSTRUCTED',
                  required_source='Cutoff-valid injury status plus prior usage' if not n.endswith('player_rows') else
                  'Canonical population of historical player-feature SOURCE ROWS, not roster size') for n in contract['excluded']]
    references=d.get('intelligence_references',{})
    for name,ref in references.items():
        require(name in {'cold_start_v1','v3_1'},'UNKNOWN_INTERFACE')
        require(ref.get('content_hash') and ref.get('version'),'SOURCE_VERSION_MISMATCH')
        if ref.get('status') in {'BLOCKED','NOT_EVALUABLE'}:
            issues.append(dict(classification='EXPECTED_UNAVAILABLE',reason='REFERENCE_ONLY_'+name))
    unavailable=[n for n,v in features.items() if v['value'] is None]
    transition=('TRANSITION_BLOCKED' if any(i['classification'] in {'LEAKAGE','STATE_TRANSITION_GAP'} for i in issues)
                else 'NOT_EVALUABLE' if not complete else 'TRANSITION_PARTIAL' if unavailable else 'TRANSITION_COMPLETE')
    input_for_hash=deepcopy(d)
    for key in ('bootstrap','transitions','schedule'):
        if key in input_for_hash: input_for_hash[key]=sorted(input_for_hash[key],key=lambda x:x['game_id'])
    state_id=digest(dict(input=input_for_hash,contract_hash=digest(contract),implementation=VERSION))
    output=dict(state_manifest=dict(context=c,state_version=state_id,parent_state_version=c.get('parent_state_version'),
                feature_contract_hash=digest(contract),margin_count=72,total_count=64),
                feature_reconstruction=[features[n] for n in contract['margin']],
                total_features=[features[n] for n in contract['total']],replay_comparison=comparison,
                transition_validation=dict(status=transition,unavailable_features=unavailable,SCORE_ONLY_SUFFICIENT='NO',
                    MINIMUM_REQUIRED_PRIMITIVE_CLASSES=list(PRIMITIVES),issues=issues),missing_feature_status=missing,
                lineage=dict(history=history,applied_event_ids=sorted(applied),observed_root_cutoff=root_cutoff,
                             intelligence_references=references))
    output['audit']=dict(contract=VERSION,result_hash=digest(output),safety=SAFETY.copy(),
                         validation_population=d.get('input_mode','EXPLICIT_SNAPSHOT'),
                         temporal_rules_satisfied=not issues and complete,
                         historical_availability_proven=d.get('input_mode')=='EXPLICIT_HISTORICAL_SNAPSHOT' and not issues and complete,
                         replay_equality_status='PASS' if comparison and all(r['classification']=='PASS' for r in comparison) else 'NOT_EVALUABLE_OR_MISMATCH',
                         monte_carlo_authorized=False)
    return output
