"""Diagnostic arithmetic on frozen draws; no RNG, selector calls, or tuning.

Analytical pool moments enumerate existing rows with equal weights. They are
reference expectations, not new stochastic realizations or valid future pools.
"""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SHADOW=HERE.parent
ROOT=SHADOW.parent
DIST=SHADOW/'distribution_validation_v1'
PROV=SHADOW/'historical_asof_provenance_v1'
EXPECTED='bae4aa02d6c92632972f366ca8252d21b850a6547571c668cc84e200bff4e711'
BANK_HASH='614b079e45c4cda7c09c58b1f6820825f221f6a40596d356c26d93d199064c6b'
M=('plays','pass_attempts','carries','passing_yards','rushing_yards','passing_tds','rushing_tds')
N=('plays','pass_rate','pass_ypa','rush_ypc','passing_tds','rushing_tds')


def encode(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(x): return hashlib.sha256(x).hexdigest()
def require(x,why):
    if not x: raise ValueError(why)
def readcsv(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))
def stats(x):
    x=np.asarray(x,dtype=float); d=x-x.mean(); sd=x.std()
    return dict(n=len(x),mean=float(x.mean()),sd=float(sd),variance=float(sd**2),
                quantiles={f'P{p}':float(np.quantile(x,p/100)) for p in (1,5,10,25,50,75,90,95,99)},
                skew=float(np.mean(d**3)/sd**3) if sd else 0.)
def cov(x,y): return float(np.mean((x-x.mean())*(y-y.mean())))
def freq(x):
    return dict(n=len(x),negative=float(np.mean(x<0)),P0=float(np.mean(x==0)),P1=float(np.mean(x==1)),
                P2=float(np.mean(x==2)),P3plus=float(np.mean(x>=3)))
def halfup(x):
    base=np.floor(x); return base+(x-base>=.5)
def pack(p,r,y,z,td,rd):
    a=p*r; c=p-a
    return np.stack((p,a,c,a*y,c*z,td,rd),axis=-1)
def stages(c,e):
    v=c+e; p,r,y,z,td,rd=np.moveaxis(v,-1,0)
    raw=pack(p,r,y,z,td,rd)
    require(np.all(p>=0),'NEGATIVE_PLAYS')
    clipped=pack(p,np.clip(r,0,1),np.maximum(y,0),np.maximum(z,0),np.maximum(td,0),np.maximum(rd,0))
    pr=halfup(p); a=halfup(pr*np.clip(r,0,1)); carry=pr-a
    opp=np.stack((pr,a,carry,a*np.maximum(y,0),carry*np.maximum(z,0),np.maximum(td,0),np.maximum(rd,0)),axis=-1)
    final=opp.copy()
    final[...,3:]=halfup(final[...,3:])
    return raw,clipped,opp,final

def centers(c):
    p,r,y,z,td,rd=np.moveaxis(c,-1,0)
    a=np.rint(p*r); carry=p-a
    return np.stack((p,a,carry,np.rint(a*y),np.rint(carry*z),td,rd),axis=-1)

def source_hashes():
    files=[p for folder in (DIST,PROV) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    names=('distribution_validation_v1.py','bounded_walk_forward_validation_v1.py','temporal_residual_selector_v1.py',
           'historical_game_paired_residual_sampler_v1.py','seeded_joint_residual_sampler_v1.py','RESIDUAL_BANK_V1.csv',
           'RESIDUAL_BANK_V1_MANIFEST.json','CALIBRATION_V1.json')
    files.extend(SHADOW/n for n in names)
    files.extend(ROOT/'simulator_primitive_generator_v1'/n for n in ('intelligence.py','scoring.py'))
    files.append(ROOT/'processed/forecast_v1_team_game_training.csv')
    return {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sorted(set(files))}

def load():
    raw=(DIST/'run_1.json').read_bytes()
    require(sha(raw.rstrip(b'\n'))==EXPECTED,'FROZEN_HASH')
    require(raw==(DIST/'run_2.json').read_bytes(),'FROZEN_RUN_MISMATCH')
    frozen=json.loads(raw)
    manifest=json.loads((DIST/'manifest.json').read_text())
    for name,h in manifest['PROTECTED_SHA256'].items():
        require(sha((ROOT/name).read_bytes())==h,'FROZEN_DEPENDENCY:'+name)
    require(sha((SHADOW/'RESIDUAL_BANK_V1.csv').read_bytes())==BANK_HASH,'BANK_HASH')
    require(json.loads((PROV/'manifest.json').read_text())['HISTORICAL_ASOF_PROVENANCE']=='UNVERIFIED','PROVENANCE_STATUS')
    bank=readcsv(SHADOW/'RESIDUAL_BANK_V1.csv')
    training=readcsv(ROOT/'processed/forecast_v1_team_game_training.csv')
    kicks={r['game_id']:r['game_date']+'T'+r['gametime']+':00' for r in training}
    observations=[(g,t) for g in frozen['observations'] for t in g['teams']]
    require(len(observations)==570 and len(frozen['records'])==28500,'COUNTS')
    C=np.array([[t['center']['expected_'+n] for n in N] for g,t in observations],float)
    Y=np.array([[t['actual'][m] for m in M] for g,t in observations],float)
    E=np.array([[float(r['resid_'+n]) for n in N] for r in bank])
    BC=np.array([[float(r['expected_'+n]) for n in N] for r in bank])
    idx={(g['game_id'],t['team']):i for i,(g,t) in enumerate(observations)}
    selected=np.empty((570,100),int); final=np.empty((570,100,7)); orientations=np.empty((570,100),int)
    seen=set(); pairrows=defaultdict(list)
    for i,r in enumerate(bank): pairrows[r['game_id']].append(i)
    for record in frozen['records']:
        pair=sorted(pairrows[record['historical_game_id']],key=lambda i:bank[i]['team'])
        require(len(pair)==2,'PAIR_COUNT')
        oriented=pair if record['orientation']==0 else pair[::-1]
        require(kicks[record['historical_game_id']]<record['kickoff'],'TEMPORAL_LEAKAGE')
        for slot,t in enumerate(record['teams']):
            i=idx[(record['game_id'],t['team'])]; j=record['realization_index']; bi=t['residual_identity']['row_index']
            require((i,j) not in seen,'DUPLICATE_DRAW'); seen.add((i,j))
            require(bi==oriented[slot],'ORIENTATION_IDENTITY')
            r=bank[bi]
            require(all(str(t['residual_identity'][k])==r[k] for k in ('game_id','team','season','week','opponent_team')),'BANK_IDENTITY')
            selected[i,j]=bi; orientations[i,j]=record['orientation']; final[i,j]=[t['simulated'][m] for m in M]
    require(len(seen)==57000,'DRAW_COUNT')
    return frozen,observations,C,Y,E,BC,bank,kicks,selected,final,orientations,pairrows


def run():
    frozen,obs,C,Y,E,BC,bank,kicks,selected,F,orientations,pairrows=load()
    center=centers(C); sampled=E[selected]
    raw,clipped,opp,recovered=stages(C[:,None,:],sampled)
    require(np.array_equal(F,recovered),'TRANSFORMATION_RECOVERY')
    bankcenter=centers(BC)
    bankraw=stages(BC,E)[0]
    bank_effective=bankraw-bankcenter
    delta=raw-center[:,None,:]
    # Arithmetic expectations over all existing rows, no random selections.
    poolmeans=[]; poolseconds=[]; fullmeans=[]; fullseconds=[]; native_pool=[]; native_pool2=[]; poolrows=[]
    for i,(g,t) in enumerate(obs):
        ids=np.array([j for j,r in enumerate(bank) if kicks[r['game_id']]<g['kickoff']])
        ef=stages(C[i],E[ids])[-1]; ff=stages(C[i],E)[-1]
        poolmeans.append(ef.mean(axis=0)); poolseconds.append((ef**2).mean(axis=0))
        fullmeans.append(ff.mean(axis=0)); fullseconds.append((ff**2).mean(axis=0))
        native_pool.append(E[ids].mean(axis=0)); native_pool2.append((E[ids]**2).mean(axis=0))
        poolrows.append(dict(game_id=g['game_id'],team=t['team'],week=int(g['game_id'].split('_')[1]),
                             eligible_games=len(ids)//2,mean=dict(zip(M,ef.mean(axis=0).tolist())),
                             sd=dict(zip(M,np.sqrt((ef**2).mean(axis=0)-ef.mean(axis=0)**2).tolist()))))
    pm=np.array(poolmeans); ps=np.array(poolseconds); fm=np.array(fullmeans); fs=np.array(fullseconds)
    nm=np.array(native_pool); ns=np.array(native_pool2)
    # Exact orientation balancing over the already selected pair identities.
    partner={a:b for pair in pairrows.values() for a,b in (pair,pair[::-1])}
    other=np.array([[partner[j] for j in row] for row in selected])
    opposite=stages(C[:,None,:],E[other])[-1]
    balanced=np.stack((F,opposite),axis=-1)
    primitives={}; td={}; strata=[]
    for k,m in enumerate(M):
        y=Y[:,k]; c=center[:,k]; f=F[:,:,k].ravel(); pre=raw[:,:,k].ravel(); d=delta[:,:,k].ravel()
        cb=np.repeat(c,100); real_error=y-c; sim_error=f-cb
        bankstats=stats(bank_effective[:,k]); sampled_stats=stats(d)
        target_cov=2*cov(c,real_error); sim_cov=2*cov(cb,sim_error)
        require(abs(y.var()-(c.var()+real_error.var()+target_cov))<1e-8,'REALIZED_VARIANCE_CLOSURE')
        require(abs(f.var()-(cb.var()+sim_error.var()+sim_cov))<1e-8,'SIM_VARIANCE_CLOSURE')
        require(abs(f.mean()-y.mean()-(c.mean()-y.mean()+d.mean()+(f-pre).mean()))<1e-9,'MEAN_CLOSURE')
        pool_mean=float(pm[:,k].mean()); pool_sd=float(np.sqrt(ps[:,k].mean()-pool_mean**2))
        full_mean=float(fm[:,k].mean()); full_sd=float(np.sqrt(fs[:,k].mean()-full_mean**2))
        mean_bal=float(balanced[:,:,k,:].mean()); sd_bal=float(balanced[:,:,k,:].std())
        low,high=np.quantile(y,[.1,.9]); mask=(f<low)|(f>high); ym=(y<low)|(y>high)
        variance=dict(center_variance=float(c.var()),realized_center_error_variance=float(real_error.var()),
                      realized_twice_center_error_covariance=target_cov,simulated_error_variance=float(sim_error.var()),
                      simulated_twice_center_error_covariance=sim_cov,
                      covariance_difference=sim_cov-target_cov,error_variance_difference=float(sim_error.var()-real_error.var()),
                      excess_final_variance=float(f.var()-y.var()))
        transformations={}
        previous=raw[:,:,k]
        for label,arr in (('constraints',clipped),('opportunity_rounding',opp),('final_rounding',F)):
            nxt=arr[:,:,k]; diff=nxt-previous
            transformations[label]=dict(mean_shift=float(diff.mean()),variance_shift=float(nxt.var()-previous.var()),
                                        changed_fraction=float(np.mean(diff!=0)),mean_absolute_change=float(np.mean(abs(diff))))
            previous=nxt
        primitives[m]=dict(CENTER_BIAS=float(c.mean()-y.mean()),CENTER_SD=float(c.std()),REALIZED_SD=float(y.std()),
                           RESIDUAL_MEAN=bankstats['mean'],RESIDUAL_SD=bankstats['sd'],
                           SAMPLED_RESIDUAL_MEAN=sampled_stats['mean'],SAMPLED_RESIDUAL_SD=sampled_stats['sd'],
                           TRANSFORMATION_MEAN_SHIFT=float((f-pre).mean()),FINAL_BIAS=float(f.mean()-y.mean()),FINAL_SD_RATIO=float(f.std()/y.std()),
                           center=stats(c),realized=stats(y),bank_effective_residual=bankstats,sampled_effective_residual=sampled_stats,
                           pre_transform=stats(pre),final=stats(f),variance_decomposition=variance,transformations=transformations,
                           tails=dict(thresholds=[float(low),float(high)],simulated_fraction=float(mask.mean()),realized_fraction=float(ym.mean()),
                                      simulated_tail_variance_contribution=float(np.mean((f-f.mean())**2*mask)),
                                      realized_tail_variance_contribution=float(np.mean((y-y.mean())**2*ym)),
                                      note='Centered second-moment contributions, not effects of removing tails.'),
                           temporal_pool=dict(eligible_exact_mean=pool_mean,eligible_exact_sd=pool_sd,full_bank_reference_mean=full_mean,full_bank_reference_sd=full_sd,
                                              eligibility_mean_difference=pool_mean-full_mean,eligibility_sd_difference=pool_sd-full_sd,
                                              observed_minus_eligible_mean=float(f.mean()-pool_mean),observed_minus_eligible_sd=float(f.std()-pool_sd)),
                           pairing=dict(orientation_balanced_mean=mean_bal,orientation_balanced_sd=sd_bal,
                                        observed_minus_balanced_mean=float(f.mean()-mean_bal),observed_minus_balanced_sd=float(f.std()-sd_bal),
                                        theoretical_pairing_marginal_effect=0.,
                                        explanation='Uniform pair plus unbiased A/B orientation gives uniform team-row marginal; pair correlation affects joint outcomes, not marginal expectation. Balance uses stored pairs only.'))
        require(abs(primitives[m]['FINAL_BIAS']-frozen['summary'][m]['bias'])<1e-10,'FROZEN_METRIC_BIAS')
        require(abs(primitives[m]['FINAL_SD_RATIO']-frozen['summary'][m]['dispersion_ratio'])<1e-10,'FROZEN_METRIC_SD')
        if m in ('passing_tds','rushing_tds'):
            nk=N.index(m); r=sampled[:,:,nk].ravel(); change=f-pre
            td[m]=dict(deterministic_expectation_mean=float(c.mean()),realized_mean=float(y.mean()),raw_bank_residual_mean=float(E[:,nk].mean()),
                       sampled_residual_mean=float(r.mean()),pre_transform_mean=float(pre.mean()),post_transform_mean=float(f.mean()),
                       zero_floor_frequency=float(np.mean(pre<0)),zero_floor_mean_shift=float(np.maximum(-pre,0).mean()),
                       rounding_frequency=float(np.mean(F[:,:,k]!=opp[:,:,k])),rounding_mean_shift=float(np.mean(F[:,:,k]-opp[:,:,k])),
                       rounding_mean_absolute_change=float(np.mean(abs(F[:,:,k]-opp[:,:,k]))),
                       frequencies=dict(realized=freq(y),deterministic_expected=freq(c),pre_transform=freq(pre),simulated_final=freq(f)))
        if m in ('passing_tds','rushing_tds','carries','rushing_yards'):
            week=np.array([int(g['game_id'].split('_')[1]) for g,t in obs]); home=np.array([t['is_home'] for g,t in obs])
            groups=[('time','weeks_01_05',week<=5),('time','weeks_06_10',(week>=6)&(week<=10)),
                    ('time','weeks_11_15',(week>=11)&(week<=15)),('time','weeks_16_postseason',week>=16),
                    ('venue','home',home),('venue','away',~home)]
            if m.endswith('tds'): cuts=[c<=0,c==1,c>=2]; labels=['low_0','mid_1','high_2plus']
            else:
                q1,q2=np.quantile(c,[1/3,2/3]); cuts=[c<=q1,(c>q1)&(c<=q2),c>q2]; labels=[f'low_le_{q1:g}',f'mid_le_{q2:g}',f'high_gt_{q2:g}']
            groups += [('expectation',label,mask) for label,mask in zip(labels,cuts)]
            for dimension,label,ix in groups:
                if not ix.any(): continue
                val=F[ix,:,k]; ry=y[ix]; cc=c[ix]
                strata.append(dict(primitive=m,dimension=dimension,bucket=label,team_observations=int(ix.sum()),
                                   game_count=len({obs[i][0]['game_id'] for i in np.flatnonzero(ix)}),draws=int(val.size),
                                   sparse=int(ix.sum())<30,center_bias=float(cc.mean()-ry.mean()),final_bias=float(val.mean()-ry.mean()),
                                   realized_sd=float(ry.std()),simulated_sd=float(val.std()),sd_ratio=float(val.std()/ry.std()) if ry.std() else None,
                                   zero_floor_mean_shift=float((clipped[ix,:,k]-raw[ix,:,k]).mean()),
                                   eligible_exact_mean=float(pm[ix,k].mean()),eligible_exact_sd=float(np.sqrt(ps[ix,k].mean()-pm[ix,k].mean()**2)),
                                   sampled_effective_residual_mean=float(delta[ix,:,k].mean())))
    native={n:dict(bank=stats(E[:,j]),sampled=stats(sampled[:,:,j].ravel()),
                   target_weighted_eligible_mean=float(nm[:,j].mean()),target_weighted_eligible_sd=float(np.sqrt(ns[:,j].mean()-nm[:,j].mean()**2))) for j,n in enumerate(N)}
    # Derived-equation variance shares include covariance via Cov(component,total).
    p,r,yp,yc,_,_=np.moveaxis(C[:,None,:],-1,0); ep,er,eyp,eyc,_,_=np.moveaxis(sampled,-1,0)
    carry_center=p*(1-r); carry_delta=(1-r)*ep-p*er-ep*er
    terms_c={'center':np.broadcast_to(carry_center,ep.shape),'plays_residual':(1-r)*ep,'pass_rate_residual':-p*er,'interaction':-ep*er}
    terms_y={'center':np.broadcast_to(carry_center*yc,ep.shape),'carry_effect':yc*carry_delta,'rush_efficiency_effect':carry_center*eyc,'interaction':carry_delta*eyc}
    equation={}
    for name,terms in (('carries',terms_c),('rushing_yards',terms_y)):
        total=sum(terms.values()).ravel()
        require(np.allclose(total,raw[:,:,M.index(name)].ravel(),atol=1e-10),'EQUATION_IDENTITY')
        equation[name]={label:dict(mean=float(v.mean()),variance=float(v.var()),covariance_allocated_variance=cov(v.ravel(),total)) for label,v in terms.items()}
    operations={n:dict(count=int(np.sum(mask)),fraction=float(np.mean(mask))) for n,mask in
                [('pass_rate_clamp',(C[:,None,1]+sampled[:,:,1]<0)|(C[:,None,1]+sampled[:,:,1]>1)),
                 ('pass_ypa_floor',C[:,None,2]+sampled[:,:,2]<0),('rush_ypc_floor',C[:,None,3]+sampled[:,:,3]<0)]}
    correlations={}
    for k,m in enumerate(M):
        # Target rows retain the frozen two-team adjacency.
        correlations[m]=dict(realized_pair_correlation=float(np.corrcoef(Y[::2,k],Y[1::2,k])[0,1]),
                             simulated_pair_correlation=float(np.corrcoef(F[::2,:,k].ravel(),F[1::2,:,k].ravel())[0,1]))
    result=dict(version='CALIBRATION_CANDIDATE_AUDIT_V1_001',status='PASS_SHADOW_ONLY',frozen_distribution_hash=EXPECTED,
                historical_asof_provenance='UNVERIFIED',counts=dict(games=285,observations=570,stored_team_realizations=57000,bank_rows=1678),
                primitives=primitives,native_mechanisms=native,td=td,stratification=strata,temporal_pools=poolrows,
                equation_variance=equation,operation_counts=operations,pair_correlations=correlations,
                methodology=dict(residual='Native additive residual for plays/TDs. For attempts/carries/yards, effective raw primitive minus deterministic primitive center; bank values at original bank centers, sampled values at target centers. Not interchangeable native residual parameters.',
                                 transformations='raw continuous equations -> constraints -> opportunity rounding -> final rounding; variance shifts depend on this declared ordering.',
                                 pool='Exact finite-population moments, no RNG. Full bank is descriptive reference containing ineligible rows, never supplied to temporal selector.',
                                 pairing='Unbiased orientation yields exact uniform-row marginal. Stored-pair orientation averaging isolates finite orientation variation; does not estimate joint counterfactuals.',
                                 inference='Algebraic attribution is descriptive, not out-of-sample causal evidence. No uncertainty estimate treats 57000 draws as independent historical observations.'),
                fail_closed_condition_fired=False,TUNING_PERFORMED=False,FROZEN_PARAMETERS_CHANGED=False,MONTE_CARLO_AUTHORIZED=False,PRODUCTION_INFLUENCE='NONE')
    return result


def main():
    before=source_hashes()
    first=run(); second=run(); raw=encode(first)
    h1=sha(raw); h2=sha(encode(second)); require(raw==encode(second),'AUDIT_NONDETERMINISM')
    require(source_hashes()==before,'FROZEN_MUTATION')
    outputs={'summary.json':json.dumps(first,sort_keys=True,indent=2)+'\n',
             'manifest.json':json.dumps(dict(STATUS='PASS_SHADOW_ONLY',CREATED_AT=datetime.now(timezone.utc).isoformat(),
                  RUN_1_HASH=h1,RUN_2_HASH=h2,DETERMINISM='PASS',source_sha256=before,FROZEN_FILES_UNCHANGED=True,
                  NEW_STOCHASTIC_REALIZATIONS=0,TUNING_PERFORMED=False,FROZEN_PARAMETERS_CHANGED=False,
                  MONTE_CARLO_AUTHORIZED=False,PRODUCTION_INFLUENCE='NONE'),sort_keys=True,indent=2)+'\n',
             'result.sha256':h1+'\n','td_frequencies.json':json.dumps(first['td'],sort_keys=True,indent=2)+'\n',
             'stratification.json':json.dumps(first['stratification'],sort_keys=True,indent=2)+'\n'}
    out=io.StringIO(); columns=['primitive','CENTER_BIAS','CENTER_SD','REALIZED_SD','RESIDUAL_MEAN','RESIDUAL_SD','SAMPLED_RESIDUAL_MEAN','SAMPLED_RESIDUAL_SD','TRANSFORMATION_MEAN_SHIFT','FINAL_BIAS','FINAL_SD_RATIO']
    writer=csv.DictWriter(out,fieldnames=columns,lineterminator='\n'); writer.writeheader()
    for m in M: writer.writerow(dict(primitive=m,**{k:first['primitives'][m][k] for k in columns[1:]}))
    outputs['decomposition.csv']=out.getvalue()
    for name,data in outputs.items():
        with (HERE/name).open('x') as f:f.write(data)
    print(f'RUN_1_HASH={h1}\nRUN_2_HASH={h2}\nDETERMINISM=PASS')

if __name__=='__main__': main()
