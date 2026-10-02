"""Fixed authorized validation grid, run twice; no forecasting or integration API.

Protocol fixed before execution: residuals only; ddof=1; linear quantiles;
average-tie Spearman. Numeric discrepancies are descriptive, with no statistical
acceptance thresholds or uniformity rejection. PASS denotes exact validation
checks plus descriptive recovery, not a significance or equivalence test.
"""
import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import scipy
from scipy.stats import rankdata

from . import historical_game_paired_residual_sampler_v1 as sampler

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
STEM = 'PAIRED_SAMPLER_DISTRIBUTION_RECOVERY_V1'
FIELDS = sampler.frozen.RESIDUAL_FIELDS
CENTER = dict(zip(sampler.frozen.CENTER_FIELDS, (100, .5, 7, 4, 2, 1, 24)))
QUANTILES = ('p05', 'p25', 'p50', 'p75', 'p95')


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def protected():
    paths = []
    for name in ('simulator_primitive_generator_v1', 'simulator_recursive_state_replay_v1'):
        paths.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(BASE/name for name in (
        'CALIBRATION_V1.json', 'RESIDUAL_BANK_V1.csv',
        'seeded_joint_residual_sampler_v1.py', 'test_seeded_joint_residual_sampler_v1.py',
        'historical_game_paired_residual_sampler_v1.py',
        'test_historical_game_paired_residual_sampler_v1.py', 'GAME_COUPLING_AUDIT_V1.json'))
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sorted(paths)}


def summary(values):
    return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)),
                **dict(zip(QUANTILES, map(float, np.quantile(values, [.05,.25,.5,.75,.95], method='linear')))))


def compare(source, sampled):
    a, b = summary(source), summary(sampled)
    return dict(source=a, sampled=b, absolute_mean_difference=abs(b['mean']-a['mean']),
                relative_std_difference=(b['std']-a['std'])/a['std'] if a['std'] else None,
                absolute_relative_std_difference=abs(b['std']-a['std'])/a['std'] if a['std'] else None,
                quantile_differences={k:b[k]-a[k] for k in QUANTILES},
                absolute_quantile_differences={k:abs(b[k]-a[k]) for k in QUANTILES})


def correlations(pairs):
    result = {'pearson':{}, 'spearman':{}}
    for i, field in enumerate(FIELDS):
        x, y = pairs[:,0,i], pairs[:,1,i]
        result['pearson'][field] = float(np.corrcoef(x,y)[0,1])
        result['spearman'][field] = float(np.corrcoef(rankdata(x,method='average'),rankdata(y,method='average'))[0,1])
    return result


def transforms(pairs):
    sums = pairs.sum(axis=1)
    return dict(sum_resid_plays=sums[:,0],
                absolute_difference_resid_plays=np.abs(pairs[:,0,0]-pairs[:,1,0]),
                sum_resid_passing_tds=sums[:,4], sum_resid_rushing_tds=sums[:,5],
                combined_game_offensive_td_residual=sums[:,4]+sums[:,5])


def reference():
    rows,digest = sampler.frozen._load_bank(sampler.BANK_PATH)
    pairs = sampler._paired_population(rows)
    assert len(rows)==1678 and len(pairs)==839
    assert CENTER['expected_plays']+min(r['resid_plays'] for r in rows)>=0
    sampler.frozen._validate_center(CENTER)
    vectors=np.array([[[r[k] for k in FIELDS] for r in pair] for pair in pairs])
    targets=json.loads((BASE/'GAME_COUPLING_AUDIT_V1.json').read_text())
    corr=correlations(vectors)
    symcorr=correlations(np.concatenate((vectors,vectors[:,::-1,:]),axis=0))
    for method in corr:
        for field in FIELDS:
            assert abs(corr[method][field]-targets['cross_team_'+method][field])<1e-12
            assert abs(symcorr[method][field]-targets['symmetrized_ordering_sensitivity'][method][field])<1e-12
    for key, values in transforms(vectors).items():
        stored=targets['transformed_game_summaries'][key]
        for stat,value in summary(values).items():
            alias={'std':'std_sample','p50':'median'}.get(stat,stat)
            assert abs(value-stored[alias])<1e-12
    return pairs,vectors,targets,digest


def run_once(run):
    """Exactly 10,000 unmodified public sampler calls; retain residuals in memory."""
    state=random.getstate()
    npstate=np.random.get_state()
    pairs,source,targets,digest=reference()
    lookup={a['game_id']:(a,b) for a,b in pairs}
    counts={key:0 for key in sorted(lookup)}
    orientations=[0,0]
    sampled=np.empty((10000,2,6),dtype=float)
    selection_trace=hashlib.sha256()
    for seed in range(10000):
        output=sampler.sample_paired_residual(CENTER,CENTER,seed=seed)
        assert output['sampler_version']==sampler.SAMPLER_VERSION
        assert output['residual_bank_sha256']==digest and output['seed']==seed
        game=output['selected_historical_game']
        a,b=lookup[game['game_id']]
        assert game==dict(season=a['season'],week=a['week'],game_id=a['game_id'],historical_team_a=a['team'],historical_team_b=b['team'])
        orientation=output['orientation']
        assert type(orientation) is int and orientation in (0,1)
        expected=(a,b) if orientation==0 else (b,a)
        for i,row in enumerate(expected):
            team=output['team_'+str(i+1)]
            assert team['center']==CENTER
            assert team['selected_historical_residual_identity']=={k:row[k] for k in (*sampler.frozen.IDENTITY_FIELDS,'row_index')}
            assert team['residuals']=={k:row[k] for k in FIELDS}
            sampled[seed,i,:]=[team['residuals'][k] for k in FIELDS]
        assert output['pair_integrity']==dict(historical_same_game=True,historical_reciprocal_opponents=True,joint_vectors_preserved=True)
        assert output['authorization']==dict(monte_carlo=False,production_influence='NONE',fanduel_solver_influence='NONE')
        counts[game['game_id']]+=1
        orientations[orientation]+=1
        selection_trace.update(encode([seed,game['game_id'],orientation,sampled[seed].tolist()]))
        if (seed+1)%1000==0: print(f'Run {run}: {seed+1}/10000 calls',flush=True)
    source_rows=source.reshape(-1,6)
    sampled_rows=sampled.reshape(-1,6)
    marginal={k:compare(source_rows[:,i],sampled_rows[:,i]) for i,k in enumerate(FIELDS)}
    scorr=correlations(sampled)
    cross={}
    for field in FIELDS:
        cross[field]={}
        for method in ('pearson','spearman'):
            target=targets['cross_team_'+method][field]
            symmetric=targets['symmetrized_ordering_sensitivity'][method][field]
            value=scorr[method][field]
            cross[field][method]=dict(source_canonical=target,source_symmetrized=symmetric,sampled=value,
                absolute_difference=abs(value-target),absolute_difference_symmetrized=abs(value-symmetric))
    source_matrix=np.corrcoef(source_rows,rowvar=False)
    sampled_matrix=np.corrcoef(sampled_rows,rowvar=False)
    difference=np.abs(source_matrix-sampled_matrix)
    within=dict(fields=list(FIELDS),source_matrix=source_matrix.tolist(),sampled_matrix=sampled_matrix.tolist(),
                absolute_difference_matrix=difference.tolist(),maximum_absolute_cell_difference=float(difference.max()),
                mean_absolute_offdiagonal_difference=float(difference[~np.eye(6,dtype=bool)].mean()))
    source_t, sampled_t=transforms(source),transforms(sampled)
    transformed={k:dict(**compare(source_t[k],sampled_t[k]),frozen_reference_summary=targets['transformed_game_summaries'][k]) for k in source_t}
    frequencies=np.array(list(counts.values()))
    frequency=dict(count_per_historical_game=counts,min_count=int(frequencies.min()),max_count=int(frequencies.max()),
                   mean_count=float(frequencies.mean()),std_count=float(frequencies.std(ddof=1)),
                   orientation_0=orientations[0],orientation_1=orientations[1],diagnostic_only=True)
    assert sum(counts.values())==10000 and sum(orientations)==10000
    assert random.getstate()==state
    npafter=np.random.get_state()
    assert npstate[0]==npafter[0] and np.array_equal(npstate[1],npafter[1]) and npstate[2:]==npafter[2:]
    return dict(audit='PAIRED_SAMPLER_DISTRIBUTION_RECOVERY_AUDIT_V1',sampler_version=sampler.SAMPLER_VERSION,
                residual_bank_sha256=digest,seeds='0..9999 inclusive',paired_draws=10000,team_vector_observations=20000,
                neutral_center_both_teams=CENTER,source_games=839,source_rows=1678,malformed_games=0,
                method=dict(std='sample ddof=1 including frequency counts',quantiles='linear interpolation',
                            spearman='Pearson correlation of average-tie ranks',primary_data='selected historical residuals only',
                            recovery='descriptive; no statistical acceptance thresholds',
                            reference_reconstruction_tolerance=1e-12,
                            orientation='random orientation converges to symmetrized source; canonical targets also reported'),
                marginal=marginal,cross_team=cross,within_team=within,game_level_transforms=transformed,
                frequency=frequency,selection_trace_sha256=selection_trace.hexdigest(),global_rng_untouched=True,
                all_draws_pair_and_vector_integrity=True,reference_reconstruction='PASS',
                versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))


def table(headers, rows):
    def fmt(x): return f'{x:.9f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(fmt,row))+' |' for row in rows])


def markdown(report):
    p=report['primary_results']; status=report['status']
    parts=['# Paired Sampler Distribution Recovery Audit V1',
           'PASS is a descriptive sampler-recovery assessment with exact integrity and reproducibility checks. '
           'No statistical equivalence or significance threshold was applied. All discrepancies are reported below. '
           'Frequency and orientation counts are diagnostics only.',
           'Each complete run called the unchanged frozen sampler once for every seed 0..9999: 10,000 pairs and '
           '20,000 team vectors. Two runs were performed (20,000 calls total across the reproducibility check). '
           'Both teams used the fixed center '+json.dumps(CENTER,sort_keys=True)+'. '
           'Only selected residuals enter the comparisons; realized primitives were neither analyzed nor interpreted.',
           'Standard deviations use ddof=1; quantiles use linear interpolation; Spearman uses average-tie ranks. '
           'Source summaries and correlations were reconstructed and checked against GAME_COUPLING_AUDIT_V1 within 1e-12 '
           '(floating-point reference-check tolerance, not a recovery threshold).',
           table(['Status field','Value'],status.items()),'## Marginal recovery']
    stats=('mean','std',*QUANTILES)
    for field,values in p['marginal'].items():
        parts.extend(['### '+field,table(['Population',*stats],[[name,*[values[name][k] for k in stats]] for name in ('source','sampled')]),
                      table(['Absolute mean difference','Relative std difference','Absolute relative std difference'],[[values[k] for k in ('absolute_mean_difference','relative_std_difference','absolute_relative_std_difference')]]),
                      table(['Quantile','Sample minus source','Absolute difference'],[[k,values['quantile_differences'][k],values['absolute_quantile_differences'][k]] for k in QUANTILES])])
    parts.extend(['## Cross-team dependence',
                  'The requested targets use lexicographic historical A/B ordering. Random orientation targets the '
                  'symmetrized population instead; both comparisons are shown. This small reference difference '
                  'is structural, not sampler tuning. Plays, pass rate, and passing TDs are listed explicitly with all mechanisms.'])
    for method in ('pearson','spearman'):
        parts.extend(['### '+method,table(['Residual','Canonical target','Sampled','Absolute difference','Symmetrized target','Absolute symmetrized difference'],
                     [[f,*[p['cross_team'][f][method][k] for k in ('source_canonical','sampled','absolute_difference','source_symmetrized','absolute_difference_symmetrized')]] for f in FIELDS])])
    parts.append('## Within-team dependence')
    for key in ('source_matrix','sampled_matrix','absolute_difference_matrix'):
        parts.extend([key,table(['Residual',*FIELDS],[[f,*row] for f,row in zip(FIELDS,p['within_team'][key])])])
    parts.append('## Game-level residual transforms')
    for field,values in p['game_level_transforms'].items():
        parts.extend(['### '+field,table(['Population',*stats],[[name,*[values[name][k] for k in stats]] for name in ('source','sampled')]),
                      table(['Absolute mean difference','Relative std difference'],[[values['absolute_mean_difference'],values['relative_std_difference']]]),
                      table(['Quantile','Sample minus source','Absolute difference'],[[k,values['quantile_differences'][k],values['absolute_quantile_differences'][k]] for k in QUANTILES])])
    parts.extend(['## Frequency diagnostics','All 839 game counts, including any zero counts, are recorded in the JSON primary_results.frequency.count_per_historical_game. '
                  'Summary and orientation counts appear in the status table above. No uniformity threshold was used.',
                  '## Reproducibility and limits',
                  'The SHA256 values cover canonical sorted-key, indented UTF-8 JSON of primary_results with a trailing newline. '
                  'This excludes the enclosing reproducibility metadata to avoid self-referential hashing. '
                  'Both full runs were independently computed. Python and NumPy global RNG states were unchanged. '
                  'Protected before/after hashes and all runtime versions are in the JSON.',
                  'This audit describes recovery of a frozen empirical residual population. It establishes no forecast '
                  'accuracy, future-game calibration, production readiness, or FanDuel usefulness. Historical temporal '
                  'provenance remains UNVERIFIED. MONTE_CARLO_RUN=false and MONTE_CARLO_AUTHORIZED=false denote '
                  'no forecast/production Monte Carlo; the only repeated draws were this explicitly authorized fixed validation grid.',
                  '## Files created or modified','\n'.join('- `'+f+'`' for f in report['files_created_or_modified']),
                  'Reproduce only within the same authorized audit scope: '
                  '`python3 -B -m future_primitive_distribution_shadow_v1.audit_paired_sampler_distribution_recovery_v1`'])
    return '\n\n'.join(parts)+'\n'


def main():
    before=protected()
    # Persist before evidence before any sampler call, including the method/script hash.
    evidence=dict(protected_before=before,script_sha256=sha(Path(__file__).read_bytes()),
                  protocol='Fixed seeds 0..9999 twice; residuals only; descriptive recovery, no significance thresholds')
    (BASE/(STEM+'_HASHES_BEFORE.json')).write_bytes(encode(evidence))
    first=run_once(1)
    first_bytes=encode(first)
    second=run_once(2)
    second_bytes=encode(second)
    assert first_bytes==second_bytes
    after=protected()
    assert before==after
    assert evidence['script_sha256']==sha(Path(__file__).read_bytes())
    f=first['frequency']; w=first['within_team']
    status=dict(AUDIT_STATUS='PASS',SAMPLER_DISTRIBUTION_RECOVERY='PASS (DESCRIPTIVE)',SEEDS='0..9999',
                PAIRED_DRAWS=10000,TEAM_VECTOR_OBSERVATIONS=20000,
                MARGINAL_RECOVERY='PASS (DESCRIPTIVE)',CROSS_TEAM_DEPENDENCE_RECOVERY='PASS (DESCRIPTIVE)',
                WITHIN_TEAM_DEPENDENCE_RECOVERY='PASS (DESCRIPTIVE)',GAME_LEVEL_TRANSFORM_RECOVERY='PASS (DESCRIPTIVE)',
                MAX_CROSS_TEAM_PEARSON_DIFFERENCE=max(v['pearson']['absolute_difference'] for v in first['cross_team'].values()),
                MAX_CROSS_TEAM_SPEARMAN_DIFFERENCE=max(v['spearman']['absolute_difference'] for v in first['cross_team'].values()),
                MAX_WITHIN_TEAM_CORRELATION_DIFFERENCE=w['maximum_absolute_cell_difference'],
                MEAN_WITHIN_TEAM_OFFDIAGONAL_DIFFERENCE=w['mean_absolute_offdiagonal_difference'],
                HISTORICAL_GAME_SELECTION_COUNT_MIN=f['min_count'],HISTORICAL_GAME_SELECTION_COUNT_MAX=f['max_count'],
                HISTORICAL_GAME_SELECTION_COUNT_MEAN=f['mean_count'],HISTORICAL_GAME_SELECTION_COUNT_STD=f['std_count'],
                ORIENTATION_0_COUNT=f['orientation_0'],ORIENTATION_1_COUNT=f['orientation_1'],
                REPRODUCIBLE=True,RESULT_SHA256_RUN_1=sha(first_bytes),RESULT_SHA256_RUN_2=sha(second_bytes),
                GLOBAL_RNG_UNTOUCHED=True,PAIRED_SAMPLER_UNCHANGED=True,SEEDED_SAMPLER_UNCHANGED=True,
                RESIDUAL_BANK_UNCHANGED=True,CALIBRATION_SPEC_UNCHANGED=True,FROZEN_GENERATOR_UNCHANGED=True,
                FROZEN_REPLAY_UNCHANGED=True,HISTORICAL_TEMPORAL_PROVENANCE='UNVERIFIED',
                MONTE_CARLO_RUN=False,MONTE_CARLO_AUTHORIZED=False,PRODUCTION_INFLUENCE='NONE',FANDUEL_SOLVER_INFLUENCE='NONE')
    files=[str((BASE/name).relative_to(ROOT)) for name in (Path(__file__).name,STEM+'.json',STEM+'.md',STEM+'_HASHES_BEFORE.json')]
    report=dict(status=status,primary_results=first,reproducibility=dict(complete_runs=2,total_sampler_calls=20000,
                primary_bytes_equal=True,hash_scope='encode(primary_results): sorted keys, indent=2, allow_nan=False, UTF-8, trailing newline'),
                protected_before=before,protected_after=after,script_sha256=evidence['script_sha256'],files_created_or_modified=files)
    (BASE/(STEM+'.json')).write_bytes(encode(report))
    (BASE/(STEM+'.md')).write_text(markdown(report))
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':
    main()
