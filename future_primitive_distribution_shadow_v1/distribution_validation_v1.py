"""Measurement only: frozen 2025 universe, 100 draws/game, no production writes.

Run with python -B -m future_primitive_distribution_shadow_v1.distribution_validation_v1.
The authority loader is cached only within a serial run, as in the frozen bounded
validator; the unchanged public selector is called for every realization.
"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import random
from pathlib import Path
import platform
import sys

import numpy as np

from . import bounded_walk_forward_validation_v1 as bounded
from . import temporal_residual_selector_v1 as temporal
from . import seeded_joint_residual_sampler_v1 as seeded

VERSION = 'DISTRIBUTION_VALIDATION_V1_001'
CHECKPOINT_HASH = 'e164ba652a15a591c7f6e158b2701c335f431a857b6ff7d4018dc214e919560f'
SELECTOR_VERSION = 'TEMPORAL_RESIDUAL_SELECTOR_V1_001'
PRIMITIVES = ('plays',) + bounded.PRIMITIVES
DIRECTORY = Path(__file__).resolve().parent
OUTPUT = DIRECTORY / 'distribution_validation_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def realization_seed(game_id, index):
    require(isinstance(game_id, str) and bool(game_id), 'INVALID_GAME_ID')
    require(type(index) is int and 0 <= index < 100, 'INVALID_REALIZATION_INDEX')
    return int.from_bytes(hashlib.sha256(canonical([game_id, index])).digest(), 'big')


def quantiles(values, probabilities):
    return np.quantile(np.asarray(values, dtype=float), probabilities, method='linear')


def interval(actual, simulations, level):
    low, high = quantiles(simulations, [(1-level)/2, (1+level)/2])
    return dict(lower=float(low), upper=float(high), width=float(high-low),
                covered=bool(low <= actual <= high))


def prepare():
    # Verify the checkpoint's semantic hash without rerunning a completed stage.
    checkpoint = json.loads(bounded.REPORT.read_text())
    for key in ('result_hash_run_1', 'result_hash_run_2'):
        require(checkpoint.pop(key) == CHECKPOINT_HASH, 'CHECKPOINT_RECORDED_HASH')
    checkpoint.pop('global_rng_untouched')
    require(sha(canonical(checkpoint)) == CHECKPOINT_HASH, 'CHECKPOINT_CONTENT_HASH')
    frozen = json.loads((DIRECTORY / 'BOUNDED_WALK_FORWARD_VALIDATION_V1_HASHES_BEFORE.json').read_text())['files']
    for name, digest in frozen.items():
        require(sha((bounded.ROOT / name).read_bytes()) == digest, f'FROZEN_DEPENDENCY_CHANGED:{name}')
    require(temporal.SELECTOR_VERSION == SELECTOR_VERSION, 'SELECTOR_VERSION')
    training, outcomes = bounded._read(bounded.TRAINING), bounded._read(bounded.OUTCOMES)
    games = bounded._integrity(training, outcomes)
    authority = temporal._load_authorities()
    outcome_map = {(r['game_id'], r['team']): r for r in outcomes}
    targets = sorted((g for g, rows in games.items() if rows[0]['season'] == '2025'),
                     key=lambda g: (bounded._kickoff(games[g][0]), g))
    require(len(targets) == 285 and targets == [g['game_id'] for g in checkpoint['accepted']], 'BOUNDED_UNIVERSE_MISMATCH')
    prepared = []
    for gid, old in zip(targets, checkpoint['accepted']):
        rows = games[gid]
        require([r['team'] for r in rows] == [r['team'] for r in old['teams']], f'TEAM_ORDER:{gid}')
        require(sorted(r['is_home'] for r in rows) == ['0', '1'], f'HOME_AWAY_AMBIGUOUS:{gid}')
        kickoff = bounded._kickoff(rows[0]).isoformat()
        require(kickoff == old['kickoff'], f'KICKOFF_MISMATCH:{gid}')
        teams = []
        for row, previous in zip(rows, old['teams']):
            observed = outcome_map[(gid, row['team'])]
            actual = {m: float(observed['attempts' if m == 'pass_attempts' else m]) for m in bounded.PRIMITIVES}
            actual['plays'] = actual['pass_attempts'] + actual['carries']
            require(actual == previous['actual'], f'REALIZED_AUTHORITY_MISMATCH:{gid}')
            require(all(np.isfinite(v) for v in actual.values()), 'NONFINITE_ACTUAL')
            teams.append(dict(team=row['team'], opponent=row['opponent_team'],
                              is_home=row['is_home'] == '1', actual=actual, center=bounded._generate(row)[1]))
        prepared.append(dict(game_id=gid, kickoff=kickoff, teams=teams))
    return prepared, authority


@contextmanager
def cached_authority(authority):
    original = temporal._load_authorities
    temporal._load_authorities = lambda *args, **kwargs: authority
    try:
        yield
    finally:
        temporal._load_authorities = original


def check_selection(selection, target, authority, pair_map):
    _, chronology, _ = authority
    gid = selection['selected_historical_game']['game_id']
    require(gid != target['game_id'], 'SAME_GAME_LEAKAGE')
    require(chronology[gid] < chronology[target['game_id']], 'STRICT_BEFORE_FAILURE')
    require(selection['selected_historical_kickoff'] == chronology[gid].isoformat(), 'SELECTED_CHRONOLOGY_MISMATCH')
    require(selection['target_kickoff'] == target['kickoff'], 'TARGET_CHRONOLOGY_MISMATCH')
    require(selection['selector_version'] == SELECTOR_VERSION, 'SELECTOR_VERSION')
    require(selection['orientation'] in (0, 1), 'INVALID_ORIENTATION')
    pair = pair_map[gid]
    oriented = pair if selection['orientation'] == 0 else pair[::-1]
    for index, row in enumerate(oriented, 1):
        identity = {k: row[k] for k in (*seeded.IDENTITY_FIELDS, 'row_index')}
        require(selection[f'team_{index}_residual_identity'] == identity, 'PAIR_ORIENTATION_FAILURE')
        require(selection[f'team_{index}_residuals'] == {k: row[k] for k in seeded.RESIDUAL_FIELDS}, 'JOINT_VECTOR_FAILURE')
    require(all(selection['temporal_integrity'].values()), 'TEMPORAL_GATE_FAILURE')
    require(all(selection['pair_integrity'].values()), 'PAIR_GATE_FAILURE')
    require(selection['temporal_audit']['rng_draws'] == 2, 'RNG_DRAWS')


def summarize(observations, records):
    summary, detailed = {}, []
    # Stable game/team/index sorting makes aggregation independent of file/list ordering.
    records = sorted(records, key=lambda r: (r['kickoff'], r['game_id'], r['realization_index']))
    distributions = {(g['game_id'], t['team']): {m: [] for m in PRIMITIVES}
                     for g in observations for t in g['teams']}
    for record in records:
        for team in record['teams']:
            for m in PRIMITIVES:
                distributions[(record['game_id'], team['team'])][m].append(team['simulated'][m])
    for g in observations:
        for t in g['teams']:
            metrics = {}
            for m in PRIMITIVES:
                sims = distributions[(g['game_id'], t['team'])][m]
                require(len(sims) == 100, 'TEAM_REALIZATION_COUNT')
                metrics[m] = {str(int(level*100)): interval(t['actual'][m], sims, level) for level in (.5, .8, .9)}
            detailed.append(dict(game_id=g['game_id'], team=t['team'], intervals=metrics))
    for m in PRIMITIVES:
        actual = np.array([t['actual'][m] for g in observations for t in g['teams']], dtype=float)
        sim = np.array([t['simulated'][m] for r in records for t in r['teams']], dtype=float)
        rq, sq = quantiles(actual, [.1,.25,.5,.75,.9]), quantiles(sim, [.1,.25,.5,.75,.9])
        summary[m] = dict(realized_mean=float(actual.mean()), simulated_mean=float(sim.mean()),
                          bias=float(sim.mean()-actual.mean()), realized_sd=float(actual.std(ddof=0)),
                          simulated_sd=float(sim.std(ddof=0)), dispersion_ratio=float(sim.std(ddof=0)/actual.std(ddof=0)),
                          quantiles={f'P{p}': dict(realized=float(a), simulated=float(b)) for p,a,b in zip((10,25,50,75,90),rq,sq)},
                          intervals={level: dict(coverage=sum(d['intervals'][m][level]['covered'] for d in detailed)/len(detailed),
                                                 mean_width=float(np.mean([d['intervals'][m][level]['width'] for d in detailed]))) for level in ('50','80','90')},
                          tails=dict(lower_threshold=float(rq[0]), upper_threshold=float(rq[-1]),
                                     realized_lower_rate=float(np.mean(actual < rq[0])), simulated_lower_rate=float(np.mean(sim < rq[0])),
                                     realized_upper_rate=float(np.mean(actual > rq[-1])), simulated_upper_rate=float(np.mean(sim > rq[-1])),
                                     realized_p01_p99=quantiles(actual,[.01,.99]).tolist(), simulated_p01_p99=quantiles(sim,[.01,.99]).tolist()))
    return summary, detailed


def run_validation():
    targets, authority = prepare()
    pairs = {p[0]['game_id']: p for p in authority[0]}
    records = []
    with cached_authority(authority):
        for target in targets:
            seeds = [realization_seed(target['game_id'], i) for i in range(100)]
            require(len(set(seeds)) == 100, 'SEED_COLLISION')
            for index, seed in enumerate(seeds):
                selection = temporal.select_temporal_paired_residual(target['game_id'], target['kickoff'], seed)
                check_selection(selection, target, authority, pairs)
                teams = []
                for slot, team in enumerate(target['teams'], 1):
                    mechanisms, primitives, reconciliation = seeded._realize(team['center'], selection[f'team_{slot}_residuals'])
                    primitives['plays'] = mechanisms['plays']
                    require(set(primitives) == set(PRIMITIVES), 'PRIMITIVE_SCHEMA')
                    teams.append(dict(team=team['team'], is_home=team['is_home'], target_slot=slot,
                                      simulated=primitives, reconciliation=reconciliation,
                                      residual_identity=selection[f'team_{slot}_residual_identity']))
                records.append(dict(game_id=target['game_id'], kickoff=target['kickoff'], realization_index=index,
                                    seed=seed, historical_game_id=selection['selected_historical_game']['game_id'],
                                    historical_kickoff=selection['selected_historical_kickoff'], orientation=selection['orientation'],
                                    eligible_game_count=selection['eligible_game_count'], teams=teams))
    require(len(records) == 28500 and sum(len(r['teams']) for r in records) == 57000, 'REALIZATION_COUNTS')
    require(Counter(r['game_id'] for r in records) == Counter({g['game_id']: 100 for g in targets}), 'GAME_COUNTS')
    summary, intervals = summarize(targets, records)
    return dict(version=VERSION, authority=authority[2], observations=targets, records=records,
                intervals=intervals, summary=summary)


def protected_hashes():
    paths = [p for directory in (DIRECTORY, bounded.ROOT/'simulator_primitive_generator_v1', bounded.ROOT/'simulator_recursive_state_replay_v1')
             for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and OUTPUT not in p.parents]
    paths += [bounded.TRAINING, bounded.OUTCOMES]
    return {str(p.relative_to(bounded.ROOT)): sha(p.read_bytes()) for p in sorted(paths)}


def result_bytes(result):
    # No clock, output path, or directory traversal data enters the result hash.
    return canonical(result)


def main():
    OUTPUT.mkdir(exist_ok=False)
    before = protected_hashes()
    py_state, np_state = random.getstate(), np.random.get_state()
    manifest = dict(STATUS='FAIL_CLOSED', VERSION=VERSION, CREATED_AT=datetime.now(timezone.utc).isoformat(),
                    GAME_COUNT=285, REALIZATIONS_PER_GAME=100, GAME_REALIZATION_COUNT=0, TEAM_REALIZATION_COUNT=0,
                    TEMPORAL_SELECTOR_VERSION=SELECTOR_VERSION, STRICT_BEFORE='NOT_COMPLETED', SAME_GAME_EXCLUSION='NOT_COMPLETED',
                    FUTURE_EXCLUSION='NOT_COMPLETED', RUN_1_HASH=None, RUN_2_HASH=None, DETERMINISM='NOT_COMPLETED',
                    MONTE_CARLO_AUTHORIZED=False, PRODUCTION_INFLUENCE='NONE', TUNING_PERFORMED=False,
                    HISTORICAL_TEMPORAL_PROVENANCE='UNVERIFIED', PYTHON=platform.python_version(), NUMPY=np.__version__,
                    methodology=dict(sd='population ddof=0', quantiles='NumPy linear interpolation',
                                     coverage='inclusive central equal-tail intervals; 570 equally weighted observations',
                                     tails='strictly below realized P10 / above realized P90; ties excluded',
                                     plays='frozen canonical plays = pass_attempts + carries; excludes sacks',
                                     seed='SHA256(canonical JSON [game_id, zero-based realization_index]); full 256-bit integer',
                                     orientation='frozen target row slots retained; historical alphabetical A/B or B/A; target is_home retained',
                                     scope='authorized repeated shadow measurement only; historical feature as-of provenance unverified'))
    try:
        for run in (1,2):
            result = run_validation()
            payload = result_bytes(result)
            digest = sha(payload)
            (OUTPUT/f'run_{run}.json').write_bytes(payload + b'\n')
            manifest[f'RUN_{run}_HASH'] = digest
            print(f'RUN_{run}_HASH={digest}', flush=True)
            if run == 1:
                first_payload = payload
            else:
                require(payload == first_payload, 'DETERMINISM_FAILURE')
        require(py_state == random.getstate(), 'GLOBAL_PYTHON_RNG_MUTATED')
        now = np.random.get_state()
        require(np_state[0] == now[0] and np.array_equal(np_state[1],now[1]) and np_state[2:] == now[2:], 'GLOBAL_NUMPY_RNG_MUTATED')
        require(protected_hashes() == before, 'PROTECTED_FILES_CHANGED')
        manifest.update(STATUS='PASS_SHADOW_ONLY', GAME_REALIZATION_COUNT=28500, TEAM_REALIZATION_COUNT=57000,
                        STRICT_BEFORE='PASS', SAME_GAME_EXCLUSION='PASS', FUTURE_EXCLUSION='PASS',
                        DETERMINISM='PASS', GAME_PAIR_JOINT_ORIENTATION='PASS', GLOBAL_RNG_UNTOUCHED=True,
                        FROZEN_FILES_UNCHANGED=True, FAIL_CLOSED_CONDITION_FIRED=False,
                        AUTHORITIES=result['authority'], PROTECTED_SHA256=before)
        (OUTPUT/'summary.json').write_text(json.dumps(result['summary'], sort_keys=True, indent=2)+'\n')
        (OUTPUT/'result.sha256').write_text(manifest['RUN_1_HASH']+'  canonical run JSON (excluding terminal newline)\n')
        print('DETERMINISM=PASS', flush=True)
    except Exception as exc:
        manifest.update(ERROR=f'{type(exc).__name__}:{exc}', FAIL_CLOSED_CONDITION_FIRED=True)
        raise
    finally:
        (OUTPUT/'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')


if __name__ == '__main__':
    main()
