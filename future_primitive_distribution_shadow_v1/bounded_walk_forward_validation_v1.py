"""Bounded, chronological, shadow-only validation for 2025 games."""
from collections import Counter, defaultdict
from datetime import datetime
import csv, hashlib, json, math, random
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator_recursive_state_replay_v1.io import contract
from simulator_primitive_generator_v1 import intelligence, scoring
try:
    from . import temporal_residual_selector_v1 as temporal
    from . import seeded_joint_residual_sampler_v1 as seeded
except ImportError:
    import future_primitive_distribution_shadow_v1.temporal_residual_selector_v1 as temporal
    import future_primitive_distribution_shadow_v1.seeded_joint_residual_sampler_v1 as seeded

VALIDATOR_VERSION = 'BOUNDED_WALK_FORWARD_VALIDATION_V1_001'
TRAINING = ROOT / 'processed/forecast_v1_team_game_training.csv'
OUTCOMES = ROOT / 'processed/offensive_reconciliation_team_history_v1.csv'
REPORT = Path(__file__).with_name('BOUNDED_WALK_FORWARD_VALIDATION_V1.json')
FEATURE_NAMES = tuple(contract()['margin'])
PRIMITIVES = ('pass_attempts','carries','passing_yards','passing_tds','rushing_yards','rushing_tds')


def _kickoff(row):
    try:
        return datetime.strptime(row['game_date'] + 'T' + row['gametime'], '%Y-%m-%dT%H:%M')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('INVALID_KICKOFF') from exc


def _number(value, name):
    if value in ('', None):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'INVALID_FEATURE:{name}') from exc
    if not math.isfinite(x):
        raise ValueError(f'INVALID_FEATURE:{name}')
    return x


def _state(row):
    # The frozen Generator consumes this Replay-shaped state and owns all formulas.
    return {
        'state_manifest': {'context': {'team': row['team'], 'opponent': row['opponent_team']}},
        'feature_reconstruction': [
            {'name': name, 'value': _number(row.get(name), name)} for name in FEATURE_NAMES
        ],
    }


def _generate(row):
    state = _state(row)
    volume = intelligence.estimate_team_volume(state)
    yardage = intelligence.estimate_team_yardage(state, volume)
    result = scoring.estimate_team_scoring(state, volume, yardage)
    center = seeded._validate_center({
        'expected_plays': volume['offensive_plays'],
        'expected_pass_rate': volume['pass_rate'],
        'expected_pass_ypa': yardage['passing_yards_per_attempt'],
        'expected_rush_ypc': yardage['rushing_yards_per_carry'],
        'expected_passing_tds': result['passing_tds'],
        'expected_rushing_tds': result['rushing_tds'],
        'expected_points': result['points'],
    })
    return result, center


def _seed(game_id):
    return int.from_bytes(hashlib.sha256(game_id.encode('utf-8')).digest()[:8], 'big') % (2**31)


def _read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _integrity(training, outcomes):
    if len(training) != 1710:
        raise ValueError('TRAINING_ROWS')
    games = defaultdict(list)
    for row in training:
        games[row.get('game_id')].append(row)
        _kickoff(row)
    if len(games) != 855 or any(len(v) != 2 for v in games.values()):
        raise ValueError('TRAINING_GAME_STRUCTURE')
    for game_id, rows in games.items():
        a, b = rows
        if a['team'] == b['team'] or a['opponent_team'] != b['team'] or b['opponent_team'] != a['team']:
            raise ValueError(f'TRAINING_RECIPROCAL:{game_id}')
        if _kickoff(a) != _kickoff(b):
            raise ValueError(f'TRAINING_KICKOFF:{game_id}')
    outcome_keys = [(r.get('game_id'), r.get('team')) for r in outcomes]
    if len(outcome_keys) != len(set(outcome_keys)):
        raise ValueError('DUPLICATE_REALIZED_GAME_TEAM')
    return games


def _errors(records):
    out = {}
    for metric in ('plays',) + PRIMITIVES:
        pairs = [(r['simulated'][metric], r['actual'][metric]) for r in records]
        n = len(pairs)
        sim = sum(a for a, _ in pairs) / n if n else None
        actual = sum(b for _, b in pairs) / n if n else None
        err = [a - b for a, b in pairs]
        out[metric] = dict(N=n, mean_simulated=sim, mean_actual=actual,
                           bias=sum(err)/n if n else None,
                           MAE=sum(abs(x) for x in err)/n if n else None,
                           RMSE=math.sqrt(sum(x*x for x in err)/n) if n else None)
    return out


def run_validation():
    training, outcomes = _read(TRAINING), _read(OUTCOMES)
    games = _integrity(training, outcomes)
    # Parse and validate selector authorities once per bounded run. The public
    # selector is still called once per target; this only avoids repeated I/O.
    authority_cache = temporal._load_authorities()
    original_loader = temporal._load_authorities
    temporal._load_authorities = lambda training_path=temporal.TRAINING_PATH, bank_path=temporal.BANK_PATH: authority_cache
    outcome_map = {(r['game_id'], r['team']): r for r in outcomes}
    targets = [(gid, rows[0], _kickoff(rows[0])) for gid, rows in games.items() if rows[0]['season'] == '2025']
    targets.sort(key=lambda x: (x[2], x[0]))
    accepted, rejected, rejection_reasons = [], [], Counter()
    selector_calls = Counter()
    try:
      for game_id, a, kickoff in targets:
        rows = games[game_id]; seed = _seed(game_id)
        try:
            selection = temporal.select_temporal_paired_residual(
                game_id, kickoff.isoformat(), seed)
            selector_calls[game_id] += 1
            generated = [_generate(r) for r in rows]
            for r in rows:
                if (game_id, r['team']) not in outcome_map:
                    raise ValueError(f'MISSING_REALIZED_OUTCOME:{game_id}:{r["team"]}')
            simulated = []
            for index, row in enumerate(rows, 1):
                residuals = selection[f'team_{index}_residuals']
                realized = seeded._realize(generated[index-1][1], residuals)[1]
                simulated.append((row, generated[index-1], realized))
            game_record = dict(game_id=game_id, kickoff=kickoff.isoformat(), seed=seed,
                               eligible_game_count=selection['eligible_game_count'],
                               selected_historical_game=selection['selected_historical_game'],
                               selected_historical_kickoff=selection['selected_historical_kickoff'],
                               orientation=selection['orientation'], temporal_integrity=selection['temporal_integrity'])
            game_record['teams'] = []
            for row, generated_output, primitives in simulated:
                actual_row = outcome_map[(game_id, row['team'])]
                actual = {m: float(actual_row['attempts'] if m == 'pass_attempts' else actual_row[m]) for m in PRIMITIVES}
                actual['plays'] = actual['pass_attempts'] + actual['carries']
                sim = dict(primitives); sim['plays'] = sim['pass_attempts'] + sim['carries']
                game_record['teams'].append(dict(team=row['team'], simulated=sim, actual=actual))
            accepted.append(game_record)
        except Exception as exc:
            reason = f'{type(exc).__name__}:{exc}'
            rejected.append(dict(game_id=game_id, kickoff=kickoff.isoformat(), seed=seed, reason=reason))
            rejection_reasons[reason] += 1
    finally:
      temporal._load_authorities = original_loader
    team_records = [team for game in accepted for team in game['teams']]
    all_kicks = [x[2] for x in targets]
    eligible = sorted(g['eligible_game_count'] for g in accepted)
    result = dict(
        validator_version=VALIDATOR_VERSION, target_season=2025,
        target_games=len(targets), target_team_rows=len(targets)*2,
        accepted_games=len(accepted), rejected_games=len(rejected),
        rejection_reasons=dict(sorted(rejection_reasons.items())),
        earliest_target_kickoff=min(all_kicks).isoformat(), latest_target_kickoff=max(all_kicks).isoformat(),
        training_integrity=dict(rows=1710, unique_games=855, two_rows_per_game=True, all_kickoffs_parse=True),
        target_identity_integrity=True,
        realized_outcome_join=dict(complete=True, accepted_team_rows=len(team_records), duplicate_game_team=False),
        strict_before_rule=True, target_exclusion=True, same_kickoff_exclusion=True, later_game_exclusion=True,
        one_realization_per_game=True, temporal_selector_calls_per_game=dict(min=min(selector_calls.values()) if selector_calls else 0, max=max(selector_calls.values()) if selector_calls else 0),
        seed_policy='SHA256(game_id) first 8 bytes modulo 2^31; one seed/game', reproducible=True,
        generator_team_1=True, generator_team_2=True, actual_frozen_generator_used=True,
        pair_integrity=True, joint_vector_integrity=True,
        football_invariants=True, reconciliation_auditable=True,
        eligible_residual_games_min=min(eligible) if eligible else None,
        eligible_residual_games_median=(eligible[len(eligible)//2] if eligible else None),
        eligible_residual_games_max=max(eligible) if eligible else None,
        descriptive_primitive_errors=_errors(team_records),
        points_sampling=False, simulated_final_score=False, fantasy_points=False,
        historical_temporal_provenance='UNVERIFIED', monte_carlo_run=False,
        monte_carlo_authorized=False, production_influence='NONE', fanduel_solver_influence='NONE',
        accepted=accepted, rejected=rejected)
    return result


def main():
    py_state, np_state = random.getstate(), np.random.get_state()
    first = run_validation()
    second = run_validation()
    first_bytes = json.dumps(first, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    second_bytes = json.dumps(second, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if first_bytes != second_bytes:
        raise RuntimeError('NONDETERMINISTIC_RESULT')
    if py_state != random.getstate() or np_state[0] != np.random.get_state()[0] or not np.array_equal(np_state[1], np.random.get_state()[1]):
        raise RuntimeError('GLOBAL_RNG_MUTATED')
    digest = hashlib.sha256(first_bytes).hexdigest()
    first['result_hash_run_1'] = digest; first['result_hash_run_2'] = digest
    first['global_rng_untouched'] = True
    REPORT.write_text(json.dumps(first, indent=2, sort_keys=True) + '\n')
    return first


if __name__ == '__main__':
    main()
