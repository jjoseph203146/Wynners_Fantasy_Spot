"""One-game shadow adapter: frozen Generator mechanisms -> one paired sample.

No RNG, retries, seed loops, forecasting, replay transitions, or score generation.
Accepts complete frozen Replay states; feature reconstruction remains Replay-owned.
"""
from collections.abc import Mapping
from copy import deepcopy

from simulator_primitive_generator_v1 import core as primitive_contract
from simulator_primitive_generator_v1 import intelligence, scoring
from simulator_recursive_state_replay_v1 import core as replay
from simulator_recursive_state_replay_v1.io import contract as replay_contract
from . import historical_game_paired_residual_sampler_v1 as paired
from . import seeded_joint_residual_sampler_v1 as seeded

HARNESS_VERSION = 'SINGLE_GAME_SIMULATION_HARNESS_V1_001'
CENTER_MAPPING = {
    'expected_plays': ('volume', 'offensive_plays'),
    'expected_pass_rate': ('volume', 'pass_rate'),
    'expected_pass_ypa': ('yardage', 'passing_yards_per_attempt'),
    'expected_rush_ypc': ('yardage', 'rushing_yards_per_carry'),
    'expected_passing_tds': ('scoring', 'passing_tds'),
    'expected_rushing_tds': ('scoring', 'rushing_tds'),
    'expected_points': ('scoring', 'points'),
}
PRIMITIVES = {'pass_attempts', 'carries', 'passing_yards', 'passing_tds',
              'rushing_yards', 'rushing_tds'}
PAIR_INTEGRITY = dict(historical_same_game=True, historical_reciprocal_opponents=True,
                      joint_vectors_preserved=True)
SAMPLER_AUTHORIZATION = dict(monte_carlo=False, production_influence='NONE',
                             fanduel_solver_influence='NONE')


class HarnessError(ValueError):
    """Invalid integration inputs or outputs; no replacement is authorized."""


def require(condition, reason):
    if not condition:
        raise HarnessError(reason)


def text(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def same(left, right):
    # JSON equality distinguishes booleans from numeric values and rejects NaN.
    return primitive_contract.canonical_json(left) == primitive_contract.canonical_json(right)


def timestamp(value, label):
    result = primitive_contract._parse_utc(value, label)
    require(result is not None, label + ' required')
    return result


def _validate_input(data):
    require(isinstance(data, Mapping), 'input must be an object')
    require(set(data) == {'game_identity', 'simulation_context', 'team_1_generator_input',
                          'team_2_generator_input', 'seed'}, 'exact single-game input fields required')
    require(type(data['seed']) is int, 'explicit integer seed required')
    game, context = data['game_identity'], data['simulation_context']
    require(isinstance(game, Mapping), 'game_identity required')
    required = {'season', 'week', 'game_id', 'team_1', 'team_2'}
    require(required <= set(game) <= required | {'home_team', 'away_team'}, 'invalid game identity fields')
    for field in ('season', 'week'):
        require(type(game[field]) is int and game[field] > 0, field + ' must be positive integer')
    for field in ('game_id', 'team_1', 'team_2'):
        require(text(game[field]), field + ' must be nonempty canonical identity')
    require(game['team_1'] != game['team_2'], 'future teams must be distinct')
    if 'home_team' in game or 'away_team' in game:
        require({'home_team', 'away_team'} <= set(game), 'both explicit home/away identities required')
        require({game['home_team'], game['away_team']} == {game['team_1'], game['team_2']},
                'explicit home/away identity mismatch')
    require(isinstance(context, Mapping) and set(context) == {'kickoff', 'completion', 'simulated_at', 'cutoff'},
            'explicit simulation chronology required')
    times = {key: timestamp(context[key], key) for key in context}
    # Same SIMULATED chronology inequalities as frozen core.validate_primitive_result.
    # No seven-primitive result is fabricated to invoke that validator (points are absent).
    require(times['kickoff'] < times['completion'] <= times['simulated_at'] <= times['cutoff'],
            'SIMULATED chronology requires kickoff < completion <= simulated_at <= cutoff')
    contract = replay_contract()
    replay.validate_contract(contract)
    for name, opponent in (('team_1', 'team_2'), ('team_2', 'team_1')):
        state = data[name + '_generator_input']
        require(isinstance(state, Mapping), 'Generator input must be frozen Replay state')
        manifest = state['state_manifest']
        c = manifest['context']
        require(all(same(c[k], game[k]) for k in ('season', 'week', 'game_id')),
                'Generator state game identity mismatch')
        require(c['team'] == game[name] and c['opponent'] == game[opponent], 'Generator team identity mismatch')
        require(timestamp(c['kickoff'], 'state kickoff') == times['kickoff'], 'state kickoff mismatch')
        # Frozen Replay pregame rule, separate from the result's later availability cutoff.
        require(replay.time(c['evidence_cutoff']) < replay.time(c['kickoff']), 'LEAKAGE_CUTOFF_NOT_PREGAME')
        require(c['feature_contract_version'] == contract['version'] and
                manifest['feature_contract_hash'] == replay.digest(contract), 'Replay feature contract mismatch')
        require(state['transition_validation']['status'] == 'TRANSITION_COMPLETE' and
                state['transition_validation']['issues'] == [] and
                state['audit']['temporal_rules_satisfied'] is True, 'unvalidated Replay state')
        require(replay.digest({k:v for k,v in state.items() if k != 'audit'}) == state['audit']['result_hash'],
                'Replay state hash mismatch')
        features = state['feature_reconstruction']
        require(isinstance(features, list), 'feature_reconstruction must be a list')
        names = set()
        for row in features:
            require(isinstance(row, Mapping) and text(row.get('name')), 'invalid feature row')
            require(row['name'] not in names, 'duplicate feature name')
            names.add(row['name'])
            if row.get('value') is not None:
                seeded._number(row['value'], row['name'])
        # Required feature availability is checked by the actual Generator, not reimplemented here.


def _generate(state):
    volume = intelligence.estimate_team_volume(state)
    yardage = intelligence.estimate_team_yardage(state, volume)
    score = scoring.estimate_team_scoring(state, volume, yardage)
    output = dict(volume=volume, yardage=yardage, scoring=score)
    center = seeded._validate_center({key: output[stage][field]
                                     for key, (stage, field) in CENTER_MAPPING.items()})
    return output, center


class _OneShotSampler:
    """Invocation-local call budget; even a failed call consumes its only slot."""
    def __init__(self):
        self.calls = 0

    def call(self, center_1, center_2, seed):
        require(self.calls == 0, 'more than one stochastic call forbidden')
        self.calls += 1
        return paired.sample_paired_residual(center_1, center_2, seed=seed)


def _validate_sample(result, centers, seed):
    require(isinstance(result, Mapping), 'invalid paired output')
    require(set(result) == {'sampler_version', 'seed', 'residual_bank_sha256', 'selected_historical_game',
                            'orientation', 'team_1', 'team_2', 'pair_integrity', 'authorization'},
            'invalid paired output fields')
    require(result['sampler_version'] == paired.SAMPLER_VERSION, 'sampler version mismatch')
    require(type(result['seed']) is int and result['seed'] == seed, 'sampler seed mismatch')
    require(result['residual_bank_sha256'] == seeded.FROZEN_BANK_SHA256, 'wrong residual-bank hash')
    require(same(result['authorization'], SAMPLER_AUTHORIZATION), 'sampler authorization mismatch')
    require(same(result['pair_integrity'], PAIR_INTEGRITY), 'pair-integrity failure')
    orientation = result['orientation']
    require(type(orientation) is int and orientation in (0, 1), 'invalid orientation')
    # Read-only integrity validation; no sampling or RNG in this verification.
    rows, _ = seeded._load_bank(seeded.BANK_PATH)
    population = paired._paired_population(rows)
    game = result['selected_historical_game']
    selected = [pair for pair in population if pair[0]['game_id'] == game['game_id']]
    require(len(selected) == 1, 'unknown selected historical game')
    a, b = selected[0]
    require(same(game, dict(season=a['season'], week=a['week'], game_id=a['game_id'],
                           historical_team_a=a['team'], historical_team_b=b['team'])),
            'selected historical game mismatch')
    assigned = (a, b) if orientation == 0 else (b, a)
    for name, row, center in zip(('team_1', 'team_2'), assigned, centers):
        team = result[name]
        require(set(team) == {'center', 'selected_historical_residual_identity', 'residuals',
                              'realized_mechanisms', 'realized_primitives', 'reconciliation'}, 'invalid team output')
        require(same(team['center'], center), 'sampler center changed')
        identity = {k: row[k] for k in (*seeded.IDENTITY_FIELDS, 'row_index')}
        require(same(team['selected_historical_residual_identity'], identity), 'historical identity/orientation mismatch')
        require(same(team['residuals'], {k: row[k] for k in seeded.RESIDUAL_FIELDS}), 'joint vector integrity failure')
        m, p = team['realized_mechanisms'], team['realized_primitives']
        require(set(p) == PRIMITIVES, 'exact six simulated primitives required')
        require(all(type(v) is int and v >= 0 for v in p.values()), 'negative/noninteger simulated primitive')
        require(type(m['plays']) is int and m['plays'] >= 0 and
                p['pass_attempts'] + p['carries'] == m['plays'], 'post-simulation volume invariant failure')
        # Reuse frozen reconciliation as validation, never reimplement or repair it.
        expected = seeded._realize(center, row)
        require(same([m, p, team['reconciliation']], list(expected)), 'reconciliation/mechanism integrity failure')


def simulate_single_game(data):
    """Return one shadow game; data is the exact five-field input documented above.

    Generator and sampler errors propagate unchanged. Invalid integration structure
    raises HarnessError. The input and returned Generator outputs are never edited.
    """
    try:
        _validate_input(data)
        snapshot = deepcopy(data)
        output_1, center_1 = _generate(snapshot['team_1_generator_input'])
        output_2, center_2 = _generate(snapshot['team_2_generator_input'])
        generator_outputs = deepcopy((output_1, output_2))
        centers = deepcopy((center_1, center_2))
        gate = _OneShotSampler()
        sample = gate.call(center_1, center_2, snapshot['seed'])
        require(gate.calls == 1, 'exactly one stochastic call required')
        _validate_sample(sample, centers, snapshot['seed'])
        teams = {}
        for name, output, center in zip(('team_1', 'team_2'), generator_outputs, centers):
            team = deepcopy(sample[name])
            del team['center']
            teams[name] = dict(identity=snapshot['game_identity'][name], generator_output=output,
                               sampler_center=center, **team)
        return dict(harness_version=HARNESS_VERSION, game_identity=deepcopy(snapshot['game_identity']),
                    simulation_context=deepcopy(snapshot['simulation_context']), seed=snapshot['seed'],
                    generator_authority=dict(package='simulator_primitive_generator_v1', status='VALIDATED_FROZEN'),
                    paired_sampler_authority=dict(sampler_version=sample['sampler_version'],
                                                  residual_bank_sha256=sample['residual_bank_sha256']),
                    center_mapping={k: '.'.join(v) for k,v in CENTER_MAPPING.items()}, **teams,
                    selected_historical_game=deepcopy(sample['selected_historical_game']), orientation=sample['orientation'],
                    pair_integrity=deepcopy(sample['pair_integrity']),
                    invariants=dict(team_1_volume_conservation=True, team_2_volume_conservation=True,
                                    nonnegative_opportunities=True, nonnegative_yards=True, nonnegative_tds=True),
                    authorization=dict(stochastic_calls=gate.calls, monte_carlo=False, repeated_game_simulation=False,
                                       production_influence='NONE', fanduel_solver_influence='NONE'),
                    historical_temporal_provenance='UNVERIFIED')
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise HarnessError(f'invalid single-game structure: {exc}') from exc
