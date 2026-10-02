"""One historical opponent pair, two local draws; shadow-only single-call API."""
from collections import defaultdict
import random

from . import seeded_joint_residual_sampler_v1 as frozen

SAMPLER_VERSION = 'HISTORICAL_GAME_PAIRED_RESIDUAL_SAMPLER_V1_001'
FROZEN_BANK_SHA256 = frozen.FROZEN_BANK_SHA256
BANK_PATH = frozen.BANK_PATH
PAIRED_GAME_COUNT = 839
SamplerError = frozen.SamplerError


def _paired_population(rows):
    """Validate every row; sort game_id then team, with no football interpretation."""
    games = defaultdict(list)
    for row in rows:
        games[row['game_id']].append(row)
    if len(games) != PAIRED_GAME_COUNT:
        raise SamplerError('exactly 839 historical games required')
    pairs = []
    for game_id in sorted(games):
        group = games[game_id]
        if len(group) != 2:
            raise SamplerError(f'{game_id}: exactly two rows required')
        a, b = sorted(group, key=lambda row: row['team'])
        if a['team'] == b['team']:
            raise SamplerError(f'{game_id}: distinct teams required')
        if a['opponent_team'] != b['team'] or b['opponent_team'] != a['team']:
            raise SamplerError(f'{game_id}: reciprocal opponents required')
        if (a['season'], a['week']) != (b['season'], b['week']):
            raise SamplerError(f'{game_id}: inconsistent season/week')
        for row in (a, b):
            for field in frozen.RESIDUAL_FIELDS:
                frozen._number(row.get(field), field)
        pairs.append((a, b))
    return tuple(pairs)


def _team_result(center, row):
    mechanisms, primitives, audit = frozen._realize(center, row)
    return dict(center=center,
                selected_historical_residual_identity={
                    key: row[key] for key in (*frozen.IDENTITY_FIELDS, 'row_index')},
                residuals={key: row[key] for key in frozen.RESIDUAL_FIELDS},
                realized_mechanisms=mechanisms, realized_primitives=primitives,
                reconciliation=audit)


def sample_paired_residual(center_team_1, center_team_2, *, seed=None, bank_path=BANK_PATH):
    """Apply one intact historical game pair to two validated deterministic centers.

    Draw 1 selects floor(random()*839); draw 2 selects floor(random()*2).
    Orientation 0 maps A/B to team_1/team_2; orientation 1 maps B/A.
    expected_points is retained only in center metadata. No retries or batches.
    """
    if type(seed) is not int:
        raise SamplerError('explicit integer seed required (bool is invalid)')
    center_1 = frozen._validate_center(center_team_1)
    center_2 = frozen._validate_center(center_team_2)
    rows, digest = frozen._load_bank(bank_path)
    pairs = _paired_population(rows)
    rng = random.Random(seed)
    a, b = pairs[int(rng.random() * len(pairs))]
    orientation = int(rng.random() * 2)
    row_1, row_2 = (a, b) if orientation == 0 else (b, a)
    return dict(
        sampler_version=SAMPLER_VERSION, seed=seed, residual_bank_sha256=digest,
        selected_historical_game=dict(season=a['season'], week=a['week'],
                                      game_id=a['game_id'], historical_team_a=a['team'],
                                      historical_team_b=b['team']),
        orientation=orientation,
        team_1=_team_result(center_1, row_1), team_2=_team_result(center_2, row_2),
        pair_integrity=dict(historical_same_game=True,
                            historical_reciprocal_opponents=True,
                            joint_vectors_preserved=True),
        authorization=dict(monte_carlo=False, production_influence='NONE',
                           fanduel_solver_influence='NONE'))
