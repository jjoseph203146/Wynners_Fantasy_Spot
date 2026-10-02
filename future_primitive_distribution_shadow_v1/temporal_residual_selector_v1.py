"""Shadow-only temporal selection; no realization or simulation."""
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import io
from pathlib import Path
import random
import re

from . import historical_game_paired_residual_sampler_v1 as paired

SELECTOR_VERSION = 'TEMPORAL_RESIDUAL_SELECTOR_V1_001'
TRAINING_PATH = Path(__file__).resolve().parents[1] / 'processed/forecast_v1_team_game_training.csv'
BANK_PATH = paired.BANK_PATH
SelectorError = paired.SamplerError


def _kickoff(value):
    # Source has no timezone. Require the same naive wall-clock convention;
    # never silently interpret an offset or invent a timezone authority.
    if not isinstance(value, str) or not re.fullmatch(
            r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?', value):
        raise SelectorError('explicit naive kickoff YYYY-MM-DDTHH:MM[:SS] required')
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise SelectorError('invalid kickoff') from exc


def _validate_chronology(rows, pairs):
    if len(rows) != 1710:
        raise SelectorError('exactly 1710 training rows required')
    games = defaultdict(list)
    required = ('game_id', 'season', 'week', 'team', 'opponent_team', 'game_date', 'gametime')
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
            raise SelectorError('missing training identity/date/time')
        games[row['game_id']].append(row)
    if len(games) != 855:
        raise SelectorError('exactly 855 training games required')
    chronology = {}
    identities = {}
    for game_id, group in games.items():
        if len(group) != 2:
            raise SelectorError('exactly two training rows per game required')
        a, b = sorted(group, key=lambda r: r['team'])
        if a['team'] == b['team'] or a['opponent_team'] != b['team'] or b['opponent_team'] != a['team']:
            raise SelectorError('ambiguous training team identity')
        if (a['season'], a['week']) != (b['season'], b['week']):
            raise SelectorError('ambiguous training season/week')
        ka = _kickoff(a['game_date'] + 'T' + a['gametime'])
        kb = _kickoff(b['game_date'] + 'T' + b['gametime'])
        if ka != kb:
            raise SelectorError('ambiguous game kickoff')
        identity = (ka, a['team'], b['team'])
        if identity in identities:
            raise SelectorError('multiple game IDs for same matchup/kickoff')
        identities[identity] = game_id
        chronology[game_id] = ka
    for pair in pairs:
        game_id = pair[0]['game_id']
        if game_id not in chronology:
            raise SelectorError('residual chronology join incomplete')
        training = {r['team']: r for r in games[game_id]}
        for row in pair:
            match = training.get(row['team'])
            if match is None or any(str(row[k]) != match[k] for k in ('season', 'week', 'opponent_team')):
                raise SelectorError('residual/training game identity mismatch')
    return chronology


def _load_authorities(training_path=TRAINING_PATH, bank_path=BANK_PATH):
    rows, bank_digest = paired.frozen._load_bank(bank_path)
    pairs = paired._paired_population(rows)
    data = Path(training_path).read_bytes()
    try:
        reader = csv.DictReader(io.StringIO(data.decode('utf-8')), strict=True)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise SelectorError('ambiguous training schema')
        training = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in training):
            raise SelectorError('malformed training row')
        chronology = _validate_chronology(training, pairs)
    except (UnicodeError, csv.Error) as exc:
        raise SelectorError('invalid training CSV') from exc
    audit = dict(training_rows=len(training), training_unique_games=len(chronology),
                 training_two_rows_per_game=True, kickoff_parse=True,
                 residual_chronology_join=True, residual_paired_games=len(pairs),
                 training_sha256=hashlib.sha256(data).hexdigest(),
                 residual_bank_sha256=bank_digest,
                 kickoff_convention='naive source game_date + gametime; no timezone inferred')
    return pairs, chronology, audit


def select_temporal_paired_residual(target_game_id=None, target_kickoff=None, seed=None,
                                   *, training_path=TRAINING_PATH, bank_path=BANK_PATH):
    """Validate all authority rows before two local draws from strictly earlier pairs.

    Target must exist in training and its explicit kickoff must match that authority.
    Sorted game_id/team order and orientation 0=A/B, 1=B/A match the frozen sampler.
    Errors propagate without retries; empty populations consume zero draws.
    """
    if not isinstance(target_game_id, str) or not target_game_id.strip():
        raise SelectorError('explicit target_game_id required')
    target = _kickoff(target_kickoff)
    if type(seed) is not int:
        raise SelectorError('explicit integer seed required (bool invalid)')
    pairs, chronology, audit = _load_authorities(training_path, bank_path)
    if target_game_id not in chronology:
        raise SelectorError('target game absent from training chronology')
    if chronology[target_game_id] != target:
        raise SelectorError('target kickoff disagrees with training chronology')
    eligible = tuple(p for p in pairs if chronology[p[0]['game_id']] < target)
    if not eligible:
        raise SelectorError('no eligible historical residual games')
    if any(p[0]['game_id'] == target_game_id or chronology[p[0]['game_id']] >= target for p in eligible):
        raise SelectorError('temporal population invariant failed')
    rng = random.Random(seed)
    a, b = eligible[int(rng.random() * len(eligible))]
    orientation = int(rng.random() * 2)
    row_1, row_2 = (a, b) if orientation == 0 else (b, a)
    result = dict(
        selector_version=SELECTOR_VERSION, target_game_id=target_game_id,
        target_kickoff=target.isoformat(), seed=seed, eligible_game_count=len(eligible),
        selected_historical_game=dict(game_id=a['game_id'], season=a['season'], week=a['week'],
                                      historical_team_a=a['team'], historical_team_b=b['team']),
        selected_historical_kickoff=chronology[a['game_id']].isoformat(), orientation=orientation,
        temporal_integrity=dict(selected_strictly_before_target=True, target_game_excluded=True,
                                same_kickoff_excluded=True, later_games_excluded=True),
        temporal_audit=dict(audit, rule='historical_kickoff < target_kickoff',
                            same_kickoff_game_count=sum(chronology[p[0]['game_id']] == target for p in pairs),
                            later_game_count=sum(chronology[p[0]['game_id']] > target for p in pairs),
                            rng_draws=2),
        pair_integrity=dict(historical_same_game=True, historical_reciprocal_opponents=True,
                            joint_vectors_preserved=True),
        authorization=dict(monte_carlo=False, repeated_game_simulation=False,
                           production_influence='NONE', fanduel_solver_influence='NONE'),
        historical_temporal_provenance='UNVERIFIED')
    for index, row in ((1, row_1), (2, row_2)):
        result[f'team_{index}_residual_identity'] = {k: row[k] for k in (*paired.frozen.IDENTITY_FIELDS, 'row_index')}
        result[f'team_{index}_residuals'] = {k: row[k] for k in paired.frozen.RESIDUAL_FIELDS}
    return result
