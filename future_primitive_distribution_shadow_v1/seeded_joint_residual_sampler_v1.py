"""Single empirical joint-vector draw; shadow only, no batch or integration API."""
from collections.abc import Mapping
import csv
import hashlib
import io
import math
from pathlib import Path
import random

SAMPLER_VERSION = 'SEEDED_JOINT_RESIDUAL_SAMPLER_V1_001'
FROZEN_BANK_SHA256 = '614b079e45c4cda7c09c58b1f6820825f221f6a40596d356c26d93d199064c6b'
BANK_PATH = Path(__file__).with_name('RESIDUAL_BANK_V1.csv')
MECHANISMS = ('plays', 'pass_rate', 'pass_ypa', 'rush_ypc', 'passing_tds', 'rushing_tds')
RESIDUAL_FIELDS = tuple('resid_' + key for key in MECHANISMS)
CENTER_FIELDS = tuple('expected_' + key for key in (*MECHANISMS, 'points'))
IDENTITY_FIELDS = ('season', 'week', 'game_id', 'team', 'opponent_team')
BANK_SCHEMA = (*IDENTITY_FIELDS, 'team_history_games', *CENTER_FIELDS,
               'actual_plays', 'actual_pass_rate', 'actual_pass_ypa', 'actual_rush_ypc',
               'passing_tds', 'rushing_tds', *RESIDUAL_FIELDS)


class SamplerError(ValueError):
    """Invalid sampler inputs or output; never substituted with defaults."""


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SamplerError(f'{field}: numeric value required')
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise SamplerError(f'{field}: finite value required')
    return value


def _parse_bank(data):
    """Structural validation separate from hash gate for focused corruption tests."""
    try:
        reader = csv.DictReader(io.StringIO(data.decode('utf-8')), strict=True)
        if tuple(reader.fieldnames or ()) != BANK_SCHEMA:
            raise SamplerError('bank schema mismatch')
        rows, seen = [], set()
        for index, raw in enumerate(reader):
            if set(raw) != set(BANK_SCHEMA) or any(v is None or not v.strip() for v in raw.values()):
                raise SamplerError('missing/extra bank field')
            row = {key: raw[key] for key in IDENTITY_FIELDS}
            for key in ('season', 'week'):
                row[key] = int(row[key])
                if row[key] <= 0:
                    raise SamplerError('invalid row identity')
            identity = tuple(row[key] for key in ('season', 'week', 'game_id', 'team'))
            if identity in seen:
                raise SamplerError('duplicate canonical residual-row identity')
            seen.add(identity)
            for key in RESIDUAL_FIELDS:
                row[key] = _number(float(raw[key]), key)
                if key in ('resid_passing_tds', 'resid_rushing_tds') and not row[key].is_integer():
                    raise SamplerError('TD residual must be integer-valued')
            row['row_index'] = index
            rows.append(row)
        if not rows:
            raise SamplerError('empty bank')
        return rows
    except (UnicodeError, csv.Error, ValueError, TypeError) as exc:
        raise SamplerError(f'invalid residual bank: {exc}') from exc


def _load_bank(path):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != FROZEN_BANK_SHA256:
        raise SamplerError('residual-bank hash mismatch')
    return _parse_bank(data), digest


def _validate_center(center):
    if not isinstance(center, Mapping):
        raise SamplerError('center must be a mapping')
    result = {}
    for key in CENTER_FIELDS:
        value = _number(center.get(key), key)
        if value < 0 or (key == 'expected_pass_rate' and value > 1):
            raise SamplerError(f'{key}: invalid center domain')
        if key in ('expected_passing_tds', 'expected_rushing_tds') and value != int(value):
            raise SamplerError(f'{key}: deterministic integer TD required')
        result[key] = value
    return result


def _realize(center, row):
    audit = []

    def changed(field, operation, raw, value):
        if value != raw:
            audit.append(dict(field=field, operation=operation, raw=raw, reconciled=value))
        return value

    def add(key):
        return _number(center['expected_' + key] + row['resid_' + key], key)

    def integer(field, value):
        _number(value, field)
        # Half-up for nonnegative quantities, avoiding an addition overflow.
        base = math.floor(value)
        return changed(field, 'round_half_up', value, base + int(value - base >= 0.5))

    raw_plays = add('plays')
    if raw_plays < 0:
        raise SamplerError('negative sampled plays; no replacement authorized')
    plays = integer('plays', raw_plays)
    raw_rate = add('pass_rate')
    rate = changed('pass_rate', 'clamp_0_1', raw_rate, min(1, max(0, raw_rate)))
    attempts = integer('pass_attempts', plays * rate)
    carries = plays - attempts
    efficiencies = {}
    for key in ('pass_ypa', 'rush_ypc'):
        raw = add(key)
        efficiencies[key] = changed(key, 'floor_zero', raw, max(0, raw))
    primitives = dict(pass_attempts=attempts, carries=carries,
                      passing_yards=integer('passing_yards', attempts * efficiencies['pass_ypa']),
                      rushing_yards=integer('rushing_yards', carries * efficiencies['rush_ypc']))
    for key in ('passing_tds', 'rushing_tds'):
        raw = add(key)
        primitives[key] = integer(key, changed(key, 'floor_zero', raw, max(0, raw)))
    if (any(type(v) is not int or v < 0 for v in primitives.values())
            or attempts + carries != plays or not 0 <= rate <= 1
            or any(v < 0 or not math.isfinite(v) for v in efficiencies.values())):
        raise SamplerError('generated primitive invariant violation')
    return dict(plays=plays, pass_rate=rate, **efficiencies), primitives, audit


def sample_joint_residual(center, *, seed=None, bank_path=BANK_PATH):
    """Validate frozen bytes and draw exactly one complete row using a local RNG.

    center uses the seven expected_* names in CENTER_FIELDS. No input is mutated.
    The caller supplies one team's deterministic Generator V1 expectations.
    """
    if type(seed) is not int:
        raise SamplerError('explicit integer seed required (bool is invalid)')
    clean_center = _validate_center(center)
    rows, digest = _load_bank(bank_path)
    # One local Random.random() draw; no global state, no per-dimension draw.
    # random() seeded with an integer has Python's reproducibility guarantee.
    index = int(random.Random(seed).random() * len(rows))
    row = rows[index]
    mechanisms, primitives, audit = _realize(clean_center, row)
    return dict(sampler_version=SAMPLER_VERSION, seed=seed, residual_bank_sha256=digest,
                center=clean_center, selected_residual_row=dict(row),
                realized_mechanisms=mechanisms, realized_primitives=primitives,
                reconciliation=audit,
                authorization=dict(monte_carlo=False, production_influence='NONE',
                                   fanduel_solver_influence='NONE'))
