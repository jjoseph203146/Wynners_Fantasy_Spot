"""M5A candidate: explicit live shadow orchestration and immutable publication.

No I/O at import. Authority snapshots describe current read-only WFS rows;
cutoff is caller supplied and is not a claim of historical snapshot availability.
"""
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from datetime import datetime

ANALYSIS_ONLY = True
PRODUCTION_INFLUENCE = SOLVER_INFLUENCE = PROJECTION_MUTATION = False
FORECAST_MUTATION = AVAILABILITY_AUTHORITY = INJURY_AUTHORITY = False
DEPTH_CHART_AUTHORITY = AI_ANALYST_INFLUENCE = DATABASE_MUTATION = False
CONSENSUS_ENABLED = False
SAFETY = {name: globals()[name] for name in (
    'ANALYSIS_ONLY', 'PRODUCTION_INFLUENCE', 'SOLVER_INFLUENCE',
    'PROJECTION_MUTATION', 'FORECAST_MUTATION', 'AVAILABILITY_AUTHORITY',
    'INJURY_AUTHORITY', 'DEPTH_CHART_AUTHORITY', 'AI_ANALYST_INFLUENCE',
    'DATABASE_MUTATION', 'CONSENSUS_ENABLED')}
CONTRACT = 'PLAYER_ANALYSIS_V1_3_M5A_SHADOW'
ROOT = Path(__file__).parent
OUTPUT = ROOT / 'live_shadow_artifacts'
SOURCE = 'ESPN_NFL_RSS_V1'
ENDPOINT = 'https://www.espn.com/espn/rss/nfl/news'
FILES = frozenset(('source_capture.json', 'authorities.json', 'v1_2_result.json',
                   'handoff.json', 'validation.json'))
PROTECTED = {
    'claims.py': '6debb81489ee7b73dfa09c3a9bca9c728eb9b2b7855693a4ac530416e84297f8',
    'revision_provenance.py': '1a9ac584b1f3f026f1a21db4e541a7f00991052b7f347c65476c79a5324eead7',
    'corroboration.py': '47b4566e08651a09c2a09e44536dac9817603b6aa807f6b4f236cd2e1d40c735',
    'handoff.py': '3f62c4a591ca2b275e77b3062419e28e20ab0ba122506e604fa4d720efd9a037',
}


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _copy(value):
    return json.loads(canonical(value))


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _utc(value):
    _require(isinstance(value, str) and re.fullmatch(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)', value),
        'CANONICAL_UTC_REQUIRED')
    datetime.fromisoformat(value.replace('Z', '+00:00'))
    return value


def _context(season, week, cutoff):
    _require(type(season) is int and season > 0 and type(week) is int and week > 0,
             'EXPLICIT_POSITIVE_SEASON_WEEK_REQUIRED')
    _utc(cutoff)


def _rows(value, keys):
    _require(isinstance(value, dict), 'MALFORMED_ENVELOPE')
    for key in keys:
        _require(isinstance(value.get(key), list) and
                 all(isinstance(row, dict) for row in value[key]), 'MALFORMED_ROWS:' + key)


def _safety(value, consensus=True):
    _require(isinstance(value, dict), 'MISSING_SAFETY')
    for key, expected in SAFETY.items():
        if key != 'CONSENSUS_ENABLED' or consensus:
            _require(value.get(key) is expected, 'SAFETY_VIOLATION:' + key)


def _nested_safety(value):
    if isinstance(value, dict):
        if 'safety' in value:
            _safety(value['safety'], consensus=False)
        for key, item in value.items():
            if key in SAFETY:
                _require(item is SAFETY[key], 'SAFETY_VIOLATION:' + key)
            if key == 'consensus_eligible':
                _require(item is False, 'CONSENSUS_ELIGIBILITY')
            if key == 'consensus':
                _require(item == [], 'CONSENSUS_NOT_EMPTY')
            _nested_safety(item)
    elif isinstance(value, list):
        for item in value:
            _nested_safety(item)


def _capture(value):
    _rows(value, ('events', 'quarantine', 'claims', 'consensus'))
    _require(value['claims'] == [] and value['consensus'] == [], 'SOURCE_CLAIMS')
    _safety(value.get('safety'), consensus=False)
    retrieval = value.get('retrieval')
    _require(isinstance(retrieval, dict), 'MISSING_RETRIEVAL')
    _require(retrieval.get('source') == SOURCE and retrieval.get('endpoint') == ENDPOINT,
             'SOURCE_BOUNDARY')
    _utc(retrieval.get('retrieved_at_utc'))
    _require(isinstance(retrieval.get('feed_sha256'), str) and
             re.fullmatch('[0-9a-f]{64}', retrieval['feed_sha256']), 'FEED_SHA256')
    for row in value['events'] + value['quarantine']:
        _require(row.get('source') == SOURCE and row.get('origin_id') == 'ESPN',
                 'EVENT_SOURCE_BOUNDARY')
    _nested_safety(value)


def _authorities(value, season, week, cutoff):
    _require(isinstance(value, dict) and set(value) == {'identity', 'schedule'},
             'MALFORMED_AUTHORITIES')
    for key, collection, authority, fields in (
        ('identity', 'players', 'WFS_IDENTITY_SNAPSHOT',
         ('gsis_id', 'player_name', 'team', 'position')),
        ('schedule', 'games', 'WFS_SCHEDULE_SNAPSHOT', ('game_id', 'kickoff_at_utc')),
    ):
        snapshot = value[key]
        _rows(snapshot, (collection,))
        _require(snapshot.get('authority') == authority and snapshot[collection], 'AUTHORITY_CONTRACT')
        _require(snapshot.get('available_at_utc') == cutoff, 'AUTHORITY_CUTOFF')
        ids = []
        for row in snapshot[collection]:
            _require(all(isinstance(row.get(f), str) and row[f].strip() for f in fields),
                     'MALFORMED_AUTHORITY_ROW')
            ids.append(row[fields[0]])
        _require(len(ids) == len(set(ids)), 'DUPLICATE_AUTHORITY_ID')
    _require(value['identity'].get('season') == season and
             value['identity'].get('week') == week, 'IDENTITY_CONTEXT')
    # Delegate game/time contract to the frozen handoff validator.
    from .handoff import _validate_schedule, _utc as handoff_utc
    _validate_schedule(value['schedule'], handoff_utc(cutoff))


def _results(normalized, handoff):
    from .handoff import (_validate_v12, _claims_shape, _provenance_shape,
                          _corroboration_shape)
    _validate_v12(normalized)
    _require(isinstance(handoff, dict), 'MALFORMED_HANDOFF')
    _safety(handoff.get('safety'))
    _claims_shape(handoff.get('claims'))
    _provenance_shape(handoff.get('revision_provenance'))
    _corroboration_shape(handoff.get('corroboration'))
    _rows(handoff['claims'], ('claims',))
    _rows(handoff['revision_provenance'], ('revision_provenance', 'event_families'))
    _rows(handoff['corroboration'], ('groups', 'rejected'))
    _require(isinstance(handoff.get('input_audit'), dict), 'MISSING_HANDOFF_AUDIT')
    _require(handoff.get('quarantine') == normalized['quarantine'] and
             handoff.get('source_quality') == normalized['source_quality'], 'HANDOFF_AUDIT_MISMATCH')
    for claim in handoff['claims']['claims']:
        _require(claim.get('consensus_eligible') is False, 'CLAIM_CONSENSUS')
    _nested_safety(normalized)
    _nested_safety(handoff)


def protected_hashes():
    hashes = {name: _hash((ROOT / name).read_bytes()) for name in PROTECTED}
    _require(hashes == PROTECTED, 'PROTECTED_ENGINE_CHANGED')
    return hashes


def _payloads(*, season, week, cutoff_utc, capture, authorities, normalized, handoff):
    _context(season, week, cutoff_utc)
    _capture(capture)
    _authorities(authorities, season, week, cutoff_utc)
    _results(normalized, handoff)
    validation = dict(safety=dict(SAFETY), source_event_count=len(capture['events']),
                      source_quarantine_count=len(capture['quarantine']),
                      normalization_quarantine_count=len(normalized['quarantine']),
                      claim_count=len(handoff['claims']['claims']),
                      zero_claims_valid=True)
    objects = dict(zip(('source_capture.json', 'authorities.json', 'v1_2_result.json',
                        'handoff.json', 'validation.json'),
                       (capture, authorities, normalized, handoff, validation)))
    payloads = {name: canonical(value) for name, value in objects.items()}
    manifest = dict(contract=CONTRACT, schema_version='1.3.M5A', safety=dict(SAFETY),
                    season=season, week=week, cutoff_utc=cutoff_utc, source=SOURCE,
                    feed_sha256=capture['retrieval']['feed_sha256'],
                    identity_sha256=_hash(canonical(authorities['identity'])),
                    schedule_sha256=_hash(canonical(authorities['schedule'])),
                    protected_engine_sha256=protected_hashes(),
                    files={name: dict(sha256=_hash(data), bytes=len(data))
                           for name, data in payloads.items()})
    manifest['run_id'] = _hash(canonical(manifest))
    payloads['manifest.json'] = canonical(manifest)
    return payloads


def _directory(path):
    """Open every path component without following symlinks, retaining a dir fd."""
    path = Path(os.path.abspath(path))
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read(fd, name):
    child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    with os.fdopen(child, 'rb') as handle:
        _require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), 'ARTIFACT_TYPE')
        return handle.read()


def _verify_fd(fd, expected=None):
    _require(set(os.listdir(fd)) == FILES | {'manifest.json'}, 'ARTIFACT_SET')
    data = {name: _read(fd, name) for name in FILES | {'manifest.json'}}
    objects = {name: json.loads(raw) for name, raw in data.items()}
    _require(all(canonical(objects[n]) == raw for n, raw in data.items()), 'NONCANONICAL_JSON')
    manifest = objects['manifest.json']
    _require(isinstance(manifest, dict), 'MALFORMED_MANIFEST')
    run_id = manifest.get('run_id')
    unsigned = {k: v for k, v in manifest.items() if k != 'run_id'}
    _require(run_id == _hash(canonical(unsigned)), 'MANIFEST_RUN_ID')
    if expected is not None:
        _require(run_id == expected, 'DESTINATION_RUN_ID')
    rebuilt = _payloads(season=manifest.get('season'), week=manifest.get('week'),
                        cutoff_utc=manifest.get('cutoff_utc'), capture=objects['source_capture.json'],
                        authorities=objects['authorities.json'], normalized=objects['v1_2_result.json'],
                        handoff=objects['handoff.json'])
    _require(data == rebuilt, 'BUNDLE_HASH_OR_CONTRACT_MISMATCH')
    return manifest, data


def verify_bundle(path):
    """Verify exact canonical bytes, content address, contracts and payload hashes."""
    fd = _directory(path)
    try:
        return _verify_fd(fd, Path(path).name)[0]
    finally:
        os.close(fd)


def _rename_exclusive(fd, source, destination):
    # Linux renameat2(RENAME_NOREPLACE): unlike rename(), never replaces even an
    # empty preexisting directory. Fail closed if the platform lacks support.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(fd, source.encode(), fd, destination.encode(), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def _publish(payloads):
    run_id = json.loads(payloads['manifest.json'])['run_id']
    parent = _directory(OUTPUT.parent)
    try:
        try:
            os.mkdir(OUTPUT.name, mode=0o700, dir_fd=parent)
            os.fsync(parent)
        except FileExistsError:
            pass
        output = os.open(OUTPUT.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)
    stage = None
    staging = None
    try:
        def existing():
            fd = os.open(run_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=output)
            try:
                _, actual = _verify_fd(fd, run_id)
                _require(actual == payloads, 'IMMUTABLE_RUN_CONFLICT')
            finally:
                os.close(fd)
        try:
            existing()
            return OUTPUT / run_id
        except FileNotFoundError:
            # Only a missing destination is permissible, never a broken bundle.
            try:
                os.stat(run_id, dir_fd=output, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError('INCOMPLETE_EXISTING_BUNDLE')
        stage = '.stage-' + os.urandom(16).hex()
        os.mkdir(stage, mode=0o700, dir_fd=output)
        staging = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=output)
        for name, data in payloads.items():
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=staging)
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        _verify_fd(staging, run_id)
        os.fsync(staging)
        try:
            _rename_exclusive(output, stage, run_id)
            stage = None
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            existing()
        os.fsync(output)
        return OUTPUT / run_id
    finally:
        if staging is not None:
            if stage is not None:
                for name in os.listdir(staging):
                    os.unlink(name, dir_fd=staging)
            os.close(staging)
        if stage is not None:
            os.rmdir(stage, dir_fd=output)
        os.close(output)


def run_live_shadow(*, season, week, cutoff_utc, prior_events=(), db_path=None, publish=True):
    """Fetch once, invoke validated engines, optionally publish a detached packet.

    The installed V1.2 engine requires a source-keyed approval registry and its
    existing ESPN bridge supplies raw_publication_timestamp. Only accepted
    capture events enter that bridge; source quarantine remains audit material.
    No clock replacement or temporal/identity/game repair is performed here.
    """
    _context(season, week, cutoff_utc)
    _require(type(publish) is bool, 'BOOLEAN_PUBLISH_REQUIRED')
    _require(isinstance(prior_events, (tuple, list)) and
             all(isinstance(row, dict) for row in prior_events), 'MALFORMED_PRIOR_EVENTS')
    protected_hashes()
    # Lazy imports keep even database-adjacent dependency imports out of import.
    from shadow_player_analysis_v1.espn_nfl_rss_v1 import fetch
    from shadow_player_analysis_v1_2.live_authority import build_live_authorities
    from shadow_player_analysis_v1_2.adapters import espn_records
    from shadow_player_analysis_v1_2.core import build
    from .handoff import build_shadow_handoff, _validate_v12

    capture = _copy(fetch(prior_events=_copy(prior_events)))
    _capture(capture)
    kwargs = dict(season=season, week=week, as_of_utc=cutoff_utc)
    if db_path is not None:
        kwargs['db_path'] = db_path
    authorities = _copy(build_live_authorities(**kwargs))
    _authorities(authorities, season, week, cutoff_utc)
    records = espn_records({'events': _copy(capture['events'])})
    normalized = _copy(build(records, _copy(authorities['identity']),
                             _copy(authorities['schedule']), cutoff_utc,
                             {SOURCE: ['ESPN_RSS_SHADOW_CAPTURE_ONLY']}))
    _validate_v12(normalized)
    packet = _copy(build_shadow_handoff(v1_2_result=_copy(normalized),
                                       cutoff_utc=cutoff_utc,
                                       schedule=_copy(authorities['schedule'])))
    payloads = _payloads(season=season, week=week, cutoff_utc=cutoff_utc,
                         capture=capture, authorities=authorities,
                         normalized=normalized, handoff=packet)
    manifest = json.loads(payloads['manifest.json'])
    path = _publish(payloads) if publish else None
    return dict(source_capture=capture, authorities=authorities, v1_2_result=normalized,
                handoff=packet, validation=json.loads(payloads['validation.json']),
                manifest=manifest, run_id=manifest['run_id'],
                safety=dict(SAFETY), bundle_path=str(path) if path is not None else None)
