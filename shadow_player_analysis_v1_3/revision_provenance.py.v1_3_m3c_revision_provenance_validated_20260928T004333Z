"""M3C candidate: pure audit metadata; never an eligibility input.

Knowledge uses the frozen claims.timestamp parser and M3A max(publication,
update-or-publication, retrieval, first-seen). Currentness is the M3B *record*
rule, not claim eligibility: for multi-content families the unique latest
known record wins, unless any capture has invalid chronology. Single-content
families have no M3B revision gate. Differential tests pin this compatibility.

A full chain requires strictly disjoint knowledge-time ranges for distinct
content versions. Repeated captures are retained, but share a revision position.
Overlapping ranges cannot prove a single linear history. Record links point
only to a unique earliest capture of the adjacent content version; ties among
same-content captures never select a representative lexically.

Chain links describe the captured history, including future versions; they do
not mean supersession was effective at cutoff. known_as_of_cutoff and
current_as_of_cutoff are separate. Earlier/future ties can invalidate the full
chain without invalidating an independently proven M3B current record.

Identifiers are opaque nonblank, trimmed strings (synthetic fixtures allowed).
Malformed input containers raise ValueError. Invalid identities quarantine the
family. No I/O, persistence, publication, or production integration exists.
"""

from copy import deepcopy

from .claims import SAFETY as CLAIM_SAFETY, canonical, timestamp

SAFETY = dict(CLAIM_SAFETY, CONSENSUS_ENABLED=False)
FIELDS = (
    'event_key', 'record_id', 'source', 'source_event_id', 'content_hash',
    'published_at_utc', 'updated_at_utc', 'retrieved_at_utc', 'first_seen_at_utc',
)


def _identifier(value):
    return (isinstance(value, str) and bool(value.strip())
            and value == value.strip() and not any(ord(c) < 32 for c in value))


def _knowledge(row):
    published = row.get('published_at_utc')
    return max(timestamp(published),
               timestamp(row.get('updated_at_utc') or published),
               timestamp(row.get('retrieved_at_utc')),
               timestamp(row.get('first_seen_at_utc')))


def _current_records(rows, times, cutoff, multi):
    """Frozen M3B record-set semantics, including same-content capture ties.

    Deliberately independent of full-chain proof; never used by claims.py.
    """
    if multi and any(t is None for t in times):
        return set()
    known = [(t, r['record_id']) for r, t in zip(rows, times)
             if t is not None and t <= cutoff]
    if not multi:
        return {rid for _, rid in known}
    if not known:
        return set()
    latest = max(t for t, _ in known)
    records = {rid for t, rid in known if t == latest}
    return records if len(records) == 1 else set()


def build_revision_provenance(*, source_events, cutoff_utc):
    """Return deterministic capture records and content-version family summaries.

    Accept a list/tuple of JSON-compatible normalized event dictionaries.
    Exact duplicate rows collapse. Raw provenance fields are preserved by copy.
    Invalid cutoff raises ValueError using the existing timestamp parser.
    """
    cutoff = timestamp(cutoff_utc)
    if not isinstance(source_events, (list, tuple)) or any(
            not isinstance(row, dict) for row in source_events):
        raise ValueError('NORMALIZED_SOURCE_EVENTS_REQUIRED')
    try:
        unique = {canonical(r): deepcopy(r) for r in source_events}
    except (TypeError, ValueError) as exc:
        raise ValueError('JSON_SOURCE_EVENTS_REQUIRED') from exc
    families = {}
    record_rows = {}
    for row in unique.values():
        key = canonical(row.get('event_key'))
        families.setdefault(key, []).append(row)
        record_rows.setdefault(canonical(row.get('record_id')), []).append(row)
    output, summaries = [], []
    for rows in families.values():
        reasons = set()
        for row in rows:
            for field in ('event_key', 'record_id', 'source', 'source_event_id', 'content_hash'):
                if not _identifier(row.get(field)):
                    reasons.add('INVALID_' + field.upper())
            if len(record_rows[canonical(row.get('record_id'))]) > 1:
                reasons.add('CONFLICTING_RECORD_ID')
        if len({canonical([r.get('source'), r.get('source_event_id')]) for r in rows}) > 1:
            reasons.add('EVENT_FAMILY_IDENTITY_CONFLICT')
        identity_invalid = bool(reasons)
        times = []
        for row in rows:
            try:
                times.append(_knowledge(row))
            except (ValueError, TypeError, OverflowError):
                times.append(None)
                reasons.add('INVALID_REVISION_CHRONOLOGY')
        versions = {}
        for row, time in zip(rows, times):
            versions.setdefault(canonical(row.get('content_hash')), []).append((row, time))
        multi = len(versions) > 1
        current = set() if identity_invalid else _current_records(rows, times, cutoff, multi)
        ordered = []
        if not reasons:
            # Only timestamps decide order. Tied distinct versions fail below.
            ordered = sorted(versions.values(), key=lambda v: min(t for _, t in v))
            for left, right in zip(ordered, ordered[1:]):
                if max(t for _, t in left) >= min(t for _, t in right):
                    reasons.add('OVERLAPPING_OR_EQUAL_REVISION_KNOWLEDGE')
        status = ('AMBIGUOUS_REVISION_ORDER' if reasons else
                  'ORDERED_REVISION_HISTORY' if multi else 'SINGLE_VERSION')
        positions, representatives = {}, []
        if status == 'ORDERED_REVISION_HISTORY':
            for i, version in enumerate(ordered):
                positions[canonical(version[0][0]['content_hash'])] = i
                first = min(t for _, t in version)
                ids = {r['record_id'] for r, t in version if t == first}
                representatives.append(next(iter(ids)) if len(ids) == 1 else None)
        known = [(t, r.get('record_id')) for r, t in zip(rows, times)
                 if t is not None and t <= cutoff]
        if multi and known and not current and not identity_invalid and all(t is not None for t in times):
            reasons.add('M3B_LATEST_KNOWN_RECORD_TIE')
        for row, time in zip(rows, times):
            future = time is not None and time > cutoff
            row_reasons = set(reasons)
            if future:
                row_reasons.add('NOT_KNOWN_AS_OF_CUTOFF')
            index = positions.get(canonical(row.get('content_hash')))
            previous = following = None
            if index is not None:
                if index:
                    previous = representatives[index - 1]
                    if previous is None:
                        row_reasons.add('NON_UNIQUE_ADJACENT_CAPTURE')
                if index + 1 < len(representatives):
                    following = representatives[index + 1]
                    if following is None:
                        row_reasons.add('NON_UNIQUE_ADJACENT_CAPTURE')
            output.append(dict(
                {f: deepcopy(row.get(f)) for f in FIELDS},
                knowledge_at_utc=time.isoformat() if time is not None else None,
                cutoff_utc=cutoff.isoformat(), family_provenance_status=status,
                provenance_status='FUTURE_REVISION' if future else status,
                known_as_of_cutoff=time <= cutoff if time is not None else None,
                current_as_of_cutoff=(not identity_invalid and time is not None
                                     and time <= cutoff and row.get('record_id') in current),
                supersedes_record_id=previous, superseded_by_record_id=following,
                revision_position=index + 1 if index is not None else None,
                reason_codes=sorted(row_reasons), safety=dict(SAFETY),
            ))
        summaries.append(dict(
            event_key=deepcopy(rows[0].get('event_key')), provenance_status=status,
            distinct_content_versions=len(versions), capture_count=len(rows),
            cutoff_utc=cutoff.isoformat(), current_record_ids=sorted(current),
            reason_codes=sorted(reasons), safety=dict(SAFETY),
        ))
    # Presentation only: chronology and all relationships have been decided.
    return dict(revision_provenance=sorted(output, key=canonical),
                event_families=sorted(summaries, key=canonical), safety=dict(SAFETY))
