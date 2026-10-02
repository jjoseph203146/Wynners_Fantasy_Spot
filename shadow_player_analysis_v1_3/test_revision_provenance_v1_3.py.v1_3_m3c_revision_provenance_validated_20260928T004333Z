"""Offline M3C positive cases and differential compatibility with frozen M3B."""
import copy
import itertools
import unittest

from . import claims
from .revision_provenance import SAFETY, build_revision_provenance
from . import test_claim_lifecycle_v1_3 as lifecycle


def event(rid='a', hour=14, content=None, **changes):
    time = f'2026-09-27T{hour:02d}:00:00Z'
    row = lifecycle.ClaimLifecycleV13Tests().event(
        record_id=rid, event_key='family', content_hash=content or 'content-' + rid,
        published_at_utc=time, updated_at_utc=time,
        retrieved_at_utc=time, first_seen_at_utc=time,
    )
    row.update(changes)
    return row


def build(events, cutoff='2026-09-27T16:00:00Z'):
    return build_revision_provenance(source_events=events, cutoff_utc=cutoff)


def records(result):
    return {r['record_id']: r for r in result['revision_provenance']}


def current(result):
    return {r['record_id'] for r in result['revision_provenance'] if r['current_as_of_cutoff']}


def m3b_current(events, cutoff):
    fixture = lifecycle.ClaimLifecycleV13Tests()
    ids = {r['record_id'] for r in events}
    result = claims.build_claims(
        source_events=events,
        classified_evidence=[fixture.evidence(record_id=rid) for rid in ids],
        entity_links=[fixture.player_link(record_id=rid) for rid in ids],
        cutoff_utc=cutoff,
    )
    return {r['record_id'] for r in result['claims']}


class RevisionProvenanceTests(unittest.TestCase):
    def test_single(self):
        r = records(build([event()]))['a']
        self.assertEqual(r['provenance_status'], 'SINGLE_VERSION')
        self.assertTrue(r['current_as_of_cutoff'])
        self.assertIsNone(r['revision_position'])
        self.assertIsNone(r['supersedes_record_id'])
        self.assertIsNone(r['superseded_by_record_id'])

    def test_two_ordered(self):
        r = records(build([event(), event('b', 15)]))
        self.assertEqual(r['a']['superseded_by_record_id'], 'b')
        self.assertEqual(r['b']['supersedes_record_id'], 'a')
        self.assertEqual(r['b']['revision_position'], 2)
        self.assertEqual(r['a']['family_provenance_status'], 'ORDERED_REVISION_HISTORY')

    def test_three_ordered(self):
        r = records(build([event(), event('b', 15), event('c', 16)]))
        self.assertEqual([r[k]['revision_position'] for k in 'abc'], [1, 2, 3])
        self.assertEqual(r['b']['superseded_by_record_id'], 'c')
        self.assertEqual(r['c']['supersedes_record_id'], 'b')

    def test_historical_cutoff(self):
        result = build([event(), event('b', 18)])
        self.assertEqual(current(result), {'a'})
        self.assertEqual(records(result)['b']['provenance_status'], 'FUTURE_REVISION')

    def test_later_cutoff(self):
        self.assertEqual(current(build([event(), event('b', 18)], '2026-09-27T19:00:00Z')), {'b'})

    def test_three_cutoffs(self):
        events = [event(), event('b', 15), event('c', 16)]
        for hour, rid in [(14, 'a'), (15, 'b'), (16, 'c')]:
            self.assertEqual(current(build(events, f'2026-09-27T{hour}:30:00Z')), {rid})

    def test_exact_cutoff(self):
        self.assertEqual(current(build([event(), event('b', 16)])), {'b'})

    def test_identical_duplicate_collapses(self):
        a = event()
        self.assertEqual(build([a]), build([a, copy.deepcopy(a)]))

    def test_distinct_same_content_captures(self):
        result = build([event(), event('copy', 15, content='content-a')])
        self.assertEqual(result['event_families'][0]['distinct_content_versions'], 1)
        self.assertEqual(current(result), {'a', 'copy'})
        self.assertTrue(all(r['revision_position'] is None for r in result['revision_provenance']))

    def test_independent_families(self):
        other = event('other', 15, event_key='other', source_event_id='other')
        result = build([event(), event('b', 15), other])
        self.assertEqual(current(result), {'b', 'other'})
        self.assertIsNone(records(result)['other']['supersedes_record_id'])

    def test_all_permutations(self):
        events = [event(), event('b', 15), event('c', 16)]
        expected = build(events)
        for perm in itertools.permutations(events):
            self.assertEqual(build(perm), expected)

    def test_raw_fields_preserved_and_inputs_unchanged(self):
        events = [event(updated_at_utc=None), event('b', 15)]
        before = copy.deepcopy(events)
        r = records(build(events))['a']
        for field, value in events[0].items():
            if field in r:
                self.assertEqual(r[field], value)
        self.assertEqual(events, before)
        self.assertEqual(r['knowledge_at_utc'], '2026-09-27T14:00:00+00:00')

    def test_each_timestamp_controls_knowledge(self):
        for field in ('published_at_utc', 'updated_at_utc', 'retrieved_at_utc', 'first_seen_at_utc'):
            result = build([event(**{field: '2026-09-27T17:00:00Z'})])
            self.assertEqual(current(result), set())
            self.assertEqual(records(result)['a']['knowledge_at_utc'], '2026-09-27T17:00:00+00:00')

    def test_m3b_differential_chronologies(self):
        # Exhaustive, deterministic ties, earlier ties, future ties, and boundaries.
        for hours in itertools.product((14, 15, 16, 17), repeat=3):
            events = [event(rid, hour) for rid, hour in zip('abc', hours)]
            for hour in (14, 15, 16, 17):
                cutoff = f'2026-09-27T{hour}:00:00Z'
                self.assertEqual(current(build(events, cutoff)), m3b_current(events, cutoff))

    def test_empty(self):
        self.assertEqual(build([]), {'revision_provenance': [], 'event_families': [], 'safety': SAFETY})


if __name__ == '__main__':
    unittest.main()
