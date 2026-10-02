"""M3C fail-closed, permutation, compatibility, and no-side-effect tests."""
import copy
import itertools
import random
import subprocess
import sys
import unittest

from . import test_revision_provenance_v1_3 as fixtures

build, event, records, current = fixtures.build, fixtures.event, fixtures.records, fixtures.current


class RevisionProvenanceAdversarialTests(unittest.TestCase):
    def assert_ambiguous(self, events):
        result = build(events)
        self.assertTrue(all(f['provenance_status'] == 'AMBIGUOUS_REVISION_ORDER'
                            for f in result['event_families']))
        for r in result['revision_provenance']:
            self.assertIsNone(r['supersedes_record_id'])
            self.assertIsNone(r['superseded_by_record_id'])
            self.assertIsNone(r['revision_position'])
        self.assertEqual(result, build(list(reversed(events))))
        return result

    def test_equal_latest(self):
        self.assertEqual(current(self.assert_ambiguous([event(), event('b', 14)])), set())

    def test_equal_earlier_preserves_m3b_currentness(self):
        events = [event(), event('b', 14), event('c', 16)]
        result = self.assert_ambiguous(events)
        self.assertEqual(current(result), {'c'})
        self.assertEqual(current(result), fixtures.m3b_current(events, '2026-09-27T16:00:00Z'))

    def test_malformed_latest(self):
        self.assertEqual(current(self.assert_ambiguous([event(), event('b', 15, updated_at_utc='bad')])), set())

    def test_malformed_earlier(self):
        self.assertEqual(current(self.assert_ambiguous([event(retrieved_at_utc='bad'), event('b', 15)])), set())

    def test_missing_required_chronology(self):
        for field in ('published_at_utc', 'retrieved_at_utc', 'first_seen_at_utc'):
            row = event()
            del row[field]
            self.assertEqual(current(self.assert_ambiguous([row, event('b', 15)])), set())

    def test_optional_update_falls_back_like_m3b(self):
        for value in (None, '', False):
            events = [event(), event('b', 15, updated_at_utc=value)]
            self.assertEqual(current(build(events)), fixtures.m3b_current(events, '2026-09-27T16:00:00Z'))

    def test_blank_and_malformed_event_key(self):
        for value in ('', ' ', None, [], {}, 42, ' padded', 'bad\nkey'):
            self.assertEqual(current(self.assert_ambiguous([event(event_key=value)])), set())

    def test_blank_and_malformed_content_hash(self):
        for value in ('', ' ', None, [], {}, 42):
            self.assertEqual(current(self.assert_ambiguous([event(content_hash=value), event('b', 15)])), set())

    def test_similar_content_unrelated_families(self):
        result = build([event(), event('b', 15, content='content-a', event_key='other', source_event_id='other')])
        self.assertEqual(len(result['event_families']), 2)
        self.assertTrue(all(r['supersedes_record_id'] is None for r in result['revision_provenance']))

    def test_opposite_lexical_record_ids(self):
        r = records(build([event('z', 14), event('a', 15)]))
        self.assertEqual(r['a']['supersedes_record_id'], 'z')

    def test_opposite_lexical_content_hashes(self):
        r = records(build([event('a', 14, content='zzz'), event('b', 15, content='aaa')]))
        self.assertEqual(r['b']['supersedes_record_id'], 'a')

    def test_future_tie_cannot_rewrite_history(self):
        events = [event(), event('b', 18), event('c', 18)]
        self.assertEqual(current(self.assert_ambiguous(events)), {'a'})

    def test_latest_tie_with_older_valid(self):
        self.assertEqual(current(self.assert_ambiguous([event(), event('b', 15), event('c', 15)])), set())

    def test_duplicate_plus_later_revision(self):
        result = build([event(), event(), event('copy', 14, content='content-a'), event('b', 15)])
        self.assertEqual(result['event_families'][0]['distinct_content_versions'], 2)
        r = records(result)
        self.assertEqual(r['a']['revision_position'], r['copy']['revision_position'])
        self.assertIsNone(r['b']['supersedes_record_id'])
        self.assertIn('NON_UNIQUE_ADJACENT_CAPTURE', r['b']['reason_codes'])
        self.assertEqual(current(result), {'b'})

    def test_three_one_unorderable(self):
        self.assertEqual(current(self.assert_ambiguous([event(), event('b', 15, first_seen_at_utc=None), event('c', 16)])), set())

    def test_seeded_permutations(self):
        events = [event(), event('b', 15), event('c', 15), event('copy', 14, content='content-a')]
        expected = build(events)
        rng = random.Random(1303)
        for _ in range(40):
            rng.shuffle(events)
            self.assertEqual(build(events), expected)

    def test_identity_collision(self):
        for field in ('source', 'source_event_id'):
            self.assertEqual(current(self.assert_ambiguous([event(), event('b', 15, **{field: 'unrelated'})])), set())

    def test_record_id_collision(self):
        self.assertEqual(current(self.assert_ambiguous([event(), event('a', 15, content='other')])), set())

    def test_recapture_crosses_content_revision(self):
        events = [event(), event('b', 15), event('copy', 16, content='content-a')]
        self.assertEqual(current(self.assert_ambiguous(events)), {'copy'})
        self.assertEqual(current(build(events)), fixtures.m3b_current(events, '2026-09-27T16:00:00Z'))

    def test_same_content_latest_record_tie_matches_m3b(self):
        events = [event(), event('b', 15), event('copy', 15, content='content-b')]
        self.assertEqual(current(build(events)), set())
        self.assertEqual(current(build(events)), fixtures.m3b_current(events, '2026-09-27T16:00:00Z'))
        self.assertEqual(build(events)['event_families'][0]['distinct_content_versions'], 2)

    def test_naive_timestamp_and_equal_offset_instants(self):
        self.assert_ambiguous([event(), event('b', 15, retrieved_at_utc='2026-09-27T15:00:00')])
        other = event('b', 14)
        for field in ('published_at_utc', 'updated_at_utc', 'retrieved_at_utc', 'first_seen_at_utc'):
            other[field] = '2026-09-27T10:00:00-04:00'
        self.assertEqual(current(self.assert_ambiguous([event(), other])), set())

    def test_invalid_cutoff_and_container_raise(self):
        for cutoff in ('bad', '', '2026-09-27T16:00:00'):
            with self.assertRaises(ValueError):
                build([event()], cutoff)
        for events in (None, {}, ['bad']):
            with self.assertRaises(ValueError):
                build(events)

    def test_bad_future_chronology_matches_m3b_fail_closed(self):
        events = [event(), event('b', 18, retrieved_at_utc='bad')]
        self.assertEqual(current(self.assert_ambiguous(events)), set())
        self.assertEqual(current(build(events)), fixtures.m3b_current(events, '2026-09-27T16:00:00Z'))

    def test_safety_and_output_mutation_isolation(self):
        expected = dict(ANALYSIS_ONLY=True, PRODUCTION_INFLUENCE=False,
                        SOLVER_INFLUENCE=False, PROJECTION_MUTATION=False,
                        FORECAST_MUTATION=False, AVAILABILITY_AUTHORITY=False,
                        INJURY_AUTHORITY=False, DEPTH_CHART_AUTHORITY=False,
                        AI_ANALYST_INFLUENCE=False, DATABASE_MUTATION=False,
                        CONSENSUS_ENABLED=False)
        result = build([event()])
        self.assertEqual(result['safety'], expected)
        self.assertEqual(records(result)['a']['safety'], expected)
        self.assertEqual(result['event_families'][0]['safety'], expected)
        result['safety']['PRODUCTION_INFLUENCE'] = True
        self.assertEqual(build([event()])['safety'], expected)

    def test_fresh_import_and_call_have_no_io_or_publication(self):
        # A fresh process prevents an already-imported module masking import effects.
        code = r'''
import sys, socket, sqlite3, subprocess, builtins, os
from unittest.mock import patch

def deny(*args, **kwargs):
    raise AssertionError('Forbidden side effect')

def audit(name, args):
    if name.startswith(('socket.', 'sqlite3.', 'subprocess.')):
        deny()
    if name == 'open' and (isinstance(args[1], str) and any(c in args[1] for c in 'wax+') or args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
        deny()
    if name.startswith(('os.remove', 'os.rename', 'os.mkdir', 'os.system')):
        deny()
sys.addaudithook(audit)
with patch.object(socket, 'socket', deny), patch.object(sqlite3, 'connect', deny), patch.object(subprocess, 'Popen', deny), patch.object(builtins, 'open', deny):
    from shadow_player_analysis_v1_3.revision_provenance import build_revision_provenance
    row = dict(event_key='e', record_id='r', source='s', source_event_id='n', content_hash='h', published_at_utc='2026-09-27T14:00:00Z', retrieved_at_utc='2026-09-27T14:00:00Z', first_seen_at_utc='2026-09-27T14:00:00Z')
    result = build_revision_provenance(source_events=[row], cutoff_utc='2026-09-27T16:00:00Z')
    assert result['revision_provenance'][0]['current_as_of_cutoff']
    assert set(result) == {'revision_provenance', 'event_families', 'safety'}
    assert not any(result['safety'][k] for k in result['safety'] if k != 'ANALYSIS_ONLY')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
