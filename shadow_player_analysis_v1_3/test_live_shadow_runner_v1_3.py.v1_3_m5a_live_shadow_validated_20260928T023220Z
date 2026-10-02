"""Offline M5A integration and publication tests; no real network or live DB."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from . import live_shadow_runner as runner
from shadow_player_analysis_v1 import espn_nfl_rss_v1 as espn
from shadow_player_analysis_v1_2 import core, live_authority
from . import handoff

CUTOFF = '2026-09-27T16:00:00Z'


def capture(ambiguous=False):
    # Exercise the real offline connector separately; the runner must never ingest.
    date = 'Sun, 27 Sep 2026 10:00:00 ' + ('EST' if ambiguous else 'GMT')
    raw = ('<rss><channel><title>NFL</title><item><title>John Player role will increase</title>'
           '<description>John Player role will increase</description>'
           '<guid>test-1</guid><link>https://example.invalid/article</link>'
           '<pubDate>' + date + '</pubDate></item></channel></rss>').encode()
    return espn.ingest(raw, '2026-09-27T11:00:00+00:00')


def authorities():
    return dict(identity=dict(authority='WFS_IDENTITY_SNAPSHOT', available_at_utc=CUTOFF,
                              season=2026, week=3, players=[dict(gsis_id='p1',
                              player_name='John Player', team='BUF', position='WR')]),
                schedule=dict(authority='WFS_SCHEDULE_SNAPSHOT', available_at_utc=CUTOFF,
                              games=[dict(game_id='g1', kickoff_at_utc='2026-09-27T20:00:00Z')]))


class Harness(unittest.TestCase):
    def setUp(self):
        self.capture = capture()
        self.authorities = authorities()
        self.tmp = tempfile.TemporaryDirectory(dir=runner.ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / 'live_shadow_artifacts'
        for target, kwargs in (
            (runner, dict(OUTPUT=self.output)),
            (espn, dict(fetch=lambda prior_events=(): copy.deepcopy(self.capture))),
            (live_authority, dict(build_live_authorities=lambda **kw: copy.deepcopy(self.authorities))),
        ):
            context = patch.multiple(target, **kwargs)
            context.start()
            self.addCleanup(context.stop)
        for target in ('socket.create_connection', 'urllib.request.OpenerDirector.open',
                       'sqlite3.connect', 'shadow_player_analysis_v1.espn_nfl_rss_v1.ingest'):
            context = patch(target, side_effect=AssertionError('FORBIDDEN_IO_OR_REINGEST'))
            context.start()
            self.addCleanup(context.stop)

    def run_shadow(self, **kwargs):
        options = dict(season=2026, week=3, cutoff_utc=CUTOFF, publish=False)
        options.update(kwargs)
        return runner.run_live_shadow(**options)


class LiveRunnerTests(Harness):
    def test_import_no_io(self):
        import builtins
        with patch.object(builtins, 'open', side_effect=AssertionError('open')), \
             patch.object(os, 'open', side_effect=AssertionError('os.open')), \
             patch.object(os, 'mkdir', side_effect=AssertionError('mkdir')), \
             patch.object(Path, 'resolve', side_effect=AssertionError('resolve')), \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('read')):
            importlib.reload(runner)
        runner.OUTPUT = self.output

    def test_engine_order_cutoff_and_approval(self):
        calls = []
        original_build, original_handoff = core.build, handoff.build_shadow_handoff
        def fetch(prior_events=()):
            calls.append('fetch')
            self.assertEqual(prior_events, [])
            return self.capture
        def authority(**kw):
            calls.append('authority')
            self.assertEqual(kw, dict(season=2026, week=3, as_of_utc=CUTOFF))
            return self.authorities
        def build(records, identity, schedule, cutoff, approvals):
            calls.append('build')
            self.assertEqual(cutoff, CUTOFF)
            self.assertEqual(approvals, {runner.SOURCE: ['ESPN_RSS_SHADOW_CAPTURE_ONLY']})
            self.assertEqual(records[0]['raw_publication_timestamp'],
                             self.capture['events'][0]['raw_feed_fields']['pubDate'])
            return original_build(records, identity, schedule, cutoff, approvals)
        def packet(**kw):
            calls.append('handoff')
            self.assertEqual(kw['cutoff_utc'], CUTOFF)
            return original_handoff(**kw)
        with patch.object(espn, 'fetch', fetch), \
             patch.object(live_authority, 'build_live_authorities', authority), \
             patch.object(core, 'build', build), \
             patch.object(handoff, 'build_shadow_handoff', packet):
            self.run_shadow()
        self.assertEqual(calls, ['fetch', 'authority', 'build', 'handoff'])

    def test_capture_and_both_quarantines_preserved(self):
        result = self.run_shadow()
        self.assertEqual(result['source_capture'], self.capture)
        self.assertEqual(result['source_capture']['retrieval'], self.capture['retrieval'])
        self.assertTrue(result['v1_2_result']['quarantine'])
        self.assertEqual(result['handoff']['quarantine'], result['v1_2_result']['quarantine'])

    def test_zero_claims_valid_no_game_inference(self):
        result = self.run_shadow()
        self.assertEqual(result['handoff']['claims']['claims'], [])
        self.assertNotIn('game_id', result['v1_2_result']['source_events'][0])
        self.assertTrue(result['validation']['zero_claims_valid'])

    def test_ambiguous_quarantine_never_promoted(self):
        # Generate fixture with original offline function while ingest remains forbidden in run.
        with patch.object(espn, 'ingest', self._original_ingest):
            self.capture = capture(True)
        result = self.run_shadow()
        self.assertEqual(result['source_capture'], self.capture)
        self.assertEqual(result['v1_2_result']['source_events'], [])
        self.assertEqual(result['handoff']['claims']['claims'], [])
        self.assertIsNone(result['source_capture']['quarantine'][0]['published_at_utc'])

    _original_ingest = staticmethod(espn.ingest)

    def test_no_fuzzy_identity(self):
        self.authorities['identity']['players'][0]['player_name'] = 'Jon Player'
        result = self.run_shadow()
        self.assertFalse(any(x.get('gsis_id') == 'p1' for x in result['v1_2_result']['entity_links']))

    def test_no_publication_when_disabled(self):
        self.run_shadow()
        self.assertFalse(self.output.exists())

    def test_explicit_database_forwarded(self):
        with patch.object(live_authority, 'build_live_authorities', return_value=self.authorities) as mocked:
            self.run_shadow(db_path='explicit.sqlite')
        self.assertEqual(mocked.call_args.kwargs['db_path'], 'explicit.sqlite')

    def test_detached_inputs_and_handoff(self):
        source, auth = copy.deepcopy(self.capture), copy.deepcopy(self.authorities)
        saved = []
        original = handoff.build_shadow_handoff
        def packet(**kw):
            value = original(**kw)
            saved.append(value)
            return value
        with patch.object(handoff, 'build_shadow_handoff', packet):
            result = self.run_shadow()
        self.assertEqual(self.capture, source)
        self.assertEqual(self.authorities, auth)
        self.assertEqual(result['handoff'], saved[0])
        result['handoff']['quarantine'].clear()
        self.assertTrue(saved[0]['quarantine'])
        result['source_capture']['events'].clear()
        self.assertEqual(self.capture, source)

    def test_safety_contract(self):
        result = self.run_shadow()
        self.assertEqual(result['safety'], runner.SAFETY)
        self.assertEqual(result['handoff']['safety'], runner.SAFETY)
        self.assertEqual(result['handoff']['claims']['consensus'], [])
        for key, value in runner.SAFETY.items():
            self.assertIs(value, key == 'ANALYSIS_ONLY')

    def test_publication_deterministic_idempotent(self):
        first = self.run_shadow(publish=True)
        second = self.run_shadow(publish=True)
        self.assertEqual(first, second)
        path = Path(first['bundle_path'])
        manifest = runner.verify_bundle(path)
        self.assertEqual(path.name, manifest['run_id'])
        unsigned = {k: v for k, v in manifest.items() if k != 'run_id'}
        self.assertEqual(path.name, hashlib.sha256(runner.canonical(unsigned)).hexdigest())
        self.assertEqual(set(p.name for p in path.iterdir()), runner.FILES | {'manifest.json'})
        self.assertEqual(list(self.output.iterdir()), [path])
        for name, meta in manifest['files'].items():
            raw = (path / name).read_bytes()
            self.assertEqual(meta, dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))

    def test_changed_input_new_address_preserves_history(self):
        first = self.run_shadow(publish=True)
        path = Path(first['bundle_path'])
        before = {p.name: p.read_bytes() for p in path.iterdir()}
        self.capture['retrieval']['feed_title'] = 'changed'
        second = self.run_shadow(publish=True)
        self.assertNotEqual(first['run_id'], second['run_id'])
        self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})

    def test_protected_hashes(self):
        self.assertEqual(runner.protected_hashes(), runner.PROTECTED)

    def test_nonzero_claims_are_preserved_without_consensus(self):
        from .test_handoff_v1_3 import fixture
        normalized, schedule = fixture()
        self.authorities['schedule']['games'] = schedule['games']
        # Fixture hashes bind the exact schedule; use this run's authority hash.
        from shadow_player_analysis_v1.core import digest
        for link in normalized['entity_links']:
            if link['entity_type'] == 'GAME':
                link['schedule_sha256'] = digest(self.authorities['schedule'])
        expected = handoff.build_shadow_handoff(v1_2_result=normalized,
                    cutoff_utc=CUTOFF, schedule=self.authorities['schedule'])
        self.assertTrue(expected['claims']['claims'])
        with patch.object(core, 'build', return_value=normalized):
            result = self.run_shadow()
        self.assertEqual(result['handoff'], expected)
        self.assertTrue(all(c['consensus_eligible'] is False
                            for c in result['handoff']['claims']['claims']))

    def test_cold_import_audit(self):
        import subprocess
        import sys
        code = '''
import sys
# Audit reads of Python module code are permitted; data I/O and writes are not.
def guard(event, args):
    if event.startswith(('socket.', 'sqlite3.')) or event in ('os.mkdir', 'os.rename', 'os.remove'):
        raise AssertionError((event, args))
    if event == 'open':
        path, mode, flags = args
        if isinstance(path, str) and path.endswith(('.py', '.pyc')) and not (flags & 3):
            return
        raise AssertionError((event, args))
sys.addaudithook(guard)
import shadow_player_analysis_v1_3.live_shadow_runner
'''
        completed = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
