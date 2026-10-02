"""Deterministic offline bridge tests. All fixture writes stay in a temp directory."""
import builtins
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import importlib
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from . import live_authority as live
from .identity import Authority
from .core import build
from .adapters import espn_records
from shadow_player_analysis_v1.core import SAFETY
from shadow_player_analysis_v1.espn_nfl_rss_v1 import ingest
from shadow_player_analysis_v1.test_espn_nfl_rss_v1 import feed, item

AS_OF = '2026-09-27T16:00:00Z'


class LiveAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'fixture.db'
        with closing(sqlite3.connect(self.path)) as c, c:
            c.executescript('''
                CREATE TABLE player_identity (gsis_id, full_name, latest_team, position);
                CREATE TABLE games (game_id, season, week, game_date, gametime, completed);
                INSERT INTO player_identity VALUES ('exact-001','Zay Flowers','BAL','WR');
                INSERT INTO games VALUES ('game-1',2026,3,'2026-09-27','13:00',0);
            ''')
        for target in ('socket.socket.connect', 'socket.create_connection', 'socket.getaddrinfo'):
            guard = patch(target, side_effect=AssertionError('NETWORK_FORBIDDEN'))
            guard.start()
            self.addCleanup(guard.stop)

    def sql(self, query, values=()):
        with closing(sqlite3.connect(self.path)) as c, c:
            c.execute(query, values)

    def snapshots(self, **kwargs):
        return live.build_live_authorities(db_path=self.path, **dict(season=2026, week=3, as_of_utc=AS_OF, **kwargs))

    def authority(self):
        s = self.snapshots()
        return Authority(s['identity'], s['schedule'], AS_OF)

    def test_import_no_io(self):
        with patch.object(sqlite3, 'connect', side_effect=AssertionError('DB')), patch.object(builtins, 'open', side_effect=AssertionError('FILE')):
            importlib.reload(live)

    def test_read_only_connection_and_no_write_statements(self):
        connect = sqlite3.connect
        traces = []
        def checked(database_uri, **kwargs):
            self.assertTrue(database_uri.endswith('?mode=ro'))
            self.assertEqual(kwargs, {'uri': True})
            c = connect(database_uri, **kwargs)
            with self.assertRaises(sqlite3.OperationalError):
                c.execute('CREATE TABLE forbidden (id)')
            c.set_trace_callback(traces.append)
            return c
        with patch.object(sqlite3, 'connect', side_effect=checked):
            self.snapshots()
        self.assertEqual(len(traces), 3)
        self.assertEqual(traces[0], 'BEGIN')
        self.assertTrue(all(s.startswith('SELECT ') for s in traces[1:]))

    def test_exact_player_fields(self):
        self.assertEqual(self.snapshots()['identity']['players'], [dict(player_name='Zay Flowers', team='BAL', position='WR', gsis_id='exact-001')])

    def incomplete(self, field):
        for value in (None, '', '   '):
            with self.subTest(value=value):
                row = dict(gsis_id='other', full_name='Other Player', latest_team='CHI', position='QB')
                row[field] = value
                self.sql('INSERT INTO player_identity VALUES (?,?,?,?)', tuple(row.values()))
                self.assertEqual(len(self.snapshots()['identity']['players']), 1)
                self.sql("DELETE FROM player_identity WHERE gsis_id IS NULL OR gsis_id != 'exact-001'")

    def test_missing_gsis(self): self.incomplete('gsis_id')
    def test_missing_name(self): self.incomplete('full_name')
    def test_missing_team(self): self.incomplete('latest_team')
    def test_missing_position(self): self.incomplete('position')

    def test_conflicting_gsis(self):
        self.sql("INSERT INTO player_identity VALUES ('exact-001','Other Player','CHI','QB')")
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_GSIS'): self.snapshots()

    def test_identical_duplicate_gsis(self):
        self.sql('INSERT INTO player_identity SELECT * FROM player_identity')
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_GSIS'): self.snapshots()

    def test_incomplete_duplicate_gsis(self):
        self.sql("INSERT INTO player_identity VALUES ('exact-001',NULL,'CHI','QB')")
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_GSIS'): self.snapshots()

    def test_shared_name_retains_ambiguity(self):
        self.sql("INSERT INTO player_identity VALUES ('exact-002','Zay Flowers','CHI','WR')")
        a = self.authority()
        self.assertEqual(a.player({'name': 'Zay Flowers'})['identity_status'], 'AMBIGUOUS')
        self.assertEqual(a.player({'name': 'Zay Flowers', 'team': 'CHI'})['gsis_id'], 'exact-002')

    def test_no_fuzzy_or_surname_matching(self):
        for name in ('Flowers', 'Zay Flowars'):
            self.assertIsNone(self.authority().player({'name': name})['gsis_id'])

    def test_explicit_season_week(self):
        s = self.snapshots()['identity']
        self.assertEqual((s['season'], s['week']), (2026, 3))
        for season, week in ((None, 3), (2026, None), (True, 3), (2026, 0)):
            with self.assertRaises(ValueError):
                live.build_live_authorities(db_path=self.path, season=season, week=week)

    def test_only_requested_schedule(self):
        self.sql("INSERT INTO games VALUES (NULL,2025,3,NULL,NULL,0)")
        self.sql("INSERT INTO games VALUES (NULL,2026,4,NULL,NULL,0)")
        self.assertEqual(len(self.snapshots()['schedule']['games']), 1)

    def test_completed_retained(self):
        self.sql("INSERT INTO games VALUES ('completed',2026,3,'2026-09-24','20:15',1)")
        self.assertEqual(len(self.snapshots()['schedule']['games']), 2)

    def kickoff(self, date, time, expected):
        self.sql('UPDATE games SET game_date=?,gametime=?', (date, time))
        self.assertEqual(self.snapshots()['schedule']['games'][0]['kickoff_at_utc'], expected)

    def test_september_afternoon(self):
        self.kickoff('2026-09-27','13:00','2026-09-27T17:00:00Z')
        self.kickoff('2026-09-27','16:05','2026-09-27T20:05:00Z')
        self.kickoff('2026-09-27','16:25','2026-09-27T20:25:00Z')

    def test_evening_utc_date_boundary(self):
        self.kickoff('2026-09-27','20:20','2026-09-28T00:20:00Z')
        self.kickoff('2026-09-28','20:15','2026-09-29T00:15:00Z')

    def test_winter_est(self):
        self.kickoff('2026-12-27','13:00','2026-12-27T18:00:00Z')

    def invalid(self, field, values):
        for value in values:
            with self.subTest(value=value):
                self.sql('UPDATE games SET '+field+'=?', (value,))
                with self.assertRaises(ValueError): self.snapshots()

    def test_invalid_date(self): self.invalid('game_date', (None, '', 'tomorrow', '2026-02-30', '2026-9-27'))
    def test_invalid_time(self): self.invalid('gametime', (None, '', '13:99', '24:00', '1:00', '13:00 EST'))
    def test_missing_game_id(self): self.invalid('game_id', (None, '', '  '))

    def test_duplicate_game_id(self):
        self.sql('INSERT INTO games SELECT * FROM games')
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_GAME_ID'): self.snapshots()

    def test_conflicting_game_id(self):
        self.sql("INSERT INTO games VALUES ('game-1',2026,3,'2026-09-28','20:15',1)")
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_GAME_ID'): self.snapshots()

    def test_dst_fold_and_gap_rejected(self):
        for date, time in (('2026-11-01','01:30'), ('2026-03-08','02:30')):
            self.sql('UPDATE games SET game_date=?,gametime=?', (date, time))
            with self.assertRaisesRegex(ValueError, 'AMBIGUOUS_OR_NONEXISTENT'): self.snapshots()

    def test_exact_injected_as_of_both_snapshots(self):
        value = '2026-09-27T16:00:00.123456+00:00'
        s = live.build_live_authorities(db_path=self.path, season=2026, week=3, as_of_utc=value)
        self.assertEqual(s['identity']['available_at_utc'], value)
        self.assertEqual(s['schedule']['available_at_utc'], value)

    def test_runtime_clock_captured_once_after_reads(self):
        instant = datetime(2026, 9, 27, 16, tzinfo=timezone.utc)
        with patch.object(live, 'datetime', wraps=datetime) as clock:
            clock.now.return_value = instant
            s = live.build_live_authorities(db_path=self.path, season=2026, week=3)
            clock.now.assert_called_once_with(timezone.utc)
        self.assertEqual(s['identity']['available_at_utc'], AS_OF)
        self.assertEqual(s['schedule']['available_at_utc'], AS_OF)

    def test_non_utc_as_of_rejected(self):
        for value in ('2026-09-27T12:00:00', '2026-09-27T12:00:00-04:00', 'bad'):
            with self.assertRaises(ValueError):
                live.build_live_authorities(db_path=self.path, season=2026, week=3, as_of_utc=value)

    def test_no_writes_network_or_publication(self):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        files = sorted(Path(self.temp.name).iterdir())
        with patch.object(builtins, 'open', side_effect=AssertionError('FILE_PUBLICATION')), patch.object(Path, 'open', side_effect=AssertionError('FILE_PUBLICATION')):
            self.snapshots()
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(files, sorted(Path(self.temp.name).iterdir()))

    def test_safety_flags(self):
        for name, value in SAFETY.items():
            self.assertIs(getattr(live, name), value)
            self.assertIs(value, name == 'ANALYSIS_ONLY')

    def test_missing_database_not_created(self):
        path = self.path.with_name('absent.db')
        with self.assertRaises(sqlite3.OperationalError):
            live.build_live_authorities(db_path=path, season=2026, week=3)
        self.assertFalse(path.exists())

    def test_empty_authority_rejected(self):
        self.sql('DELETE FROM games')
        with self.assertRaisesRegex(ValueError, 'EMPTY_LIVE_AUTHORITY'): self.snapshots()

    def test_espn_ambiguous_timestamp_stays_ineligible(self):
        capture = ingest(feed(item(title='Zay Flowers will play', pub='Sun, 27 Sep 2026 10:00:00 EST')), AS_OF)
        records = espn_records(capture)
        records[0]['game_id'] = 'game-1'  # Explicit evidence association, never inferred.
        s = self.snapshots()
        result = build(records=records, **s, cutoff=AS_OF,
                       approvals={'ESPN_NFL_RSS_V1': ['ESPN_RSS_SHADOW_CAPTURE_ONLY']})
        event = result['source_events'][0]
        self.assertEqual(event['temporal_status'], 'AMBIGUOUS_SOURCE_TIMESTAMP')
        self.assertEqual(event['temporal_confidence'], 'UNRESOLVED')
        self.assertIsNone(event['published_at_utc'])
        row = result['classified_evidence'][0]
        self.assertIn('TEMPORAL_UNRESOLVED', row['temporal_reason_codes'])
        self.assertFalse(row['pregame_eligible'])
        for row in result['classified_evidence'] + result['entity_links']:
            self.assertFalse(row['claim_eligible'])
            self.assertFalse(row['consensus_eligible'])
            self.assertEqual(row['safety'], SAFETY)
        self.assertEqual(result['source_quality'][0]['consensus_votes'], 0)
        records[0].pop('game_id')
        result = build(records=records, **s, cutoff=AS_OF, approvals={})
        self.assertEqual(self.authority().game(None)['identity_status'], 'UNRESOLVED')
        self.assertFalse(any(r['entity_type'] == 'GAME' for r in result['entity_links']))


if __name__ == '__main__':
    unittest.main()
