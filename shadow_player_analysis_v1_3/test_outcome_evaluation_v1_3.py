"""Offline M5B fixtures and focused tests. No live fetch or claim rebuild."""
import copy
import hashlib
import itertools
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import outcome_evaluation as m5b


ROOT = Path(__file__).resolve().parent.parent
STAMP = '2026-09-27T23:00:00Z'
PROTECTED = {
    'claims.py': '6debb81489ee7b73dfa09c3a9bca9c728eb9b2b7855693a4ac530416e84297f8',
    'revision_provenance.py': '1a9ac584b1f3f026f1a21db4e541a7f00991052b7f347c65476c79a5324eead7',
    'corroboration.py': '47b4566e08651a09c2a09e44536dac9817603b6aa807f6b4f236cd2e1d40c735',
    'handoff.py': '3f62c4a591ca2b275e77b3062419e28e20ab0ba122506e604fa4d720efd9a037',
    'live_shadow_runner.py': '4dc3f96048a9b94ff5d6b923132a0da645aec425808893aab3d67f7c0971fa9a',
}


def claim(cid='claim-a', signal='ROLE_INCREASE', **changes):
    """Already-built claim-shaped fixture; no upstream engines invoked."""
    row = dict(claim_id=cid, record_id='record-a', event_key='event-a',
               source='source-a', source_event_id='native-a', origin_id='origin-a',
               content_hash='content-a', gsis_id='00-TEST-001', player_name='Same Name',
               game_id='game-1', signal_type=signal,
               direction={'ROLE_INCREASE': 'INCREASE', 'ROLE_DECREASE': 'DECREASE'}
                         .get(signal, 'UNSPECIFIED'),
               evidence_kind='REPORT', evidence_quote='Frozen pregame evidence.',
               published_at_utc='2026-09-27T14:00:00Z', cutoff_utc='2026-09-27T16:00:00Z',
               knowledge_at_utc='2026-09-27T15:00:00Z',
               effective_from_utc='2026-09-27T15:00:00Z',
               kickoff_at_utc='2026-09-27T17:00:00Z',
               valid_until_utc='2026-09-27T17:00:00Z', schedule_sha256='a' * 64,
               claim_eligible=True, consensus_eligible=False,
               consensus_gate='NOT_ENABLED_V1_3', eligibility_status='ELIGIBLE',
               corroboration_state='NOT_EVALUATED',
               safety={k: v for k, v in m5b.SAFETY.items() if k != 'CONSENSUS_ENABLED'})
    row.update(changes)
    return row


class OutcomeFixture(unittest.TestCase):
    """Temporary files are confined to the workspace and removed by cleanup."""
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='m5b-test-', dir=ROOT)
        self.addCleanup(tmp.cleanup)
        self.db = Path(tmp.name) / 'fixture.sqlite'
        self.conn = sqlite3.connect(self.db)
        self.addCleanup(self.conn.close)
        # Deliberately no unique constraints: duplicates must be detected by M5B.
        self.conn.executescript('''
          CREATE TABLE games(game_id TEXT, season INTEGER, week INTEGER,
                             game_type TEXT, completed, updated_at TEXT);
          CREATE TABLE player_game_stats(game_id TEXT, player_id TEXT,
              season INTEGER, week INTEGER, team TEXT, updated_at TEXT,
              player_display_name TEXT, fanduel_points REAL, receiving_tds REAL,
              target_share REAL, air_yards_share REAL);
          CREATE TABLE player_weekly_usage(game_id TEXT, player_id TEXT,
              season INTEGER, week INTEGER, team TEXT, updated_at TEXT,
              player_display_name TEXT, opportunities, offense_pct, targets, carries,
              usage_3g, snap_pct_3g, target_3g, carry_3g,
              usage_delta, snap_delta, target_delta, carry_delta,
              role_expansion_flag, role_decline_flag, target_share,
              offense_snaps REAL, air_yards_share REAL, fanduel_points REAL);
          CREATE TABLE player_pregame_features(game_id TEXT, player_id TEXT,
              season INTEGER, week INTEGER, team TEXT, updated_at TEXT,
              history_games, opportunities_avg_3, snap_pct_avg_3,
              targets_avg_3, carries_avg_3, target_share_avg_3);
        ''')
        self.insert('games', dict(game_id='game-1', season=2026, week=3,
                                 game_type='REG', completed=1, updated_at=STAMP))
        identity = dict(game_id='game-1', player_id='00-TEST-001', season=2026,
                        week=3, team='AAA', updated_at=STAMP)
        self.insert('player_game_stats', dict(identity, player_display_name='Same Name',
                    fanduel_points=0, receiving_tds=0, target_share=.3, air_yards_share=.9))
        self.insert('player_weekly_usage', dict(identity, player_display_name='Same Name',
                    opportunities=15, offense_pct=.6, targets=5, carries=10,
                    usage_3g=10, snap_pct_3g=.5, target_3g=4, carry_3g=6,
                    usage_delta=5, snap_delta=.1, target_delta=1, carry_delta=4,
                    role_expansion_flag=1, role_decline_flag=0, target_share=.3,
                    offense_snaps=50, air_yards_share=.9, fanduel_points=0))
        self.insert('player_pregame_features', dict(identity, history_games=3,
                    opportunities_avg_3=10, snap_pct_avg_3=.5, targets_avg_3=4,
                    carries_avg_3=6, target_share_avg_3=.2))

    def insert(self, table, row):
        self.conn.execute('INSERT INTO ' + table + '(' + ','.join(row) + ') VALUES ('
                          + ','.join('?' for _ in row) + ')', tuple(row.values()))
        self.conn.commit()

    def change(self, table, **values):
        self.conn.execute('UPDATE ' + table + ' SET ' + ','.join(k + '=?' for k in values),
                          tuple(values.values()))
        self.conn.commit()

    def delete(self, table):
        self.conn.execute('DELETE FROM ' + table)
        self.conn.commit()

    def duplicate(self, table):
        self.conn.execute('INSERT INTO ' + table + ' SELECT * FROM ' + table)
        self.conn.commit()

    def evaluate(self, *rows):
        return m5b.evaluate_outcomes(claims=list(rows) if rows else [claim()], db_path=self.db)

    def row(self, *rows):
        result = self.evaluate(*rows)
        all_rows = result['evaluations'] + result['unevaluated']
        self.assertEqual(len(all_rows), 1, result)
        return all_rows[0]

    def declining(self):
        self.change('player_weekly_usage', opportunities=5, targets=2, carries=3,
                    offense_pct=.4, usage_delta=-5, snap_delta=-.1,
                    target_delta=-2, carry_delta=-3,
                    role_expansion_flag=0, role_decline_flag=1)


class OutcomeEvaluationTests(OutcomeFixture):
    def test_role_increase_supported(self):
        self.assertEqual(self.row()['outcome_state'], 'SUPPORTED')

    def test_role_increase_not_supported(self):
        self.declining()
        self.assertEqual(self.row()['outcome_state'], 'NOT_SUPPORTED')

    def test_role_decrease_supported(self):
        self.declining()
        self.assertEqual(self.row(claim(signal='ROLE_DECREASE'))['outcome_state'], 'SUPPORTED')

    def test_role_decrease_not_supported(self):
        self.assertEqual(self.row(claim(signal='ROLE_DECREASE'))['outcome_state'], 'NOT_SUPPORTED')

    def test_target_share_increase_supported(self):
        row = self.row(claim(signal='TARGET_SHARE', direction='INCREASE'))
        self.assertEqual(row['outcome_state'], 'SUPPORTED')
        self.assertEqual(row['pregame_baseline']['player_pregame_features']['target_share_avg_3'], .2)

    def test_target_share_increase_not_supported(self):
        self.change('player_weekly_usage', target_share=.1)
        self.assertEqual(self.row(claim(signal='TARGET_SHARE', direction='INCREASE'))['outcome_state'], 'NOT_SUPPORTED')

    def test_target_share_decrease_supported(self):
        self.change('player_weekly_usage', target_share=.1)
        self.assertEqual(self.row(claim(signal='TARGET_SHARE', direction='DECREASE'))['outcome_state'], 'SUPPORTED')

    def test_target_share_decrease_not_supported(self):
        self.assertEqual(self.row(claim(signal='TARGET_SHARE', direction='DECREASE'))['outcome_state'], 'NOT_SUPPORTED')

    def test_target_share_equal_not_supported(self):
        self.change('player_weekly_usage', target_share=.2)
        self.assertEqual(self.row(claim(signal='TARGET_SHARE', direction='INCREASE'))['outcome_state'], 'NOT_SUPPORTED')

    def test_target_share_unspecified_indeterminate(self):
        for share in (0, .3, 1):
            self.change('player_weekly_usage', target_share=share)
            self.assertEqual(self.row(claim(signal='TARGET_SHARE'))['outcome_state'], 'INDETERMINATE')

    def test_no_evidence_interpretation_for_majority_or_larger(self):
        for quote in ('Same Name is expected to command the majority target share',
                      'Same Name is expected to have a larger target share'):
            self.assertEqual(self.row(claim(signal='TARGET_SHARE', evidence_quote=quote))['outcome_state'], 'INDETERMINATE')

    def test_incomplete_game(self):
        self.change('games', completed=0)
        self.assertEqual(self.row()['evaluation_status'], 'NOT_FINAL')

    def test_missing_game(self):
        self.delete('games')
        self.assertEqual(self.row()['evaluation_status'], 'NO_OUTCOME')

    def test_duplicate_games(self):
        self.duplicate('games')
        self.assertEqual(self.row()['evaluation_status'], 'AMBIGUOUS_OUTCOME')

    def test_missing_stats(self):
        self.delete('player_game_stats')
        self.assertEqual(self.row()['evaluation_status'], 'NO_OUTCOME')

    def test_missing_usage(self):
        self.delete('player_weekly_usage')
        self.assertEqual(self.row()['evaluation_status'], 'NO_OUTCOME')

    def test_missing_baseline(self):
        self.delete('player_pregame_features')
        self.assertEqual(self.row()['reason_codes'], ['MISSING_BASELINE'])

    def test_exact_identity_no_name_dependency(self):
        self.change('player_weekly_usage', player_display_name='Unrelated Name')
        self.assertEqual(self.row()['outcome_state'], 'SUPPORTED')

    def test_no_name_match(self):
        self.assertEqual(self.row(claim(gsis_id='different-id'))['evaluation_status'], 'NO_OUTCOME')

    def test_no_fuzzy_identity(self):
        self.assertEqual(self.row(claim(gsis_id='00-TEST-01'))['evaluation_status'], 'NO_OUTCOME')

    def test_exact_game(self):
        self.assertEqual(self.row(claim(game_id='game-2'))['evaluation_status'], 'NO_OUTCOME')

    def test_original_claim_unchanged(self):
        original = claim(extra_audit={'items': [1, 2]})
        before = copy.deepcopy(original)
        row = self.row(original)
        self.assertEqual(original, before)
        self.assertEqual(row['pregame_claim'], before)
        self.assertEqual(row['pregame_claim_sha256'], m5b._digest(before))

    def test_output_is_detached(self):
        original = claim(extra_audit={'items': [1, 2]})
        output = self.row(original)
        output['pregame_claim']['extra_audit']['items'].append(3)
        output['safety']['PRODUCTION_INFLUENCE'] = True
        self.assertEqual(original['extra_audit']['items'], [1, 2])
        self.assertFalse(self.row()['safety']['PRODUCTION_INFLUENCE'])

    def test_deterministic_ids(self):
        self.assertEqual(self.evaluate(), self.evaluate())
        self.assertEqual(len(self.row()['evaluation_id']), 64)

    def test_permutation_invariance_and_ordering(self):
        rows = [claim('z'), claim('a'), claim('m', signal='TARGET_SHARE')]
        expected = self.evaluate(*rows)
        for permutation in itertools.permutations(rows):
            self.assertEqual(self.evaluate(*permutation), expected)
        self.assertEqual([r['claim_id'] for r in expected['evaluations']], ['a', 'm', 'z'])

    def test_identical_duplicate_collapse(self):
        result = self.evaluate(claim(), claim())
        self.assertEqual(len(result['evaluations']), 1)
        self.assertEqual(result['summary']['duplicate_count'], 1)
        self.assertEqual(result['evaluations'], self.evaluate()['evaluations'])

    def test_conflicting_duplicate_ids(self):
        rows = [claim(), claim(game_id='game-2')]
        first = self.evaluate(*rows)
        self.assertFalse(first['evaluations'])
        self.assertEqual(first, self.evaluate(*reversed(rows)))
        self.assertTrue(all(r['reason_codes'] == ['CONFLICTING_CLAIM_ID'] for r in first['unevaluated']))

    def test_safety_flags_exact(self):
        expected = {'ANALYSIS_ONLY': True, **{k: False for k in (
            'PRODUCTION_INFLUENCE', 'SOLVER_INFLUENCE', 'PROJECTION_MUTATION',
            'FORECAST_MUTATION', 'AVAILABILITY_AUTHORITY', 'INJURY_AUTHORITY',
            'DEPTH_CHART_AUTHORITY', 'AI_ANALYST_INFLUENCE', 'DATABASE_MUTATION',
            'CONSENSUS_ENABLED')}}
        result = self.evaluate()
        self.assertEqual(result['safety'], expected)
        self.assertEqual(result['evaluations'][0]['safety'], expected)
        for key, value in expected.items():
            self.assertIs(getattr(m5b, key), value)

    def test_consensus_preserved(self):
        c = self.row()['pregame_claim']
        self.assertIs(c['consensus_eligible'], False)
        self.assertEqual(c['consensus_gate'], 'NOT_ENABLED_V1_3')
        self.assertEqual(c['corroboration_state'], 'NOT_EVALUATED')

    def test_summary_counts_only(self):
        def check(value):
            for child in value.values():
                if isinstance(child, dict):
                    check(child)
                else:
                    self.assertIs(type(child), int)
        check(self.evaluate()['summary'])

    def test_audit_source_fields(self):
        row = self.row()
        self.assertEqual(set(row['postgame_outcome']), {'player_game_stats', 'player_weekly_usage'})
        self.assertEqual(row['postgame_outcome']['player_weekly_usage']['updated_at'], STAMP)
        self.assertEqual(row['postgame_outcome']['player_weekly_usage']['usage_delta'], 5)
        self.assertEqual(row['pregame_baseline']['player_pregame_features']['opportunities_avg_3'], 10)
        self.assertEqual(row['completed_game']['completed'], 1)

    def test_database_bytes_and_schema_unchanged(self):
        before = self.db.read_bytes()
        schema = self.conn.execute('SELECT * FROM sqlite_master').fetchall()
        self.evaluate()
        self.assertEqual(self.db.read_bytes(), before)
        self.assertEqual(self.conn.execute('SELECT * FROM sqlite_master').fetchall(), schema)
        self.assertEqual(self.conn.execute('PRAGMA quick_check').fetchone(), ('ok',))

    def test_database_uri_readonly_and_statement_trace(self):
        real_connect = sqlite3.connect
        statements, connections = [], []
        def connect(filename, **kwargs):
            self.assertTrue(filename.endswith('?mode=ro'))
            self.assertTrue(kwargs['uri'])
            connection = real_connect(filename, **kwargs)
            connection.set_trace_callback(statements.append)
            connections.append(filename)
            return connection
        with patch.object(m5b.sqlite3, 'connect', side_effect=connect):
            self.assertEqual(self.row()['outcome_state'], 'SUPPORTED')
        self.assertEqual(len(connections), 1)
        self.assertTrue(all(s.startswith(('SELECT ', 'BEGIN')) for s in statements), statements)

    def test_import_no_db_network_filesystem_side_effects(self):
        code = '''
import sys
def guard(event, args):
    if event.startswith(('socket.', 'sqlite3.')) or event in (
        'os.mkdir', 'os.rename', 'os.remove', 'os.rmdir', 'os.truncate'):
        raise AssertionError((event, args))
    if event == 'open':
        path, mode, flags = args
        if isinstance(path, str) and path.endswith(('.py', '.pyc')) and not (flags & 3):
            return
        raise AssertionError((event, args))
sys.addaudithook(guard)
import shadow_player_analysis_v1_3.outcome_evaluation
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_protected_hashes(self):
        for name, expected in PROTECTED.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest(), expected)

    def test_empty_input_no_db_access(self):
        with patch.object(m5b.sqlite3, 'connect', side_effect=AssertionError('DB accessed')):
            self.assertEqual(m5b.evaluate_outcomes(claims=[])['evaluations'], [])


if __name__ == '__main__':
    unittest.main()
