"""Absent-week identity regression and partial-snapshot fail-closed contract."""
import sqlite3
import unittest
from unittest.mock import patch

import pandas as pd
import injury_consensus as c


class RolloverTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.db.executescript('''
            CREATE TABLE weekly_rosters(season, week, gsis_id, full_name, team,
                position, status, status_description_abbr, updated_at);
            CREATE TABLE player_identity(gsis_id, full_name, latest_team, position);
            CREATE TABLE depth_charts(gsis_id, team, player_name, pos_abb,
                pos_name, snapshot_dt, updated_at);
        ''')
        self.people = [
            ('00-0034855', 'Baker Mayfield', 'TB', 'QB'),
            ('00-0036358', 'CeeDee Lamb', 'DAL', 'WR'),
            ('00-0037120', 'Tyler Goodson', 'DAL', 'RB'),
        ]
        self.ids = {p[0] for p in self.people}
        self.stamp = '2026-10-06T14:08:49Z'
        for gsis, name, team, pos in self.people:
            self.db.execute('INSERT INTO player_identity VALUES(?,?,?,?)', (gsis,name,team,pos))
            self.db.execute('INSERT INTO depth_charts VALUES(?,?,?,?,?,?,?)',
                            (gsis,team,name,pos,pos,self.stamp,self.stamp))
            self.db.execute('INSERT INTO weekly_rosters VALUES(2026,4,?,?,?,?,?,?,?)',
                            (gsis,name,team,pos,'OUT','RES',self.stamp))
        self.secondary = pd.DataFrame([
            dict(gsis_id=g, player_name=n, team=t, practice_signal='',
                 availability_signal='', signal_strength='LOW', source='test',
                 source_timestamp=self.stamp, headline='', news_signal_count=1)
            for g,n,t,p in self.people
        ])
        self.structured = pd.DataFrame(columns=[
            'gsis_id','team','player_name','position','report_status','practice_status',
        ])
        self.forecast = pd.DataFrame([
            dict(player_id=g, entity_name=n, team=t, position=p,
                 entity_type='OFFENSE_PLAYER', game_id='2026_05_TEST')
            for g,n,t,p in self.people
        ])

    def resolve(self):
        return c.current_roster_identity(self.db, 2026, 5, self.ids)

    def expand(self, base):
        with patch.object(c, 'resolve_current_offensive_universe', return_value=self.forecast):
            return c._gav2_expand_global_consensus(self.db, base, 2026, 5)

    def test_absent_week_resolves_without_historical_availability(self):
        for historical_status in ['ACT', 'OUT']:
            self.db.execute('UPDATE weekly_rosters SET status=?', (historical_status,))
            roster = self.resolve()
            self.assertEqual(set(roster.gsis_id), self.ids)
            base = c._wfs_v1_build_consensus(self.structured, self.secondary, roster, 2026, 5)
            result = self.expand(base)
            self.assertTrue(result.consensus_status.eq('').all())
            self.assertTrue(result.injury_gate.eq('ALLOW').all())
            self.assertEqual(self.db.execute('SELECT count(*) FROM weekly_rosters WHERE week=5').fetchone()[0], 0)

    def test_gav2_generic_missing_roster_remains_unknown(self):
        roster = self.resolve()
        base = c._wfs_v1_build_consensus(self.structured, self.secondary, roster, 2026, 5)
        result = self.expand(base.iloc[:0])
        self.assertTrue(result.availability_risk.eq('UNKNOWN').all())
        self.assertTrue(result.injury_gate.eq('ALLOW').all())
        self.assertTrue(result.consensus_status.eq('').all())

    def test_partial_snapshot_does_not_fallback(self):
        self.db.execute('INSERT INTO weekly_rosters SELECT season,5,gsis_id,full_name,team,position,status,status_description_abbr,updated_at FROM weekly_rosters LIMIT 1')
        with self.assertRaisesRegex(RuntimeError, 'missing from current roster'):
            self.resolve()

    def test_present_snapshot_uses_roster_without_depth(self):
        self.db.execute('UPDATE weekly_rosters SET week=5')
        self.db.execute('DELETE FROM depth_charts')
        self.assertEqual(set(self.resolve().gsis_id), self.ids)

    def test_duplicate_identity_fails_closed(self):
        self.db.execute('INSERT INTO player_identity SELECT * FROM player_identity LIMIT 1')
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            self.resolve()

    def test_missing_exact_identity_no_name_match(self):
        self.db.execute("UPDATE player_identity SET gsis_id='different-id' WHERE position='QB'")
        with self.assertRaisesRegex(RuntimeError, 'one exact'):
            self.resolve()

    def test_depth_team_mismatch(self):
        self.db.execute("UPDATE depth_charts SET team='SEA' WHERE pos_abb='QB'")
        with self.assertRaisesRegex(RuntimeError, 'contradiction'):
            self.resolve()

    def test_tied_depth_team_conflict(self):
        self.db.execute("INSERT INTO depth_charts SELECT gsis_id,'SEA',player_name,pos_abb,pos_name,snapshot_dt,updated_at FROM depth_charts LIMIT 1")
        with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
            self.resolve()

    def test_depth_position_conflict(self):
        self.db.execute("UPDATE depth_charts SET pos_abb='TE' WHERE pos_abb='QB'")
        with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
            self.resolve()

    def test_old_depth_does_not_replace_current_absence(self):
        self.db.execute("UPDATE depth_charts SET snapshot_dt='2026-09-29T00:00:00Z' WHERE pos_abb='QB'")
        with self.assertRaisesRegex(RuntimeError, 'missing or ambiguous'):
            self.resolve()

    def test_secondary_team_mismatch(self):
        self.secondary.loc[0, 'team'] = 'SEA'
        with self.assertRaisesRegex(RuntimeError, 'team mismatch'):
            c._wfs_v1_build_consensus(self.structured, self.secondary, self.resolve(), 2026, 5)

    def test_gav2_forecast_team_mismatch(self):
        base = c._wfs_v1_build_consensus(self.structured, self.secondary, self.resolve(), 2026, 5)
        self.forecast.loc[0, 'team'] = 'SEA'
        with self.assertRaisesRegex(RuntimeError, 'contradiction'):
            self.expand(base)


if __name__ == '__main__':
    unittest.main()
