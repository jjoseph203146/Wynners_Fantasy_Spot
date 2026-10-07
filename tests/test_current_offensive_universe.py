"""Independent identity universe: schedule, evidence, and GAV2 integration."""
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import current_offensive_universe as u
import injury_consensus as c


class UniverseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = sqlite3.connect(str(Path(temporary.name) / 'fixture.db'))
        self.addCleanup(self.db.close)
        self.now = datetime(2026, 10, 6, 15, tzinfo=timezone.utc)
        self.db.executescript('''
            CREATE TABLE games(season,week,game_type,game_date,completed,game_id,home_team,away_team);
            INSERT INTO games VALUES(2026,4,'REG','2026-10-05',1,'2026_04_TB_DAL','DAL','TB');
            INSERT INTO games VALUES(2026,5,'REG','2026-10-09',0,'2026_05_TB_DAL','DAL','TB');
            CREATE TABLE weekly_rosters(season,week,gsis_id,full_name,team,position,
                status,status_description_abbr,updated_at);
            CREATE TABLE player_identity(gsis_id,full_name,latest_team,position);
            CREATE TABLE depth_charts(gsis_id,player_name,team,pos_abb,snapshot_dt,updated_at,pos_name);
        ''')
        for gsis, name, team, pos in [('00-0000001','Test QB','TB','QB'),
                                     ('00-0000002','Test WR','DAL','WR')]:
            self.db.execute('INSERT INTO player_identity VALUES(?,?,?,?)', (gsis,name,team,pos))
            self.db.execute('INSERT INTO weekly_rosters VALUES(2026,5,?,?,?,?,?,?,?)',
                            (gsis,name,team,pos,'ACT','A','2026-10-06T14:00:00Z'))
            self.db.execute('INSERT INTO depth_charts VALUES(?,?,?,?,?,?,?)',
                            (gsis,name,team,pos,'2026-10-06T14:00:00Z','2026-10-06T14:00:00Z',pos))
        self.db.commit()

    def resolve(self, season=2026, week=5):
        self.db.commit()  # Existing schedule resolver reads via its own RO connection.
        return u.resolve_current_offensive_universe(self.db, season, week, now=self.now)

    def test_schedule_identity_only_read_only_and_deterministic(self):
        self.db.execute('PRAGMA query_only=ON')
        result = self.resolve()
        pd.testing.assert_frame_equal(result, self.resolve())
        self.assertEqual(list(result.columns), u.COLUMNS)
        self.assertEqual(set(result.game_id), {'2026_05_TB_DAL'})
        self.assertEqual(set(result.team), {'TB','DAL'})
        self.assertTrue(result.season.eq(2026).all())
        self.assertTrue(result.week.eq(5).all())
        self.assertEqual(result.player_id.nunique(), len(result))
        self.assertTrue(result.position.isin(u.POSITIONS).all())

    def test_wrong_week(self):
        with self.assertRaisesRegex(RuntimeError, 'WRONG_PLANNING'):
            self.resolve(week=4)

    def test_wrong_season(self):
        with self.assertRaisesRegex(RuntimeError, 'WRONG_PLANNING'):
            self.resolve(season=2025)

    def test_wrong_game_prefix(self):
        self.db.execute("UPDATE games SET game_id='2026_04_TB_DAL' WHERE week=5")
        with self.assertRaisesRegex(RuntimeError, 'WRONG_WEEK'):
            self.resolve()

    def test_duplicate_identity_fails(self):
        self.db.execute('INSERT INTO player_identity SELECT * FROM player_identity LIMIT 1')
        with self.assertRaisesRegex(RuntimeError, 'DUPLICATE_PLAYER_IDENTITY'):
            self.resolve()

    def test_duplicate_depth_gsis_fails(self):
        self.db.execute('INSERT INTO depth_charts SELECT * FROM depth_charts LIMIT 1')
        with self.assertRaisesRegex(RuntimeError, 'DUPLICATE_DEPTH_GSIS'):
            self.resolve()

    def test_duplicate_roster_fails(self):
        self.db.execute('INSERT INTO weekly_rosters SELECT * FROM weekly_rosters LIMIT 1')
        with self.assertRaisesRegex(RuntimeError, 'DUPLICATE_CURRENT_ROSTER'):
            self.resolve()

    def test_identity_team_contradiction(self):
        self.db.execute("UPDATE player_identity SET latest_team='SEA' WHERE position='QB'")
        with self.assertRaisesRegex(RuntimeError, 'IDENTITY_TEAM'):
            self.resolve()

    def test_complete_roster_team_conflict_excludes_depth_membership(self):
        # Both roster teams are scheduled; unique current roster wins over depth.
        self.db.execute("INSERT INTO weekly_rosters SELECT season,week,'00-9999999',full_name,team,position,status,status_description_abbr,updated_at FROM weekly_rosters WHERE team='TB'")
        self.db.execute("UPDATE weekly_rosters SET team='DAL',status='DEV' WHERE gsis_id='00-0000001'")
        result = self.resolve()
        self.assertNotIn('00-0000001', set(result.player_id))
        self.assertEqual(result.attrs['membership_evidence']['excluded_depth_identities'], [
            dict(player_id='00-0000001',depth_team='TB',roster_team='DAL',
                 reason='CURRENT_ROSTER_TEAM_MISMATCH')])
        self.assertEqual(list(result.columns),u.COLUMNS)

    def test_current_roster_position_explicitly_wins(self):
        self.db.execute("UPDATE player_identity SET position='FB' WHERE latest_team='DAL'")
        self.db.execute("UPDATE depth_charts SET pos_abb='TE' WHERE team='DAL'")
        self.db.execute("UPDATE weekly_rosters SET position='RB' WHERE team='DAL'")
        result = self.resolve()
        self.assertEqual(result.loc[result.team.eq('DAL'),'position'].item(), 'RB')
        self.assertIn('00-0000002', result.attrs['membership_evidence']['roster_position_override_ids'])

    def test_fb_role_admitted_only_with_canonical_roster_position(self):
        self.db.execute("UPDATE depth_charts SET pos_abb='FB' WHERE team='DAL'")
        self.db.execute("UPDATE weekly_rosters SET position='RB' WHERE team='DAL'")
        self.assertEqual(self.resolve().loc[lambda d:d.team.eq('DAL'),'position'].item(), 'RB')

    def test_invalid_roster_position_not_replaced_by_depth(self):
        self.db.execute("UPDATE weekly_rosters SET position='DB' WHERE team='DAL'")
        with self.assertRaisesRegex(RuntimeError, 'INVALID_CURRENT_ROSTER_POSITION'):
            self.resolve()

    def test_whole_week_absence_uses_current_identity_not_history(self):
        self.db.execute("UPDATE weekly_rosters SET week=4,status='OUT',position='DB'")
        result = self.resolve()
        self.assertEqual(result.attrs['membership_evidence']['roster_snapshot'], 'WHOLE_WEEK_ABSENT')
        self.assertEqual(set(result.position), {'QB','WR'})
        self.assertEqual(list(result.columns),u.COLUMNS)

    def test_absent_roster_position_contradiction_fails(self):
        self.db.execute('UPDATE weekly_rosters SET week=4')
        self.db.execute("UPDATE player_identity SET position='TE' WHERE latest_team='DAL'")
        with self.assertRaisesRegex(RuntimeError, 'ABSENT_ROSTER_POSITION_CONTRADICTION'):
            self.resolve()

    def test_partial_week_missing_team_fails(self):
        self.db.execute("DELETE FROM weekly_rosters WHERE team='DAL'")
        with self.assertRaisesRegex(RuntimeError, 'PARTIAL_TARGET_WEEK'):
            self.resolve()

    def test_complete_roster_depth_only_player_excluded_without_fallback(self):
        self.db.execute("UPDATE weekly_rosters SET gsis_id='00-9999999' WHERE team='DAL'")
        result = self.resolve()
        self.assertNotIn('00-0000002', set(result.player_id))
        evidence = result.attrs['membership_evidence']
        self.assertEqual(evidence['roster_snapshot'], 'CURRENT_WEEK_PRESENT')
        self.assertEqual(evidence['excluded_depth_identities'], [
            dict(player_id='00-0000002',depth_team='DAL',roster_team='',
                 reason='ABSENT_FROM_CURRENT_ROSTER')])
        self.assertEqual(list(result.columns),u.COLUMNS)

    def test_contradictory_duplicate_roster_team_fails(self):
        self.db.execute("INSERT INTO weekly_rosters SELECT season,week,gsis_id,full_name,'DAL',position,status,status_description_abbr,updated_at FROM weekly_rosters WHERE team='TB'")
        with self.assertRaisesRegex(RuntimeError, 'DUPLICATE_CURRENT_ROSTER'):
            self.resolve()

    def test_duplicate_roster_outside_depth_still_fails(self):
        for _ in range(2):
            self.db.execute("INSERT INTO weekly_rosters SELECT season,week,'00-9999999',full_name,team,position,status,status_description_abbr,updated_at FROM weekly_rosters WHERE gsis_id='00-0000001'")
        with self.assertRaisesRegex(RuntimeError, 'DUPLICATE_CURRENT_ROSTER'):
            self.resolve()

    def test_invalid_offensive_roster_schedule_team_fails(self):
        self.db.execute("INSERT INTO weekly_rosters SELECT season,week,'00-9999999',full_name,'SEA',position,status,status_description_abbr,updated_at FROM weekly_rosters LIMIT 1")
        with self.assertRaisesRegex(RuntimeError, 'INVALID_CURRENT_ROSTER_SCHEDULE_TEAM'):
            self.resolve()

    def test_malformed_offensive_roster_identity_fails(self):
        self.db.execute("INSERT INTO weekly_rosters SELECT season,week,'not-gsis',full_name,team,position,status,status_description_abbr,updated_at FROM weekly_rosters LIMIT 1")
        with self.assertRaisesRegex(RuntimeError, 'MALFORMED_CURRENT_ROSTER_GSIS'):
            self.resolve()

    def test_duplicate_identity_for_excluded_depth_is_not_ignored(self):
        self.db.execute("UPDATE weekly_rosters SET gsis_id='00-9999999' WHERE team='DAL'")
        self.db.execute("INSERT INTO player_identity SELECT * FROM player_identity WHERE latest_team='DAL'")
        with self.assertRaisesRegex(RuntimeError, 'DUPLICATE_PLAYER_IDENTITY'):
            self.resolve()

    def test_roster_status_never_changes_membership_or_assigns_availability(self):
        baseline = self.resolve()
        for status in ['ACT','DEV','RES','RET','CUT','EXE']:
            with self.subTest(status=status):
                self.db.execute('UPDATE weekly_rosters SET status=?',(status,))
                result = self.resolve()
                pd.testing.assert_frame_equal(result,baseline)
                self.assertEqual(list(result.columns),u.COLUMNS)

    def test_excluded_depth_does_not_require_identity_team_override(self):
        self.db.execute("UPDATE weekly_rosters SET gsis_id='00-9999999' WHERE team='DAL'")
        self.db.execute("UPDATE player_identity SET latest_team='SEA' WHERE gsis_id='00-0000002'")
        result = self.resolve()
        self.assertNotIn('00-0000002',set(result.player_id))

    def test_missing_team_depth_fails(self):
        self.db.execute("DELETE FROM depth_charts WHERE team='DAL'")
        with self.assertRaisesRegex(RuntimeError, 'MISSING_TEAM_DEPTH'):
            self.resolve()

    def test_stale_depth_fails(self):
        self.db.execute("UPDATE depth_charts SET snapshot_dt='2026-10-03T14:00:00Z'")
        with self.assertRaisesRegex(RuntimeError, 'STALE_OR_INVALID'):
            self.resolve()

    def test_future_depth_fails(self):
        self.db.execute("UPDATE depth_charts SET snapshot_dt='2026-10-07T14:00:00Z'")
        with self.assertRaisesRegex(RuntimeError, 'STALE_OR_INVALID'):
            self.resolve()

    def test_named_depth_without_gsis_no_name_match(self):
        self.db.execute("UPDATE depth_charts SET gsis_id='' WHERE team='DAL'")
        with self.assertRaisesRegex(RuntimeError, 'WITHOUT_GSIS'):
            self.resolve()

    def test_vacant_slot_is_not_player_and_old_player_not_carried_forward(self):
        self.db.execute("INSERT INTO depth_charts VALUES('00-0000003','Old TE','DAL','TE','2026-10-05T14:00:00Z','','TE')")
        self.db.execute("INSERT INTO depth_charts VALUES(NULL,NULL,'DAL','TE','2026-10-06T14:00:00Z','','TE')")
        result=self.resolve()
        self.assertEqual(len(result),2)
        self.assertEqual(result.attrs['membership_evidence']['vacant_depth_slots'],1)

    def test_outside_schedule_is_not_member(self):
        self.db.execute("INSERT INTO depth_charts VALUES('00-0000003','Other QB','SEA','QB','2026-10-06T14:00:00Z','','QB')")
        self.assertEqual(len(self.resolve()),2)

    def test_missing_identity_fails(self):
        self.db.execute("DELETE FROM player_identity WHERE latest_team='DAL'")
        with self.assertRaisesRegex(RuntimeError, 'MISSING_EXACT_GSIS'):
            self.resolve()

    def test_gav2_real_resolver_never_reads_numeric_forecast(self):
        self.db.commit()
        structured=pd.DataFrame(columns=['gsis_id','team','player_name','position','report_status','practice_status'])
        base=c._wfs_v1_build_consensus(structured,pd.DataFrame(),pd.DataFrame(),2026,5)
        with patch.object(c,'resolve_current_offensive_universe',
                          side_effect=lambda conn,season,week:u.resolve_current_offensive_universe(conn,season,week,now=self.now)), \
             patch.object(c.pd,'read_parquet',side_effect=AssertionError('numeric forecast read')):
            result=c._gav2_expand_global_consensus(self.db,base,2026,5)
        self.assertEqual(len(result),2)
        self.assertTrue(result.week.eq(5).all())

    def test_gav2_keeps_defensive_wrong_week_guard(self):
        universe=self.resolve()
        universe['game_id']='2026_04_TB_DAL'
        with patch.object(c,'resolve_current_offensive_universe',return_value=universe):
            with self.assertRaisesRegex(RuntimeError,'outside the requested season/week'):
                c._gav2_expand_global_consensus(self.db,pd.DataFrame(),2026,5)


if __name__ == '__main__':
    unittest.main()
