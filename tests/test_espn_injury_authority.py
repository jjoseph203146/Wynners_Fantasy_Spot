"""Contract tests using synthetic identities and a fixed schedule, without network."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import espn_injury_ingest as e
import injury_consensus as c


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.executescript('''
        CREATE TABLE games(season,week,game_type,game_date,home_team,away_team);
        INSERT INTO games VALUES(2026,3,'REG','2026-09-27','WAS','SEA');
        CREATE TABLE weekly_rosters(season,week,gsis_id,espn_id,team,position);
        INSERT INTO weekly_rosters VALUES(2026,3,'00-0000001','123','WAS','QB');
        CREATE TABLE player_identity(gsis_id,espn_id,source_roster);
        INSERT INTO player_identity VALUES('00-0000001','123',1);
        ''')
        self.payload = dict(timestamp='2026-09-26T04:00:00Z',status='success',season=dict(year=2026,type=2),injuries=[dict(id='28',injuries=[dict(id='99',status='Out',date='2026-09-25T17:00Z',athlete=dict(displayName='Synthetic Player',links=[dict(href='https://www.espn.com/nfl/player/_/id/123/example')],team=dict(id='28',abbreviation='WSH'),position=dict(abbreviation='QB')))])])
        self.structured = pd.DataFrame([dict(season=2026,week=3,game_type='REG',gsis_id='00-0000001',team='WAS',position='QB',player_name='Synthetic Player',report_status='',practice_status='DNP',primary_injury='',secondary_injury='')])

    def tearDown(self):
        self.db.close()

    def parse(self):
        return e.parse_feed(self.payload,self.db,2026,3,now='2026-09-26T04:10:00Z')

    def test_out_blocks_and_practice_preserved(self):
        evidence, stats = self.parse()
        merged = e.merge_structured(self.structured,evidence,2026,3)
        result = c.build_consensus(merged,pd.DataFrame(),pd.DataFrame(),2026,3)
        result = e.attach_lineage(result,evidence)
        self.assertEqual((result.iloc[0].consensus_status,result.iloc[0].injury_gate),('OUT','BLOCK'))
        self.assertEqual(result.iloc[0].consensus_practice_status,'DNP')
        self.assertEqual(result.iloc[0].authoritative_source,e.SOURCE)

    def test_active_never_erases_hard_source(self):
        for status in ['OUT','INACTIVE','DOUBTFUL']:
            self.payload['injuries'][0]['injuries'][0]['status']='Active'
            evidence,_=self.parse()
            self.structured['report_status']=status
            self.assertEqual(e.merge_structured(self.structured,evidence,2026,3).iloc[0].report_status,status)

    def test_active_preserves_roster_hard_block_lineage(self):
        self.payload['injuries'][0]['injuries'][0]['status']='Active'
        evidence,_=self.parse()
        merged=e.merge_structured(self.structured,evidence,2026,3)
        result=c.build_consensus(merged,pd.DataFrame(),pd.DataFrame(),2026,3)
        result['consensus_status']='OUT'
        result['injury_gate']='BLOCK'
        result['authoritative_source']='CURRENT_WEEKLY_ROSTER'
        result=e.attach_lineage(result,evidence)
        self.assertEqual(result.iloc[0].injury_gate,'BLOCK')
        self.assertEqual(result.iloc[0].authoritative_source,'CURRENT_WEEKLY_ROSTER')
        self.assertEqual(result.iloc[0].espn_authority_decision,'PRESERVE_EXISTING_HARD_BLOCK')

    def test_questionable_and_doubtful_policy(self):
        for status, gate in [('Questionable','ALLOW'),('Doubtful','BLOCK'),('Inactive','BLOCK')]:
            self.payload['injuries'][0]['injuries'][0]['status']=status
            evidence,_=self.parse()
            merged=e.merge_structured(self.structured,evidence,2026,3)
            result=c.build_consensus(merged,pd.DataFrame(),pd.DataFrame(),2026,3)
            self.assertEqual(result.iloc[0].injury_gate,gate)

    def test_ambiguous_crosswalk_quarantined(self):
        self.db.execute("INSERT INTO player_identity VALUES('00-0000002','123',1)")
        evidence,stats=self.parse()
        self.assertEqual(stats['ambiguous_rows'],1)
        self.assertEqual(e.merge_structured(self.structured,evidence,2026,3).iloc[0].report_status,'')

    def test_duplicate_gsis_quarantined(self):
        self.payload['injuries'][0]['injuries'] *= 2
        _,stats=self.parse()
        self.assertEqual(stats['duplicate_gsis_rows'],2)
        self.assertEqual(stats['mapped_rows'],0)

    def test_conflicting_athlete_ids(self):
        self.payload['injuries'][0]['injuries'][0]['athlete']['id']='456'
        _,stats=self.parse()
        self.assertEqual(stats['ambiguous_rows'],1)

    def test_team_and_position_mismatch(self):
        for column,value in [('team','SEA'),('position','WR')]:
            self.db.execute(f'UPDATE weekly_rosters SET {column}=?',(value,))
            _,stats=self.parse()
            self.assertEqual(stats['mapped_rows'],0)
            self.db.execute("UPDATE weekly_rosters SET team='WAS',position='QB'")

    def test_current_feed_hard_status_survives_six_day_boundary(self):
        record = self.payload['injuries'][0]['injuries'][0]

        # Current fresh feed still carries an exact-mapped hard status,
        # but the originating record is just outside the normal T-6d
        # target-game window.
        record['status'] = 'Out'
        record['date'] = '2026-09-20T23:59Z'

        evidence, _ = self.parse()
        row = evidence.iloc[0]

        self.assertEqual(row.espn_identity_result, 'MAPPED')
        self.assertTrue(row.recognized)
        self.assertTrue(
            row.applicable,
            'fresh-feed exact-mapped hard status was discarded '
            'solely by the normal record-date applicability boundary'
        )

    def test_old_hard_status_does_not_persist_indefinitely(self):
        record = self.payload['injuries'][0]['injuries'][0]
        record['status'] = 'Out'
        record['date'] = '2026-09-10T17:00Z'

        _, stats = self.parse()
        self.assertEqual(stats['total_relevant_rows'], 0)

    def test_stale_record_unknown_status_and_wrong_season(self):
        record=self.payload['injuries'][0]['injuries'][0]
        record['date']='2026-09-10T17:00Z'
        _,stats=self.parse();self.assertEqual(stats['total_relevant_rows'],0)
        record['date']='2026-09-25T17:00Z';record['status']='Unknown'
        _,stats=self.parse();self.assertEqual(stats['total_relevant_rows'],0)
        self.payload['season']['year']=2025
        with self.assertRaises(RuntimeError):self.parse()

    def test_network_failure_preserves_last_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'data/espn_injuries/last_valid.json';p.parent.mkdir(parents=True);p.write_text('previous evidence')
            failure=e.requests.exceptions.ConnectionError('offline')
            with patch.object(e,'ROOT',root),patch.object(e.requests,'get',side_effect=failure) as get,patch.object(e.time,'sleep') as sleep:
                with self.assertRaises(RuntimeError):e.ingest(self.db,2026,3)
            self.assertEqual(get.call_count,3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list],[5,10])
            self.assertEqual(p.read_text(),'previous evidence')
            self.assertEqual(json.loads((p.parent/'status.json').read_text())['status'],'FAIL_CLOSED')

    def test_transport_timeout_retries_then_recovers(self):
        class Response:
            def raise_for_status(self):
                return None
            def json(self):
                return self_payload

        self_payload=self.payload
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            failures=[
                e.requests.exceptions.Timeout('temporary timeout'),
                e.requests.exceptions.ConnectionError('temporary connection failure'),
                Response(),
            ]
            real_parse_feed=e.parse_feed
            def fixed_parse_feed(payload,conn,season,week):
                return real_parse_feed(payload,conn,season,week,now='2026-09-26T04:10:00Z')
            with patch.object(e,'ROOT',root),patch.object(e.requests,'get',side_effect=failures) as get,patch.object(e.time,'sleep') as sleep,patch.object(e,'parse_feed',side_effect=fixed_parse_feed):
                frame=e.ingest(self.db,2026,3)
            self.assertEqual(get.call_count,3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list],[5,10])
            self.assertFalse(frame.empty)
            self.assertEqual(json.loads((root/'data/espn_injuries/status.json').read_text())['status'],'PASS')

    def test_non_transport_failure_is_not_retried(self):
        class Response:
            def raise_for_status(self):
                return None
            def json(self):
                raise ValueError('invalid json')

        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(e,'ROOT',root),patch.object(e.requests,'get',return_value=Response()) as get,patch.object(e.time,'sleep') as sleep:
                with self.assertRaises(RuntimeError):e.ingest(self.db,2026,3)
            self.assertEqual(get.call_count,1)
            sleep.assert_not_called()
            self.assertEqual(json.loads((root/'data/espn_injuries/status.json').read_text())['status'],'FAIL_CLOSED')


if __name__ == '__main__':
    unittest.main()
