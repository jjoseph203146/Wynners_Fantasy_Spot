"""Synthetic fixtures only; no production files, services or network calls."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from .core import run, STATS, CURRENT, STARTER, MANIFEST, AUDIT, CANDIDATE, EXIT_CODES
from .production import Production, readonly_connect

NOW = datetime(2030, 10, 6, 16, tzinfo=timezone.utc)
GAME = '2030_05_AAA_BBB'


class Fixture:
    root = Path('/synthetic')

    def __init__(self):
        self.available = True
        self.mtime_value = NOW.timestamp()-60
        self.game = dict(game_id=GAME, home_team='AAA', away_team='BBB', mode='PREGAME')
        self.context = dict(season=2030, week=5, games=[self.game], pregame_teams=['AAA','BBB'])
        self.forecast = pd.DataFrame([dict(entity_type='OFFENSE_PLAYER', game_id=GAME, team=t,
             player_id=p, position='QB', **{f: 2.0 for f in STATS}) for t,p in [('AAA','00-0000001'),('BBB','00-0000002')]])
        self.starters = pd.DataFrame([dict(team=t, position='QB', depth_gsis_id=p, rotowire_gsis_id=p,
            verification_status='AGREE', comparable_to_rotowire=True, depth_availability_present=True,
            rw_availability_present=True, depth_injury_gate='ALLOW', rw_injury_gate='ALLOW')
            for t,p in [('AAA','00-0000001'),('BBB','00-0000002')]])
        self.meta = dict(contract='WFS_STARTER_VERIFICATION_CURRENT_V1', source_contract='WFS_STARTER_VERIFICATION_V1_1',
            analysis_only=True, production_influence=False, season=2030, week=5, game_type='REG', identity_gate='PASS',
            identity_unresolved_count=0, identity_ambiguous_count=0, generated_at_utc=NOW.isoformat(), role_rows=2)
        self.reconciliation = self.forecast[['game_id','team','player_id','position']].copy()
        self.reconciliation['primary_qb_id'] = self.reconciliation.player_id
        self.reconciliation['reconciliation_role'] = 'PRIMARY_QB'
        self.observation = dict(context=dict(mode='PREGAME',available=True,source_path=str(self.root/CURRENT)),
            selected=self.forecast.to_dict('records'), used=self.forecast.to_dict('records'))
        self.snapshot_checked = False
        self.racing = False

    def service(self, name): return self.available
    def http_health(self): return self.available
    def storage(self): return dict(status='HEALTHY',reason='STORAGE_READABLE')
    def updater(self, now): return dict(status='HEALTHY',reason='UPDATER_COMPLETED')
    def schedule(self, now): return self.context
    def live(self, context, now): return dict(status='HEALTHY',reason='NO_ACTIVE_GAME_EXPECTED')
    def frame(self, name): return self.forecast.copy() if name == CURRENT else self.starters.copy()
    def mtime(self, name): return self.mtime_value
    def digest(self, name): return 'synthetic-hash'
    def read(self, name): return self.reconciliation.to_csv(index=False).encode()
    def json(self, name):
        if name == MANIFEST: return self.meta
        return dict(version='WFS_OFFENSIVE_TEAM_RECONCILIATION_SHADOW_V3',status='PASS', generated_at_utc=NOW.isoformat(),
            rows=len(self.reconciliation),outputs={'player_shadow':{'sha256':hashlib.sha256(self.read(CANDIDATE)).hexdigest()}})
    def observe(self, game): return copy.deepcopy(self.observation)
    def validate_snapshot(self, game, observation): self.snapshot_checked = True
    def changed(self): return self.racing


class CoreTests(unittest.TestCase):
    def setUp(self): self.f = Fixture()
    def result(self): return run(self.f,NOW)
    def assertReason(self, reason):
        r = self.result()
        self.assertIn(reason,[c['reason_code'] for c in r['checks']])
        self.assertNotEqual(r['overall_status'],'HEALTHY')
        return r
    def test_all_healthy(self):
        r=self.result(); self.assertEqual(r['overall_status'],'HEALTHY')
        self.assertEqual(EXIT_CODES[r['overall_status']],0)
        json.dumps(r,allow_nan=False)
        for c in r['checks']:
            self.assertEqual(set(c),{'check_id','status','reason_code','scope','summary','evidence'})
    def test_app_unavailable(self):
        self.f.available=False
        self.assertEqual(self.assertReason('APP_UNAVAILABLE')['overall_status'],'CRITICAL')
    def test_stale_required_forecast(self):
        self.f.mtime_value=NOW.timestamp()-7201
        self.assertReason('STALE_FORECAST_EVIDENCE')
    def test_stale_starter(self):
        self.f.meta['generated_at_utc']='2030-10-06T13:00:00+00:00'
        self.assertReason('STALE_STARTER_EVIDENCE')
    def test_unresolved_starter(self):
        self.f.meta['identity_unresolved_count']=1
        self.assertReason('UNRESOLVED_STARTER_IDENTITY')
    def test_starter_primary_mismatch(self):
        self.f.reconciliation.loc[0,['player_id','primary_qb_id']]='00-0000003'
        self.assertReason('STARTER_PRIMARY_QB_MISMATCH')
    def test_primary_public_mismatch(self):
        self.f.observation['selected'][0]['player_id']='00-0000003'
        self.assertReason('PRIMARY_PUBLIC_QB_MISMATCH')
    def test_correct_identity_wrong_numbers(self):
        self.f.observation['used'][0]['expected_passing_yards']=301
        self.assertReason('QB_NUMERICAL_FORECAST_MISMATCH')
    def test_wrong_artifact_same_numbers(self):
        self.f.observation['context']['source_path']='/synthetic/data/parquet/current_unified_stat_forecasts.parquet'
        self.assertReason('CONSUMER_SOURCE_MISMATCH')
    def test_duplicate_qb(self):
        self.f.forecast=pd.concat([self.f.forecast,self.f.forecast.iloc[[0]]])
        self.assertReason('DUPLICATE_FORECAST_IDENTITY')
    def test_duplicate_public_qb(self):
        self.f.observation['selected'].append(dict(self.f.observation['selected'][0]))
        self.assertReason('PUBLIC_QB_COVERAGE_OR_DUPLICATE')
    def test_live_postgame_frozen(self):
        for mode in ['LIVE','POSTGAME']:
            with self.subTest(mode=mode):
                self.f=Fixture(); self.f.game['mode']=mode; self.f.context['pregame_teams']=[]
                self.f.observation['context'].update(mode=mode,frozen_at_kickoff=True,source_kind='KICKOFF_SNAPSHOT',
                    source_path=str(self.f.root/f'data/forecast_snapshots/{GAME}/stat_forecast_kickoff.parquet'))
                # Old kickoff QB must not be compared with current starters.
                self.f.observation['selected'][0]['player_id']='00-0000999'
                self.assertEqual(self.result()['overall_status'],'HEALTHY')
                self.assertTrue(self.f.snapshot_checked)
    def test_live_rejects_current_substitution(self):
        self.f.game['mode']='LIVE'; self.f.context['pregame_teams']=[]
        self.f.observation['context']['mode']='LIVE'
        self.assertReason('CONSUMER_SOURCE_MISMATCH')
    def test_precedence(self):
        self.f.available=False; self.f.meta['identity_gate']='FAIL'
        r=self.result(); self.assertEqual(r['overall_status'],'OWNER_ACTION_REQUIRED')
        self.assertEqual(EXIT_CODES[r['overall_status']],30)
        self.assertGreater(r['summary_counts']['CRITICAL'],0)
    def test_unknown_schema(self):
        self.f.forecast=self.f.forecast.drop(columns=['expected_attempts'])
        self.assertReason('REQUIRED_SCHEMA_MISSING')
    def test_exception_fail_closed_and_redacted(self):
        def fail(): raise RuntimeError('SECRET_DO_NOT_PRINT')
        self.f.http_health=fail
        r=self.assertReason('REQUIRED_CHECK_EXCEPTION')
        self.assertNotIn('SECRET_DO_NOT_PRINT',json.dumps(r))
    def test_snapshot_race(self):
        self.f.racing=True
        r=self.assertReason('PRODUCTION_CHANGED_DURING_OBSERVATION')
        self.assertFalse(any(c['check_id']=='qb_invariant' for c in r['checks']))

    def test_nullable_qb_receiving_fields(self):
        for field in STATS[8:]:
            self.f.forecast[field] = float('nan')
            for row in self.f.observation['used']: row[field] = None
        self.assertEqual(self.result()['overall_status'], 'HEALTHY')
    def test_blocked_rotowire_starter_does_not_hide_other_teams(self):
        self.f.starters.loc[0,'rw_injury_gate']='BLOCK'
        r=self.assertReason('ROTOWIRE_STARTER_BLOCKED')
        self.assertTrue(any(c['reason_code']=='QB_IDENTITIES_AND_VALUES_AGREE' and c['scope'].endswith('/BBB') for c in r['checks']))


class AdapterTests(unittest.TestCase):
    def test_sqlite_cannot_write_or_create(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'fixture.db'
            c = sqlite3.connect(path)
            c.execute('CREATE TABLE example (id INTEGER)'); c.commit(); c.close()
            before=path.read_bytes()
            with readonly_connect(path) as c:
                with self.assertRaises(sqlite3.OperationalError): c.execute('INSERT INTO example VALUES (1)')
            self.assertEqual(before,path.read_bytes())
            with self.assertRaises(FileNotFoundError): readonly_connect(Path(temp)/'missing.db')
            self.assertFalse((Path(temp)/'missing.db').exists())
    def test_updater_completion_failure_missing_and_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'logs').mkdir(); path=root/'logs/cron.log'
            start='NFL hourly pipeline launcher started: Sun Oct  6 11:07:01 EDT 2030\n'
            # NOW is noon EDT: current completed hourly run.
            path.write_text(start+'NFL hourly pipeline completed successfully.\nFinished: Sun Oct  6 11:17:10 EDT 2030\n')
            self.assertEqual(Production(root).updater(NOW)['status'],'HEALTHY')
            path.write_text(start+'NFL hourly pipeline exited with code: 45\nFinished: Sun Oct  6 11:17:10 EDT 2030\n')
            self.assertEqual(Production(root).updater(NOW)['reason'],'UPDATER_FAILED')

            # Exit 34 without the explicit post-authority marker is a real failure.
            path.write_text(start+'NFL hourly pipeline exited with code: 34\nFinished: Sun Oct  6 11:17:10 EDT 2030\n')
            ordinary_34 = Production(root).updater(NOW)
            self.assertEqual(ordinary_34['status'], 'CRITICAL')
            self.assertEqual(ordinary_34['reason'], 'UPDATER_FAILED')
            self.assertEqual(ordinary_34['exit_code'], 34)

            # Post-authority LIVE deferral is warning-only, even though it exits 34.
            path.write_text(
                start+
                'NFL hourly pipeline authority completed; downstream analytics deferred for LIVE.\n'
                'NFL hourly pipeline exited with code: 34\n'
                'Finished: Sun Oct  6 11:17:10 EDT 2030\n'
            )
            deferred = Production(root).updater(NOW)
            self.assertEqual(deferred['status'], 'WARNING')
            self.assertEqual(deferred['reason'], 'UPDATER_DOWNSTREAM_DEFERRED_FOR_LIVE')
            self.assertEqual(deferred['exit_code'], 34)

            # A completed launcher invocation deferred by the LIVE lock is warning-only.
            path.write_text(
                start+
                'Updater skipped because shared updater/LIVE lock remained busy for 60 seconds.\n'
                'NFL hourly pipeline deferred because LIVE lock remained busy.\n'
                'Finished: Sun Oct  6 11:17:10 EDT 2030\n'
            )
            lock_busy = Production(root).updater(NOW)
            self.assertEqual(lock_busy['status'], 'WARNING')
            self.assertEqual(lock_busy['reason'], 'UPDATER_DEFERRED_FOR_LIVE')
            self.assertNotIn('exit_code', lock_busy)

            path.write_text(start)
            self.assertEqual(Production(root).updater(NOW)['reason'],'UPDATER_EXCESSIVE_RUNTIME')
    def test_live_idle_active_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'data').mkdir()
            c=sqlite3.connect(root/'data/wfs_live.db')
            c.execute('CREATE TABLE live_ingest_audit (ingest_id INTEGER,event_id TEXT,captured_at_utc TEXT,ingest_status TEXT,ambiguous_occurrences INTEGER,unresolved_occurrences INTEGER)')
            c.execute("INSERT INTO live_ingest_audit VALUES (1,'fixture','2030-10-06T15:50:00+00:00','PASS',0,0)")
            c.commit()
            a=Production(root)
            with patch.object(a,'service',return_value=True):
                self.assertEqual(a.live({'games':[]},NOW)['reason'],'NO_ACTIVE_GAME_EXPECTED')
                context={'games':[dict(game_id=GAME, event_id='fixture', live_expected=True)]}
                self.assertEqual(a.live(context,NOW)['status'],'CRITICAL')
                c.execute("UPDATE live_ingest_audit SET captured_at_utc='2030-10-06T15:59:00+00:00'"); c.commit()
                self.assertEqual(a.live(context,NOW)['status'],'HEALTHY')
                c.execute('UPDATE live_ingest_audit SET unresolved_occurrences=1'); c.commit()
                from .core import Invalid
                with self.assertRaises(Invalid) as raised: a.live(context,NOW)
                self.assertEqual(raised.exception.reason,'LIVE_IDENTITY_UNRESOLVED')
            c.close()
    def test_updater_ignores_inner_finished_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'logs').mkdir()
            (root/'logs/cron.log').write_text('NFL hourly pipeline launcher started: Sun Oct  6 11:57:01 EDT 2030\nFinished: 2030-10-06 11:57:37\n')
            self.assertEqual(Production(root).updater(NOW)['reason'],'UPDATER_RUNNING')
    def test_updater_missed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'logs').mkdir()
            (root/'logs/cron.log').write_text('NFL hourly pipeline launcher started: Sun Oct  6 09:07:01 EDT 2030\nNFL hourly pipeline completed successfully.\nFinished: Sun Oct  6 09:17:10 EDT 2030\n')
            self.assertEqual(Production(root).updater(NOW)['reason'],'UPDATER_MISSED_EXPECTED_RUN')
    def test_write_guard(self):
        import subprocess, sys
        with tempfile.TemporaryDirectory() as temp:
            target=str(Path(temp)/'must_not_exist')
            code='from health_monitor.production import install_write_guard; install_write_guard(); open('+repr(target)+', "w")'
            result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(Path(target).exists())


if __name__ == '__main__': unittest.main()
