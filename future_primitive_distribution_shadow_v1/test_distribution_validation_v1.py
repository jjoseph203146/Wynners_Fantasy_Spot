"""Focused unit checks plus verification of both complete validation artifacts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from . import distribution_validation_v1 as v


class DistributionUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.targets, cls.authority = v.prepare()
        cls.target = cls.targets[0]
        cls.pairs = {p[0]['game_id']: p for p in cls.authority[0]}
        with v.cached_authority(cls.authority):
            cls.selection = v.temporal.select_temporal_paired_residual(cls.target['game_id'], cls.target['kickoff'], 7)

    def test_seed_derivation(self):
        expected = int.from_bytes(hashlib.sha256(b'["2025_01_DAL_PHI",0]').digest(), 'big')
        self.assertEqual(v.realization_seed('2025_01_DAL_PHI', 0), expected)
        self.assertNotEqual(expected, v.realization_seed('2025_01_DAL_PHI', 1))
        for bad in (-1,100,True,1.5):
            with self.assertRaises(ValueError): v.realization_seed('game', bad)

    def test_100_unique_seeds_every_game(self):
        for game in self.targets:
            self.assertEqual(len({v.realization_seed(game['game_id'],i) for i in range(100)}),100)

    def test_strict_before(self):
        v.check_selection(self.selection, self.target, self.authority, self.pairs)
        chronology = self.authority[1]
        self.assertLess(chronology[self.selection['selected_historical_game']['game_id']], chronology[self.target['game_id']])

    def test_same_game_fails_closed(self):
        s = copy.deepcopy(self.selection)
        s['selected_historical_game']['game_id'] = self.target['game_id']
        with self.assertRaisesRegex(ValueError,'SAME_GAME_LEAKAGE'):
            v.check_selection(s,self.target,self.authority,self.pairs)

    def test_future_fails_closed(self):
        s = copy.deepcopy(self.selection)
        s['selected_historical_game']['game_id'] = self.targets[-1]['game_id']
        with self.assertRaisesRegex(ValueError,'STRICT_BEFORE_FAILURE'):
            v.check_selection(s,self.target,self.authority,self.pairs)

    def test_same_kickoff_fails_closed(self):
        authority = (self.authority[0], dict(self.authority[1]), self.authority[2])
        authority[1][self.selection['selected_historical_game']['game_id']] = authority[1][self.target['game_id']]
        with self.assertRaisesRegex(ValueError,'STRICT_BEFORE_FAILURE'):
            v.check_selection(self.selection,self.target,authority,self.pairs)

    def test_pair_and_orientation_corruption(self):
        for field in ('orientation','team_1_residuals','team_1_residual_identity'):
            s = copy.deepcopy(self.selection)
            if field == 'orientation': s[field] = 1-s[field]
            elif field == 'team_1_residuals': s[field]['resid_plays'] += 1
            else: s[field]['team'] = 'INVALID'
            with self.assertRaises(ValueError):
                v.check_selection(s,self.target,self.authority,self.pairs)

    def test_quantiles(self):
        self.assertEqual(v.quantiles([0,10,20,30,40],[.1,.25,.5,.75,.9]).tolist(),[4,10,20,30,36])

    def test_interval_coverage_inclusive(self):
        self.assertEqual(v.interval(10,[0,10,20,30,40],.5),dict(lower=10.,upper=30.,width=20.,covered=True))
        self.assertFalse(v.interval(9,[0,10,20,30,40],.5)['covered'])
        self.assertTrue(v.interval(0,[0]*100,.9)['covered'])

    def test_hash_independent_of_filesystem_order(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/name for name in ('z.json','a.json')]
            for p in paths: p.write_text(json.dumps({'file':p.name}))
            def collect(order): return {p.name:json.loads(p.read_text()) for p in order}
            self.assertEqual(v.sha(v.result_bytes(collect(paths))),v.sha(v.result_bytes(collect(paths[::-1]))))


class CompleteRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.first = json.loads((v.OUTPUT/'run_1.json').read_text())
        cls.second = json.loads((v.OUTPUT/'run_2.json').read_text())
        cls.manifest = json.loads((v.OUTPUT/'manifest.json').read_text())

    def test_repeated_run_determinism(self):
        self.assertEqual(self.first,self.second)
        self.assertEqual(v.sha(v.result_bytes(self.first)),self.manifest['RUN_1_HASH'])
        self.assertEqual(self.manifest['RUN_1_HASH'],self.manifest['RUN_2_HASH'])
        self.assertEqual(self.manifest['DETERMINISM'],'PASS')

    def test_expected_counts_and_order(self):
        for result in (self.first,self.second):
            records = result['records']
            self.assertEqual(len(result['observations']),285)
            self.assertEqual(len(result['intervals']),570)
            self.assertEqual(len(records),28500)
            self.assertEqual(sum(len(r['teams']) for r in records),57000)
            keys = [(r['kickoff'],r['game_id'],r['realization_index']) for r in records]
            self.assertEqual(keys, sorted(keys))
            self.assertEqual(len(set(keys)),28500)
            for offset in range(0,28500,100):
                self.assertEqual([r['realization_index'] for r in records[offset:offset+100]],list(range(100)))

    def test_expected_schema(self):
        self.assertEqual(set(v.PRIMITIVES),{'plays','pass_attempts','carries','passing_yards','rushing_yards','passing_tds','rushing_tds'})
        for record in self.first['records']:
            self.assertEqual(sorted(t['is_home'] for t in record['teams']),[False,True])
            for team in record['teams']:
                self.assertEqual(set(team['simulated']),set(v.PRIMITIVES))
                self.assertEqual(team['simulated']['plays'],team['simulated']['pass_attempts']+team['simulated']['carries'])

    def test_all_realization_temporal_integrity(self):
        for record in self.first['records']:
            self.assertNotEqual(record['game_id'],record['historical_game_id'])
            self.assertLess(record['historical_kickoff'],record['kickoff'])
            for team in record['teams']:
                self.assertEqual(team['residual_identity']['game_id'],record['historical_game_id'])

    def test_aggregate_independent_of_record_order(self):
        summary, intervals = v.summarize(self.first['observations'],self.first['records'][::-1])
        self.assertEqual(summary,self.first['summary'])
        self.assertEqual(intervals,self.first['intervals'])

    def test_frozen_files_and_scope(self):
        for name,digest in self.manifest['PROTECTED_SHA256'].items():
            self.assertEqual(v.sha((v.bounded.ROOT/name).read_bytes()),digest,name)
        self.assertFalse(self.manifest['TUNING_PERFORMED'])
        self.assertFalse(self.manifest['MONTE_CARLO_AUTHORIZED'])
        self.assertEqual(self.manifest['PRODUCTION_INFLUENCE'],'NONE')
        self.assertTrue(self.manifest['GLOBAL_RNG_UNTOUCHED'])


if __name__ == '__main__': unittest.main()
