"""Bounded contract tests: fixed seeds 0/1 and controlled draws, no experiments."""
import copy
import hashlib
import json
from pathlib import Path
import random
import unittest
from unittest.mock import Mock, patch

from . import historical_game_paired_residual_sampler_v1 as s

f = s.frozen
CENTER_1 = dict(zip(f.CENTER_FIELDS, (60, .55, 7, 4, 2, 1, 24)))
CENTER_2 = dict(zip(f.CENTER_FIELDS, (65, .6, 8, 5, 3, 2, 30)))


class PairedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = s.BANK_PATH.read_bytes()
        cls.rows = f._parse_bank(cls.data)
        cls.pairs = s._paired_population(cls.rows)
        cls.result = s.sample_paired_residual(CENTER_1, CENTER_2, seed=0)

    def sample(self, **kwargs):
        return s.sample_paired_residual(kwargs.pop('center_1', CENTER_1),
                                       kwargs.pop('center_2', CENTER_2), **dict(seed=0, **kwargs))

    def bad_rows(self, change):
        rows = copy.deepcopy(self.rows)
        change(rows)
        with self.assertRaises(s.SamplerError):
            s._paired_population(rows)

    def test_frozen_bank_hash(self):
        self.assertEqual(hashlib.sha256(self.data).hexdigest(), s.FROZEN_BANK_SHA256)
        self.assertEqual(self.result['residual_bank_sha256'], s.FROZEN_BANK_SHA256)

    def test_wrong_hash_fails_before_rng(self):
        with patch.object(Path, 'read_bytes', return_value=self.data+b'\n'), patch.object(s.random, 'Random') as rng:
            with self.assertRaises(s.SamplerError): self.sample()
            rng.assert_not_called()

    def test_839_games(self): self.assertEqual(len(self.pairs), 839)
    def test_two_rows(self): self.assertTrue(all(len(p) == 2 for p in self.pairs))
    def test_reciprocal_opponents(self):
        self.assertTrue(all(a['team'] == b['opponent_team'] and b['team'] == a['opponent_team'] for a,b in self.pairs))
    def test_missing_row(self): self.bad_rows(lambda r: r.pop())
    def test_extra_row(self): self.bad_rows(lambda r: r.append(dict(r[0])))
    def test_wrong_population(self): self.bad_rows(lambda r: r[0].update(game_id='EXTRA'))
    def test_same_teams(self):
        team = self.pairs[0][0]['team']
        idx = self.pairs[0][1]['row_index']
        self.bad_rows(lambda r: r[idx].update(team=team))
    def test_nonreciprocal(self): self.bad_rows(lambda r: r[0].update(opponent_team='WRONG'))
    def test_inconsistent_season(self): self.bad_rows(lambda r: r[0].update(season=1))
    def test_inconsistent_week(self): self.bad_rows(lambda r: r[0].update(week=99))
    def test_canonical_order(self):
        self.assertEqual(s._paired_population(list(reversed(self.rows))), self.pairs)
        self.assertEqual([a['game_id'] for a,b in self.pairs], sorted(a['game_id'] for a,b in self.pairs))
    def test_same_seed(self): self.assertEqual(self.result, self.sample())
    def test_different_seed(self):
        other = s.sample_paired_residual(CENTER_1, CENTER_2, seed=1)
        self.assertNotEqual(other['selected_historical_game'], self.result['selected_historical_game'])
    def test_orientation_deterministic(self): self.assertEqual(self.result['orientation'], self.sample()['orientation'])
    def test_same_historical_game(self):
        for team in ('team_1','team_2'):
            self.assertEqual(self.result[team]['selected_historical_residual_identity']['game_id'], self.result['selected_historical_game']['game_id'])
    def vector(self, team):
        result = self.result[team]
        identity = result['selected_historical_residual_identity']
        row = self.rows[identity['row_index']]
        self.assertEqual(identity, {k:row[k] for k in (*f.IDENTITY_FIELDS,'row_index')})
        self.assertEqual(result['residuals'], {k:row[k] for k in f.RESIDUAL_FIELDS})
    def test_team_1_vector(self): self.vector('team_1')
    def test_team_2_vector(self): self.vector('team_2')
    def test_no_cross_team_mixing_both_orientations(self):
        for orientation in (0,1):
            rng = Mock(spec=['random']); rng.random.side_effect = [0, orientation*.5]
            with patch.object(s.random,'Random',return_value=rng): result = self.sample()
            self.assertEqual(result['orientation'],orientation)
            for name,row in zip(('team_1','team_2'),self.pairs[0][::1 if orientation==0 else -1]):
                self.assertEqual(result[name]['residuals'],{k:row[k] for k in f.RESIDUAL_FIELDS})
                self.assertEqual(result[name]['selected_historical_residual_identity']['team'],row['team'])
    def test_exactly_two_local_draws(self):
        rng = Mock(spec=['random']); rng.random.side_effect = [0.0, .5]
        with patch.object(s.random,'Random',return_value=rng) as factory: result = self.sample()
        factory.assert_called_once_with(0)
        self.assertEqual(rng.random.call_count,2)
        self.assertEqual(result['selected_historical_game']['game_id'],self.pairs[0][0]['game_id'])
        self.assertEqual(result['orientation'],1)
    def test_global_rng_untouched(self):
        state=random.getstate(); self.sample(); self.assertEqual(state,random.getstate())
    def unmutated(self, name, center):
        original=copy.deepcopy(center); result=self.sample()
        result[name]['center']['expected_plays']=999
        self.assertEqual(center,original)
    def test_center_1_not_mutated(self): self.unmutated('team_1',CENTER_1)
    def test_center_2_not_mutated(self): self.unmutated('team_2',CENTER_2)
    def volume(self,name):
        t=self.result[name]; p=t['realized_primitives']
        self.assertEqual(p['pass_attempts']+p['carries'],t['realized_mechanisms']['plays'])
    def test_volume_team_1(self): self.volume('team_1')
    def test_volume_team_2(self): self.volume('team_2')
    def nonnegative(self, fields):
        for name in ('team_1','team_2'):
            t=self.result[name]; values={**t['realized_mechanisms'],**t['realized_primitives']}
            for field in fields: self.assertGreaterEqual(values[field],0)
    def test_opportunities(self): self.nonnegative(('plays','pass_attempts','carries'))
    def test_yards(self): self.nonnegative(('passing_yards','rushing_yards'))
    def test_tds(self): self.nonnegative(('passing_tds','rushing_tds'))
    def test_efficiencies(self): self.nonnegative(('pass_ypa','rush_ypc'))
    def audit(self,name):
        t=self.result[name]
        self.assertEqual((t['realized_mechanisms'],t['realized_primitives'],t['reconciliation']),f._realize(t['center'],t['residuals']))
    def test_audit_team_1(self): self.audit('team_1')
    def test_audit_team_2(self): self.audit('team_2')
    def test_reconciliation_edge_cases_both_teams(self):
        for name in ('team_1','team_2'):
            center=dict(CENTER_1,expected_plays=2.5,expected_pass_rate=.5)
            row=dict(self.rows[0],**dict(zip(f.RESIDUAL_FIELDS,(0,1,-99,-99,-99,-99))))
            t=s._team_result(f._validate_center(center),row)
            self.assertEqual(t['realized_mechanisms'],dict(plays=3,pass_rate=1,pass_ypa=0,rush_ypc=0))
            self.assertEqual(t['realized_primitives'],dict(pass_attempts=3,carries=0,passing_yards=0,rushing_yards=0,passing_tds=0,rushing_tds=0))
            self.assertEqual({a['operation'] for a in t['reconciliation']},{'round_half_up','clamp_0_1','floor_zero'})
            self.assertEqual(len(t['reconciliation']),6)
    def test_negative_plays_fails_no_retry(self):
        with self.assertRaisesRegex(s.SamplerError,'negative sampled plays'):
            s._team_result(CENTER_1,dict(self.rows[0],resid_plays=-100))
    def test_no_points(self):
        for name in ('team_1','team_2'):
            t=self.result[name]
            self.assertEqual(set(t['realized_primitives']),{'pass_attempts','carries','passing_yards','rushing_yards','passing_tds','rushing_tds'})
            self.assertNotIn('points',t['realized_mechanisms'])
            self.assertNotIn('resid_points',t['residuals'])
    def test_points_diagnostic(self):
        result=self.sample(center_1=dict(CENTER_1,expected_points=999),center_2=dict(CENTER_2,expected_points=888))
        for name,center in (('team_1',CENTER_1),('team_2',CENTER_2)):
            result[name]['center']['expected_points']=center['expected_points']
        self.assertEqual(result,self.result)
    def test_invalid_seed(self):
        for seed in (None,True,False,1.0,'1'):
            with self.assertRaises(s.SamplerError): s.sample_paired_residual(CENTER_1,CENTER_2,seed=seed)
        with self.assertRaises(s.SamplerError): s.sample_paired_residual(CENTER_1,CENTER_2)
    def test_missing_center_field(self):
        for name,center in (('center_1',CENTER_1),('center_2',CENTER_2)):
            for field in f.CENTER_FIELDS:
                bad=dict(center); del bad[field]
                with self.assertRaises(s.SamplerError): self.sample(**{name:bad})
    def test_nonfinite_center(self):
        for name,center in (('center_1',CENTER_1),('center_2',CENTER_2)):
            for field in f.CENTER_FIELDS:
                for value in (float('nan'),float('inf'),-float('inf')):
                    with self.assertRaises(s.SamplerError): self.sample(**{name:dict(center,**{field:value})})
    def test_invalid_center_domains(self):
        for name in ('center_1','center_2'):
            for changes in ({'expected_plays':-1},{'expected_pass_rate':1.1},{'expected_passing_tds':.5},{'expected_points':True}):
                with self.assertRaises(s.SamplerError): self.sample(**{name:dict(CENTER_1,**changes)})
    def test_missing_residual(self):
        for field in f.RESIDUAL_FIELDS: self.bad_rows(lambda r: r[0].pop(field))
    def test_nonfinite_residual(self):
        for field in f.RESIDUAL_FIELDS:
            for value in (float('nan'),float('inf'),-float('inf')):
                self.bad_rows(lambda r: r[0].update({field:value}))
    def protected(self, selector):
        root=s.BANK_PATH.parent.parent
        hashes=json.loads(s.BANK_PATH.with_name('PAIRED_SAMPLER_V1_FROZEN_HASHES_BEFORE.json').read_text())
        paths=[p for p in hashes if selector(p)]
        self.assertTrue(paths)
        for p in paths: self.assertEqual(hashlib.sha256((root/p).read_bytes()).hexdigest(),hashes[p],p)
    def test_frozen_seeded_sampler(self): self.protected(lambda p:'seeded_joint_residual_sampler_v1.py' in p)
    def test_frozen_bank(self): self.protected(lambda p:p.endswith('RESIDUAL_BANK_V1.csv'))
    def test_frozen_calibration(self): self.protected(lambda p:p.endswith('CALIBRATION_V1.json'))
    def test_frozen_generator(self): self.protected(lambda p:p.startswith('simulator_primitive_generator_v1/'))
    def test_frozen_replay(self): self.protected(lambda p:p.startswith('simulator_recursive_state_replay_v1/'))
    def test_authorization(self):
        self.assertEqual(self.result['authorization'],dict(monte_carlo=False,production_influence='NONE',fanduel_solver_influence='NONE'))
    def test_pair_integrity(self):
        self.assertEqual(self.result['pair_integrity'],dict(historical_same_game=True,historical_reciprocal_opponents=True,joint_vectors_preserved=True))
    def test_json_serializable(self): self.assertEqual(json.loads(json.dumps(self.result,allow_nan=False)),self.result)


if __name__ == '__main__': unittest.main()
