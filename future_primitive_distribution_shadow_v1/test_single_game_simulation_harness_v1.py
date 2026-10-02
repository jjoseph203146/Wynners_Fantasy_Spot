"""Bounded integration tests: only seeds 0 and 1; no distribution experiment."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from simulator_recursive_state_replay_v1.fixtures import sample
from . import single_game_simulation_harness_v1 as h


def fixture():
    contract = h.replay_contract()
    a = sample(contract)
    b = copy.deepcopy(a)
    b['context'].update(team='B', opponent='A')
    return dict(game_identity=dict(season=2025, week=1, game_id='G2', team_1='A', team_2='B'),
                simulation_context=dict(kickoff='2025-01-05T17:00:00Z', completion='2025-01-05T20:00:00Z',
                                        simulated_at='2025-01-05T21:00:00Z', cutoff='2025-01-12T16:00:00Z'),
                team_1_generator_input=h.replay.build(a, contract),
                team_2_generator_input=h.replay.build(b, contract), seed=0)


def resign(state):
    state['audit']['result_hash'] = h.replay.digest({k:v for k,v in state.items() if k != 'audit'})


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input = fixture()
        actual = h.paired.sample_paired_residual
        cls.captured = []
        def capture(*args, **kwargs):
            output = actual(*args, **kwargs)
            cls.captured.append(copy.deepcopy(output))
            return output
        with patch.object(h.paired, 'sample_paired_residual', side_effect=capture) as sampler:
            cls.result = h.simulate_single_game(cls.input)
            cls.initial_sampler_calls = sampler.call_count
        cls.sample = cls.captured[0]

    def invalid_input(self, change, expected=ValueError):
        data = copy.deepcopy(self.input)
        change(data)
        with patch.object(h.paired, 'sample_paired_residual') as sampler:
            with self.assertRaises(expected): h.simulate_single_game(data)
            sampler.assert_not_called()

    def invalid_sample(self, change):
        result = copy.deepcopy(self.sample)
        change(result)
        with patch.object(h.paired, 'sample_paired_residual', return_value=result) as sampler:
            with self.assertRaises(ValueError): h.simulate_single_game(self.input)
            self.assertEqual(sampler.call_count, 1)

    def test_valid_execution(self):
        self.assertEqual(self.result['harness_version'], h.HARNESS_VERSION)
        self.assertEqual(self.result['game_identity'], self.input['game_identity'])
        self.assertEqual(self.result['simulation_context'], self.input['simulation_context'])
        self.assertEqual(self.result['historical_temporal_provenance'], 'UNVERIFIED')

    def actual_generator(self, name):
        originals = (h.intelligence.estimate_team_volume, h.intelligence.estimate_team_yardage,
                     h.scoring.estimate_team_scoring)
        with patch.object(h.intelligence, 'estimate_team_volume', wraps=originals[0]) as volume, \
             patch.object(h.intelligence, 'estimate_team_yardage', wraps=originals[1]) as yardage, \
             patch.object(h.scoring, 'estimate_team_scoring', wraps=originals[2]) as scoring, \
             patch.object(h.paired, 'sample_paired_residual', return_value=copy.deepcopy(self.sample)):
            result = h.simulate_single_game(self.input)
        idx = 0 if name == 'team_1' else 1
        for mock in (volume, yardage, scoring):
            self.assertEqual(mock.call_count, 2)
            self.assertEqual(mock.call_args_list[idx].args[0], self.input[name+'_generator_input'])
        self.assertEqual(result[name]['generator_output']['scoring']['team'], self.input['game_identity'][name])

    def test_actual_generator_team_1(self): self.actual_generator('team_1')
    def test_actual_generator_team_2(self): self.actual_generator('team_2')

    def test_no_duplicated_generator_formulas(self):
        tree = ast.parse(inspect.getsource(h._generate))
        self.assertFalse(any(isinstance(n, ast.BinOp) for n in ast.walk(tree)))
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertEqual(calls, ['estimate_team_volume','estimate_team_yardage','estimate_team_scoring','_validate_center','items'])

    def mapping(self, name):
        t = self.result[name]; g = t['generator_output']
        expected = dict(expected_plays=g['volume']['offensive_plays'],expected_pass_rate=g['volume']['pass_rate'],
                        expected_pass_ypa=g['yardage']['passing_yards_per_attempt'],expected_rush_ypc=g['yardage']['rushing_yards_per_carry'],
                        expected_passing_tds=g['scoring']['passing_tds'],expected_rushing_tds=g['scoring']['rushing_tds'],
                        expected_points=g['scoring']['points'])
        self.assertEqual(t['sampler_center'],expected)
        self.assertEqual({k:g['scoring'][k] for k in (*h.PRIMITIVES,'points')},
                         h.scoring.generate_team_primitives(self.input[name+'_generator_input']))

    def test_mapping_team_1(self): self.mapping('team_1')
    def test_mapping_team_2(self): self.mapping('team_2')

    def generator_unmutated(self, name):
        generated = []; original = h._generate
        def capture(state):
            output, center = original(state)
            generated.append((output,copy.deepcopy(output)))
            return output,center
        with patch.object(h,'_generate',side_effect=capture), \
             patch.object(h.paired,'sample_paired_residual',return_value=copy.deepcopy(self.sample)):
            result = h.simulate_single_game(self.input)
        idx = 0 if name=='team_1' else 1
        self.assertEqual(*generated[idx])
        result[name]['generator_output']['scoring']['points']=999
        self.assertEqual(*generated[idx])

    def test_generator_output_not_mutated_team_1(self): self.generator_unmutated('team_1')
    def test_generator_output_not_mutated_team_2(self): self.generator_unmutated('team_2')
    def test_single_sampler_call(self): self.assertEqual(self.initial_sampler_calls,1)
    def test_explicit_seed_forwarded(self):
        with patch.object(h.paired,'sample_paired_residual',return_value=copy.deepcopy(self.sample)) as sampler:
            h.simulate_single_game(self.input)
        sampler.assert_called_once_with(self.result['team_1']['sampler_center'],self.result['team_2']['sampler_center'],seed=0)
    def test_same_input_same_seed(self): self.assertEqual(h.simulate_single_game(self.input),self.result)
    def test_different_seed(self):
        data=copy.deepcopy(self.input); data['seed']=1
        other=h.simulate_single_game(data)
        self.assertNotEqual(other['selected_historical_game'],self.result['selected_historical_game'])
    def test_same_selected_game(self):
        for name in ('team_1','team_2'):
            self.assertEqual(self.result[name]['selected_historical_residual_identity']['game_id'],self.result['selected_historical_game']['game_id'])
    def test_pair_integrity(self): self.assertEqual(self.result['pair_integrity'],h.PAIR_INTEGRITY)
    def test_joint_vectors(self):
        rows,_=h.seeded._load_bank(h.seeded.BANK_PATH)
        for name in ('team_1','team_2'):
            t=self.result[name]; row=rows[t['selected_historical_residual_identity']['row_index']]
            self.assertEqual(t['residuals'],{k:row[k] for k in h.seeded.RESIDUAL_FIELDS})
    def test_orientation_preserved(self): self.assertEqual(self.result['orientation'],self.sample['orientation'])
    def volume(self,name):
        t=self.result[name]
        self.assertEqual(t['realized_primitives']['pass_attempts']+t['realized_primitives']['carries'],t['realized_mechanisms']['plays'])
    def test_team_1_volume(self): self.volume('team_1')
    def test_team_2_volume(self): self.volume('team_2')
    def nonnegative(self,fields):
        for name in ('team_1','team_2'):
            t=self.result[name]; values={**t['realized_primitives'],**t['realized_mechanisms']}
            for key in fields: self.assertGreaterEqual(values[key],0)
    def test_nonnegative_opportunities(self): self.nonnegative(('plays','pass_attempts','carries'))
    def test_nonnegative_yards(self): self.nonnegative(('passing_yards','rushing_yards'))
    def test_nonnegative_tds(self): self.nonnegative(('passing_tds','rushing_tds'))
    def test_nonnegative_efficiencies(self): self.nonnegative(('pass_ypa','rush_ypc'))
    def test_reconciliation_team_1(self): self.assertEqual(self.result['team_1']['reconciliation'],self.sample['team_1']['reconciliation'])
    def test_reconciliation_team_2(self): self.assertEqual(self.result['team_2']['reconciliation'],self.sample['team_2']['reconciliation'])
    def test_no_points_sampling(self):
        for name in ('team_1','team_2'):
            self.assertEqual(set(self.result[name]['realized_primitives']),h.PRIMITIVES)
            self.assertEqual(set(self.result[name]['residuals']),set(h.seeded.RESIDUAL_FIELDS))
            self.assertNotIn('points',self.result[name]['realized_mechanisms'])
    def test_points_diagnostic_only(self):
        original=h.scoring.estimate_team_scoring
        def changed(*args):
            out=original(*args); out['points']+=100
            return out
        with patch.object(h.scoring,'estimate_team_scoring',side_effect=changed): other=h.simulate_single_game(self.input)
        for name in ('team_1','team_2'):
            self.assertEqual(other[name]['sampler_center']['expected_points'],self.result[name]['sampler_center']['expected_points']+100)
            for key in ('residuals','realized_mechanisms','realized_primitives','reconciliation'):
                self.assertEqual(other[name][key],self.result[name][key])
    def test_no_final_score(self): self.assertNotIn('final_score',json.dumps(self.result))
    def test_no_fantasy_points(self): self.assertNotIn('fantasy',json.dumps(self.result))
    def test_invalid_seed(self):
        for value in (None,True,False,1.,'0',[0,1]): self.invalid_input(lambda d:d.update(seed=value))
    def test_same_teams(self): self.invalid_input(lambda d:d['game_identity'].update(team_2='A'))
    def test_missing_game_identity(self): self.invalid_input(lambda d:d.pop('game_identity'))
    def test_invalid_game_identity(self):
        for field,value in (('season',True),('week',0),('team_1',''),('team_2',' '),('game_id',' G2 ')):
            self.invalid_input(lambda d:d['game_identity'].update({field:value}))
    def test_missing_generator_input(self):
        for name in ('team_1','team_2'): self.invalid_input(lambda d:d.pop(name+'_generator_input'))
    def test_missing_required_feature(self):
        def change(d,name):
            state=d[name+'_generator_input']
            state['feature_reconstruction']=[r for r in state['feature_reconstruction'] if r['name']!='team_pass_rate_avg_3']
            resign(state)
        for name in ('team_1','team_2'):
            self.invalid_input(lambda d:change(d,name),h.intelligence.PrimitiveIntelligenceError)
    def test_nonfinite_feature(self):
        for name in ('team_1','team_2'):
            for value in (float('nan'),float('inf'),-float('inf')):
                self.invalid_input(lambda d:d[name+'_generator_input']['feature_reconstruction'][0].update(value=value))
    def test_generator_failure(self):
        error=h.intelligence.PrimitiveIntelligenceError('intentional')
        with patch.object(h.intelligence,'estimate_team_volume',side_effect=error),patch.object(h.paired,'sample_paired_residual') as sampler:
            with self.assertRaises(h.intelligence.PrimitiveIntelligenceError) as caught: h.simulate_single_game(self.input)
            self.assertIs(caught.exception,error); sampler.assert_not_called()
    def test_sampler_failure_no_retry(self):
        error=h.seeded.SamplerError('intentional')
        with patch.object(h.paired,'sample_paired_residual',side_effect=error) as sampler:
            with self.assertRaises(h.seeded.SamplerError) as caught: h.simulate_single_game(self.input)
            self.assertIs(caught.exception,error); self.assertEqual(sampler.call_count,1)
    def test_invalid_paired_output(self):
        for value in (None,{},[],{'team_1':{}}):
            with patch.object(h.paired,'sample_paired_residual',return_value=value) as sampler:
                with self.assertRaises(ValueError): h.simulate_single_game(self.input)
                self.assertEqual(sampler.call_count,1)
    def test_wrong_bank_hash_output(self): self.invalid_sample(lambda r:r.update(residual_bank_sha256='WRONG'))
    def test_wrong_bank_bytes_actual_sampler(self):
        original=Path.read_bytes
        def read(path):
            data=original(path)
            return data+b'\n' if path==h.seeded.BANK_PATH else data
        with patch.object(Path,'read_bytes',read),patch.object(h.paired,'sample_paired_residual',wraps=h.paired.sample_paired_residual) as sampler:
            with self.assertRaisesRegex(h.seeded.SamplerError,'hash mismatch'): h.simulate_single_game(self.input)
            self.assertEqual(sampler.call_count,1)
    def test_post_simulation_volume_failure(self): self.invalid_sample(lambda r:r['team_1']['realized_primitives'].update(carries=999))
    def test_post_simulation_negative_yards(self): self.invalid_sample(lambda r:r['team_2']['realized_primitives'].update(passing_yards=-1))
    def test_post_simulation_negative_tds(self): self.invalid_sample(lambda r:r['team_1']['realized_primitives'].update(rushing_tds=-1))
    def test_global_rng_untouched(self):
        before=random.getstate(); h.simulate_single_game(self.input); self.assertEqual(before,random.getstate())
    def protected(self,match):
        base=Path(h.__file__).parent
        before=json.loads((base/'SINGLE_GAME_SIMULATION_HARNESS_V1_FROZEN_HASHES_BEFORE.json').read_text())
        selected={k:v for k,v in before.items() if match(k)}
        self.assertTrue(selected)
        for p,digest in selected.items(): self.assertEqual(hashlib.sha256((base.parent/p).read_bytes()).hexdigest(),digest,p)
    def test_frozen_paired(self): self.protected(lambda p:'historical_game_paired_residual_sampler_v1.py' in p)
    def test_frozen_seeded(self): self.protected(lambda p:'seeded_joint_residual_sampler_v1.py' in p)
    def test_frozen_bank(self): self.protected(lambda p:p.endswith('RESIDUAL_BANK_V1.csv'))
    def test_frozen_calibration(self): self.protected(lambda p:p.endswith('CALIBRATION_V1.json'))
    def test_frozen_generator(self): self.protected(lambda p:p.startswith('simulator_primitive_generator_v1/'))
    def test_frozen_replay(self): self.protected(lambda p:p.startswith('simulator_recursive_state_replay_v1/'))
    def test_all_existing_shadow_files_unchanged(self): self.protected(lambda p:p.startswith('future_primitive_distribution_shadow_v1/'))
    def test_input_not_mutated(self):
        data=copy.deepcopy(self.input); before=copy.deepcopy(data)
        with patch.object(h.paired,'sample_paired_residual',return_value=copy.deepcopy(self.sample)): result=h.simulate_single_game(data)
        self.assertEqual(data,before)
        result['game_identity']['team_1']='OTHER'
        self.assertEqual(data,before)
    def test_second_stochastic_call_forbidden(self):
        gate=h._OneShotSampler()
        with patch.object(h.paired,'sample_paired_residual',return_value={}) as sampler:
            gate.call({}, {}, 0)
            with self.assertRaisesRegex(h.HarnessError,'more than one'): gate.call({}, {}, 0)
            self.assertEqual(sampler.call_count,1)
    def test_failed_call_consumes_budget(self):
        gate=h._OneShotSampler()
        with patch.object(h.paired,'sample_paired_residual',side_effect=h.seeded.SamplerError('fail')) as sampler:
            with self.assertRaises(h.seeded.SamplerError): gate.call({}, {}, 0)
            with self.assertRaises(h.HarnessError): gate.call({}, {}, 1)
            self.assertEqual(sampler.call_count,1)
    def test_repeated_simulation_parameter_rejected(self): self.invalid_input(lambda d:d.update(num_simulations=2))
    def test_invalid_chronology(self):
        for changes in ({'simulated_at':'2025-01-05T19:00:00Z'}, {'cutoff':'2025-01-05T20:30:00Z'},
                        {'completion':'2025-01-05T17:00:00Z'}, {'simulated_at':'2025-01-05T21:00:00'},
                        {'completion':None}, {'kickoff':'invalid'}):
            self.invalid_input(lambda d:d['simulation_context'].update(changes))
    def test_missing_chronology(self): self.invalid_input(lambda d:d['simulation_context'].pop('completion'))
    def test_future_evidence_rejected(self):
        def change(d):
            s=d['team_1_generator_input']; s['state_manifest']['context']['evidence_cutoff']='2025-01-05T17:00:00Z'; resign(s)
        self.invalid_input(change)
    def test_state_identity_mismatch(self):
        def change(d):
            s=d['team_2_generator_input']; s['state_manifest']['context']['team']='A'; resign(s)
        self.invalid_input(change)
    def test_state_hash_mismatch(self): self.invalid_input(lambda d:d['team_1_generator_input']['audit'].update(result_hash='WRONG'))
    def test_unvalidated_state(self): self.invalid_input(lambda d:d['team_1_generator_input']['transition_validation'].update(status='TRANSITION_BLOCKED'))
    def test_wrong_feature_contract(self): self.invalid_input(lambda d:d['team_2_generator_input']['state_manifest'].update(feature_contract_hash='WRONG'))
    def test_duplicate_feature(self):
        def change(d):
            s=d['team_1_generator_input']; s['feature_reconstruction'].append(copy.deepcopy(s['feature_reconstruction'][0])); resign(s)
        self.invalid_input(change)
    def test_missing_center_field(self):
        original=h.intelligence.estimate_team_volume
        def bad(state):
            out=original(state); del out['pass_rate']; return out
        with patch.object(h.intelligence,'estimate_team_volume',side_effect=bad),patch.object(h.paired,'sample_paired_residual') as sampler:
            with self.assertRaises(h.HarnessError): h.simulate_single_game(self.input)
            sampler.assert_not_called()
    def test_nonfinite_center(self):
        original=h.intelligence.estimate_team_yardage
        def bad(*args):
            out=original(*args); out['passing_yards_per_attempt']=float('inf'); return out
        with patch.object(h.intelligence,'estimate_team_yardage',side_effect=bad),patch.object(h.paired,'sample_paired_residual') as sampler:
            with self.assertRaises(h.seeded.SamplerError): h.simulate_single_game(self.input)
            sampler.assert_not_called()
    def test_pair_flag_failure(self): self.invalid_sample(lambda r:r['pair_integrity'].update(historical_same_game=False))
    def test_cross_game_vector_mixing(self): self.invalid_sample(lambda r:r['team_1']['residuals'].update(resid_plays=999))
    def test_orientation_mismatch(self): self.invalid_sample(lambda r:r.update(orientation=1-r['orientation']))
    def test_invalid_orientation(self): self.invalid_sample(lambda r:r.update(orientation=True))
    def test_changed_reconciliation(self): self.invalid_sample(lambda r:r['team_2'].update(reconciliation=[]))
    def test_points_injection(self): self.invalid_sample(lambda r:r['team_1']['realized_primitives'].update(points=24))
    def test_changed_center(self): self.invalid_sample(lambda r:r['team_2']['center'].update(expected_points=999))
    def test_changed_seed(self): self.invalid_sample(lambda r:r.update(seed=1))
    def test_changed_authorization(self): self.invalid_sample(lambda r:r['authorization'].update(monte_carlo=True))
    def test_no_rng_in_harness(self):
        tree=ast.parse(inspect.getsource(h))
        self.assertFalse(any(isinstance(n,ast.Import) and any(x.name in ('random','numpy') for x in n.names) for n in ast.walk(tree)))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='sample_paired_residual']
        self.assertEqual(len(calls),1)
    def test_authorization(self):
        self.assertEqual(self.result['authorization'],dict(stochastic_calls=1,monte_carlo=False,repeated_game_simulation=False,production_influence='NONE',fanduel_solver_influence='NONE'))
    def test_no_inferred_home_away(self):
        self.assertNotIn('home_team',self.result['game_identity']); self.assertNotIn('away_team',self.result['game_identity'])
    def test_explicit_home_away_preserved(self):
        data=copy.deepcopy(self.input); data['game_identity'].update(home_team='B',away_team='A')
        with patch.object(h.paired,'sample_paired_residual',return_value=copy.deepcopy(self.sample)):
            out=h.simulate_single_game(data)
        self.assertEqual(out['game_identity'],data['game_identity'])
    def test_json_serializable(self): self.assertEqual(json.loads(json.dumps(self.result,allow_nan=False)),self.result)


if __name__=='__main__': unittest.main()
