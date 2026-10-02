import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .core import build,SAFETY
from .fixtures import sample,child,golden_reference
from . import io

class ReplayTests(unittest.TestCase):
    def setUp(self): self.c=io.contract();self.d=sample(self.c)
    def run_state(self): return build(self.d,self.c)
    def f(self,n): return next(r for r in self.run_state()['feature_reconstruction'] if r['name']==n)
    def reference(self):
        self.d['reference']=golden_reference(self.c)
    def test_mapping72(self): self.assertEqual({r['name'] for r in self.run_state()['feature_reconstruction']},set(self.c['margin']))
    def test_mapping64(self): self.assertEqual(len(self.run_state()['total_features']),64)
    def test_missing20_inventory(self):
        rows=self.run_state()['missing_feature_status'];self.assertEqual({r['name'] for r in rows},set(self.c['excluded']))
        self.assertTrue(all(r['value'] is None and r['status']=='UNRESOLVED' for r in rows))
    def test_rest(self): self.assertEqual(self.f('team_rest')['value'],7)
    def test_cutoff(self):
        self.d['context']['evidence_cutoff']=self.d['context']['kickoff']
        with self.assertRaisesRegex(ValueError,'LEAKAGE'): self.run_state()
    def test_future_leakage(self):
        self.d['bootstrap'][1]['observation_time']='2025-01-06T00:00:00Z'
        self.assertEqual(self.run_state()['transition_validation']['status'],'TRANSITION_BLOCKED')
    def test_completion_boundary(self):
        self.d['bootstrap'][0]['observation_time']='2024-12-22T19:00:00Z'
        self.assertIn('LEAKAGE',str(self.run_state()['transition_validation']))
    def test_zero_unknown(self):
        self.assertEqual(self.f('opp_rushing_tds_avg_3')['semantic_state'],'KNOWN_ZERO')
        self.d['bootstrap'][1]['teams'][1]['rushing_tds']=None
        self.assertEqual(self.f('opp_rushing_tds_avg_3')['semantic_state'],'UNKNOWN')
    def test_reference_semantics(self):
        self.reference();self.d['reference']['features']['opp_rushing_tds_avg_3']=dict(value=0,semantic_state='NOT_APPLICABLE')
        row=next(r for r in self.run_state()['replay_comparison'] if r['feature']=='opp_rushing_tds_avg_3')
        self.assertEqual(row['classification'],'STATE_TRANSITION_GAP')
    def test_arithmetic_independent(self):
        expected={'team_points_for_avg_3':21,'team_passing_yards_avg_3':205,'team_history_games':2,
                  'coach_prior_win_pct':1,'team_opponent_pass_yards_allowed_avg_3':205,
                  'opp_opponent_pass_yards_allowed_avg_3':150,'team_offensive_plays_avg_3':50.5}
        for n,v in expected.items():self.assertAlmostEqual(self.f(n)['value'],v)
    def test_comparison_pass(self):
        self.reference();self.assertTrue(all(r['classification']=='PASS' for r in self.run_state()['replay_comparison']))
    def test_tolerance(self):
        self.reference();self.d['reference']['features']['team_rest']+=1e-10
        self.assertTrue(all(r['classification']=='PASS' for r in self.run_state()['replay_comparison']))
        self.d['reference']['features']['team_rest']+=.01
        self.assertIn('NUMERIC_TOLERANCE',str(self.run_state()['replay_comparison']))
    def test_source_version(self):
        self.reference();self.d['reference']['source_hash']='wrong'
        self.assertEqual(self.run_state()['replay_comparison'][0]['classification'],'SOURCE_VERSION_MISMATCH')
    def test_identity_mismatch(self):
        self.d['context']['opponent']='C'
        with self.assertRaisesRegex(ValueError,'IDENTITY_MISMATCH'):self.run_state()
    def test_duplicate_game(self):
        self.d['bootstrap'].append(copy.deepcopy(self.d['bootstrap'][0]))
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):self.run_state()
    def test_team_binding(self):
        self.d['bootstrap'][0]['teams'][1]['team']='C'
        with self.assertRaisesRegex(ValueError,'IDENTITY'):self.run_state()
    def test_player_rows_definition(self): self.assertIn('SOURCE ROWS',next(r for r in self.run_state()['missing_feature_status'] if r['name']=='team_player_rows')['required_source'])
    def test_rolling_transition(self):
        self.d=child(self.d,self.run_state());self.assertAlmostEqual(self.f('team_points_for_avg_3')['value'],64/3)
    def test_season_continuity(self):self.assertEqual(self.f('team_history_games')['value'],2)
    def test_score_only_partial(self):
        self.d=child(self.d,self.run_state(),score_only=True)
        self.assertEqual(self.run_state()['transition_validation']['status'],'TRANSITION_PARTIAL')
        self.assertIsNone(self.f('team_pass_attempts_avg_3')['value'])
        self.assertIsNotNone(self.f('team_points_for_avg_3')['value'])
    def test_rich_transition(self):
        self.d=child(self.d,self.run_state());self.assertEqual(self.run_state()['transition_validation']['status'],'TRANSITION_COMPLETE')
    def test_missing_primitive(self):
        del self.d['bootstrap'][0]['teams'][0]['passing_yards'];self.assertIsNone(self.f('team_passing_yards_avg_3')['value'])
    def test_simulated_branch(self):
        self.d=child(self.d,self.run_state(),simulated=True);self.assertEqual(self.run_state()['transition_validation']['status'],'TRANSITION_COMPLETE')
    def test_observed_rejects_simulated(self):
        self.d['bootstrap'][0]['provenance']='SIMULATED'
        with self.assertRaisesRegex(ValueError,'BRANCH'):self.run_state()
    def test_branch_local(self):
        self.d=child(self.d,self.run_state(),simulated=True);self.d['transitions'][0]['scenario_version']='other'
        with self.assertRaisesRegex(ValueError,'BRANCH'):self.run_state()
    def test_future_actual_in_simulated(self):
        self.d=child(self.d,self.run_state());self.d['context']['provenance']='SIMULATED'
        self.assertEqual(self.run_state()['transition_validation']['status'],'TRANSITION_BLOCKED')
    def test_parent_immutability(self):
        self.d=child(self.d,self.run_state());before=copy.deepcopy(self.d);self.run_state();self.assertEqual(before,self.d)
    def test_parent_tamper(self):
        self.d=child(self.d,self.run_state());self.d['parent']['lineage']['applied_event_ids']=[]
        with self.assertRaisesRegex(ValueError,'PARENT_MUTATION'):self.run_state()
    def test_determinism(self):
        first=self.run_state()['state_manifest']['state_version'];self.d['bootstrap'].reverse()
        self.assertEqual(first,self.run_state()['state_manifest']['state_version'])
    def test_duplicate_transition(self):
        self.d=child(self.d,self.run_state());self.d['transitions']=[sample(self.c)['bootstrap'][0]]
        with self.assertRaisesRegex(ValueError,'DUPLICATE_TRANSITION'):self.run_state()
    def test_references_only(self):
        before=self.run_state()['feature_reconstruction']
        self.d['intelligence_references']={n:dict(content_hash='synthetic',version='v1',status='NOT_EVALUABLE') for n in ('cold_start_v1','v3_1')}
        self.assertEqual(before,self.run_state()['feature_reconstruction'])
        self.assertFalse(self.run_state()['audit']['historical_availability_proven'])
    def test_production_isolation(self):
        with patch('builtins.open',side_effect=AssertionError('IO')):
            self.assertEqual(self.run_state()['audit']['safety'],SAFETY)
    def test_missing_time_not_evaluable(self):
        self.d['bootstrap'][0]['observation_time']=None
        self.assertEqual(self.run_state()['transition_validation']['status'],'NOT_EVALUABLE')
        self.assertFalse(self.run_state()['audit']['historical_availability_proven'])
    def test_leakage_overrides_numeric_comparison(self):
        self.reference();self.d['bootstrap'][0]['observation_time']='2025-01-06T00:00:00Z'
        self.assertTrue(all(r['classification']=='LEAKAGE' for r in self.run_state()['replay_comparison']))
    def test_zero_denominator(self):
        for e in self.d['bootstrap']:
            e['teams'][0].update(attempts=0,carries=0)
        self.assertEqual(self.f('team_pass_rate_avg_3')['semantic_state'],'KNOWN_ZERO')
    def test_coach_tie(self):
        for e in self.d['bootstrap']:e['teams'][0]['points']=10
        self.assertEqual(self.f('coach_prior_win_pct')['value'],0)
    def test_coach_change(self):
        self.d['coach_assignments']['A']['coach_id']='NEW'
        self.assertEqual(self.f('coach_prior_games')['value'],0)
        self.assertIsNone(self.f('coach_prior_win_pct')['value'])
    def test_contract_excluded_exact(self):
        self.assertEqual(sum('player_rows' in n for n in self.c['excluded']),2)
        self.assertEqual(sum('injury' in n or 'questionable' in n or 'doubtful' in n or 'out_count' in n for n in self.c['excluded']),18)
    def test_publisher(self):
        with tempfile.TemporaryDirectory() as t,patch.object(io,'ROOT',Path(t)/'shadow'):
            path=io.publish(self.d,self.c);self.assertEqual(path,io.publish(self.d,self.c))
    def test_unknown_history(self):
        self.d['history_coverage_complete']=False
        self.assertEqual(self.run_state()['transition_validation']['status'],'NOT_EVALUABLE')
    def test_unknown_rest(self):
        self.d['schedule']=self.d['schedule'][-1:];self.d['bootstrap']=[]
        self.assertIsNone(self.f('team_rest')['value'])

if __name__=='__main__':unittest.main()
