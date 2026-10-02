import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .core import build, SAFETY
from .fixtures import sample, child, quantity
from . import publication

class ReconciliationTests(unittest.TestCase):
    def setUp(self): self.d=sample()
    def result(self): return build(self.d)
    def check(self,name): return next(r for r in self.result()['reconciliation_checks'] if r['check']==name)
    def env(self,name): return self.result()['team_opportunity_envelope'][name]
    def revise(self): self.d=child(self.d,self.result())

    def test_exact_qb(self): self.assertEqual(self.result()['qb_context']['canonical_starter_gsis_id'],'Q')
    def test_ambiguous_identity(self):
        self.d['identity']['players'].append(dict(self.d['identity']['players'][0],gsis_id='Q2'))
        self.assertEqual(self.check('QB_IDENTITY')['status'],'BLOCKED')
    def test_future_input(self):
        self.d['envelopes']['PASS_ATTEMPTS']['available_at']='2025-09-01T13:00:00Z'
        self.assertIsNone(self.env('PASS_ATTEMPTS')['value'])
        self.assertEqual(self.check('INPUT_PASS_ATTEMPTS')['status'],'BLOCKED')
    def test_unknown_participation(self):
        self.d['qb_state']['expected_participation']=None
        self.assertIsNone(self.result()['qb_context']['expected_participation'])
        self.assertEqual(self.check('QB_PARTICIPATION')['status'],'NOT_EVALUABLE')
    def test_out_no_haircut(self):
        self.revise(); self.d['qb_state']['availability']['status']='OUT'
        self.assertEqual(self.env('PASS_ATTEMPTS')['value'],30)
        self.assertIn('UNCHANGED_UNSUPPORTED_CAUSAL_CHANGE',self.result()['scenario_lineage']['signals'])
    def test_supported_revision(self):
        self.revise(); self.d['envelopes']['PASS_ATTEMPTS']=quantity(25,'pass_attempts')
        self.assertEqual(self.env('TARGETS')['value'],23.5)
        self.assertIn('TEAM_PASS_VOLUME_DOWN',self.result()['scenario_lineage']['signals'])
    def test_unsupported_revision(self):
        self.revise(); self.d['envelopes']['PASS_ATTEMPTS']=quantity(20,'pass_attempts')
        self.d['envelopes']['PASS_ATTEMPTS']['authorized']=False
        self.assertIsNone(self.env('PASS_ATTEMPTS')['value'])
    def test_independent_envelopes(self):
        self.assertIsNone(self.env('ROUTES')['value']); self.assertIsNone(self.env('PLAYER_SNAPS')['value'])
        self.assertEqual(self.env('DROPBACKS')['value'],35)
    def test_target_pass(self): self.assertEqual(self.check('PLAYER_TARGETS')['status'],'PASS')
    def test_target_overage(self):
        self.d['player_snapshot']['players'][2]['targets']=25
        self.assertEqual(self.check('PLAYER_TARGETS')['status'],'BLOCKED')
    def test_missing_routes(self): self.assertEqual(self.check('PLAYER_ROUTES')['status'],'NOT_EVALUABLE')
    def test_non_qb_accounting(self): self.assertEqual(self.env('NON_QB_CARRIES')['value'],23)
    def test_qb_separation(self):
        self.assertEqual(self.check('PLAYER_NON_QB_CARRIES')['known_total'],23)
        self.assertEqual(self.check('PLAYER_QB_CARRIES')['known_total'],5)
    def test_qb_increase_cannot_fund_non_qb(self):
        self.revise(); self.d['envelopes']['QB_CARRIES']=quantity(8,'all_rush_attempts')
        self.assertEqual(self.env('NON_QB_CARRIES')['value'],20)
        self.assertEqual(self.check('PLAYER_NON_QB_CARRIES')['status'],'BLOCKED')
    def test_residual(self):
        self.d['player_snapshot']['players'][2]['targets']=18
        r=next(r for r in self.result()['residuals'] if r['metric']=='TARGETS')
        self.assertAlmostEqual(r['residual'],2); self.assertEqual(r['status'],'WARNING')
    def test_negative_residual(self):
        self.d['player_snapshot']['players'][2]['targets']=30
        r=next(r for r in self.result()['residuals'] if r['metric']=='TARGETS')
        self.assertLess(r['residual'],0); self.assertEqual(r['status'],'BLOCKED')
    def test_no_allocation_mutation(self):
        self.revise(); self.d['envelopes']['PASS_ATTEMPTS']=quantity(20,'pass_attempts')
        before=copy.deepcopy(self.d); self.result(); self.assertEqual(before,self.d)
    def test_reallocation_request(self):
        self.revise(); self.d['envelopes']['PASS_ATTEMPTS']=quantity(20,'pass_attempts')
        self.assertEqual(self.result()['scenario_lineage']['reallocation_request']['status'],'REQUEST_ONLY_NO_ALLOCATION')
    def test_parent_immutable(self):
        self.revise(); old=copy.deepcopy(self.d['parent']); self.result(); self.assertEqual(old,self.d['parent'])
    def test_version_determinism(self): self.assertEqual(self.result(),self.result())
    def test_duplicate_revision(self):
        self.revise(); self.d['context']['scenario_version']='s1'
        with self.assertRaisesRegex(ValueError,'DUPLICATE_SCENARIO_REVISION'): self.result()
    def test_parent_hash_tamper(self):
        self.revise(); self.d['parent']['qb_context']['total_carries']=99
        with self.assertRaisesRegex(ValueError,'PARENT_HASH_MISMATCH'): self.result()
    def test_cold_start_mismatch(self):
        self.d['cold_start_reference']=dict(scenario_version='old',input_sha256='abc')
        self.assertEqual(self.check('COLD_START_LINEAGE')['reason'],'BLOCKED_VERSION_MISMATCH')
    def test_cold_reference_unchanged(self):
        self.d['cold_start_reference']={k:self.d['context'][k] for k in ('scenario_version','baseline_version','qb_context_version','evidence_cutoff')}
        self.d['cold_start_reference']['input_sha256']='fixture-hash'
        before=copy.deepcopy(self.d); self.result(); self.assertEqual(before,self.d)
    def test_production_isolation(self):
        self.result()
        with patch('builtins.open',side_effect=AssertionError('IO')),patch('sqlite3.connect',side_effect=AssertionError('DB')),patch('socket.socket',side_effect=AssertionError('NETWORK')):
            self.assertEqual(self.result()['manifest']['safety'],SAFETY)
    def test_incomplete_known_excess(self):
        self.d['player_snapshot']['coverage_complete']=False
        self.d['player_snapshot']['players'][2]['targets']=40
        self.assertEqual(self.check('PLAYER_TARGETS')['status'],'BLOCKED')
    def test_missing_player_quantity(self):
        self.d['player_snapshot']['players'][2]['targets']=None
        self.assertEqual(self.check('PLAYER_TARGETS')['status'],'NOT_EVALUABLE')
    def test_no_metric_equivalence(self):
        self.d['relations']={}
        self.assertEqual(self.check('ATTEMPTS_WITHIN_DROPBACKS')['status'],'NOT_EVALUABLE')
    def test_attempts_dropbacks(self):
        self.d['envelopes']['DROPBACKS']=quantity(20,'dropbacks')
        self.assertEqual(self.check('ATTEMPTS_WITHIN_DROPBACKS')['status'],'BLOCKED')
    def test_snaps_individual_not_sum(self): self.assertEqual(self.check('INDIVIDUAL_SNAPS_WITHIN_PLAYS')['status'],'PASS')
    def test_qb_exceeds_team(self):
        self.d['envelopes']['QB_CARRIES']=quantity(30,'all_rush_attempts')
        self.assertEqual(self.check('RUSH_PARTITION')['status'],'BLOCKED')
    def test_duplicate_players(self):
        self.d['player_snapshot']['players'].append(self.d['player_snapshot']['players'][0])
        with self.assertRaisesRegex(ValueError,'DUPLICATE_PLAYER'): self.result()
    def test_no_post_kickoff(self):
        self.d['context']['evidence_cutoff']=self.d['context']['kickoff']
        with self.assertRaisesRegex(ValueError,'PREGAME_ONLY'): self.result()
    def test_low_confidence_no_revision(self):
        self.d['envelopes']['PASS_ATTEMPTS']['confidence']='COLD_START_INFERENCE'
        self.assertIsNone(self.env('PASS_ATTEMPTS')['value'])
    def test_same_source_version_changed_value(self):
        self.revise(); self.d['envelopes']['PASS_ATTEMPTS']['value']=25
        with self.assertRaisesRegex(ValueError,'CONFLICTING_SOURCE_VERSION'): self.result()
    def test_envelope_version_reuse(self):
        self.revise(); self.d['envelopes']['PASS_ATTEMPTS']=quantity(25,'pass_attempts')
        self.d['context']['opportunity_envelope_version']='e1'
        with self.assertRaisesRegex(ValueError,'REUSED_ENVELOPE_VERSION'): self.result()
    def test_parent_missing(self):
        self.d['context']['parent_scenario_version']='missing'
        with self.assertRaisesRegex(ValueError,'MISSING_PARENT'): self.result()
    def test_absolute_only(self):
        self.d['envelopes']['PASS_ATTEMPTS']['representation']='DELTA'
        self.assertIsNone(self.env('PASS_ATTEMPTS')['value'])
    def test_incompatible_rush_definitions(self):
        self.d['envelopes']['NON_QB_CARRIES']=quantity(23,'different_rushing')
        with self.assertRaisesRegex(ValueError,'INCOMPATIBLE_RUSH_DEFINITION'): self.result()
    def test_wrong_scope(self):
        self.d['envelopes']['PASS_ATTEMPTS']['scope']['team']='NYJ'
        self.assertIsNone(self.env('PASS_ATTEMPTS')['value'])
    def test_player_identity(self):
        self.d['player_snapshot']['players'][1]['gsis_id']='unknown'
        self.assertEqual(self.check('PLAYER_IDENTITY_unknown')['status'],'BLOCKED')
    def test_out_participation_contradiction(self):
        self.d['qb_state']['availability']['status']='OUT'
        self.assertEqual(self.check('UNAVAILABLE_STARTER')['status'],'BLOCKED')
    def test_post_transfer_requires_lineage(self):
        self.d['player_snapshot']['representation']='POST_TRANSFER_ABSOLUTE'
        self.assertEqual(self.check('PLAYER_SNAPSHOT_LINEAGE')['status'],'BLOCKED')
    def test_no_state_change_reported_for_identical_qb(self):
        self.revise()
        self.assertFalse(self.result()['scenario_lineage']['qb_state_changed'])
    def test_routes_explicit_budget(self):
        self.d['envelopes']['ROUTES']=quantity(50,'player_routes')
        self.assertEqual(self.check('PLAYER_ROUTES')['status'],'PASS')
    def test_qb_components_disjoint(self):
        self.d['relations']['QB_COMPONENTS_DISJOINT_SUBSET']=True
        self.d['qb_state']['designed_carries']=quantity(4,'designed_carries')
        self.d['qb_state']['scrambles']=quantity(3,'scrambles')
        self.assertEqual(self.check('QB_COMPONENTS')['status'],'BLOCKED')
    def test_snaps_over_plays(self):
        self.d['player_snapshot']['players'][0]['snaps']=70
        self.assertEqual(self.check('INDIVIDUAL_SNAPS_WITHIN_PLAYS')['status'],'BLOCKED')
    def test_no_rush_component_inference(self):
        self.assertIsNone(self.result()['qb_context']['scrambles'])
        self.assertIsNone(self.result()['qb_context']['designed_carries'])

    def test_publisher_idempotence_and_conflict(self):
        with tempfile.TemporaryDirectory() as t,patch.object(publication,'ROOT',Path(t)/'shadow'):
            p=publication.publish(self.d); self.assertEqual(p,publication.publish(self.d))
            self.d['envelopes']['PASS_ATTEMPTS']=quantity(29,'pass_attempts')
            with self.assertRaisesRegex(ValueError,'IMMUTABLE_SCENARIO_CONFLICT'): publication.publish(self.d)

if __name__=='__main__': unittest.main()
