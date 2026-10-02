import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .core import build, METRICS, SAFETY
from .fixtures import sample, add_claim
from . import publication


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.data = sample()

    def result(self):
        return build(self.data)

    def player(self, pid, metric='targets'):
        return next(r for r in self.result()['players'] if r['gsis_id'] == pid and r['metric'] == metric)

    def balance(self, pid='D', metric='targets'):
        return next(r for r in self.result()['balances'] if r['donor_gsis_id'] == pid and r['metric'] == metric)

    def set_value(self, table, pid, metric, value):
        next(r for r in self.data[table] if r['gsis_id'] == pid and r['metric'] == metric)['value'] = value

    def test_exact_lane_candidate_isolation(self):
        self.data['claims'] = []
        self.assertTrue(self.player('R')['candidate'])
        self.assertFalse(self.player('S')['candidate'])
        self.data['depth'][1]['pos_slot'] = 2
        self.assertFalse(self.player('R')['candidate'])

    def test_ambiguous_donor(self):
        self.data['identity']['players'].append(dict(self.data['roster'][0], gsis_id='D2'))
        self.assertEqual(self.result()['balances'], [])
        self.assertIn('BLOCKED_DONOR_IDENTITY', str(self.result()['manifest']['quarantine']))

    def test_ambiguous_beneficiary(self):
        self.data['identity']['players'].append(dict(self.data['roster'][1], gsis_id='R2'))
        row = next(r for r in self.result()['ledger'] if r['claim_id'] == 'DRtargets')
        self.assertIsNone(row['beneficiary_gsis_id'])
        self.assertEqual(row['allocated_quantity'], 0)
        self.assertEqual(self.balance()['residual'], 7)

    def test_unavailable_beneficiary(self):
        self.data['availability'][1]['status'] = 'OUT'
        self.assertEqual(self.player('R')['allocated_credit'], 0)

    def test_targets_conservation(self):
        b = self.balance()
        self.assertEqual((b['vacated'], b['allocated'], b['residual']), (8, 6, 2))
        self.assertTrue(all(r['conservation_status'] == 'PASS' for r in self.result()['manifest']['team_conservation']))

    def test_carries_conservation(self):
        add_claim(self.data, 'D', 'R', 'carries', 12)
        self.assertEqual((self.balance(metric='carries')['allocated'], self.balance(metric='carries')['residual']), (12, 4))

    def test_routes_missing(self):
        self.set_value('baselines', 'D', 'routes', None)
        self.assertIsNone(self.balance(metric='routes')['vacated'])
        self.assertIsNone(self.player('R', 'routes')['allocated_credit'])
        self.assertEqual(self.balance(metric='routes')['conservation_status'], 'NOT_EVALUABLE')

    def test_snaps_missing(self):
        self.data['baselines'] = [r for r in self.data['baselines'] if r['metric'] != 'snaps']
        self.assertIsNone(self.balance(metric='snaps')['residual'])
        self.assertIsNone(self.player('R', 'snaps')['hypothetical_post_transfer'])

    def test_metric_independence(self):
        self.assertEqual(self.player('R')['allocated_credit'], 5)
        for metric in ('routes', 'snaps', 'carries'):
            self.assertEqual(self.player('R', metric)['allocated_credit'], 0)

    def test_supported_cross_lane(self):
        self.assertEqual(self.player('S')['allocated_credit'], 1)
        self.assertEqual(self.player('S')['classification'], 'SECONDARY_BENEFICIARY')

    def test_unsupported_cross_lane(self):
        self.data['evidence'][1]['donor_absence_or_role_change'] = False
        self.assertEqual(self.player('S')['allocated_credit'], 0)

    def second_donor(self):
        self.data['availability'][3]['status'] = 'OUT'
        self.data['donors'].append(self.data['roster'][3])
        self.set_value('baselines', 'U', 'targets', 10)
        add_claim(self.data, 'U', 'S', 'targets', 4)

    def test_multiple_vacancies(self):
        self.second_donor()
        self.assertEqual(len(self.result()['balances']), 8)
        self.assertEqual(self.player('S')['allocated_credit'], 5)

    def test_claims_exceed_supply_proportional(self):
        self.data['depth'] = []  # both supported claims are secondary
        self.data['claims'][0]['requested_quantity'] = 10
        self.data['evidence'][0]['increment'] = 10
        self.data['claims'][1]['requested_quantity'] = 6
        self.data['evidence'][1]['increment'] = 6
        self.assertEqual(self.player('R')['allocated_credit'], 5)
        self.assertEqual(self.player('S')['allocated_credit'], 3)
        self.assertEqual(self.balance()['residual'], 0)

    def test_direct_priority(self):
        self.data['claims'][1]['requested_quantity'] = 20
        self.data['evidence'][1]['increment'] = 20
        self.assertEqual(self.player('R')['allocated_credit'], 5)
        self.assertEqual(self.player('S')['allocated_credit'], 3)

    def test_shared_recipient_capacity(self):
        self.second_donor()
        self.set_value('capacities', 'S', 'targets', 4)  # baseline 2, headroom 2
        self.assertAlmostEqual(self.player('S')['allocated_credit'], 2)
        self.assertAlmostEqual(self.balance()['allocated'], 5.4)
        self.assertAlmostEqual(self.balance('U')['allocated'], 1.6)

    def test_capacity_clipped_stays_residual(self):
        self.set_value('capacities', 'R', 'targets', 3)  # direct credit capped at 1
        self.data['claims'][1]['requested_quantity'] = 20
        self.data['evidence'][1]['increment'] = 20
        self.assertEqual(self.player('S')['allocated_credit'], 3)
        self.assertEqual(self.balance()['residual'], 4)

    def test_duplicate_idempotence(self):
        before = self.result()
        self.data['claims'].append(copy.deepcopy(self.data['claims'][0]))
        self.data['donors'].append(copy.deepcopy(self.data['donors'][0]))
        after = self.result()
        for key in ('players', 'ledger', 'balances'):
            self.assertEqual(before[key], after[key])
        self.assertEqual(after, self.result())

    def test_duplicate_edge_rejected(self):
        self.data['claims'].append(dict(self.data['claims'][0], claim_id='another'))
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_TRANSFER_EDGE'):
            self.result()

    def test_order_independence(self):
        before = self.result()
        self.data['claims'].reverse()
        self.data['roster'].reverse()
        for key in ('players', 'ledger', 'balances'):
            self.assertEqual(before[key], self.result()[key])

    def test_residual_no_evidence(self):
        self.data['evidence'] = []
        self.assertEqual(self.balance()['residual'], 8)

    def test_unknown_baseline_not_zero(self):
        self.set_value('baselines', 'D', 'targets', None)
        self.assertIsNone(self.balance()['allocated'])
        self.assertIsNone(self.player('D')['debit'])
        self.assertEqual(self.balance()['conservation_status'], 'NOT_EVALUABLE')

    def test_full_team_coverage(self):
        self.assertEqual(len(self.result()['players']), len(self.data['roster']) * 4)
        self.assertEqual({r['gsis_id'] for r in self.result()['players']}, {p['gsis_id'] for p in self.data['roster']})

    def test_cold_start_zero(self):
        self.data['claims'][0]['confidence'] = 'COLD_START_INFERENCE'
        self.assertEqual(self.player('R')['allocated_credit'], 0)
        self.assertEqual(self.player('R')['classification'], 'UNCERTAIN_BENEFICIARY')

    def test_unchanged_teammate(self):
        r = self.player('U')
        self.assertEqual((r['baseline'], r['hypothetical_post_transfer']), (2, 2))
        self.assertEqual(r['classification'], 'UNCHANGED_TEAMMATE')

    def test_qb_unchanged(self):
        add_claim(self.data, 'D', 'Q', 'carries', 4)
        for metric in METRICS:
            r = self.player('Q', metric)
            self.assertEqual(r['baseline'], r['hypothetical_post_transfer'])
            self.assertFalse(r['candidate'])

    def test_qb_cannot_fund(self):
        self.data['donors'].append(self.data['roster'][4])
        self.data['availability'][4]['status'] = 'OUT'
        add_claim(self.data, 'Q', 'R', 'carries', 4)
        self.assertEqual(self.player('R', 'carries')['allocated_credit'], 0)
        self.assertFalse(any(b['donor_gsis_id'] == 'Q' for b in self.result()['balances']))

    def test_cutoff_and_version(self):
        self.data['evidence'][0]['available_at'] = '2025-09-01T13:00:00Z'
        self.assertEqual(self.player('R')['allocated_credit'], 0)
        self.data['evidence'][1]['baseline_version'] = 'wrong'
        self.assertEqual(self.player('S')['allocated_credit'], 0)
        r = self.player('R')
        self.assertEqual(r['qb_context_version'], 'qb-v1')
        self.assertEqual(r['scenario_version'], 'fixture-v1')

    def test_not_pregame(self):
        self.data['context']['evidence_cutoff'] = self.data['context']['kickoff']
        with self.assertRaisesRegex(ValueError, 'PREGAME_ONLY'):
            self.result()

    def test_questionable_does_not_vacate(self):
        self.data['availability'][0]['status'] = 'QUESTIONABLE'
        self.assertEqual(self.result()['balances'], [])

    def test_missing_capacity(self):
        self.data['capacities'] = []
        self.assertEqual(self.balance()['allocated'], 0)

    def test_supported_routes_and_snaps(self):
        add_claim(self.data, 'D', 'R', 'routes', 24)
        add_claim(self.data, 'D', 'R', 'snaps', 40)
        self.assertEqual(self.balance(metric='routes')['residual'], 6)
        self.assertEqual(self.balance(metric='snaps')['residual'], 10)

    def test_duplicate_canonical_edge_different_context_fields(self):
        other = copy.deepcopy(self.data['claims'][0])
        other['claim_id'] = 'different'
        other['beneficiary']['source_ref'] = 'same-person-different-source'
        self.data['claims'].append(other)
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_TRANSFER_EDGE'):
            self.result()

    def test_incompatible_metric_definition(self):
        self.data['baselines'][0]['definition'] = 'incompatible-targets'
        with self.assertRaisesRegex(ValueError, 'INCOMPATIBLE_TEAM_METRIC_DEFINITIONS'):
            self.result()

    def test_wrong_identity_id(self):
        self.data['claims'][0]['beneficiary'] = dict(self.data['roster'][1], gsis_id='WRONG')
        self.assertEqual(self.player('R')['allocated_credit'], 0)

    def test_tied_lane_candidate_fails_closed(self):
        row = dict(self.data['depth'][1], gsis_id='S')
        self.data['depth'].append(row)
        self.data['claims'] = []
        self.assertFalse(self.player('R')['candidate'])
        self.assertIn('AMBIGUOUS_SUCCESSOR', str(self.result()['manifest']['quarantine']))

    def test_unknown_availability_no_lane_promotion(self):
        self.data['availability'][1]['status'] = 'UNKNOWN'
        self.data['claims'] = []
        self.assertFalse(self.player('R')['candidate'])

    def test_stale_baseline_version_rejected(self):
        self.data['baselines'][0]['version'] = 'old'
        with self.assertRaisesRegex(ValueError, 'BASELINE_LINEAGE'):
            self.result()

    def test_prior_cannot_establish_supply(self):
        self.data['baselines'][0]['authorized_pregame'] = False
        with self.assertRaisesRegex(ValueError, 'UNAUTHORIZED_BASELINE'):
            self.result()

    def test_negative_quantity_rejected(self):
        self.set_value('baselines', 'D', 'targets', -1)
        with self.assertRaisesRegex(ValueError, 'INVALID_QUANTITY'):
            self.result()

    def test_input_not_mutated(self):
        original = copy.deepcopy(self.data)
        self.result()
        self.assertEqual(self.data, original)

    def test_no_io_and_flags(self):
        # Warm the existing pure identity resolver's dependencies before denying I/O.
        self.result()
        with patch('builtins.open', side_effect=AssertionError('I/O')), patch('sqlite3.connect', side_effect=AssertionError('DB')), patch('socket.socket', side_effect=AssertionError('NETWORK')):
            result = self.result()
        self.assertEqual(result['manifest']['safety'], SAFETY)

    def test_publication_isolated_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(publication, 'SHADOW_ROOT', Path(temp) / 'shadow'):
                path = publication.publish(self.data)
                self.assertEqual(path, publication.publish(self.data))
                self.assertEqual({f.name for f in path.iterdir()}, {'players.json', 'ledger.json', 'balances.json', 'manifest.json'})
                self.assertEqual(json.loads((path / 'manifest.json').read_text())['safety'], SAFETY)

    def test_writer_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'link').symlink_to(root / 'other')
            with patch.object(publication, 'SHADOW_ROOT', root / 'link'):
                with self.assertRaisesRegex(ValueError, 'UNSAFE_SHADOW_ROOT'):
                    publication.publish(self.data)


if __name__ == '__main__':
    unittest.main()
