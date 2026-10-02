"""Focused M4A tests; fixtures describe already-built claims, not eligibility."""
import copy
import itertools
import unittest

from shadow_player_analysis_v1_3 import corroboration as m4a
from shadow_player_analysis_v1_3 import claims as frozen


def claim(n='a', origin=None, direction='INCREASE', strict=True, **changes):
    row = dict(claim_id='claim-' + n, record_id='record-' + n,
               event_key='event-' + n, source='source-' + n, source_event_id='native-' + n,
               origin_id=origin if origin is not None else 'origin-' + n,
               content_hash='content-' + n, gsis_id='00-TEST-001', signal_type='COACH_INTENT',
               direction=direction, evidence_kind='REPORTED_EXPECTATION',
               evidence_quote='Example Player will have a larger role',
               published_at_utc='2026-09-27T14:00:00Z', cutoff_utc='2026-09-27T16:00:00Z',
               eligibility_status='ELIGIBLE', claim_eligible=True, consensus_eligible=False,
               consensus_gate='NOT_ENABLED_V1_3', corroboration_state='NOT_EVALUATED',
               safety=dict(frozen.SAFETY))
    if strict:
        row.update(game_id='game-1', kickoff_at_utc='2026-09-27T17:00:00Z',
                   schedule_sha256='a' * 64, knowledge_at_utc='2026-09-27T15:00:00Z',
                   effective_from_utc='2026-09-27T15:00:00Z', valid_until_utc='2026-09-27T17:00:00Z')
    row.update(changes)
    return row


class CorroborationTests(unittest.TestCase):
    def status(self, *rows):
        result = m4a.build_corroboration(list(rows))
        self.assertEqual(len(result['groups']), 1, result)
        return result['groups'][0]['status']

    def test_agreement(self):
        self.assertEqual(self.status(claim(), claim('b')), 'AGREEMENT')

    def test_repeated_origin(self):
        self.assertEqual(self.status(claim(origin='o'), claim('b', origin='o')), 'SINGLE_ORIGIN')

    def test_different_sources_same_origin(self):
        self.assertEqual(self.status(claim(origin='o', source='A'), claim('b', origin='o', source='B')), 'SINGLE_ORIGIN')

    def test_increase_decrease(self):
        self.assertEqual(self.status(claim(), claim('b', direction='DECREASE')), 'CONFLICT')

    def test_positive_negative(self):
        self.assertEqual(self.status(claim(direction='POSITIVE'), claim('b', direction='NEGATIVE')), 'CONFLICT')

    def test_same_origin_opposition(self):
        self.assertEqual(self.status(claim(origin='o'), claim('b', origin='o', direction='DECREASE')), 'MIXED')

    def test_single(self):
        self.assertEqual(self.status(claim()), 'SINGLE_ORIGIN')

    def test_no_origin(self):
        self.assertEqual(self.status(claim(origin='')), 'INSUFFICIENT')

    def test_unspecified(self):
        self.assertEqual(self.status(claim(direction='UNSPECIFIED'), claim('b', direction='UNSPECIFIED')), 'INSUFFICIENT')

    def test_neutral(self):
        self.assertEqual(self.status(claim(direction='NEUTRAL'), claim('b', direction='NEUTRAL')), 'INSUFFICIENT')

    def test_players(self):
        self.assertEqual(len(m4a.build_corroboration([claim(), claim('b', gsis_id='other')])['groups']), 2)

    def test_games(self):
        self.assertEqual(len(m4a.build_corroboration([claim(), claim('b', game_id='other')])['groups']), 2)

    def test_signals(self):
        for signal in m4a.SIGNALS - {'COACH_INTENT', 'ROLE_DECREASE'}:
            with self.subTest(signal=signal):
                self.assertEqual(len(m4a.build_corroboration([claim(), claim('b', signal_type=signal)])['groups']), 2)

    def test_horizons(self):
        other = claim('b', knowledge_at_utc='2026-09-27T15:01:00Z', effective_from_utc='2026-09-27T15:01:00Z')
        self.assertEqual(len(m4a.build_corroboration([claim(), other])['groups']), 2)

    def test_duplicate_capture(self):
        self.assertEqual(m4a.build_corroboration([claim()]), m4a.build_corroboration([claim(), claim()]))

    def test_permutations(self):
        rows = [claim(), claim('b'), claim('c', direction='DECREASE')]
        expected = m4a.build_corroboration(rows)
        for order in itertools.permutations(rows):
            self.assertEqual(m4a.build_corroboration(order), expected)

    def test_deterministic_group_ids(self):
        a = m4a.build_corroboration([claim()])['groups'][0]['group_id']
        b = m4a.build_corroboration([claim('b')])['groups'][0]['group_id']
        self.assertEqual(a, b)

    def test_input_unchanged_and_output_detached(self):
        rows = [claim(conditions={'weather': ['dry']}), claim('b')]
        before = copy.deepcopy(rows)
        result = m4a.build_corroboration(rows)
        result['groups'][0]['comparison'].clear()
        self.assertEqual(rows, before)

    def test_ineligible_status(self):
        result = m4a.build_corroboration([claim(), claim('b', eligibility_status='QUARANTINED')])
        self.assertEqual(result['groups'][0]['status'], 'SINGLE_ORIGIN')
        self.assertFalse(result['claim_states']['claim-b']['voting'])

    def test_ineligible_boolean(self):
        self.assertFalse(m4a.build_corroboration([claim(claim_eligible=False)])['groups'])

    def test_consensus_stays_disabled(self):
        row = claim()
        result = m4a.build_corroboration([row, claim('b')])
        self.assertFalse(row['consensus_eligible'])
        self.assertEqual(row['consensus_gate'], 'NOT_ENABLED_V1_3')
        self.assertFalse(result['safety']['CONSENSUS_ENABLED'])
        self.assertFalse(frozen.CONSENSUS_ENABLED)
        self.assertFalse(m4a.build_corroboration([claim(consensus_eligible=True)])['groups'])

    def test_safety(self):
        self.assertEqual(m4a.build_corroboration([])['safety'], m4a.SAFETY)
        for key, value in m4a.SAFETY.items():
            self.assertIs(value, key == 'ANALYSIS_ONLY')
            self.assertIs(getattr(m4a, key), value)

    def test_actual_frozen_claims(self):
        from shadow_player_analysis_v1_3.test_claim_lifecycle_v1_3 import ClaimLifecycleV13Tests
        row = ClaimLifecycleV13Tests().build()['claims'][0]
        result = m4a.build_corroboration([row])
        self.assertFalse(result['rejected'])
        self.assertEqual(result['groups'][0]['status'], 'INSUFFICIENT')
        self.assertEqual(row['direction'], 'UNSPECIFIED')

    def test_legacy_separate(self):
        result = m4a.build_corroboration([claim(), claim('b', strict=False)])
        self.assertEqual(len(result['groups']), 2)

    def test_three_origins(self):
        result = m4a.build_corroboration([claim(), claim('b'), claim('c'), claim('d', origin='origin-a')])
        self.assertEqual(result['groups'][0]['independent_origin_count'], 3)
        self.assertEqual(result['groups'][0]['status'], 'AGREEMENT')


if __name__ == '__main__':
    unittest.main()
