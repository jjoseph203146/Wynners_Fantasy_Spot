"""Adversarial M4A boundaries and executable no-I/O guards."""
import copy
import hashlib
import itertools
from pathlib import Path
import subprocess
import sys
import unittest

from shadow_player_analysis_v1_3 import corroboration as m4a
from shadow_player_analysis_v1_3.test_corroboration_v1_3 import claim


class CorroborationAdversarialTests(unittest.TestCase):
    def rejected(self, row):
        result = m4a.build_corroboration([row])
        self.assertEqual(result['groups'], [], result)
        self.assertTrue(result['rejected'])
        return result

    def test_all_required_identifiers(self):
        for field in m4a.IDENTIFIERS:
            for bad in ('', ' ', ' x', 'x ', 'x\ny', '\ud800', 'x\u200by', 1, True, [], {}, None):
                with self.subTest(field=field, value=bad):
                    self.rejected(claim(**{field: bad}))

    def test_missing_required_fields(self):
        for field in set(m4a.REQUIRED) - {'origin_id'}:
            with self.subTest(field=field):
                row = claim()
                del row[field]
                self.rejected(row)

    def test_malformed_origins_never_vote(self):
        for bad in ('', ' ', ' o', 'o\nx', 'UNKNOWN', 'unknown', 'UNSPECIFIED',
                    'UNRESOLVED', 'NONE', 'NULL', 'N/A', [], {}, 7, False, None):
            with self.subTest(origin=bad):
                row = claim()
                row['origin_id'] = bad
                result = m4a.build_corroboration([row, claim('b')])
                group = result['groups'][0]
                self.assertEqual(group['status'], 'SINGLE_ORIGIN')
                self.assertEqual(group['unknown_origin_count'], 1)
                self.assertFalse(result['claim_states']['claim-a']['voting'])

    def test_missing_origin_does_not_use_source(self):
        row = claim()
        del row['origin_id']
        result = m4a.build_corroboration([row])
        self.assertEqual(result['groups'][0]['status'], 'INSUFFICIENT')
        self.assertEqual(result['groups'][0]['independent_origin_count'], 0)

    def test_malformed_directions(self):
        for bad in ('UP', 'increase', '', ' INCREASE', None, [], {}, 1):
            with self.subTest(direction=bad):
                self.rejected(claim(direction=bad))

    def test_malformed_signal(self):
        for bad in ('ROLE', 'NEW_SIGNAL', [], {}):
            self.rejected(claim(signal_type=bad))

    def test_role_direction_contradiction(self):
        self.rejected(claim(signal_type='ROLE_INCREASE', direction='DECREASE'))
        self.rejected(claim(signal_type='ROLE_DECREASE', direction='INCREASE'))

    def test_role_signals_never_aliased(self):
        result = m4a.build_corroboration([claim(signal_type='ROLE_INCREASE'),
                                       claim('b', signal_type='ROLE_DECREASE', direction='DECREASE')])
        self.assertEqual(len(result['groups']), 2)

    def test_named_signal_pairs_separate(self):
        for a, b in [('TARGET_SHARE', 'BACKFIELD_SHARE'), ('RED_ZONE_ROLE', 'DEEP_TARGET_ROLE'),
                     ('ROLE_INCREASE', 'TARGET_SHARE'), ('STARTER_CHANGE', 'COACH_INTENT')]:
            self.assertEqual(len(m4a.build_corroboration([claim(signal_type=a), claim('b', signal_type=b)])['groups']), 2)

    def test_same_origin_mixed_plus_second_origin(self):
        result = m4a.build_corroboration([claim(origin='o'), claim('b', origin='o', direction='DECREASE'), claim('c')])
        self.assertEqual(result['groups'][0]['status'], 'MIXED')

    def test_clean_conflict_survives_inconsistent_third_origin(self):
        result = m4a.build_corroboration([claim(), claim('b', direction='DECREASE'),
                                       claim('c', origin='mixed'), claim('d', origin='mixed', direction='DECREASE')])
        self.assertEqual(result['groups'][0]['status'], 'CONFLICT')

    def test_non_opposing_labels_not_conflict(self):
        result = m4a.build_corroboration([claim(), claim('b', direction='POSITIVE')])
        self.assertEqual(result['groups'][0]['status'], 'INSUFFICIENT')

    def test_neutral_does_not_add_vote(self):
        for direction in ('NEUTRAL', 'UNSPECIFIED'):
            result = m4a.build_corroboration([claim(), claim('b', direction=direction)])
            self.assertEqual(result['groups'][0]['status'], 'SINGLE_ORIGIN')
            self.assertEqual(result['groups'][0]['independent_origin_count'], 1)

    def test_lexical_claim_ids_not_semantics(self):
        for first, second in [('a', 'z'), ('z', 'a')]:
            result = m4a.build_corroboration([claim(claim_id=first), claim('b', claim_id=second, direction='DECREASE')])
            self.assertEqual(result['groups'][0]['status'], 'CONFLICT')

    def test_lexical_origin_ids_not_semantics(self):
        for first, second in [('a', 'z'), ('z', 'a')]:
            result = m4a.build_corroboration([claim(origin=first), claim('b', origin=second, direction='DECREASE')])
            self.assertEqual(result['groups'][0]['status'], 'CONFLICT')

    def test_revision_family_quarantined_no_winner(self):
        rows = [claim(event_key='shared'), claim('z', event_key='shared', direction='DECREASE')]
        for order in itertools.permutations(rows):
            result = m4a.build_corroboration(order)
            self.assertFalse(result['groups'])
            self.assertTrue(all('UNRESOLVED_REVISION_FAMILY' in x['reason_codes'] for x in result['claim_states'].values()))

    def test_revision_provenance_cannot_rescue(self):
        rows = [claim(event_key='shared', revision_provenance={'current': False}),
                claim('b', event_key='shared', revision_provenance={'current': True, 'revision_position': 2})]
        self.assertFalse(m4a.build_corroboration(rows)['groups'])

    def test_ineligible_conflicting_revision_still_quarantines_family(self):
        rows = [claim(event_key='shared'), claim('b', event_key='shared', claim_eligible=False)]
        self.assertFalse(m4a.build_corroboration(rows)['groups'])

    def test_claim_id_collision_quarantines_both(self):
        result = m4a.build_corroboration([claim(), claim('b', claim_id='claim-a')])
        self.assertFalse(result['groups'])
        self.assertIn('CONTRADICTORY_CLAIM_ID', result['claim_states']['claim-a']['reason_codes'])

    def test_record_metadata_collision(self):
        rows = [claim(), claim('b', record_id='record-a')]
        self.assertFalse(m4a.build_corroboration(rows)['groups'])

    def test_same_event_cannot_manufacture_origins(self):
        rows = [claim(event_key='shared', content_hash='same'),
                claim('b', event_key='shared', content_hash='same')]
        self.assertFalse(m4a.build_corroboration(rows)['groups'])

    def test_native_event_cannot_bypass_revision_quarantine(self):
        rows = [claim(), claim('b', source='source-a', source_event_id='native-a')]
        self.assertFalse(m4a.build_corroboration(rows)['groups'])

    def test_identical_event_recaptures_one_origin(self):
        first = claim()
        second = dict(first, claim_id='claim-b', record_id='record-b')
        result = m4a.build_corroboration([first, second])
        self.assertFalse(result['rejected'])
        self.assertEqual(result['groups'][0]['status'], 'SINGLE_ORIGIN')

    def test_historical_revision_snapshots_separate_cutoffs(self):
        first = claim()
        second = dict(first, claim_id='claim-b', record_id='record-b', content_hash='new',
                      cutoff_utc='2026-09-27T16:01:00Z')
        result = m4a.build_corroboration([first, second])
        self.assertFalse(result['rejected'])
        self.assertEqual(len(result['groups']), 2)

    def test_unicode_diagnostics_deterministic(self):
        row = claim(evidence_quote='unpaired surrogate \ud800')
        self.assertEqual(m4a.build_corroboration([row]), m4a.build_corroboration([row, row]))

    def test_blank_game_not_absent(self):
        self.rejected(claim(game_id=''))
        self.rejected(claim(strict=False, game_id=''))
        self.rejected(claim(game_id=None))

    def test_partial_lifecycle(self):
        for field in m4a.LIFECYCLE:
            row = claim()
            del row[field]
            self.rejected(row)

    def test_cutoff_mismatch_separate(self):
        result = m4a.build_corroboration([claim(), claim('b', cutoff_utc='2026-09-27T16:01:00Z')])
        self.assertEqual(len(result['groups']), 2)

    def test_explicit_cutoff_not_retiming(self):
        result = m4a.build_corroboration([claim()], cutoff_utc='2026-09-27T16:01:00Z')
        self.assertFalse(result['groups'])
        self.assertIn('CUTOFF_MISMATCH', result['rejected'][0]['reason_codes'])

    def test_explicit_equivalent_cutoff(self):
        self.assertEqual(m4a.build_corroboration([claim()]), m4a.build_corroboration([claim()], cutoff_utc='2026-09-27T12:00:00-04:00'))

    def test_bad_cutoff_raises(self):
        for bad in ('', '2026-09-27', {}, False):
            with self.assertRaises(ValueError):
                m4a.build_corroboration([claim()], cutoff_utc=bad)

    def test_effective_knowledge_mismatch(self):
        self.rejected(claim(effective_from_utc='2026-09-27T15:01:00Z'))

    def test_valid_kickoff_mismatch(self):
        self.rejected(claim(valid_until_utc='2026-09-27T18:00:00Z'))

    def test_different_end_horizon(self):
        result = m4a.build_corroboration([claim(), claim('b', valid_until_utc='2026-09-27T18:00:00Z', kickoff_at_utc='2026-09-27T18:00:00Z')])
        self.assertEqual(len(result['groups']), 2)

    def test_no_historical_future_blend(self):
        self.rejected(claim(cutoff_utc='2026-09-27T17:00:00Z'))
        self.rejected(claim(cutoff_utc='2026-09-28T16:00:00Z'))

    def test_future_knowledge_and_publication(self):
        self.rejected(claim(published_at_utc='2026-09-27T16:01:00Z'))
        self.rejected(claim(knowledge_at_utc='2026-09-27T16:01:00Z', effective_from_utc='2026-09-27T16:01:00Z'))

    def test_same_semantics_different_hashes_independent_events(self):
        result = m4a.build_corroboration([claim(), claim('b')])
        self.assertEqual(result['groups'][0]['status'], 'AGREEMENT')

    def test_identical_content_independence_uses_origins(self):
        for origin, status in [('origin-a', 'SINGLE_ORIGIN'), ('independent', 'AGREEMENT')]:
            rows = [claim(content_hash='same'), claim('b', origin=origin, content_hash='same')]
            self.assertEqual(m4a.build_corroboration(rows)['groups'][0]['status'], status)

    def test_source_labels_do_not_add_independence(self):
        rows = [claim(origin='o', source=str(i), claim_id=str(i), record_id=str(i), event_key=str(i)) for i in range(10)]
        self.assertEqual(m4a.build_corroboration(rows)['groups'][0]['status'], 'SINGLE_ORIGIN')

    def test_comparison_semantics_separate(self):
        for field in m4a.COMPARISON:
            result = m4a.build_corroboration([claim(**{field: 'a'}), claim('b', **{field: 'b'})])
            self.assertEqual(len(result['groups']), 2)

    def test_comparison_missing_not_null(self):
        self.assertEqual(len(m4a.build_corroboration([claim(), claim('b', metric=None)])['groups']), 2)

    def test_schedule_snapshots_separate(self):
        self.assertEqual(len(m4a.build_corroboration([claim(), claim('b', schedule_sha256='b'*64)])['groups']), 2)

    def test_safety_tampering(self):
        for key in m4a.SAFETY:
            row = claim()
            row['safety'][key] = key != 'ANALYSIS_ONLY'
            self.rejected(row)

    def test_boolean_contract_not_truthiness(self):
        self.rejected(claim(claim_eligible=1))
        self.rejected(claim(consensus_eligible=0))

    def test_no_feedback_of_evaluated_claims(self):
        self.rejected(claim(corroboration_state='AGREEMENT'))

    def test_non_json_and_non_object_rows(self):
        result = m4a.build_corroboration([None, 1, [], claim(extra=float('nan')), claim(extra=object())])
        self.assertFalse(result['groups'])
        self.assertTrue(result['rejected'])

    def test_non_json_capture_cannot_hide_claim_id_collision(self):
        rows = [claim(), claim(extra=object()), claim('b')]
        for order in itertools.permutations(rows):
            result = m4a.build_corroboration(order)
            self.assertEqual(result['groups'][0]['status'], 'SINGLE_ORIGIN')
            self.assertFalse(result['claim_states']['claim-a']['voting'])

    def test_invalid_container(self):
        with self.assertRaises(ValueError):
            m4a.build_corroboration({'claims': [claim()]})

    def test_adversarial_permutation_invariance(self):
        rows = [claim(event_key='shared'), claim('b', event_key='shared'), claim('c', origin=''), claim('d')]
        before = copy.deepcopy(rows)
        expected = m4a.build_corroboration(rows)
        for order in itertools.permutations(rows):
            self.assertEqual(m4a.build_corroboration(order), expected)
        self.assertEqual(rows, before)

    def test_protected_hashes(self):
        root = Path(__file__).parent
        for name, expected in {
            'claims.py': '6debb81489ee7b73dfa09c3a9bca9c728eb9b2b7855693a4ac530416e84297f8',
            'revision_provenance.py': '1a9ac584b1f3f026f1a21db4e541a7f00991052b7f347c65476c79a5324eead7',
        }.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)

    def test_fresh_import_and_execution_no_io(self):
        code = r'''
import sys, socket, sqlite3, subprocess, builtins, os
from unittest.mock import patch

def deny(*args, **kwargs):
    raise AssertionError('M4A I/O forbidden')

def audit(name, args):
    if name.startswith(('socket.', 'sqlite3.', 'subprocess.')):
        deny()
    if name == 'open':
        mode, flags = args[1:3]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
            deny()
    if name.startswith(('os.remove', 'os.rename', 'os.mkdir', 'os.system')):
        deny()

sys.addaudithook(audit)
with patch.object(socket, 'socket', deny), patch.object(sqlite3, 'connect', deny), patch.object(subprocess, 'Popen', deny), patch.object(builtins, 'open', deny):
    from shadow_player_analysis_v1_3 import corroboration as module
    from shadow_player_analysis_v1_3.test_corroboration_v1_3 import claim
    assert module.build_corroboration([claim(), claim('b')])['groups'][0]['status'] == 'AGREEMENT'
    assert module.CONSENSUS_ENABLED is False
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
