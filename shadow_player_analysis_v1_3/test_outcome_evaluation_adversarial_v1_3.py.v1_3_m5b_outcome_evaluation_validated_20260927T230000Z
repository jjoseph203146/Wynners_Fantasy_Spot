"""Adversarial M5B tests use only disposable local SQLite fixtures."""
import copy
import sqlite3
import unittest
from unittest.mock import patch

from . import outcome_evaluation as m5b
from .test_outcome_evaluation_v1_3 import OutcomeFixture, claim


class OutcomeAdversarialTests(OutcomeFixture):
    def test_duplicate_usage(self):
        self.duplicate('player_weekly_usage')
        self.assertEqual(self.row()['evaluation_status'], 'AMBIGUOUS_OUTCOME')

    def test_duplicate_stats(self):
        self.duplicate('player_game_stats')
        self.assertEqual(self.row()['evaluation_status'], 'AMBIGUOUS_OUTCOME')

    def test_duplicate_baseline(self):
        self.duplicate('player_pregame_features')
        self.assertEqual(self.row()['evaluation_status'], 'AMBIGUOUS_OUTCOME')

    def test_duplicate_identity_different_teams(self):
        self.conn.execute("INSERT INTO player_weekly_usage SELECT * FROM player_weekly_usage")
        self.conn.execute("UPDATE player_weekly_usage SET team='BBB' WHERE rowid=2")
        self.conn.commit()
        self.assertEqual(self.row()['evaluation_status'], 'AMBIGUOUS_OUTCOME')

    def test_name_collision_does_not_make_exact_match_ambiguous(self):
        self.conn.execute('INSERT INTO player_game_stats SELECT * FROM player_game_stats')
        self.conn.execute("UPDATE player_game_stats SET player_id='other-id' WHERE rowid=2")
        self.conn.commit()
        self.assertEqual(self.row()['outcome_state'], 'SUPPORTED')

    def test_wrong_gsis_same_name(self):
        self.change('player_weekly_usage', player_id='wrong-id')
        self.assertEqual(self.row()['evaluation_status'], 'NO_OUTCOME')

    def test_wrong_game_same_gsis(self):
        self.change('player_weekly_usage', game_id='other-game')
        self.assertEqual(self.row()['evaluation_status'], 'NO_OUTCOME')

    def test_identity_case_is_exact(self):
        self.assertEqual(self.row(claim(gsis_id='00-test-001'))['evaluation_status'], 'NO_OUTCOME')

    def test_malformed_completed_flags(self):
        for flag in (None, '1', 'True', 'FINAL', 'completed', 2, -1, 1.0):
            with self.subTest(flag=flag):
                self.change('games', completed=flag)
                self.assertEqual(self.row()['evaluation_status'], 'NOT_FINAL')

    def test_incomplete_never_reads_player_outcomes(self):
        self.change('games', completed=0)
        self.conn.execute('DROP TABLE player_weekly_usage')
        self.conn.commit()
        self.assertEqual(self.row()['evaluation_status'], 'NOT_FINAL')

    def test_null_usage(self):
        self.change('player_weekly_usage', usage_delta=None)
        self.assertEqual(self.row()['outcome_state'], 'INDETERMINATE')

    def test_null_target_share(self):
        self.change('player_weekly_usage', target_share=None)
        self.assertEqual(self.row(claim(signal='TARGET_SHARE', direction='INCREASE'))['outcome_state'], 'INDETERMINATE')

    def test_contradictory_flags(self):
        self.change('player_weekly_usage', role_decline_flag=1)
        self.assertEqual(self.row()['reason_codes'], ['CONTRADICTORY_ROLE_FLAGS'])

    def test_malformed_deltas(self):
        for delta in ('five', '5', float('inf'), None):
            self.change('player_weekly_usage', usage_delta=delta)
            self.assertEqual(self.row()['outcome_state'], 'INDETERMINATE')

    def test_inconsistent_deltas(self):
        self.change('player_weekly_usage', usage_delta=99)
        self.assertEqual(self.row()['reason_codes'], ['ROLE_BASELINE_DELTA_MISMATCH'])

    def test_inconsistent_flags(self):
        self.change('player_weekly_usage', role_expansion_flag=0)
        self.assertEqual(self.row()['reason_codes'], ['ROLE_FLAG_DELTA_MISMATCH'])

    def test_malformed_flags(self):
        for flag in ('1', None, 2, 1.0):
            self.change('player_weekly_usage', role_expansion_flag=flag)
            self.assertEqual(self.row()['outcome_state'], 'INDETERMINATE')

    def test_zero_history_not_false_negative(self):
        self.change('player_pregame_features', history_games=0)
        self.assertEqual(self.row()['outcome_state'], 'INDETERMINATE')

    def test_null_baseline(self):
        self.change('player_pregame_features', opportunities_avg_3=None)
        self.assertEqual(self.row()['outcome_state'], 'INDETERMINATE')

    def test_baseline_mismatch(self):
        self.change('player_pregame_features', opportunities_avg_3=11)
        self.assertEqual(self.row()['outcome_state'], 'INDETERMINATE')

    def test_negative_counts(self):
        self.change('player_weekly_usage', targets=-5)
        self.assertEqual(self.row()['outcome_state'], 'INDETERMINATE')

    def test_out_of_range_shares(self):
        for share in (-.01, 1.01, '0.3', float('inf')):
            self.change('player_weekly_usage', target_share=share)
            self.assertEqual(self.row(claim(signal='TARGET_SHARE', direction='INCREASE'))['outcome_state'], 'INDETERMINATE')

    def test_malformed_claims(self):
        for bad in (None, [], 'claim', {}, 17, {'claim_id': []}, claim(signal_type=[])):
            result = m5b.evaluate_outcomes(claims=[bad], db_path=self.db)
            self.assertFalse(result['evaluations'])
            self.assertEqual(result['unevaluated'][0]['evaluation_status'], 'INVALID_CLAIM')

    def test_missing_required_fields(self):
        for key in ('claim_id', 'gsis_id', 'game_id', 'direction', 'signal_type', 'safety',
                    'cutoff_utc', 'kickoff_at_utc', 'evidence_quote'):
            row = claim()
            del row[key]
            self.assertEqual(self.row(row)['evaluation_status'], 'INVALID_CLAIM')

    def test_claim_eligible_false(self):
        self.assertEqual(self.row(claim(claim_eligible=False))['evaluation_status'], 'INVALID_CLAIM')

    def test_nonboolean_eligibility(self):
        self.assertEqual(self.row(claim(claim_eligible=1))['evaluation_status'], 'INVALID_CLAIM')

    def test_consensus_eligible_true(self):
        self.assertEqual(self.row(claim(consensus_eligible=True))['evaluation_status'], 'INVALID_CLAIM')

    def test_consensus_gate_mismatch(self):
        self.assertEqual(self.row(claim(consensus_gate='ENABLED'))['evaluation_status'], 'INVALID_CLAIM')

    def test_eligibility_status_mismatch(self):
        self.assertEqual(self.row(claim(eligibility_status='REJECTED'))['evaluation_status'], 'INVALID_CLAIM')

    def test_every_unsafe_flag_rejected(self):
        for flag, expected in m5b.SAFETY.items():
            row = claim()
            row['safety'][flag] = not expected
            self.assertEqual(self.row(row)['evaluation_status'], 'INVALID_CLAIM')

    def test_signal_direction_conflict(self):
        self.assertEqual(self.row(claim(direction='DECREASE'))['evaluation_status'], 'INVALID_CLAIM')

    def test_malformed_claim_timestamps(self):
        for timestamp in ('tomorrow', '2026-09-27T16:00:00', None, 0):
            self.assertEqual(self.row(claim(cutoff_utc=timestamp))['evaluation_status'], 'INVALID_CLAIM')

    def test_postgame_cutoff_rejected(self):
        self.assertEqual(self.row(claim(cutoff_utc='2026-09-28T00:00:00Z'))['evaluation_status'], 'INVALID_CLAIM')

    def test_future_knowledge_rejected(self):
        self.assertEqual(self.row(claim(knowledge_at_utc='2026-09-28T00:00:00Z'))['evaluation_status'], 'INVALID_CLAIM')

    def test_malformed_outcome_timestamp(self):
        self.change('player_weekly_usage', updated_at='bad')
        self.assertEqual(self.row()['evaluation_status'], 'NO_OUTCOME')

    def test_malformed_baseline_timestamp(self):
        self.change('player_pregame_features', updated_at='bad')
        self.assertEqual(self.row()['outcome_state'], 'INDETERMINATE')

    def test_postgame_baseline_refresh_is_audited_not_claim_mutation(self):
        self.change('player_pregame_features', updated_at='2026-10-01T00:00:00Z')
        original = claim()
        self.assertEqual(self.row(original)['pregame_claim'], original)
        self.assertEqual(self.row(original)['outcome_state'], 'SUPPORTED')

    def test_null_updated_at_is_audited(self):
        self.change('player_weekly_usage', updated_at=None)
        row = self.row()
        self.assertIsNone(row['postgame_outcome']['player_weekly_usage']['updated_at'])
        self.assertEqual(row['outcome_state'], 'SUPPORTED')

    def test_invalid_duplicate_poison_valid_variant(self):
        result = self.evaluate(claim(), claim(claim_eligible=False))
        self.assertFalse(result['evaluations'])
        self.assertEqual(len(result['unevaluated']), 2)

    def test_non_json_duplicate_poison_valid_variant(self):
        result = self.evaluate(claim(), claim(payload=object()))
        self.assertFalse(result['evaluations'])
        self.assertTrue(all(r['evaluation_status'] == 'INVALID_CLAIM' for r in result['unevaluated']))

    def test_cyclic_claim_rejected(self):
        original = claim()
        original['cycle'] = original
        self.assertEqual(self.row(original)['evaluation_status'], 'INVALID_CLAIM')
        self.assertIs(original['cycle'], original)

    def test_all_pregame_fields_survive_opposite_postgame_results(self):
        original = claim(extra_audit={'evidence': ['frozen']})
        before = copy.deepcopy(original)
        first = self.row(original)
        self.declining()
        second = self.row(original)
        self.assertNotEqual(first['outcome_state'], second['outcome_state'])
        self.assertNotEqual(first['evaluation_id'], second['evaluation_id'])
        self.assertEqual(first['pregame_claim'], before)
        self.assertEqual(second['pregame_claim'], before)
        self.assertEqual(original, before)

    def test_route_not_proxied_by_targets(self):
        self.change('player_weekly_usage', targets=100)
        self.assertEqual(self.row(claim(signal='ROUTE_PARTICIPATION'))['evaluation_status'], 'UNSUPPORTED_SIGNAL')

    def test_red_zone_not_proxied_by_touchdowns(self):
        self.change('player_game_stats', receiving_tds=5)
        self.assertEqual(self.row(claim(signal='RED_ZONE_ROLE'))['evaluation_status'], 'UNSUPPORTED_SIGNAL')

    def test_deep_role_not_proxied_by_air_yards(self):
        self.change('player_weekly_usage', air_yards_share=1)
        self.assertEqual(self.row(claim(signal='DEEP_TARGET_ROLE'))['evaluation_status'], 'UNSUPPORTED_SIGNAL')

    def test_coach_intent_not_proxied_by_usage(self):
        self.assertEqual(self.row(claim(signal='COACH_INTENT'))['evaluation_status'], 'UNSUPPORTED_SIGNAL')

    def test_starter_not_proxied_by_snaps(self):
        self.change('player_weekly_usage', offense_snaps=100)
        self.assertEqual(self.row(claim(signal='STARTER_CHANGE'))['evaluation_status'], 'UNSUPPORTED_SIGNAL')

    def test_backfield_not_redefined_as_carries(self):
        self.assertEqual(self.row(claim(signal='BACKFIELD_SHARE'))['evaluation_status'], 'UNSUPPORTED_SIGNAL')

    def test_unknown_signal_invalid(self):
        self.assertEqual(self.row(claim(signal='FANTASY_POINTS'))['evaluation_status'], 'INVALID_CLAIM')

    def test_production_does_not_grade_role(self):
        self.change('player_game_stats', fanduel_points=-10)
        self.change('player_weekly_usage', fanduel_points=-10)
        self.assertEqual(self.row()['outcome_state'], 'SUPPORTED')

    def test_conditional_claim_not_silently_scored(self):
        self.assertEqual(self.row(claim(conditions=['if teammate inactive']))['evaluation_status'], 'UNSUPPORTED_SIGNAL')

    def test_positive_label_not_inferred_as_increase(self):
        self.assertEqual(self.row(claim(signal='TARGET_SHARE', direction='POSITIVE'))['outcome_state'], 'INDETERMINATE')

    def test_metadata_team_mismatch(self):
        self.change('player_weekly_usage', team='BBB')
        self.assertEqual(self.row()['evaluation_status'], 'AMBIGUOUS_OUTCOME')

    def test_metadata_week_mismatch(self):
        self.change('player_game_stats', week=4)
        self.assertEqual(self.row()['evaluation_status'], 'AMBIGUOUS_OUTCOME')

    def test_baseline_team_mismatch(self):
        self.change('player_pregame_features', team='BBB')
        self.assertEqual(self.row()['evaluation_status'], 'AMBIGUOUS_OUTCOME')

    def test_sql_injection_identity_is_bound(self):
        result = self.row(claim(gsis_id="x'OR'1'='1"))
        self.assertEqual(result['evaluation_status'], 'NO_OUTCOME')
        self.assertEqual(self.conn.execute('SELECT count(*) FROM games').fetchone(), (1,))

    def test_uri_injection_rejected(self):
        result = m5b.evaluate_outcomes(claims=[claim()], db_path='file:' + str(self.db) + '?mode=rw')
        self.assertEqual(result['unevaluated'][0]['reason_codes'], ['DATABASE_READ_UNAVAILABLE'])

    def test_missing_db_not_created(self):
        missing = self.db.parent / 'does-not-exist.sqlite'
        result = m5b.evaluate_outcomes(claims=[claim()], db_path=missing)
        self.assertFalse(missing.exists())
        self.assertEqual(result['unevaluated'][0]['evaluation_status'], 'NO_OUTCOME')

    def test_filename_uri_metacharacters_are_encoded(self):
        destination = self.db.parent / 'db?mode=rw#name.sqlite'
        destination.write_bytes(self.db.read_bytes())
        result = m5b.evaluate_outcomes(claims=[claim()], db_path=destination)
        self.assertEqual(result['evaluations'][0]['outcome_state'], 'SUPPORTED')

    def test_no_schema_fallback(self):
        self.conn.execute('DROP TABLE player_pregame_features')
        self.conn.commit()
        self.assertEqual(self.row()['reason_codes'], ['DATABASE_READ_UNAVAILABLE'])

    def test_authorizer_rejects_writes_and_mutating_pragmas(self):
        # Exercise the actual read-only connection and actual authorizer.
        connection = sqlite3.connect(self.db.as_uri() + '?mode=ro', uri=True)
        self.addCleanup(connection.close)
        connection.set_authorizer(m5b._read_authorizer)
        for statement in ('UPDATE games SET completed=0', 'DELETE FROM games',
                          'CREATE TABLE forbidden(x)', 'DROP TABLE games',
                          'PRAGMA user_version=1', 'VACUUM',
                          "ATTACH DATABASE ':memory:' AS other"):
            with self.subTest(statement=statement), self.assertRaises(sqlite3.DatabaseError):
                connection.execute(statement)

    def test_invalid_input_no_db_access(self):
        with patch.object(m5b.sqlite3, 'connect', side_effect=AssertionError('DB accessed')):
            self.assertEqual(self.row(claim(claim_eligible=False))['evaluation_status'], 'INVALID_CLAIM')

    def test_bad_collection_rejected_before_io(self):
        with patch.object(m5b.sqlite3, 'connect', side_effect=AssertionError('DB accessed')):
            for collection in (None, {}, 'claims', iter([claim()])):
                with self.assertRaises(ValueError):
                    m5b.evaluate_outcomes(claims=collection)


if __name__ == '__main__':
    unittest.main()
