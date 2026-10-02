import hashlib, json, random
from pathlib import Path
import unittest
import numpy as np

from . import bounded_walk_forward_validation_v1 as v


class BoundedWalkForwardTests(unittest.TestCase):
    def test_authority_integrity_and_target_scope(self):
        training, outcomes = v._read(v.TRAINING), v._read(v.OUTCOMES)
        games = v._integrity(training, outcomes)
        self.assertEqual(len(training), 1710)
        self.assertEqual(len(games), 855)
        self.assertTrue(all(len(x) == 2 for x in games.values()))
        self.assertEqual(sum(rows[0]['season'] == '2025' for rows in games.values()), 285)

    def test_one_run_shape_and_temporal_audit(self):
        result = v.run_validation()
        self.assertEqual(result['target_season'], 2025)
        self.assertEqual(result['target_games'], 285)
        self.assertEqual(result['accepted_games'] + result['rejected_games'], 285)
        self.assertEqual(result['rejection_reasons'], {})
        self.assertTrue(result['one_realization_per_game'])
        self.assertEqual(result['temporal_selector_calls_per_game'], {'min': 1, 'max': 1})
        self.assertTrue(result['strict_before_rule'])
        self.assertTrue(result['target_exclusion'])
        self.assertTrue(result['same_kickoff_exclusion'])
        self.assertTrue(result['later_game_exclusion'])
        self.assertTrue(result['actual_frozen_generator_used'])
        self.assertTrue(result['football_invariants'])
        self.assertFalse(result['points_sampling'])
        self.assertEqual(result['historical_temporal_provenance'], 'UNVERIFIED')

    def test_repeatability_and_global_rng(self):
        py, np_state = random.getstate(), np.random.get_state()
        a, b = v.run_validation(), v.run_validation()
        self.assertEqual(a, b)
        self.assertEqual(py, random.getstate())
        now = np.random.get_state()
        self.assertEqual(np_state[0], now[0]); np.testing.assert_array_equal(np_state[1], now[1])
        self.assertEqual(np_state[2:], now[2:])

    def test_protected_hashes(self):
        manifest = json.loads(Path(v.REPORT.with_name('BOUNDED_WALK_FORWARD_VALIDATION_V1_HASHES_BEFORE.json')).read_text())['files']
        for filename, digest in manifest.items():
            self.assertEqual(hashlib.sha256(Path(filename).read_bytes()).hexdigest(), digest, filename)


if __name__ == '__main__':
    unittest.main()
