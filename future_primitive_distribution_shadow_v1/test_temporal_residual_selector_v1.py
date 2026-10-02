"""Focused fixed-seed selector validation; never runs a simulator."""
import ast
import copy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch
import numpy as np
from . import temporal_residual_selector_v1 as s

TARGET = '2024_08_TEN_DET'
KICKOFF = '2024-10-27T13:00:00'


class TemporalSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs, cls.chronology, cls.audit = s._load_authorities()
        with s.TRAINING_PATH.open() as handle:
            cls.training = list(csv.DictReader(handle))

    def select(self, seed=7):
        return s.select_temporal_paired_residual(TARGET, KICKOFF, seed)

    def test_chronology_counts_parse_join(self):
        self.assertEqual(self.audit['training_rows'], 1710)
        self.assertEqual(len(self.chronology), 855)
        self.assertEqual(len(self.pairs), 839)
        for game in self.chronology:
            rows = [r for r in self.training if r['game_id'] == game]
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(datetime.strptime(row['game_date']+' '+row['gametime'], '%Y-%m-%d %H:%M'), self.chronology[game])
        self.assertTrue(all(p[0]['game_id'] in self.chronology for p in self.pairs))

    def test_population_exact_and_exclusions(self):
        with s.BANK_PATH.open() as handle:
            raw = list(csv.DictReader(handle))
        dates = {r['game_id']: datetime.strptime(r['game_date']+' '+r['gametime'], '%Y-%m-%d %H:%M') for r in self.training}
        cutoff = datetime.fromisoformat(KICKOFF)
        expected = sorted({r['game_id'] for r in raw if dates[r['game_id']] < cutoff})
        captured = []
        # Observe the complete population used by the first draw, not only output flags.
        import sys
        def trace(frame, event, arg):
            if frame.f_code is s.select_temporal_paired_residual.__code__ and event == 'return':
                captured.extend(p[0]['game_id'] for p in frame.f_locals['eligible'])
            return trace
        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            result = self.select()
        finally:
            sys.settrace(previous)
        self.assertEqual(captured, expected)
        self.assertEqual(result['eligible_game_count'], len(expected))
        self.assertNotIn(TARGET, captured)
        self.assertTrue(all(dates[g] < cutoff for g in captured))
        self.assertGreater(result['temporal_audit']['same_kickoff_game_count'], 1)
        self.assertGreater(result['temporal_audit']['later_game_count'], 0)
        self.assertLess(datetime.fromisoformat(result['selected_historical_kickoff']), cutoff)

    def test_determinism_and_fixed_seed_difference(self):
        a = self.select(7)
        self.assertEqual(a, self.select(7))
        b = self.select(11)
        self.assertNotEqual(a['selected_historical_game'], b['selected_historical_game'])
        rng = random.Random(7)
        rng.random()
        self.assertEqual(a['orientation'], int(rng.random()*2))

    def test_two_local_draws_global_states_untouched(self):
        py_before, np_before = random.getstate(), np.random.get_state()
        real = random.Random
        with patch.object(s.random, 'Random', wraps=real) as constructor:
            local = real(7)
            with patch.object(local, 'random', wraps=local.random) as draws:
                constructor.return_value = local
                self.select()
                self.assertEqual(draws.call_count, 2)
                constructor.assert_called_once_with(7)
        self.assertEqual(py_before, random.getstate())
        after = np.random.get_state()
        self.assertEqual(np_before[0], after[0])
        np.testing.assert_array_equal(np_before[1], after[1])
        self.assertEqual(np_before[2:], after[2:])

    def test_pair_vectors_both_orientations(self):
        for orientation_value in (0.1, 0.9):
            with patch.object(s.random, 'Random') as factory:
                factory.return_value.random.side_effect = [0.4, orientation_value]
                result = self.select()
            game = result['selected_historical_game']['game_id']
            pair = next(p for p in self.pairs if p[0]['game_id'] == game)
            expected = pair if orientation_value < .5 else pair[::-1]
            for index, row in enumerate(expected, 1):
                identity = result[f'team_{index}_residual_identity']
                self.assertEqual(identity, {k:row[k] for k in (*s.paired.frozen.IDENTITY_FIELDS, 'row_index')})
                self.assertEqual(result[f'team_{index}_residuals'], {k:row[k] for k in s.paired.frozen.RESIDUAL_FIELDS})
                self.assertEqual(len(result[f'team_{index}_residuals']), 6)
            self.assertEqual(expected[0]['opponent_team'], expected[1]['team'])
            self.assertEqual(expected[1]['opponent_team'], expected[0]['team'])

    def test_invalid_inputs_no_rng(self):
        with patch.object(s.random, 'Random') as factory:
            for seed in (None, True, False, 1.0, '7'):
                with self.subTest(seed=seed), self.assertRaises(s.SelectorError):
                    s.select_temporal_paired_residual(TARGET, KICKOFF, seed)
            for target, kickoff in ((None,KICKOFF), ('',KICKOFF), ('missing',KICKOFF),
                    (TARGET,None), (TARGET,'bad'), (TARGET,'2024-02-30T13:00'),
                    (TARGET,KICKOFF+'Z'), (TARGET,'2024-10-28T13:00')):
                with self.subTest(target=target,kickoff=kickoff), self.assertRaises(s.SelectorError):
                    s.select_temporal_paired_residual(target,kickoff,7)
            earliest = min(self.chronology, key=self.chronology.get)
            with self.assertRaises(s.SelectorError):
                s.select_temporal_paired_residual(earliest,self.chronology[earliest].isoformat(),7)
            factory.assert_not_called()

    def test_failed_draw_not_retried(self):
        with patch.object(s.random, 'Random') as factory:
            factory.return_value.random.side_effect = RuntimeError('draw failure')
            with self.assertRaises(RuntimeError):
                self.select()
            self.assertEqual(factory.return_value.random.call_count, 1)

    def test_corrupt_chronology_fail_closed(self):
        mutations = [lambda r:r.pop(), lambda r:r[0].update(game_id='alien'),
                     lambda r:r[0].update(game_date=''), lambda r:r[0].update(gametime=''),
                     lambda r:r[0].update(gametime='99:00'), lambda r:r[0].update(gametime='12:00'),
                     lambda r:r[0].update(opponent_team='BAD'), lambda r:r[0].update(season='9999')]
        for mutate in mutations:
            rows = copy.deepcopy(self.training)
            mutate(rows)
            with self.assertRaises(s.SelectorError):
                s._validate_chronology(rows,self.pairs)
        pairs = copy.deepcopy(self.pairs)
        pairs[0][0]['game_id'] = 'missing'
        with self.assertRaises(s.SelectorError):
            s._validate_chronology(self.training,pairs)
        pairs = copy.deepcopy(self.pairs)
        pairs[0][0]['team'] = 'BAD'
        with self.assertRaises(s.SelectorError):
            s._validate_chronology(self.training,pairs)

    def test_ambiguous_identity_and_two_row_gate(self):
        rows = copy.deepcopy(self.training)
        rows[0]['game_id'] = rows[2]['game_id']
        with self.assertRaisesRegex(s.SelectorError, 'two training rows'):
            s._validate_chronology(rows, self.pairs)
        rows = copy.deepcopy(self.training)
        for i in (2, 3):
            for key in ('game_date', 'gametime', 'team', 'opponent_team'):
                rows[i][key] = rows[i-2][key]
        with self.assertRaisesRegex(s.SelectorError, 'multiple game IDs'):
            s._validate_chronology(rows, self.pairs)

    def test_second_draw_failure_not_retried(self):
        with patch.object(s.random, 'Random') as factory:
            factory.return_value.random.side_effect = [0.2, RuntimeError('orientation failure')]
            with self.assertRaises(RuntimeError):
                self.select()
            self.assertEqual(factory.return_value.random.call_count, 2)

    def test_bad_pair_rejected(self):
        rows = s.paired.frozen._load_bank(s.BANK_PATH)[0]
        rows[0]['opponent_team'] = 'BAD'
        with self.assertRaises(s.SelectorError):
            s.paired._paired_population(rows)

    def test_no_realization_or_simulation(self):
        with patch.object(s.paired, '_team_result', side_effect=AssertionError), patch.object(s.paired.frozen, '_realize', side_effect=AssertionError), patch.object(s.paired, 'sample_paired_residual', side_effect=AssertionError):
            result = self.select()
        tree = ast.parse(Path(s.__file__).read_text())
        imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any(any(x in line.lower() for x in ('generator','harness','numpy','fanduel')) for line in imports))
        self.assertFalse(any(x in json.dumps(result) for x in ('realized_primitives','fantasy_points','sampled_points','reconciliation')))
        self.assertEqual(result['historical_temporal_provenance'], 'UNVERIFIED')
        self.assertEqual(result['authorization'], dict(monte_carlo=False,repeated_game_simulation=False,production_influence='NONE',fanduel_solver_influence='NONE'))

    def test_frozen_components_unchanged(self):
        root = s.TRAINING_PATH.parents[1]
        before = json.loads(Path(s.__file__).with_name('TEMPORAL_RESIDUAL_SELECTOR_V1_FROZEN_HASHES_BEFORE.json').read_text())['files']
        for name, digest in before.items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),digest,name)


if __name__ == '__main__':
    unittest.main()
