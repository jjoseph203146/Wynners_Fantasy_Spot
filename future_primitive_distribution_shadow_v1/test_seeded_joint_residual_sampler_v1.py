"""Bounded unit tests: only seeds 0 and 1; no distribution evaluation."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from future_primitive_distribution_shadow_v1 import seeded_joint_residual_sampler_v1 as s

CENTER = dict(zip(s.CENTER_FIELDS, (60, .55, 7, 4, 2, 1, 24)))


class SamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = s.BANK_PATH.read_bytes()
        cls.rows = s._parse_bank(cls.data)
        cls.result = s.sample_joint_residual(CENTER, seed=0)

    def realize(self, residuals=None, center=None):
        row = dict.fromkeys(s.RESIDUAL_FIELDS, 0)
        row.update(residuals or {})
        return s._realize(s._validate_center(center or CENTER), row)

    def corrupted_bank(self, change):
        rows = list(csv.DictReader(io.StringIO(self.data.decode())))
        change(rows)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=s.BANK_SCHEMA)
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode()

    def test_frozen_bank_hash(self):
        self.assertEqual(hashlib.sha256(self.data).hexdigest(), s.FROZEN_BANK_SHA256)
        self.assertEqual(self.result['residual_bank_sha256'], s.FROZEN_BANK_SHA256)

    def test_wrong_bank_hash_fails(self):
        with patch.object(Path, 'read_bytes', return_value=self.data + b'\n'):
            with self.assertRaisesRegex(s.SamplerError, 'hash mismatch'):
                s.sample_joint_residual(CENTER, seed=0)

    def test_same_seed_identical_output(self):
        self.assertEqual(self.result, s.sample_joint_residual(CENTER, seed=0))

    def test_different_seeds_different_rows(self):
        self.assertNotEqual(self.result['selected_residual_row']['row_index'],
                            s.sample_joint_residual(CENTER, seed=1)['selected_residual_row']['row_index'])

    def test_all_dimensions_same_row(self):
        row = self.result['selected_residual_row']
        self.assertEqual(row, self.rows[row['row_index']])
        self.assertTrue(all(key in row for key in (*s.IDENTITY_FIELDS, *s.RESIDUAL_FIELDS)))

    def test_exactly_one_rng_draw_no_independent_sampling(self):
        rng = random.Random(0)
        with patch.object(s.random, 'Random', wraps=random.Random) as factory:
            with patch.object(rng, 'random', wraps=rng.random) as draw:
                factory.return_value = rng
                result = s.sample_joint_residual(CENTER, seed=0)
                factory.assert_called_once_with(0)
                draw.assert_called_once_with()
                self.assertEqual(result, self.result)

    def test_global_rng_untouched(self):
        before = random.getstate()
        s.sample_joint_residual(CENTER, seed=0)
        self.assertEqual(before, random.getstate())

    def test_volume_conservation(self):
        p = self.result['realized_primitives']
        self.assertEqual(p['pass_attempts'] + p['carries'], self.result['realized_mechanisms']['plays'])

    def test_no_negative_opportunities(self):
        m, p, _ = self.realize({'resid_plays': -60})
        self.assertEqual((m['plays'], p['pass_attempts'], p['carries']), (0, 0, 0))

    def test_negative_plays_rejected(self):
        with self.assertRaisesRegex(s.SamplerError, 'negative sampled plays'):
            self.realize({'resid_plays': -61})

    def test_no_negative_yardage(self):
        _, p, _ = self.realize({'resid_pass_ypa': -10, 'resid_rush_ypc': -10})
        self.assertEqual((p['passing_yards'], p['rushing_yards']), (0, 0))

    def test_no_negative_tds(self):
        _, p, _ = self.realize({'resid_passing_tds': -10, 'resid_rushing_tds': -10})
        self.assertEqual((p['passing_tds'], p['rushing_tds']), (0, 0))

    def test_rounding_and_audit(self):
        center = dict(CENTER, expected_plays=2.5, expected_pass_rate=.5,
                      expected_pass_ypa=1.25, expected_rush_ypc=1.5)
        m, p, audit = self.realize(center=center)
        self.assertEqual(m['plays'], 3)
        self.assertEqual(p, dict(pass_attempts=2, carries=1, passing_yards=3,
                                 rushing_yards=2, passing_tds=2, rushing_tds=1))
        self.assertEqual([(a['field'], a['raw'], a['reconciled']) for a in audit],
                         [('plays', 2.5, 3), ('pass_attempts', 1.5, 2),
                          ('passing_yards', 2.5, 3), ('rushing_yards', 1.5, 2)])
        self.assertTrue(all(a['operation'] == 'round_half_up' for a in audit))
        self.assertEqual((m, p, audit), self.realize(center=center))

    def assert_bound(self, key, residual, expected, operation):
        m, p, audit = self.realize({'resid_' + key: residual})
        self.assertEqual({**m, **p}[key], expected)
        self.assertIn(dict(field=key, operation=operation,
                           raw=CENTER['expected_' + key] + residual, reconciled=expected), audit)

    def test_pass_rate_lower_bound(self):
        self.assert_bound('pass_rate', -1, 0, 'clamp_0_1')

    def test_pass_rate_upper_bound(self):
        self.assert_bound('pass_rate', 1, 1, 'clamp_0_1')

    def test_pass_ypa_negative(self):
        self.assert_bound('pass_ypa', -8, 0, 'floor_zero')

    def test_rush_ypc_negative(self):
        self.assert_bound('rush_ypc', -5, 0, 'floor_zero')

    def test_passing_td_negative(self):
        self.assert_bound('passing_tds', -3, 0, 'floor_zero')

    def test_rushing_td_negative(self):
        self.assert_bound('rushing_tds', -2, 0, 'floor_zero')

    def test_missing_center_fields(self):
        for key in s.CENTER_FIELDS:
            center = dict(CENTER)
            del center[key]
            with self.subTest(key=key), self.assertRaises(s.SamplerError):
                s.sample_joint_residual(center, seed=0)

    def test_nonfinite_center_fields(self):
        for key in s.CENTER_FIELDS:
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(key=key, value=value), self.assertRaises(s.SamplerError):
                    s.sample_joint_residual(dict(CENTER, **{key: value}), seed=0)

    def test_invalid_seed(self):
        for seed in (None, True, False, 1.0, '1'):
            with self.subTest(seed=seed), self.assertRaises(s.SamplerError):
                s.sample_joint_residual(CENTER, seed=seed)
        with self.assertRaises(s.SamplerError):
            s.sample_joint_residual(CENTER)

    def test_missing_residual_fields(self):
        for key in s.RESIDUAL_FIELDS:
            data = self.corrupted_bank(lambda rows: rows[0].update({key: ''}))
            with self.subTest(key=key), self.assertRaises(s.SamplerError):
                s._parse_bank(data)

    def test_nonfinite_residual_fields(self):
        for key in s.RESIDUAL_FIELDS:
            for value in ('nan', 'inf', '-inf'):
                data = self.corrupted_bank(lambda rows: rows[0].update({key: value}))
                with self.subTest(key=key, value=value), self.assertRaises(s.SamplerError):
                    s._parse_bank(data)

    def test_schema_mismatch(self):
        with self.assertRaisesRegex(s.SamplerError, 'schema mismatch'):
            s._parse_bank(self.data.replace(b'resid_plays', b'wrong_field', 1))

    def test_duplicate_canonical_identity(self):
        data = self.corrupted_bank(lambda rows: rows.append(dict(rows[0], opponent_team='OTHER')))
        with self.assertRaisesRegex(s.SamplerError, 'duplicate canonical'):
            s._parse_bank(data)

    def test_no_points_sampling(self):
        self.assertEqual(set(self.result['realized_primitives']),
                         {'pass_attempts', 'carries', 'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds'})
        self.assertNotIn('resid_points', self.result['selected_residual_row'])

    def test_points_diagnostic_only(self):
        other = s.sample_joint_residual(dict(CENTER, expected_points=999), seed=0)
        self.assertEqual(other['center']['expected_points'], 999)
        other['center']['expected_points'] = CENTER['expected_points']
        self.assertEqual(other, self.result)

    def test_authorization(self):
        self.assertEqual(self.result['authorization'], dict(monte_carlo=False,
                         production_influence='NONE', fanduel_solver_influence='NONE'))

    def test_center_not_mutated(self):
        center = copy.deepcopy(CENTER)
        result = s.sample_joint_residual(center, seed=0)
        self.assertEqual(center, CENTER)
        result['center']['expected_plays'] = 99
        self.assertEqual(center, CENTER)

    def test_bank_artifact_unchanged(self):
        self.assertEqual(s.BANK_PATH.read_bytes(), self.data)

    def test_frozen_dependencies_unchanged(self):
        before = json.loads(s.BANK_PATH.with_name('SEEDED_SAMPLER_V1_FROZEN_HASHES_BEFORE.json').read_text())
        root = s.BANK_PATH.resolve().parent.parent
        for path, digest in before.items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((root / path).read_bytes()).hexdigest(), digest)

    def test_arithmetic_overflow_fails(self):
        with self.assertRaises(s.SamplerError):
            self.realize(center=dict(CENTER, expected_pass_ypa=1e308))

    def test_fractional_td_center_rejected(self):
        with self.assertRaises(s.SamplerError):
            s.sample_joint_residual(dict(CENTER, expected_passing_tds=1.5), seed=0)

    def test_fractional_td_residual_rejected(self):
        data = self.corrupted_bank(lambda rows: rows[0].update(resid_passing_tds='0.5'))
        with self.assertRaises(s.SamplerError):
            s._parse_bank(data)


if __name__ == '__main__':
    unittest.main()
