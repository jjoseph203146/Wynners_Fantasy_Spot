"""Offline hostile-envelope and immutable-publication checks."""
import copy
import errno
import json
import os
from pathlib import Path
from unittest.mock import patch

from . import live_shadow_runner as runner
from .test_live_shadow_runner_v1_3 import Harness
from shadow_player_analysis_v1_2 import core
from . import handoff


class AdversarialRunnerTests(Harness):
    def published(self):
        return Path(self.run_shadow(publish=True)['bundle_path'])

    def rejected(self, path):
        with self.assertRaises((ValueError, OSError, TypeError)):
            runner.verify_bundle(path)
        with self.assertRaises((ValueError, OSError, TypeError)):
            self.run_shadow(publish=True)

    def test_altered_artifact(self):
        path = self.published()
        (path / 'source_capture.json').write_bytes(b'{}')
        self.rejected(path)

    def test_extra_artifact(self):
        path = self.published()
        (path / 'extra.json').write_bytes(b'{}')
        self.rejected(path)

    def test_missing_artifact(self):
        path = self.published()
        (path / 'source_capture.json').unlink()
        self.rejected(path)

    def test_forged_run_id(self):
        path = self.published()
        file = path / 'manifest.json'
        manifest = json.loads(file.read_bytes())
        manifest['run_id'] = '0' * 64
        file.write_bytes(runner.canonical(manifest))
        self.rejected(path)

    def test_forged_hash(self):
        path = self.published()
        file = path / 'manifest.json'
        manifest = json.loads(file.read_bytes())
        manifest['files']['source_capture.json']['sha256'] = '0' * 64
        file.write_bytes(runner.canonical(manifest))
        self.rejected(path)

    def test_forged_size(self):
        path = self.published()
        file = path / 'manifest.json'
        manifest = json.loads(file.read_bytes())
        manifest['files']['source_capture.json']['bytes'] += 1
        file.write_bytes(runner.canonical(manifest))
        self.rejected(path)

    def test_rehashed_forged_manifest(self):
        path = self.published()
        file = path / 'manifest.json'
        manifest = json.loads(file.read_bytes())
        manifest['files']['source_capture.json']['sha256'] = '0' * 64
        del manifest['run_id']
        manifest['run_id'] = runner._hash(runner.canonical(manifest))
        file.write_bytes(runner.canonical(manifest))
        renamed = path.with_name(manifest['run_id'])
        path.rename(renamed)
        with self.assertRaises(ValueError):
            runner.verify_bundle(renamed)

    def test_symlink_artifact(self):
        path = self.published()
        artifact = path / 'handoff.json'
        target = Path(self.tmp.name) / 'target.json'
        artifact.rename(target)
        artifact.symlink_to(target)
        self.rejected(path)

    def test_symlink_manifest(self):
        path = self.published()
        artifact = path / 'manifest.json'
        target = Path(self.tmp.name) / 'target.json'
        artifact.rename(target)
        artifact.symlink_to(target)
        self.rejected(path)

    def test_symlink_bundle(self):
        path = self.published()
        target = Path(self.tmp.name) / 'target'
        path.rename(target)
        path.symlink_to(target, target_is_directory=True)
        self.rejected(path)

    def test_symlink_output(self):
        target = Path(self.tmp.name) / 'target'
        target.mkdir()
        self.output.symlink_to(target, target_is_directory=True)
        with self.assertRaises(OSError):
            self.run_shadow(publish=True)
        self.assertEqual(list(target.iterdir()), [])

    def test_dangling_destination_symlink(self):
        run_id = self.run_shadow()['run_id']
        self.output.mkdir()
        (self.output / run_id).symlink_to(self.output / 'absent')
        with self.assertRaises((ValueError, OSError)):
            self.run_shadow(publish=True)

    def test_conflicting_empty_destination(self):
        run_id = self.run_shadow()['run_id']
        destination = self.output / run_id
        destination.mkdir(parents=True)
        with self.assertRaises(ValueError):
            self.run_shadow(publish=True)
        self.assertEqual(list(destination.iterdir()), [])

    def test_conflicting_destination_file(self):
        run_id = self.run_shadow()['run_id']
        self.output.mkdir()
        (self.output / run_id).write_bytes(b'KEEP')
        with self.assertRaises(OSError):
            self.run_shadow(publish=True)
        self.assertEqual((self.output / run_id).read_bytes(), b'KEEP')

    def test_malformed_capture(self):
        original = copy.deepcopy(self.capture)
        for field, value in [('events', {}), ('quarantine', [None]), ('retrieval', None),
                             ('claims', [{}]), ('consensus', [{}]), ('safety', {})]:
            with self.subTest(field=field):
                self.capture = copy.deepcopy(original)
                self.capture[field] = value
                with self.assertRaises((ValueError, TypeError)):
                    self.run_shadow()

    def test_malformed_feed_hash(self):
        for value in (None, '', 'a' * 63, 'A' * 64, '../bad', 123):
            with self.subTest(value=value):
                self.capture['retrieval']['feed_sha256'] = value
                with self.assertRaises(ValueError):
                    self.run_shadow()

    def test_source_boundary(self):
        self.capture['retrieval']['endpoint'] = 'https://example.invalid/article'
        with self.assertRaises(ValueError):
            self.run_shadow()

    def test_mismatched_safety(self):
        for key in runner.SAFETY:
            with self.subTest(key=key):
                self.capture['safety'][key] = not runner.SAFETY[key]
                with self.assertRaises(ValueError):
                    self.run_shadow()
                self.capture['safety'][key] = runner.SAFETY[key]

    def test_cutoff_fails_before_fetch(self):
        from shadow_player_analysis_v1 import espn_nfl_rss_v1 as espn
        for value in (None, '', '2026-09-27', '2026-09-27T16:00:00',
                      '2026-09-27T16:00:00-04:00', '2026-09-27 16:00:00Z',
                      '2026-02-30T16:00:00Z', ' 2026-09-27T16:00:00Z'):
            with self.subTest(value=value), patch.object(espn, 'fetch') as mocked:
                with self.assertRaises(ValueError):
                    self.run_shadow(cutoff_utc=value)
                mocked.assert_not_called()

    def test_invalid_season_week(self):
        for field in ('season', 'week'):
            for value in (None, 0, -1, True, 3.0, '3'):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.run_shadow(**{field: value})

    def test_malformed_authorities(self):
        original = copy.deepcopy(self.authorities)
        for key, field, value in [('identity', 'players', []), ('identity', 'players', [{}]),
                                  ('identity', 'authority', 'ESPN'),
                                  ('schedule', 'games', [{}]), ('schedule', 'games', []),
                                  ('schedule', 'available_at_utc', '2099-01-01T00:00:00Z')]:
            with self.subTest(field=field, value=value):
                self.authorities = copy.deepcopy(original)
                self.authorities[key][field] = value
                with self.assertRaises(ValueError):
                    self.run_shadow()

    def test_malformed_v12(self):
        for value in (None, [], {}, {'source_events': []}):
            with self.subTest(value=value), patch.object(core, 'build', return_value=value):
                with self.assertRaises(ValueError):
                    self.run_shadow()

    def test_malformed_handoff(self):
        for value in (None, [], {}, {'safety': runner.SAFETY}):
            with self.subTest(value=value), patch.object(handoff, 'build_shadow_handoff', return_value=value):
                with self.assertRaises(ValueError):
                    self.run_shadow()

    def test_consensus_promotion_rejected(self):
        original = handoff.build_shadow_handoff
        def promoted(**kwargs):
            value = original(**kwargs)
            value['claims']['claims'] = [{'consensus_eligible': True}]
            return value
        with patch.object(handoff, 'build_shadow_handoff', promoted):
            with self.assertRaises(ValueError):
                self.run_shadow()

    def test_downstream_mutation_does_not_alter_capture_or_snapshot(self):
        original = core.build
        before = copy.deepcopy(self.capture)
        def mutate(records, identity, schedule, cutoff, approvals):
            result = original(records, identity, schedule, cutoff, approvals)
            records.clear()
            identity.clear()
            schedule.clear()
            return result
        with patch.object(core, 'build', mutate):
            result = self.run_shadow()
        self.assertEqual(result['source_capture'], before)
        self.assertEqual(result['authorities'], self.authorities)

    def test_write_failure_no_partial_run(self):
        with patch.object(os, 'fsync', side_effect=OSError('injected')):
            with self.assertRaises(OSError):
                self.run_shadow(publish=True)
        if self.output.exists():
            self.assertEqual(list(self.output.iterdir()), [])

    def test_rename_failure_no_partial_run(self):
        with patch.object(runner, '_rename_exclusive', side_effect=OSError(errno.EIO, 'injected')):
            with self.assertRaises(OSError):
                self.run_shadow(publish=True)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_stage_verification_failure_no_partial_run(self):
        with patch.object(runner, '_verify_fd', side_effect=ValueError('injected')):
            with self.assertRaises(ValueError):
                self.run_shadow(publish=True)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_exclusive_rename_preserves_racing_destination(self):
        original = runner._rename_exclusive
        def collision(fd, stage, destination):
            os.mkdir(destination, dir_fd=fd)
            return original(fd, stage, destination)
        with patch.object(runner, '_rename_exclusive', collision):
            with self.assertRaises(ValueError):
                self.run_shadow(publish=True)
        self.assertEqual(len(list(self.output.iterdir())), 1)
        self.assertEqual(list(next(self.output.iterdir()).iterdir()), [])

    def test_noncanonical_artifact(self):
        path = self.published()
        file = path / 'handoff.json'
        file.write_bytes(file.read_bytes() + b'\n')
        self.rejected(path)

    def test_payload_fsync_failure_cleans_stage(self):
        original = os.fsync
        count = 0
        def fail(fd):
            nonlocal count
            count += 1
            if count == 3:
                raise OSError('payload fsync injected')
            return original(fd)
        with patch.object(os, 'fsync', fail):
            with self.assertRaises(OSError):
                self.run_shadow(publish=True)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_symlink_output_parent(self):
        parent = Path(self.tmp.name) / 'linked'
        parent.symlink_to(Path(self.tmp.name), target_is_directory=True)
        with patch.object(runner, 'OUTPUT', parent / 'artifacts'):
            with self.assertRaises(OSError):
                self.run_shadow(publish=True)
