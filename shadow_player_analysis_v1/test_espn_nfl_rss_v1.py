"""Synthetic RSS only. All network entry points are mocked."""
import copy
from email.message import Message
import hashlib
import importlib
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error
from xml.sax.saxutils import escape

from . import espn_nfl_rss_v1 as rss
from . import espn_rss_publication as publication
from .core import SAFETY, consensus
from .fixtures import sample
from .publication import verify

PUB = "Sun, 07 Sep 2025 10:00:00 GMT"
SEEN = "2025-09-07T10:02:00+00:00"
TEXT = "Coach says Example Runner will take over starting role"
ARTICLE = "https://example.invalid/article-never-fetch"
OBSERVED_PUB = "Sun, 27 Sep 2026 15:03:12 EST"
OBSERVED_SEEN = "2026-09-27T19:50:02.423813+00:00"


def item(guid="stable-1", title=TEXT, pub=PUB, summary="RSS supplied summary"):
    return '<item>' + ''.join(f'<{k}>{escape(v)}</{k}>' for k, v in
        (("guid", guid), ("title", title), ("pubDate", pub), ("description", summary), ("link", ARTICLE))) + '</item>'


def feed(*items):
    return ('<rss version="2.0"><channel><title>Synthetic NFL</title>' + ''.join(items) + '</channel></rss>').encode()


class ESPNConnectorTests(unittest.TestCase):
    def ingest(self, **kw):
        return rss.ingest(feed(item(**kw)), SEEN)

    def review(self, capture=None, fixture=None):
        f = fixture or sample()
        event = (capture or self.ingest())["events"][0]
        claim = dict(f["claims"][0], evidence_quote=event["evidence_text"],
                     forward_looking_basis="will take over starting role")
        return rss.evaluate_candidate(claim, event, f["identity"], f["game"], f["cutoff_utc"])

    def mock_fetch(self, *, status=200, content_type="application/rss+xml", error=None):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = status
        response.geturl.return_value = rss.ENDPOINT
        response.headers = Message()
        response.headers['Content-Type'] = content_type
        response.read.return_value = feed(item())
        opener = MagicMock()
        opener.open.return_value = response
        opener.open.side_effect = error
        return opener

    def test_valid_item(self):
        r = self.ingest()
        self.assertEqual(len(r['events']), 1)
        self.assertEqual(r['events'][0]['source'], 'ESPN_NFL_RSS_V1')
        self.assertEqual(r['events'][0]['summary'], 'RSS supplied summary')
        self.assertEqual(r['claims'], [])

    def test_guid_identity(self):
        a, b = self.ingest(), self.ingest(title='Changed headline')
        self.assertEqual(a['events'][0]['event_key'], b['events'][0]['event_key'])

    def test_fallback_identity(self):
        a = self.ingest(guid='')['events'][0]
        b = self.ingest(guid='', title='Revision')['events'][0]
        self.assertEqual(a['source_event_id'], b['source_event_id'])
        self.assertTrue(a['source_event_id'].startswith('derived:'))

    def test_duplicate(self):
        self.assertEqual(len(rss.ingest(feed(item(), item()), SEEN)['events']), 1)

    def test_refetch(self):
        first = self.ingest()
        later = rss.ingest(feed(item()), '2025-09-07T11:00:00Z', first['events'])
        self.assertEqual(first['events'], later['events'])

    def test_unversioned_revision(self):
        first = self.ingest()
        raw = feed(item(title='Changed'), item())
        result = rss.ingest(raw, SEEN, first['events'])
        self.assertEqual(result['events'], [])
        self.assertIn('UNVERSIONED_CONTENT_CHANGE', str(result['quarantine']))
        reordered = rss.ingest(feed(item(), item(title='Changed')), SEEN, first['events'])
        self.assertEqual(result['events'], reordered['events'])
        self.assertEqual(result['quarantine'], reordered['quarantine'])

    def test_versioned_revision(self):
        first = self.ingest()
        revised = rss.ingest(feed(item(title='Changed', pub='Sun, 07 Sep 2025 11:00:00 GMT')),
                             '2025-09-07T12:00:00Z', first['events'])
        self.assertEqual(len(revised['events']), 2)
        self.assertEqual(revised['events'][1]['supersedes_revision_id'], first['events'][0]['revision_id'])

    def test_publication_timestamp(self):
        self.assertEqual(self.ingest()['events'][0]['published_at_utc'], '2025-09-07T10:00:00+00:00')

    def test_retrieval_timestamp(self):
        self.assertEqual(self.ingest()['events'][0]['retrieved_at_utc'], SEEN)

    def test_missing_timestamp(self):
        self.assertEqual(self.ingest(pub='')['events'], [])

    def test_malformed_timestamp(self):
        self.assertEqual(self.ingest(pub='garbage GMT')['events'], [])

    def test_ambiguous_timezone(self):
        self.assertEqual(self.ingest(pub='Sun, 07 Sep 2025 10:00:00')['events'], [])
        self.assertEqual(self.ingest(pub='Sun, 07 Sep 2025 10:00:00 EST')['events'], [])

    def test_malformed_xml(self):
        with self.assertRaises(rss.FeedError): rss.ingest(b'<rss>', SEEN)

    def test_empty_feed(self):
        with self.assertRaises(rss.FeedError): rss.ingest(feed(), SEEN)

    def test_unexpected_feed(self):
        with self.assertRaises(rss.FeedError): rss.ingest(b'<html/>', SEEN)

    def test_http_failure(self):
        for opener in (self.mock_fetch(status=500), self.mock_fetch(error=urllib.error.URLError('failure'))):
            with patch.object(rss.urllib.request, 'build_opener', return_value=opener):
                with self.assertRaises(rss.FeedError): rss.fetch()

    def test_timeout(self):
        with patch.object(rss.urllib.request, 'build_opener', return_value=self.mock_fetch(error=TimeoutError())):
            with self.assertRaises(rss.FeedError): rss.fetch()

    def test_content_type(self):
        with patch.object(rss.urllib.request, 'build_opener', return_value=self.mock_fetch(content_type='text/html')):
            with self.assertRaises(rss.FeedError): rss.fetch()

    def test_article_url(self):
        self.assertEqual(self.ingest()['events'][0]['article_url'], ARTICLE)

    def test_only_feed_requested(self):
        opener = self.mock_fetch()
        with patch.object(rss.urllib.request, 'build_opener', return_value=opener), patch.object(socket, 'socket', side_effect=AssertionError('network')):
            rss.fetch()
        opener.open.assert_called_once()
        args, kwargs = opener.open.call_args
        self.assertEqual(args[0].full_url, rss.ENDPOINT)
        self.assertEqual(kwargs['timeout'], rss.TIMEOUT_SECONDS)
        self.assertIn('RSS client', args[0].get_header('User-agent'))

    def test_redirect_forbidden(self):
        with self.assertRaisesRegex(rss.FeedError, 'REDIRECT_FORBIDDEN'):
            rss._NoRedirect().redirect_request(None, None, 302, '', {}, ARTICLE)

    def test_import_no_request(self):
        with patch.object(socket, 'socket', side_effect=AssertionError('network')), patch.object(rss.urllib.request, 'build_opener', side_effect=AssertionError('request')):
            importlib.reload(rss)

    def test_post_kickoff(self):
        for hour in (17, 18):
            capture = rss.ingest(feed(item(pub=f'Sun, 07 Sep 2025 {hour}:00:00 GMT')), f'2025-09-07T{hour}:01:00Z')
            self.assertIn('POST_KICKOFF_PUBLICATION', self.review(capture)['reason_codes'])

    def test_late_retrieval(self):
        capture = rss.ingest(feed(item()), '2025-09-07T19:00:00Z')
        self.assertIn('EVIDENCE_AFTER_CUTOFF', self.review(capture)['reason_codes'])

    def test_recap(self):
        capture = self.ingest(title='Example Runner catches 9 passes for 130 yards in win')
        self.assertEqual(capture['events'][0]['source_phase'], 'POST_GAME_RECAP')
        self.assertEqual(self.review(capture)['eligibility_status'], 'QUARANTINED')
        self.assertEqual(capture['claims'], [])

    def test_insufficient(self):
        capture = self.ingest(title='NFL news today')
        self.assertEqual(capture['events'][0]['source_phase'], 'INSUFFICIENT_UNRESOLVED')
        self.assertEqual(self.review(capture)['eligibility_status'], 'QUARANTINED')

    def test_unresolved_identity(self):
        f = sample()
        f['claims'][0]['player_name'] = 'Unknown Player'
        self.assertEqual(self.review(fixture=f)['eligibility_status'], 'QUARANTINED')
        self.assertIsNone(self.review(fixture=f)['gsis_id'])

    def test_ambiguous_identity(self):
        f = sample()
        f['identity']['players'].append(dict(f['identity']['players'][0], gsis_id='OTHER'))
        self.assertEqual(self.review(fixture=f)['identity_reason'], 'AMBIGUOUS_EXACT_IDENTITY')

    def test_origin_consensus(self):
        first = self.review()
        second = self.review(self.ingest(guid='second'))
        self.assertEqual(first['eligibility_status'], 'ELIGIBLE')
        group = consensus([first, second])[0]
        self.assertEqual(group['independent_origin_count'], 1)
        self.assertEqual(group['status'], 'SINGLE_ORIGIN')

    def test_deterministic_bundle(self):
        capture = self.ingest()
        self.assertEqual(publication.bundle(capture), publication.bundle(copy.deepcopy(capture)))

    def test_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, data in publication.bundle(self.ingest()).items(): (root / name).write_bytes(data)
            self.assertEqual(verify(root)['safety'], SAFETY)
            (root / 'events.json').write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'ARTIFACT_HASH'): verify(root)

    def test_immutable_publication_no_database(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / 'nfl.db'
            ledger = root / 'forecast_ledger.db'
            for path in (db, ledger): path.write_bytes(b'untouched database sentinel')
            before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (db, ledger)]
            with patch.object(publication, 'ROOT', root), patch.object(publication, 'OUTPUT', root / 'espn_nfl_rss_v1_artifacts'), patch.object(sqlite3, 'connect', side_effect=AssertionError('DB')), patch.object(socket, 'socket', side_effect=AssertionError('network')):
                capture = self.ingest()
                path = publication.publish(capture)
                self.assertEqual(path, publication.publish(capture))
                verify(path)
                self.assertEqual(len(list(path.parent.iterdir())), 1)
                (path / 'events.json').write_text('corrupt')
                with self.assertRaises(ValueError): publication.publish(capture)
                self.assertEqual((path / 'events.json').read_text(), 'corrupt')
            self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in (db, ledger)])

    def test_failed_stage_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(publication, 'ROOT', root), patch.object(publication, 'OUTPUT', root / 'espn_nfl_rss_v1_artifacts'), patch.object(publication, 'verify', side_effect=ValueError('fail')):
                with self.assertRaises(ValueError): publication.publish(self.ingest())
                self.assertEqual(list(publication.OUTPUT.iterdir()), [])

    def test_entities_forbidden(self):
        with self.assertRaises(rss.FeedError): rss.ingest(b'<!DOCTYPE rss [<!ENTITY x "bad">]><rss/>', SEEN)

    def test_future_publication(self):
        self.assertEqual(self.ingest(pub='Mon, 08 Sep 2025 10:00:00 GMT')['events'], [])

    def observed(self, raw=None):
        return rss.ingest(raw or feed(item(pub=OBSERVED_PUB)), OBSERVED_SEEN)

    def test_observed_est_unresolved(self):
        result = self.observed()
        row = result['quarantine'][0]
        self.assertEqual(result['events'], [])
        self.assertEqual(row['temporal_status'], 'AMBIGUOUS_SOURCE_TIMESTAMP')
        self.assertEqual(row['temporal_confidence'], 'UNRESOLVED')
        self.assertEqual(row['eligibility_status'], 'QUARANTINED')
        self.assertIsNone(row['published_at_utc'])

    def test_no_est_to_edt_or_literal_est_conversion(self):
        for pub in (OBSERVED_PUB, 'Sun, 07 Sep 2025 01:00:00 EST'):
            with self.assertRaisesRegex(ValueError, 'INVALID_PUBLICATION_TIMESTAMP'):
                rss._publication(pub)

    def test_observation_times_not_publication(self):
        row = self.observed()['quarantine'][0]
        for field in ('retrieved_at_utc', 'first_seen_at_utc'):
            self.assertEqual(row[field], OBSERVED_SEEN)
            self.assertNotEqual(row['published_at_utc'], row[field])

    def test_exact_raw_date_and_provenance(self):
        raw_date = '  ' + OBSERVED_PUB + '  '
        row = self.observed(feed(item(guid=' original-guid ', pub=raw_date)))['quarantine'][0]
        self.assertEqual(row['raw_feed_fields']['pubDate'], raw_date)
        self.assertEqual(row['raw_feed_fields']['guid'], ' original-guid ')
        self.assertEqual(row['source_event_id'], 'guid:original-guid')
        self.assertEqual((row['source'], row['origin_id']), (rss.SOURCE, rss.ORIGIN))
        self.assertEqual(row['headline'], TEXT)
        self.assertEqual(row['summary'], 'RSS supplied summary')
        self.assertEqual(row['article_url'], ARTICLE)
        self.assertEqual(row['content_hash'], rss.digest(row['raw_feed_fields']))

    def test_ambiguous_manual_review_fails_closed(self):
        f = sample()
        event = self.observed()['quarantine'][0]
        claim = dict(f['claims'][0], evidence_quote=event['evidence_text'],
                     forward_looking_basis='will take over starting role')
        with self.assertRaisesRegex(ValueError, 'UNRESOLVED_SOURCE_TIMESTAMP'):
            rss.evaluate_candidate(claim, event, f['identity'], f['game'], f['cutoff_utc'])

    def test_ambiguous_no_claim_or_consensus_vote(self):
        result = self.observed()
        self.assertEqual(result['claims'], [])
        self.assertEqual(result['consensus'], [])
        self.assertEqual(consensus(result['quarantine']), [])

    def test_nine_unrelated_guids_timestamp_diagnostic(self):
        items = [item(guid=f'unrelated-{i}', title=f'Unrelated story {i}', pub=OBSERVED_PUB)
                 for i in range(9)]
        result = self.observed(feed(*items))
        diagnostic = result['retrieval']['source_timestamp_diagnostics']
        self.assertEqual(diagnostic, [dict(raw_pubDate=OBSERVED_PUB,
            repeated_source_timestamp_count=9, source_timestamp_suspect=True)])
        self.assertEqual(result, self.observed(feed(*items)))
        reversed_result = self.observed(feed(*reversed(items)))
        self.assertEqual(diagnostic, reversed_result['retrieval']['source_timestamp_diagnostics'])
        self.assertEqual(result['quarantine'], reversed_result['quarantine'])
        self.assertEqual(len(result['quarantine']), 9)
        self.assertEqual(result['events'], [])
        self.assertEqual(consensus(result['quarantine']), [])
        for row in result['quarantine']:
            self.assertIsNone(row['published_at_utc'])
            self.assertEqual(row['temporal_confidence'], 'UNRESOLVED')
            self.assertEqual(row['eligibility_status'], 'QUARANTINED')

    def test_duplicate_guid_not_timestamp_corroboration(self):
        result = self.observed(feed(*[item(pub=OBSERVED_PUB)] * 9))
        diagnostic = result['retrieval']['source_timestamp_diagnostics'][0]
        self.assertEqual(diagnostic['repeated_source_timestamp_count'], 1)
        self.assertFalse(diagnostic['source_timestamp_suspect'])

    def test_numeric_offset_normal_review(self):
        result = self.ingest(pub='Sun, 07 Sep 2025 06:00:00 -0400')
        self.assertEqual(result['events'][0]['published_at_utc'], '2025-09-07T10:00:00+00:00')
        self.assertEqual(self.review(result)['eligibility_status'], 'ELIGIBLE')

    def test_numeric_offset_post_kickoff(self):
        result = rss.ingest(feed(item(pub='Sun, 07 Sep 2025 13:00:00 -0400')),
                            '2025-09-07T17:01:00Z')
        row = self.review(result)
        self.assertEqual(row['eligibility_status'], 'QUARANTINED')
        self.assertIn('POST_KICKOFF_PUBLICATION', row['reason_codes'])

    def test_future_numeric_offset_unresolved_order(self):
        result = self.observed(feed(item(pub=OBSERVED_PUB.replace('EST', '-0500'))))
        self.assertEqual(result['events'], [])
        row = result['quarantine'][0]
        self.assertEqual(row['published_at_utc'], '2026-09-27T20:03:12+00:00')
        self.assertEqual(row['temporal_status'], 'INVALID_SOURCE_TIME_ORDER')

    def test_diagnostics_do_not_change_valid_refetch(self):
        first = self.ingest()
        result = rss.ingest(feed(item(), item(guid='other')), SEEN, first['events'])
        self.assertEqual(len(result['events']), 2)
        self.assertTrue(result['retrieval']['source_timestamp_diagnostics'][0]['source_timestamp_suspect'])
        self.assertEqual(self.review(result)['eligibility_status'], 'ELIGIBLE')

    def test_ambiguous_capture_no_io_and_bundle_retention(self):
        with patch.object(sqlite3, 'connect', side_effect=AssertionError('DB')), \
             patch.object(socket, 'socket', side_effect=AssertionError('network')), \
             patch.object(rss.urllib.request, 'build_opener', side_effect=AssertionError('request')), \
             patch('builtins.open', side_effect=AssertionError('file mutation')):
            importlib.reload(rss)
            result = self.observed()
            payloads = publication.bundle(result)
        self.assertEqual(result['safety'], SAFETY)
        self.assertIn(OBSERVED_PUB.encode(), payloads['quarantine.json'])
        self.assertIn(b'AMBIGUOUS_SOURCE_TIMESTAMP', payloads['quarantine.json'])


if __name__ == '__main__':
    unittest.main()
