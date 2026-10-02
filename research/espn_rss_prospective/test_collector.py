"""Synthetic, network-free tests; temporary datasets stay beside the collector."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collector as c


RAW_EST = "Sun, 27 Sep 2026 18:48:31 EST"


def rss(pubdate=RAW_EST, copies=1, second=None):
    def item(date):
        return ('<item><guid>US-EN-123</guid><link>https://www.espn.com/nfl/story/_/id/123/test</link>'
                '<title>Test</title><description>Summary</description><pubDate>' + date + '</pubDate></item>')
    return ('<rss><channel>' + item(pubdate) * copies + (item(second) if second else '') + '</channel></rss>').encode()


API = json.dumps({"headlines": [{"id": 123, "originallyPosted": "2026-09-27T22:00:00Z",
    "published": "2026-09-27T22:35:00Z", "lastModified": "2026-09-27T18:35:00-04:00",
    "categorized": "2026-09-27T22:36:00Z", "links": {"web": {"href": "https://www.espn.com/nfl/story/_/id/123/test"}}}]}).encode()
HTML = b'''<html><link rel="canonical" href="https://www.espn.com/nfl/story/_/id/123/test">
<meta property="article:modified_time" content="2026-09-27T22:35:00Z">
<script type="application/ld+json">{"datePublished":"2026-09-27T22:00:00Z","dateModified":"2026-09-27T18:35:00-04:00"}</script>
<script>window.data={"updated":"2026-09-27T22:35:00+00:00"};</script></html>'''


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test_evidence_", dir=c.HERE)
        self.dataset = Path(self.temp.name) / "dataset"
        self.calls = []
        self.addCleanup(self.temp.cleanup)
        self.network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def fetcher(self, feed=None, failure=False):
        def fetch(url, deadline):
            self.calls.append(url)
            body = (rss() if feed is None else feed) if url == c.RSS_URL else API if url.startswith(c.API_BASE) else HTML
            fail = failure and url != c.RSS_URL
            return (b"service error" if fail else body), {
                "url": url, "started_at_utc": c.utc_now(), "ended_at_utc": c.utc_now(),
                "status": 503 if fail else 200, "headers": [["Content-Type", "test"]],
                "complete": True, "error": "HTTP 503" if fail else None}
        return fetch

    def test_new_guid_and_structured_metadata(self):
        path, run = c.capture(self.dataset, self.fetcher())
        self.assertEqual(run["status"], "PASS")
        item = run["items"][0]
        self.assertEqual(item["classification"], "NEW_GUID")
        self.assertEqual(item["content_id"], "123")
        self.assertEqual(item["metadata"]["api"]["timestamps"]["published"], "2026-09-27T22:35:00Z")
        self.assertEqual(len(item["metadata"]["html"]["structured_timestamps"]), 2)
        self.assertEqual(len(item["metadata"]["html"]["meta_timestamps"]), 1)
        self.assertEqual(len(item["metadata"]["html"]["embedded_explicit_timestamps"]), 3)
        self.assertTrue(path.is_dir())

    def test_changed_pubdate(self):
        c.capture(self.dataset, self.fetcher())
        changed = "Sun, 27 Sep 2026 19:48:31 EST"
        _, run = c.capture(self.dataset, self.fetcher(rss(changed)))
        self.assertEqual(run["items"][0]["classification"], "PUBDATE_CHANGED")
        self.assertEqual(run["items"][0]["previous_raw_pubDate"], RAW_EST)

    def test_unchanged_only_fetches_rss(self):
        c.capture(self.dataset, self.fetcher())
        self.calls.clear()
        _, run = c.capture(self.dataset, self.fetcher())
        self.assertEqual(run["items"][0]["classification"], "UNCHANGED")
        self.assertEqual(self.calls, [c.RSS_URL])

    def test_raw_est_exact_and_no_conversion(self):
        raw = "  " + RAW_EST + "  "
        path, run = c.capture(self.dataset, self.fetcher(rss(raw)))
        self.assertEqual(run["items"][0]["raw_pubDate"], raw)
        self.assertEqual((path / "raw/feed_rss.body").read_bytes(), rss(raw))
        self.assertFalse(run["timezone_conversion_performed"])
        self.assertEqual(run["timezone_decision"], "NONE")
        self.assertNotIn("published_at_utc", json.dumps(run))
        self.assertEqual(c.explicit_timestamps({"pubDate": RAW_EST}), [])

    def test_failures_keep_previous_evidence_and_raw_error_bodies(self):
        old, _ = c.capture(self.dataset, self.fetcher())
        original = {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()}
        _, run = c.capture(self.dataset, self.fetcher(rss("Mon, 28 Sep 2026 09:00:00 EST"), failure=True))
        self.assertEqual(run["status"], "PARTIAL")
        self.assertTrue(run["snapshot_valid"])
        self.assertEqual(original, {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()})
        self.assertEqual(len(list((self.dataset / "runs").iterdir())), 2)

    def test_malformed_rss_does_not_advance_snapshot(self):
        old, _ = c.capture(self.dataset, self.fetcher())
        bad, run = c.capture(self.dataset, self.fetcher(b"<rss>broken"))
        self.assertEqual(run["status"], "FAIL")
        self.assertEqual((bad / "raw/feed_rss.body").read_bytes(), b"<rss>broken")
        _, reference = c.latest_snapshot(self.dataset)
        self.assertEqual(reference["path"], str((old / "run.json").relative_to(self.dataset)))

    def test_duplicate_identical_guid_collapses_and_conflict_fails(self):
        _, run = c.capture(self.dataset, self.fetcher(rss(copies=2)))
        self.assertEqual(len(run["items"]), 1)
        self.assertEqual(run["duplicate_identical_occurrences"], 1)
        self.assertEqual(len(self.calls), 3)
        _, bad = c.capture(self.dataset, self.fetcher(rss(second="other raw date")))
        self.assertEqual(bad["status"], "FAIL")
        self.assertFalse(bad["snapshot_valid"])

    def test_atomic_commit_and_manifest_hashes(self):
        with patch.object(c.os, "rename", side_effect=OSError("interrupted commit")):
            with self.assertRaises(OSError):
                c.capture(self.dataset, self.fetcher())
        self.assertEqual(c.latest_snapshot(self.dataset), ({}, None))
        self.assertEqual(len(list(self.dataset.glob(".pending_*"))), 1)
        path, _ = c.capture(self.dataset, self.fetcher())
        for name, entry in json.loads((path / "manifest.json").read_text()).items():
            self.assertEqual(hashlib.sha256((path / name).read_bytes()).hexdigest(), entry["sha256"])
        with self.assertRaises(FileExistsError):
            c.write_new(path / "run.json", b"overwrite")

    def test_concurrent_runs_append_without_overwriting(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: c.capture(self.dataset, self.fetcher()), range(2)))
        self.assertNotEqual(results[0][0], results[1][0])
        self.assertEqual(len(list((self.dataset / "runs").glob("*/run.json"))), 2)

    def test_url_allowlist_and_id_conflict(self):
        for url in ["http://www.espn.com/a", "https://evil.example/a", "https://www.espn.com.evil/a", "https://user@www.espn.com/a"]:
            self.assertFalse(c.permitted_url(url))
        with self.assertRaises(ValueError):
            c.resolve_id({"guid": "US-EN-123", "link": "https://www.espn.com/nfl/story/_/id/456/x"})
        self.assertIsNone(c.NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.example"))

    def test_bad_shape_and_entities(self):
        for raw in [b"<html/>", b"<rss><channel/></rss>", b'<!DOCTYPE rss><rss><channel/></rss>']:
            with self.assertRaises(ValueError):
                c.parse_rss(raw)

    def test_http_expired_budget_never_opens_network(self):
        body, record = c.http_get(c.RSS_URL, 0)
        self.assertEqual(body, b"")
        self.assertFalse(record["attempted"])
        self.assertIsNotNone(record["error"])


if __name__ == "__main__":
    unittest.main()
