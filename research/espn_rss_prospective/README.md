# Manual ESPN NFL RSS prospective evidence collector

This directory is isolated research. It has no production imports, database access,
scheduled jobs, production configuration changes or timezone interpretation.
M5C remains BLOCKED. Collection does not establish same-revision correspondence.

Run manually from the project root:

```sh
python3 -B research/espn_rss_prospective/collector.py --live
```

One invocation fetches RSS once and compares it with the latest completed, valid
RSS snapshot in this dataset. On the first run every GUID is NEW_GUID. Reappearing
GUIDs absent from the immediately preceding snapshot are also NEW_GUID. Changes
are exact pubDate text comparisons; titles/descriptions alone do not trigger
article fetches. XML-decoded pubDate text retains surrounding whitespace; the
unaltered response bytes preserve the original XML encoding and lexical form.
No date parser is applied to RSS pubDate.

Each NEW_GUID/PUBDATE_CHANGED item queues its public ESPN API and HTML requests
immediately after feed parsing, using four concurrent workers. A busy queue can
delay a request; every request has its own actual start/end interval. There is
no promise of simultaneous or same-revision capture. Unchanged items retain their
classification and RSS evidence; no article requests are made for them.

Network bounds: eight-second socket timeout, twelve-second response budget,
120-second run request budget, four workers, 100 RSS item limit, four MiB per
response. The response deadline is checked between reads, so a pending socket
operation can add up to one socket timeout. Local parsing/commit time is extra.
No automatic retries, redirects, proxy credentials, API keys or login are used.
Only HTTPS requests to www.espn.com, espn.com and content.core.api.espn.com are
permitted. Error responses and partial bodies are preserved, explicitly marked
incomplete when truncated or interrupted. Complete captures are unmodified bytes.

Evidence is append-only: `evidence/runs/<UTC timestamp>_<random suffix>/` contains
`run.json`, raw bodies and request records, and a SHA256 manifest. Request records
include status, headers, retrieval intervals, artifact path and body hash. Paths
inside run.json and manifests are relative to that run directory. API timestamps
are preserved as supplied, and HTML JSON-LD/meta/embedded explicit timestamps are
extracted without converting offsets. Nested API fields can concern related
content; JSON paths identify context and do not imply a revision relationship.

Runs are built in unique `.pending_*` directories and atomically renamed into
`runs/` only after all payload files are flushed. Interrupted pending directories
are preserved but ignored by snapshot discovery. No pointer file or historical
run is overwritten. Concurrent manual invocations have independent histories
referencing whichever snapshot was committed at their start; use sequential
invocations for a simple timeline. No filesystem locks are created or touched.

Identical duplicate GUID records collapse to one observation, with occurrence
counts and all original XML retained. Conflicting duplicate records, malformed
RSS, missing GUID/pubDate, or incomplete RSS fail the run and do not advance the
previous snapshot. API/HTML failures produce PARTIAL and preserve all previous
evidence; the valid RSS snapshot still advances. Failed article requests are not
retried automatically on a later unchanged observation. Metadata extraction
failures retain the raw responses for offline inspection. PASS means successful
collection, never proof of timezone or publication semantics.

Network-free tests:

```sh
python3 -B -m unittest discover -s research/espn_rss_prospective -p test_collector.py -v
```

Tests use synthetic bytes and temporary directories inside this research directory.
No existing project files are changed by the collector. Keep future changes to
these files backed up with a UTC timestamp before editing, as requested.
