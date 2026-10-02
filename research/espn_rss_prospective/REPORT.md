# Prospective ESPN RSS collector — baseline report

COLLECTOR_PATH=research/espn_rss_prospective/collector.py
TEST_PATH=research/espn_rss_prospective/test_collector.py
TEST_RESULT=PASS (12 synthetic, network-free tests)
BASELINE_CAPTURE=research/espn_rss_prospective/evidence/runs/20260928T135719.633250Z_50f286db00e2
COLLECTOR_STATUS=PASS
RSS_ITEMS=26
NEW_GUIDS=26
PUBDATE_CHANGES=0
API_CAPTURES=26
HTML_CAPTURES=26
RAW_EST_PRESERVED=YES
TIMEZONE_CONVERSION_PERFORMED=NONE
PRODUCTION_SOURCE_FILES_CHANGED=NONE
DATABASE_WRITES=NONE
SERVICES_CHANGED=NONE
CRON_CHANGED=NONE
UPDATER_RUN=NO
M5C_STATUS=BLOCKED

Exactly one bounded live invocation completed. The baseline contains 53 raw
responses (RSS + 26 API + 26 HTML), all HTTP 200 and complete. All 107 manifest
entries were verified, including run.json and request records. The 26 parsed RSS
pubDate values match the raw feed's XML text exactly. All 38 monitored source
hashes remain unchanged. Only new files in this isolated directory were created;
no existing file required modification or backup.

Run interval: 2026-09-28T13:57:19.633547+00:00 through 2026-09-28T13:57:23.463017+00:00.
RSS retrieval interval: 2026-09-28T13:57:19.633600+00:00 through 2026-09-28T13:57:19.807364+00:00.

Focused tests: `python3 -B -m unittest discover -s research/espn_rss_prospective -p test_collector.py -v`.
12 tests passed, exit 0. They cover new/changed/unchanged observations, exact EST
preservation and no conversion, metadata extraction, failed requests and evidence
retention, malformed feeds, duplicate GUIDs, interrupted atomic commits, manifest
hashes, concurrent append safety, URL restrictions and expired request budgets.

Invoke another snapshot manually with:
`python3 -B research/espn_rss_prospective/collector.py --live`.
No background loop or schedule was created. See README.md for bounds, duplicate
handling, failed-capture behavior and append-only storage details. Successful
collection does not establish timestamp semantics or same-revision matching.

SHA256:
collector.py=769823b9ca2fcdd16ae9b57ad787c1d67eb28b4d51885c87fad788687cc58591
test_collector.py=ab9a4ce6f73d8eee0763844bcba2d6cbfe5c72493245e0008063339f3e02b34d
README.md=51894af4bfe123d9130b9d1a615ae11eaaa4dd0528108b4352688f1640e20b08
