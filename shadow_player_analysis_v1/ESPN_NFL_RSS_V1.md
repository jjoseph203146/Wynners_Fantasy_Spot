# ESPN NFL RSS shadow connector — V1.1 temporal hardening

This additive connector leaves every validated foundation file unchanged. It has
no production importer, automatic runner, CLI, service, database access, or schedule.
The foundation's synthetic-only `build()` approval gate remains unchanged.

`espn_nfl_rss_v1.fetch(prior_events=())` explicitly requests the single centralized
`ENDPOINT`, with a 15-second socket timeout, 2 MB response limit, identifying RSS
User-Agent, no proxies, no redirects and no retries. It never requests item links.
No live request has been used to validate this implementation or endpoint.

Offline use is `ingest(rss_bytes, retrieval_timestamp, prior_events=())`. Input is
RSS 2.0 XML; DTD/entity declarations, malformed/empty/unexpected documents fail
closed. HTTP and content-type failures raise `FeedError`. Invalid item timestamps
or content quarantine that item with its supplied fields. Publication dates require
an explicit numeric timezone or GMT/UTC. Raw supplied dates remain in
`raw_feed_fields`; normalized publication and retrieval dates use UTC.

Unparseable, missing, or ambiguous publication dates retain the complete capture
in `quarantine`, including GUID/source_event_id, source/origin, headline, summary,
article URL, content hash, retrieval and first-seen timestamps. The exact supplied
date is retained in `raw_feed_fields.pubDate`. These records have
`published_at_utc=null`, `temporal_status="AMBIGUOUS_SOURCE_TIMESTAMP"`,
`temporal_confidence="UNRESOLVED"`, and `eligibility_status="QUARANTINED"`.
The existing eligibility field is the pregame gate; no duplicate eligibility
boolean is introduced. They produce no claims and cannot vote in consensus.
Manual review raises `UNRESOLVED_SOURCE_TIMESTAMP` before claim evaluation.
They remain available in shadow quarantine bundles for audit and future
corroboration, without automatic promotion.

In particular, `Sun, 27 Sep 2026 15:03:12 EST` remains unresolved when retrieved
at `2026-09-27T19:50:02.423813+00:00`. EST is never reinterpreted as EDT, and no
hour correction, article-ID ordering, article crawl, retrieval-time substitution,
or first-seen-time substitution supplies publication time. Explicit numeric
offsets and GMT/UTC still follow the existing foundation checks. A parseable date
that violates observation ordering retains its parsed value for audit but has
`temporal_status="INVALID_SOURCE_TIME_ORDER"`, unresolved confidence, and remains
quarantined. Unambiguous normalized events use
`temporal_status="UNAMBIGUOUS_SOURCE_TIMESTAMP"` and resolved confidence; this
does not itself confer pregame eligibility.

`retrieval.source_timestamp_diagnostics` is sorted by exact raw pubDate and counts
distinct nonblank, trimmed GUIDs per nonblank timestamp from the current feed.
Each entry contains `raw_pubDate`, `repeated_source_timestamp_count`, and
`source_timestamp_suspect` (true for more than one distinct GUID). Repeated
copies of one GUID do not inflate the count. Missing GUIDs/dates are not counted.
This conservative signal identifies shared timestamps, not semantic unrelatedness
or proof of an error. Nine unrelated GUIDs sharing a date therefore report nine
and true. Diagnostics never repair dates or change eligibility, and are kept out
of event records so changing feed composition cannot create revision conflicts.

GUID is preferred; absent GUID, the supplied article URL identifies the event.
Without either, title plus publication time provides deterministic fallback
identity. A changed title in this last case cannot reliably be linked to the old
story. Retrieval time never contributes to event identity. `content_hash` binds
all captured item fields, while the unchanged foundation supplies event/revision
keys and evidence hashes. Feed-level SHA-256 binds the exact input XML bytes.
For unresolved dates without GUID or URL, fallback identity uses title plus the
raw pubDate instead; no normalized event/revision key is fabricated.

Pass prior accepted events for cross-fetch deduplication and revision detection.
The foundation retains the earliest observation of unchanged content; current
retrieval metadata separately records each fetch. Changed content with unchanged
publication time quarantines the conflicting logical event. A changed publication
time permits a foundation revision, conservatively treating the new publication
time as the new revision's availability. No update timestamp is invented. Historical
bundles remain immutable; this module does not maintain a mutable latest pointer.

Classification is deliberately limited to recap detection, explicit coach/starting
role language, or `INSUFFICIENT_UNRESOLVED`. These are routing labels, not extracted
claims. All normalized events are also quarantined for unresolved identity and
manual claim review. Claims and consensus outputs are empty. Every ESPN record
uses the single `ESPN` origin, including syndicated material.

`evaluate_candidate()` is an offline manual-review helper using the foundation's
exact identity and temporal checks. It additionally rejects recap/insufficient
text regardless of manual phase labels. It does not publish claims or implement a
multi-revision consensus pipeline. Snapshot provenance and manual annotation review
remain the caller's responsibility; no WFS player identity is created.
The temporal guard rejects unresolved captures before invoking the foundation.

`espn_rss_publication.publish(capture)` writes only beneath
`shadow_player_analysis_v1/espn_nfl_rss_v1_artifacts/<run_id>/`. It uses canonical
JSON, foundation code hashes and the unchanged manifest verifier, staged fsynced
files and atomic directory rename. Existing bundles are verified, never overwritten.
Artifacts: events, quarantine, empty claims/consensus, input, validation and manifest.
Every manifest includes all ten safety flags, source/retrieval metadata and hashes.
No real capture or persistent publication is generated by the test suite.

Offline validation:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B -m unittest shadow_player_analysis_v1.test_foundation shadow_player_analysis_v1.test_espn_nfl_rss_v1 -v
```

V1.1 synthetic validation: 25 foundation tests and 48 connector tests. No live
network requests, database mutations, service/scheduler changes, or production
publication are needed. All ten foundation safety flags remain unchanged.
