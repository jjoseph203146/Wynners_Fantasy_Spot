# Player Analysis Intelligence V1.2 — isolated shadow layer

V1.2 accepts explicit normalized adapter records, classifies content with deterministic
rules, resolves exact WFS entities, reports temporal gates, and publishes immutable
JSON bundles. It has no fetch, database, production publication, recommendations,
claim generation, or consensus voting capability. All ten foundation safety flags
are carried into its artifacts. ESPN V1.1 and foundation code remain unchanged.

## Identity authority and repository investigation

* `fanduel_injury_ingest.py:347–387` builds an exact WFS roster/injury identity index
  and resolves normalized full names using optional team/position constraints.
  V1.2 imports only its pure `normalize_name`, `normalize_team`,
  `normalize_position`, `resolve_identity`, and team dictionaries. This is the
  **same resolver already used by foundation V1**. No index builder or ingestion
  entry point is called. The module was inspected: its module-level initialization
  defines constants and functions, without network or database I/O.
* `scripts/starter_verification_v1.py:189–224` independently confirms the exact
  name/team/position intersection and zero/one/multiple candidate fail-closed
  rule. Its suffix-stripping normalization differs; V1.2 does not combine the
  two contracts or introduce additional aliases.
* `wfs_ai_analyst.py:805–839, 1085–1096` uses exact IDs from `player_identity`, with
  uniqueness required. `espn_injury_ingest.py:92` similarly uses the existing WFS
  crosswalk. `player_identity.py` is a database-writing builder, so it is **not
  imported**. ESPN IDs/names never create WFS identities here.
* `scripts/fanduel_schedule_identity.py` and the foundation use existing schedule
  identity. V1.2 requires an exact `game_id` occurring once in an explicit WFS
  schedule snapshot; there is no team/date-to-game guess or invented ID.

Callers supply a trusted `WFS_IDENTITY_SNAPSHOT` with `available_at_utc` and
`players` rows containing `gsis_id`, `player_name`, `team`, `position`. Export those
rows from the existing WFS identity/index authority; do not populate them from a
news adapter. The fixture snapshots use conspicuously synthetic IDs. A separate
`WFS_SCHEDULE_SNAPSHOT` supplies `available_at_utc` and `games` with `game_id` and
`kickoff_at_utc`. Both snapshots must be available by the evaluation cutoff.
Snapshot hashes are recorded. Conflicting team/position rows for one GSIS fail
closed. Caller-supplied GSIS IDs must already occur in WFS and agree with any
supplied name/context. No snapshot is changed.

Normalized exact-name matches use the WFS resolver's unique GSIS candidate set.
Team/position can disambiguate only as that resolver allows. Candidate IDs remain
in the audit. Missing names, misspellings, surname-only inputs, and conflicting
constraints never trigger an approximate lookup. Team names/aliases use the same
WFS team dictionaries, followed by an explicit membership check.

## Source-neutral input contract

`core.build(records, identity, schedule, cutoff, approvals)` is pure. The caller
owns the approvals registry (`source -> allowed approval IDs`); records cannot
approve themselves. `source_approved` is independent of content and identity.
Normalized records preserve all incoming fields and require these fields:

* `source`, `origin_id`, `source_event_id`, `approval_id`
* `content_hash`, `evidence_text`, `headline`, `summary`
* `raw_publication_timestamp`, `published_at_utc` (null if unresolved)
* `retrieved_at_utc`, `first_seen_at_utc`
* `source_phase`, `temporal_status`, `temporal_confidence`

`article_url`, GUID/raw feed fields, source categories and any additional
provenance remain intact. `game_id` supplies exact temporal game context.
`entities` optionally supplies PLAYER (`name`, optional `gsis_id`, `team`,
`position`, `evidence_quote`), TEAM (`name`), and GAME (`game_id`) mentions.
`evidence_kind` optionally preserves adapter-declared expectation, opinion,
market information, or WFS corroboration. A factual label cannot override
opinion/expectation detected in the text. WFS corroboration is an attribution
label only; dependency validation and claims remain disabled.

`adapters.espn_records(capture)` consumes the existing ESPN `events` and
`quarantine` lists offline. It preserves ambiguous raw pubDate, normalized null
publication time, GUID, URL, hash, text, origin and timestamp diagnostics carried
by individual records. The capture-level `retrieval` diagnostics remain in the
original V1.1 capture; they are not publication evidence. Only V1.1's obsolete
identity/manual-review placeholders are removed; adapter integrity failures
remain as `adapter_reason_codes`. No ESPN connector changes were needed.

## Classification and independent gates

Full WFS names are discovered in clauses using validated normalization. A narrow
capitalized full-name pattern can expose an unknown candidate, which remains
unresolved. Team full names are detected against WFS aliases. Supplied team
abbreviations resolve exactly. No surname expansion or automatic game inference
is performed. Multiple players share one source event and receive explicit
clause relationships. Clauses with multiple players and uncertain scope retain
identities but classify the relationship as insufficient. Quotes must occur in
the event and contain the supplied full name. Repeated mentions may have separate
relationships rather than silently discarding later evidence.

The controlled taxonomy is in `classification.TAXONOMY`. Routing is conservative:
noise and college/history content reject; market and opinion content remain
market/opinion; postgame box-score language takes priority over role keywords;
negated/questioned propositions fail closed. Generic carries, targets, catches,
yards and snaps are insufficient. Defensive injuries resolve normally and add
team-environment context. This does not create an IDP surface or infer effects
on specific offensive players. This deliberately limited rule set is not a
complete natural-language parser; missed/unclear evidence awaits later review.

`relevance`, `identity_status`, `source_approved`, and temporal state are separate.
Temporal validation requires resolved source publication, timezone-aware ordered
publication/update/first-seen/retrieval times, knowledge by cutoff, and an exact
schedule game whose kickoff is after publication/update/cutoff. Recaps are never
pregame eligible. Retrieval and first-seen are never substituted for publication.
The raw timestamp is never repaired. Ambiguous ESPN EST timestamps may classify
HIGH and resolve GSIS while remaining pregame-ineligible.

**V1.2 generates no claims and no consensus votes, even for otherwise valid
records.** `claim_eligible=False`, `consensus_eligible=False`, and explicit
`NOT_ENABLED_V1_2` gates are always emitted. Foundation's stricter claim horizon,
forward proposition, latest revision, and independent-origin consensus rules are
unchanged and not bypassed. A temporally valid record is not a validated claim.
Opposing evidence is retained, not reconciled. Foundation regression verifies
that independent opposing eligible claims remain `CONFLICT`.

Exact duplicate records collapse deterministically; distinct captures/revisions
remain auditable. Conflicting content hashes under one source event receive
`SOURCE_EVENT_REVISION_CONFLICT`. Syndicated sources retain their original origin;
no source or repeated player mention increases a vote count (all counts are zero).
Unknown/rejected entities, adapter failures and temporal problems have separate,
possibly multiple, quarantine reason codes. No event-wide classification silently
resolves a player's identity.

## Offline use and immutable publication

```bash
venv/bin/python -B -m shadow_player_analysis_v1_2 --synthetic-fixtures normal
venv/bin/python -B -m shadow_player_analysis_v1_2 --synthetic-fixtures ambiguous
venv/bin/python -B -m shadow_player_analysis_v1_2 --input-json /path/to/offline_inputs.json
venv/bin/python -B -m unittest shadow_player_analysis_v1.test_foundation shadow_player_analysis_v1.test_espn_nfl_rss_v1 shadow_player_analysis_v1_2.test_v1_2
```

The input JSON object contains the six named `build` arguments above. The CLI has
no network option. Output is fixed to this package's `artifacts/<manifest-hash>/`:
`source_events.json`, `classified_evidence.json`, `entity_links.json`,
`quarantine.json`, `source_quality.json`, `input.json`, `manifest.json`.
Each manifest binds input/snapshot/code hashes, approvals, cutoff, safety flags
and every payload's size/SHA-256. Repeating identical inputs/code reuses a verified
bundle. Historical files are never overwritten; corruption causes failure.
Staging, fsync, verification and same-directory atomic rename prevent exposure of
partial published bundles. Failed staging is removed. No mutable current pointer,
production table, parquet, UI, service, cron or updater integration exists.
