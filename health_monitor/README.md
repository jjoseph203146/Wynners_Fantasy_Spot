# NFL APP Health Monitor V1 core

Run from the repository root with the existing virtual environment:

```
venv/bin/python -B -m health_monitor
venv/bin/python -B -m health_monitor --json
venv/bin/python -B -m unittest health_monitor.test_core -q
```

The text report lists failures and aggregate counts. JSON includes every finding,
its scope, stable reason code, safe evidence and observation timestamp.
Exit codes: HEALTHY 0, WARNING 10, CRITICAL 20, OWNER_ACTION_REQUIRED 30;
monitor execution failure 40. Exit codes never cause repairs.

## Boundaries

No production changes, alert delivery, scheduler, runtime state or outbox.
The CLI disables bytecode writes and installs a Python filesystem-write guard.
Database adapters (including legacy consumer connections) force mode=ro,
query_only, a two-second lock timeout and a five-second SQL work budget.
Only two allowlisted systemctl status commands and the local HTTP health GET
are used. The whole CLI has a 90-second budget. No solver/build/publish/ingest
entrypoint is imported or executed.

The consumer observer calls the actual build_game_evidence and _build_stat_outlook
functions and captures rows passed to their existing formatter in memory. This
preserves source selection, starter selection and display transformations without
editing production code. It cannot prove the running Streamlit process has the
same code loaded, or certify external routing/authentication/UI rendering.

## Authority and freshness

Current pregame authority is explicitly the current consumer's
nfl_current_unified_stat_forecasts.parquet. A changed consumer path fails closed;
current_unified_stat_forecasts.parquet is never substituted.

PRIMARY_QB evidence comes from the existing reconciliation CSV verified against
its PASS builder audit, contract, row count and output SHA256. This verifies the
existing evidence, not a new reconciliation or immutable promotion preflight.
The monitor does not claim that the unified file has a manifest-bound derivation
from the reconciliation CSV. It checks primary identity independently and compares
the consumer's exact QB values to the current unified file, with 1e-8 absolute
tolerance. QB receiving fields are nullable under the current schema; missing
passing/rushing values fail. Null versus a numerical value never compares equal.

Starters require the existing AGREE/depth/RotoWire/availability qualification,
exact GSIS IDs and the 7200-second freshness limit. An unqualified team fails
closed without hiding other teams' comparisons. LIVE/POSTGAME use hash-verified
kickoff snapshots and do not compare to today's starters or PRIMARY_QB.

Schedule context uses the existing planning-week resolver and Eastern dates.
Game kickoff times are interpreted in America/New_York. Incomplete games after
kickoff require Live audit freshness even when the event feed is missing.
Idle Live checks inspect historical audit status but do not apply age deadlines.
Regular-season context unavailable (including unsupported offseason context) fails
closed. Postponements/delayed completion may require owner interpretation.

Defaults: hourly forecast/starter warning after 90 minutes, hard failure after
120 minutes; reconciliation hard age 120 minutes. Forecast age uses mtime only
alongside exact schedule coverage and schema/identity validation. It is not proof
of upstream numerical freshness when no standalone current-artifact lineage exists.
Live audit warning after 180 seconds, critical after 300 seconds when expected.
Updater is due hourly at minute 07; missing-run grace 15 minutes; unfinished-run
limit 45 minutes. Only launcher terminal records establish completion; nested
stage Finished messages do not. Prior failed runs remain critical while the next
run is in progress. A running process alone is never successful completion.

Artifact and DB/WAL signatures are checked before/after observation. If production
changes, cross-layer conclusions are replaced with a WARNING to retry manually.
This conservative check can defer during active ingestion and is not a global
atomic transaction. It does not hold production locks or block writers.

Storage checks five established authority DBs, required tables and bounded reads;
it does not run a full integrity scan. Disk warning: under 5 GiB or 5%; critical:
under 1 GiB or 2%. SQL/schema/read exceptions fail closed with sanitized types.

## Next phase

Review observations and thresholds, then design isolated dedup/outbox state under
var/health_monitor/. Owner authorization is still needed for notifications and
scheduling. This package creates neither runtime state nor scheduling definitions.
