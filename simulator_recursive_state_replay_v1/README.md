# Recursive state replay validator V1

Isolated validation only. No model training, Monte Carlo, forecasts, player allocation or
production integration. EXP004 has 72 margin features, 64 total features and 20 explicitly
excluded fields from the original 92-field inventory. This validator does not restore them.
`feature_contract.json` pins the lists and historical source hash from the EXP004 audit.

## What is validated

The pure `core.build(input, contract)` function reconstructs pregame state from prior,
explicitly supplied two-team game observations. It never reads its reference feature row
while constructing features. A separate comparison uses raw, pre-imputation values and
absolute tolerance 1e-8 / relative tolerance 1e-10. Tests include an independent 72-field
hand-calculated synthetic oracle. Synthetic equality does not establish historical NFL equality.

All 70 recursive fields and two rest fields carry values, semantic states, source versions,
hashes, cutoff and reconstruction status. The 64 totals entries reference the same reconstructed
values. Excluded injury/status/population fields remain UNRESOLVED with null values. Player-row
counts retain their historical meaning as source feature rows, not roster size. No player
population contributes to the 72 fields; future population adapters must resolve canonical IDs.

The legacy builder encodes absent primitives and empty rolling windows as zero. This validator
preserves their unknown meaning and reports EXPECTED_UNAVAILABLE against a frozen encoded zero.
It does not use median imputation to manufacture agreement. UNKNOWN, UNRESOLVED and
NOT_APPLICABLE cannot compare equal to KNOWN_ZERO. No 72-feature metric is inherently not
applicable; a reference asserting that semantic state is compared explicitly, not numerically.

## Inputs and chronology

See `fixtures.sample()` and `fixtures.child()` for the explicit snapshot schema. Inputs must
include canonical team keys; a unique schedule keyed by game; effective game dates/kickoffs;
cutoff-known coach assignments; source versions/hashes; bootstrap coverage declarations;
and the coach-history start season. Caller declarations and hashes must describe genuine
exports, not invented historical publication times. The validator checks consistency, not
external source authenticity.

Each result contains both canonical teams, seven nullable primitives per team (points, attempts,
carries, passing yards/TDs, rushing yards/TDs), completion time, observation-availability time,
source lineage, and OBSERVED or SIMULATED provenance. Results require kickoff < completion <=
observation time <= pregame cutoff < target kickoff, with completion strictly before cutoff.
Missing timestamps produce NOT_EVALUABLE; they are never replaced by file modification time.
Late/current-game evidence is quarantined as LEAKAGE and blocks transition validation.

Known historical observations seed the root. The caller must export enough warm-up history
for both teams and their prior opponents/coach careers. Coverage must be declared incomplete
when an export omits relevant history. The validator cannot discover unexported source rows.
Coach history retains the frozen finite historical scope, name-to-coach identity bindings,
and tie-as-no-win behavior. Team histories continue across seasons; rolling windows count games,
not weeks. Legacy team season/week/game ordering is compared with completion chronology and
any disagreement is surfaced rather than silently repaired.

The schedule must be the cutoff-valid effective schedule, excluding cancelled/superseded entries.
Rest is calendar game-date distance from the previous bound game, not elapsed hours. First-game
rest conventions are not guessed: missing prior schedule yields UNKNOWN. Historical stored rest
that follows another convention will be reported as a mismatch for explicit review.

Frozen definitions retained: offensive_plays = attempts + carries; pass/rush rates use that
proxy; 3/5 game means and their differences; opposing-defense histories join by current opponent.
Score alone updates scoring/coach fields but leaves 40 primitive-dependent fields unavailable.
Rich explicit primitives support transitions; there is no primitive generator here.

## Immutable states and references

Children carry exactly one content-hash-verified parent, increasing simulation step/cutoff,
the same branch/team/source snapshot, and unique transitions. Observed states reject simulated
evidence. Simulated evidence requires its branch ID. Actual observations after the branch's
observed root cutoff cannot backfill simulated state. Core functions do not mutate inputs.
State IDs hash normalized input, contract and implementation version. Duplicate transitions and
parent/source conflicts fail closed. New source snapshots require a new explicitly initialized
replay rather than silently changing a parent's observations.

Cold-Start V1 and V3.1 references contain immutable version/content hashes and status only.
They are retained without injecting any values into the 72 features. BLOCKED/NOT_EVALUABLE
references stay disclosed; they never establish simulator evidence.

## Running and results

`venv/bin/python -B -m simulator_recursive_state_replay_v1.io --input snapshot.json`

Optionally add `--frozen-csv processed/forecast_v1_team_game_training.csv`. The read-only adapter
verifies the pinned file hash and selects exactly one game/team row. It does not reconstruct
history from that reference. The supplied primitive snapshot must remain independent.

Only `processed/simulator_recursive_state_replay_v1/<state-hash>/` is written, with manifest,
features/totals, comparisons, transition validation, excluded-feature status, lineage and audit.
Existing identical bundles are reused; different/partial content fails closed. Audit is written
last. Verify its result_hash over the other decoded JSON files keyed by filename stem.

The read-only `source_readiness()` helper inspects only known `team_game_stats`/`games` schemas.
The current sources expose updated_at but not explicit original availability/completion fields.
Therefore a full as-of historical replay remains unproven without suitable exports. No completion
or observation timestamps are fabricated to overcome that limitation.

Transition statuses describe the supplied scenario, not implementation quality. Complete rich
synthetic inputs may pass; score-only inputs are PARTIAL; absent timing/coverage is NOT_EVALUABLE;
leakage is BLOCKED. Replay comparison and temporal proof remain separate. Monte Carlo remains
unauthorized even if arithmetic replay passes, pending full historical temporal coverage and a
separately validated source/generator of future primitives.

Focused tests: `venv/bin/python -B -m unittest simulator_recursive_state_replay_v1.test_replay`.
