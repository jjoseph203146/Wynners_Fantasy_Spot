# V3.1 QB / opportunity reconciliation shadow V1

This is an isolated, deterministic offline validator. It never allocates or mutates player
opportunity, queries a database, invokes a production writer, or supplies optimizer scores.
It reuses the existing exact identity resolver through `shadow_player_analysis_v1.core.resolve`.
No Cold-Start V1 code is imported or changed. The frozen interface is accepted as a reference.

Historical conclusions remain unchanged: the earlier target-only experiment's aggregate
conservation did not prove player accuracy; actual-participant MAE slightly worsened. QB guard
false positives do not justify a blanket haircut. The examined causal failures were retrospective
medical exits, not proven pregame-adjustable events. This package implements no such haircut.

## Inputs

`fixtures.sample()` documents the complete fictional schema. All snapshots cover exactly one
team/game and pregame cutoff. Real input must be marked `EXPLICIT_PREGAME_SNAPSHOT` and exported
from authorized existing sources. There is no automatic current-NFL adapter. The caller supplies
complete roster/metric coverage and immutable source versions; the validator checks declarations,
not authenticity of external files. Never present a prior or manually guessed value as an
existing authorized model output.

The context binds season, week, game, team, scenario, parent scenario, baseline, QB context,
opportunity-envelope version, cutoff and kickoff. A parent is the complete prior `build()`
result, verified against its content hash. Child versions cannot reuse an ancestor scenario;
changed context/envelopes require new versions. Envelope quantities are absolute replacements,
not deltas. Conflicting values under the same source/version are rejected.

Each quantitative record needs: nullable value, metric definition, authorized flag,
`SUPPORTED_INFERENCE` confidence, `ABSOLUTE` representation, source reference/version,
availability timestamp and exact season/week/game/team scope. Unknown and low-confidence
values do not become zero or establish numerical revisions. Observed identities/availability
are separate from expected future quantities. Participation needs its denominator (attempts,
dropbacks or snaps). QB designed carries and scrambles remain separate nullable components.

Availability is an explicit export of `WFS_STAT_FORECAST_AVAILABILITY_GATE_V1`. Starter and
player identities use the existing canonical snapshot resolver. OUT plus positive participation
is a contradiction, not a reason to guess the backup's volume. To represent a supported backup,
export its exact starter identity and evidence-qualified context. State changes alone never
apply numerical volume multipliers.

The existing `V3_ATTEMPTS_X_0_94` target policy is opt-in and explicitly labeled as a supported
model derivation, not a football identity. Unsupported explicit target inputs do not fall back
to it. Non-QB carries may be derived as compatible team carries minus QB carries; a negative
result is blocked. Routes/snaps have no invented aggregate budgets. QB components are never
assumed exhaustive. Optional cross-metric subset flags in `relations` must come from the
exporter's documented metric definitions, not guesses. They authorize comparisons only.

Player snapshots are either BASELINE or POST_TRANSFER_ABSOLUTE. The latter requires an input
hash and ledger reference. Both must match context/baseline/scenario/cutoff versions. No ledger
is applied here. Optional `cold_start_reference` has the frozen context/baseline/scenario/cutoff
and input hash; mismatches are BLOCKED_VERSION_MISMATCH. Old ledgers remain immutable. New
scenario allocation requests require an authorized pre-transfer baseline; this code never
creates that baseline or reruns Cold-Start allocation.

## Results and semantics

Outputs contain QB context, eight separate envelopes, checks, signed residuals, scenario lineage,
analysis signals, a reallocation request when needed, and a manifest with deterministic input
and result hashes. These are accounting checks, not evidence of player accuracy.

- PASS: complete compatible values satisfy the stated invariant.
- WARNING: compatible values fit, but explicit residual capacity remains.
- BLOCKED: excess, conflicting identity/lineage, or another contradiction. Excess is detected
  even with incomplete player coverage. Some malformed/contradictory structural inputs raise
  a named ValueError before any artifact can be published.
- NOT_EVALUABLE: missing evidence or incompatible definitions. Unknown is never forced to PASS.

Overall status conservatively includes all reported checks, so an otherwise valid target/rushing
fixture can remain NOT_EVALUABLE because routes or snaps lack aggregate budgets. Per-check status
and metric confidence are retained. Within-envelope signals indicate accounting feasibility only.
Residuals are not redistributed. No player records are output with rewritten workloads.

A supported context change may emit REALLOCATION_REQUIRED. This is a request descriptor, never
an executable action. Player overages do likewise. Baselines and the old allocation ledger remain
unchanged. Unknown revised envelopes are visible as unresolved, not silently carried forward as
backup-adjusted values. QB/non-QB player totals are partitioned by canonical position.

## Execution and artifacts

Run `venv/bin/python -B -m offensive_reconciliation_v3_1_shadow_v1.publication --input /path/to/input.json`.
The writer accepts no arbitrary destination. Artifacts go only under
`processed/offensive_reconciliation_v3_1_shadow_v1/<scenario-key-hash>/`:
`qb_context.json`, `team_opportunity_envelope.json`, `reconciliation_checks.json`, `residuals.json`,
`scenario_lineage.json`, and `manifest.json` (written last).

Same scenario and identical input are idempotent; conflicting content cannot overwrite that
scenario. Partial writes fail closed on retry. Symlinked output roots are rejected. To verify,
load all non-manifest JSON files keyed by filename stem and compare their canonical digest with
`manifest.result_sha256`. Synthetic validation bundles are not current NFL predictions.

Focused tests: `venv/bin/python -B -m unittest offensive_reconciliation_v3_1_shadow_v1.test_reconciliation`.
