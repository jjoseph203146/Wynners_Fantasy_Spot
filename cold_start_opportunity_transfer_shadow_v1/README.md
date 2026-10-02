# Cold-start opportunity transfer V1 shadow

This isolated package consumes explicit pregame snapshots. It does not query databases,
fetch evidence, infer missing metrics, change availability, or feed production consumers.
The existing `shadow_player_analysis_v1.core.resolve` wrapper reuses the canonical exact
FanDuel identity resolver. No fuzzy matching is added. QB opportunity is outside allocation.

## Input and execution

`fixtures.sample()` is a complete **fictional** input schema example. A real input must be
an explicitly exported, authorized snapshot, marked `EXPLICIT_PREGAME_SNAPSHOT`. The caller
must supply the complete current offensive roster for one team/game; omitted players cannot
be discovered by this offline engine. No live adapter or current-player output is included.

Every input belongs to the top-level season/week/game/team/scenario and evidence cutoff.
Roster, identity, availability, depth and baseline versions describe immutable exports for
that context. Do not mix records from other games. Source references must identify the actual
upstream evidence. This interface validates declared lineage; it does not authenticate upstream
files or turn declarations into confirmed football evidence.

- `identity`: existing `WFS_IDENTITY_SNAPSHOT` contract, available before cutoff.
- `roster`: current QB/RB/FB/WR/TE canonical identities, source and availability timestamp.
- `availability`: export of `WFS_STAT_FORECAST_AVAILABILITY_GATE_V1`, not inferred statuses.
  OUT/INACTIVE or an explicitly exported authoritative `hard_block=true` can vacate supply.
  QUESTIONABLE/DOUBTFUL alone cannot. Unknown recipient availability is ineligible.
- `baselines`: authorized pregame counts, exact GSIS/metric, version, source, timestamp,
  unit and metric definition. Null or absent is unknown. Priors cannot manufacture supply.
- `capacities`: supported total post-transfer ceilings, not percentages or invented defaults;
  same lineage requirements. Headroom equals ceiling minus recipient baseline, floored at zero.
- `depth`: canonical GSIS, exact pos_grp/pos_abb/pos_slot, positive pos_rank, version/source/time.
  Ambiguous or unknown first available successors cannot be bypassed to invent a candidate.
- `donors`: identity claims for the scenario. Invalid/unavailable-authority failures are quarantined.
- `evidence`: already established OBSERVED source evidence, ID/source/time, donor/beneficiary IDs,
  game/team/baseline version, metric/unit/definition, nonnegative incremental quantity, explicit
  absence/role-change relationship, comparable context and quantity basis. Historical increments
  must already be measured upstream; generic medians or qualitative role signals are insufficient.
- `claims`: unique claim ID, donor/beneficiary identity claims, metric, requested quantity and
  evidence ID. Optional `confidence=COLD_START_INFERENCE` forces zero numerical allocation.
  Future allocations are SUPPORTED_INFERENCE, never OBSERVED.

Supported metrics are independent targets, player-route participations, carries and offensive
player-snaps. Routes/snaps remain unresolved without compatible existing sources. No conversion
between metrics is performed. All baselines for a metric must use the same definition.

Run explicitly with `venv/bin/python -B -m cold_start_opportunity_transfer_shadow_v1.publication
--input /path/to/snapshot.json` (on one shell line). The only output root is
`processed/cold_start_opportunity_transfer_shadow_v1/<bundle-hash>/`. Existing identical bundles
are reused; differing content is never overwritten. A partial failed write fails closed on retry.
The manifest is written last; consumers must require it and verify its artifact hashes.

## Allocation and output

Direct means a resolved exact-lane successor, not a numerical entitlement. Supported direct
claims get first use of donor supply, then supported secondary claims. Oversubscribed claims
are reduced proportionally within each donor/tier. Recipient headroom is shared proportionally
across all donors in that tier, with direct headroom consumption preceding secondary claims.
Capacity-clipped amounts stay residual, including when later secondary claims exist.
Identical claims deduplicate; conflicting IDs or multiple claims for one canonical edge fail closed.

`players.json` has four rows per resolved offensive roster player, including unchanged QBs and
zero-credit teammates. Unknown aggregate credit is null; `known_allocated_credit` separately
retains any supported portion. `ledger.json` preserves claims, inferred direct candidates and
rejections. `balances.json` records V, A and R per donor/metric, with NOT_EVALUABLE for unknown
supply. `manifest.json` includes lineage, quarantine, coverage gaps, hashes and safety flags.
Residual is an accounting balance, never a fantasy projection or assumed team production.
Incomplete roster identity or metric coverage makes team conservation NOT_EVALUABLE.

QB context/version is stored only for later scenario rebinding. No V3.1 reconciliation occurs.
Production integration, statistical efficiency conversion, FanDuel ranking and solver use are
outside this package. Synthetic published artifacts prove the mechanics, not current NFL accuracy.

Focused tests: `venv/bin/python -B -m unittest cold_start_opportunity_transfer_shadow_v1.test_transfer`.
