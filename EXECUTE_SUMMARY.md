# WFS NFL DATA ENGINE — EXECUTE SUMMARY

Last operational update: 2026-09-13
Project root: /home/mwynn/nfl_data_engine

This document is the operational handoff and production-state summary for
the WFS NFL Data Engine / NFL DFS / NFL Live project.

Its purpose is to preserve validated production behavior, frozen decisions,
important safety contracts, and current architecture between development
sessions.

Research results do NOT become production behavior merely because they are
documented here. Anything explicitly marked REJECTED, RESEARCH ONLY, SHADOW,
or NOT PROMOTED must remain outside production unless separately validated
and explicitly promoted.


======================================================================
1. CORE PRODUCTION PHILOSOPHY
======================================================================

The WFS NFL system is deterministic, fail-closed, and audit-oriented.

Core rules:

- Never silently mutate production forecasts.
- Never use fuzzy player identity matching in production.
- Exact player identity is authoritative.
- Exact GSIS identity is preferred for NFL player resolution.
- Availability authority overrides depth-chart authority.
- Production and research/shadow artifacts remain separated.
- Public UI exposes football/DFS information, not backend internals.
- Hashes, model internals, diagnostics, cache details, solver internals,
  fingerprints, and audit/debug information remain admin-only.
- Validated/frozen stages remain frozen unless explicit evidence requires
  a controlled change.
- Do not promote research simply because it improves an in-sample metric.
- Production changes require explicit validation and safety gates.


======================================================================
2. FANDUEL NFL SCORING CONTRACT
======================================================================

WFS uses FanDuel NFL Classic scoring.

Important scoring rules:

- 0.5 points per reception.
- No 100-yard rushing bonus.
- No 100-yard receiving bonus.
- No 300-yard passing bonus.

Fantasy tracking includes:

- Offensive players.
- Team Defense / Special Teams (DST).

Individual defensive players are NOT part of the fantasy-player tracking
surface.

Defensive players may remain present in raw play-by-play data where needed
for football context.


======================================================================
3. PUBLIC GAME-DAY STAT PRESENTATION
======================================================================

Single-game counting-stat projections displayed publicly use WHOLE NUMBERS.

Examples include:

- Pass attempts
- Pass completions
- Passing yards
- Passing TDs
- Interceptions
- Carries
- Rushing yards
- Rushing TDs
- Targets
- Receptions
- Receiving yards
- Receiving TDs
- Field goals
- Extra points
- Sacks
- Defensive interceptions
- Defensive touchdowns
- Other discrete single-game event counts

The underlying model projections remain decimal/fractional.

Public whole-number formatting is PRESENTATION ONLY.

Do not round or mutate the underlying forecast artifacts.

Rates remain rates and may display as percentages/decimals where appropriate.

Examples:

- Expected Pass Rate is a rate and may remain a percentage.
- Single-game projected targets are a counting stat and must display as a
  whole number.

Current whole-number event formatting uses nonnegative half-up behavior.


======================================================================
4. EXPECTED PASS RATE
======================================================================

Production artifact:

processed/pregame_context_pass_expectation_v1.csv

Public display name:

Expected Pass Rate

Production contract columns:

game_id
season
week
team
opponent_team
pregame_context_pass_expectation

Production SHA256:

65f0b307d6dca3b7f9eb7f1dd7c1117f6673b6ac93230d8804ac70b278f02df1

Public placement:

- Week Forecast prediction cards.
- Under the primary prediction section.
- Above Total / WFS Margin.

NFL Data Center does NOT display Expected Pass Rate.

Matching is exact game_id + team.

No fuzzy matching.

The deployed model is coach-neutral.

No coach adjustment is applied to Expected Pass Rate.


======================================================================
5. COACHING INTELLIGENCE
======================================================================

Coaching intelligence remains ANALYSIS / INTERPRETATION ONLY.

Historical research showed useful descriptive coaching tendencies but did
not validate a production coach adjustment that improved the deployed
pregame context model.

Therefore:

- No production forecast coach adjustment.
- No solver coach weight.
- No team-environment mutation from coaching intelligence.
- No direct coach adjustment to Expected Pass Rate.

Future coaching intelligence may analyze:

- Coach philosophy
- Historical tendencies
- Game-plan tendencies
- First-time/new coach uncertainty
- Down
- Distance
- Score
- Clock
- Field position
- Play-by-play context

Any future production influence must first be introduced analysis-only and
validated independently.


======================================================================
6. STARTER-AWARE STAT OUTLOOK
======================================================================

Public Stat Outlook is starter-aware.

Primary starter identity is resolved through exact NFL identity and football
depth lanes.

Supported offensive positions:

QB
RB
FB
WR
TE

Kicker and DST remain handled separately.

Every confirmed/projected offensive starter with a usable WFS statistical
projection should receive Stat Outlook.

Starter selection is NOT simply "highest projected players."


======================================================================
7. STARTER / AVAILABILITY AUTHORITY
======================================================================

Operational hierarchy:

1. Official OUT / INACTIVE authority
2. Current authoritative starter information
3. Exact WFS depth-chart role
4. Exact next-man-up in the same depth lane
5. Projected workload
6. Recent workload / fantasy usage

RotoWire current NFL lineups are an approved public starter-information
source.

RotoWire generally becomes increasingly authoritative near kickoff when
official lineup/inactive information becomes available.

Availability always overrides depth.

OUT and INACTIVE are hard blocks.

DOUBTFUL is treated as risk unless separately established as unavailable
under the active NFL workflow.

QUESTIONABLE does not automatically block a player.


======================================================================
8. EXACT DEPTH-LANE RESOLUTION
======================================================================

Depth data is stored in:

data/nfl.db

Table:

depth_charts

Relevant depth identity includes:

snapshot_dt
team
player_name
gsis_id
pos_grp
pos_abb
pos_slot
pos_rank

Each unique football depth lane owns one current starter.

Lane identity:

pos_grp + pos_abb + pos_slot

Within a lane:

- Sort by pos_rank.
- Skip unavailable players.
- Select the first available exact GSIS player.
- If the starter is OUT/INACTIVE, move to the next available player in the
  SAME exact football lane.

Do not replace a missing starter by simply selecting the highest fantasy
projection at the position.


======================================================================
9. STAT OUTLOOK LIVE / POSTGAME FREEZE
======================================================================

PREGAME Stat Outlook may use current starter/availability information.

LIVE and POSTGAME Stat Outlook selection behavior remains frozen from the
kickoff state.

Do not allow live depth changes or late research logic to silently rewrite
the intended frozen Stat Outlook behavior.


======================================================================
10. COLD-START STARTER FALLBACK — GAME-DAY BRIDGE
======================================================================

Current game-day requirement:

A confirmed starter should not disappear from Stat Outlook solely because
the normal canonical forecast does not contain that player.

Current implementation includes a fail-closed PREGAME cold-start fallback
inside:

wfs_ai_analyst.py

Marker:

WFS_STAT_OUTLOOK_COLD_START_FALLBACK_V1

The fallback:

- Applies to PREGAME presentation.
- Requires exact player identity.
- Requires an existing cold-start rookie shadow forecast.
- Allows the existing exact depth-lane starter resolver to select the player.
- Does not override a canonical forecast.
- Does not permit OUT/INACTIVE players.
- Does not mutate the canonical production forecast.
- Does not ingest adjusted-incumbent R8V rows into canonical production.

Current shadow source:

data/parquet/current_cold_start_full_team_shadow_v2.parquet

Only rows with:

shadow_row_type == COLD_START_ROOKIE

are eligible for this presentation fallback.

This is a GAME-DAY BRIDGE.

It is not equivalent to promotion of the complete R8 cold-start architecture
into the unified-stat publisher.


======================================================================
11. CURRENT COLD-START EXAMPLES
======================================================================

Jeremiyah Love
ARI
RB
GSIS: 00-0041027

Carnell Tate
TEN
WR
GSIS: 00-0041438

Both were identified as exact current rank-1 depth starters while missing
normal canonical statistical forecast rows.

The cold-start fallback allows them to receive public Stat Outlook without
mutating the canonical artifact.


======================================================================
12. R8Z HARD FEASIBILITY SAFETY
======================================================================

R8Z validated a mathematical hard feasibility rule:

hybrid_carries =
min(
    absolute_prior_carries,
    projected_non_qb_rush_pool
)

This is a structural safety cap.

It is NOT a validated role-share model.

For Jeremiyah Love, the current hard-feasibility carry value is approximately:

13.813809 carries

The game-day cold-start presentation fallback may use this capped carry value
and preserve the original rushing efficiency when scaling rushing yards/TDs.

This does NOT mean Love should receive the entire non-QB rushing workload in
a future full-team production model.

Long-term cold-start production must use coherent displacement /
redistribution.


======================================================================
13. QB PASSING TD / RECEIVER TD DISPLAY CONSISTENCY
======================================================================

Marker:

WFS_STAT_OUTLOOK_TD_DISPLAY_RECONCILIATION_V1

Public Stat Outlook now reconciles displayed passing TDs with displayed
receiving TDs.

Problem addressed:

A QB could have an underlying expectation such as:

1.54 passing TDs

which publicly displays as:

2 passing TDs

while several individual receivers each had fractional TD expectations below
0.5 and independently displayed:

0 receiving TDs

That produced an incoherent public game prediction.


Current presentation rule:

- Determine the selected starting QB.
- Convert his underlying passing-TD expectation to the public whole-number
  total.
- Consider the displayed eligible RB/FB/WR/TE receivers.
- Use their underlying receiving-TD expectations as deterministic weights.
- Allocate the QB's displayed passing-TD total across those receivers.
- The displayed receiver receiving-TD total must equal the displayed QB
  passing-TD total when positive receiving-TD weights exist.

This applies GENERICALLY to teams/QBs.

It is not hardcoded to Jacoby Brissett.

The allocation is PRESENTATION ONLY.

Underlying fractional passing-TD and receiving-TD forecasts are not mutated.

If all eligible receiving-TD weights are zero, the system does not fabricate
a TD scorer merely to force an allocation.


======================================================================
14. CURRENT ANALYST PRODUCTION STATE
======================================================================

Current validated wfs_ai_analyst.py SHA256:

387d04451c362717e4eb6e6ffc835b95a39494cd9f8a32cc8d74f61fe84995c2

Current canonical unified-stat forecast:

data/parquet/current_unified_stat_forecasts.parquet

Frozen canonical SHA256:

6067d8ea4a31ae60c5667c647c13c6bede4667c1369724c7861cf32929af13b9

The canonical SHA remained unchanged through the game-day cold-start and TD
presentation changes.


======================================================================
15. OTHER FROZEN PRODUCTION HASHES
======================================================================

Public app baseline:

c514299f64b2f6f58381ddbe0e8d12f7015dc76bfed9f6324fd185c6dc56bf26

Unified-stat publisher baseline:

54a590a170685c2ef22015d7494f9b3796132822f1b2bb4be0c97d7e12fe9d75

Expected Pass Rate artifact:

65f0b307d6dca3b7f9eb7f1dd7c1117f6673b6ac93230d8804ac70b278f02df1

Do not assume a changed hash is safe.

Investigate the cause before accepting a production hash change.


======================================================================
16. R8 COLD-START RESEARCH — IMPORTANT STATUS
======================================================================

R8 research explored cold-start statistical forecasting for rookies/new
starters with insufficient NFL history.

Core conceptual framework:

Team Role Opportunity
× Expected Role Share / Opportunity Prior
× Player Efficiency
× Game Environment

Research investigated:

- Rookie historical priors
- First-start priors
- Hierarchical priors
- Team pass/rush opportunity
- Target-pool estimation
- DvP adjustment
- Role-share modeling
- Hard feasibility
- Exact depth-lane displacement

Research results must NOT be confused with production promotion.


======================================================================
17. R8 MATCHUP / DVP CONCLUSIONS
======================================================================

Generic DvP adjustment was NOT validated for cold-start production.

Locked conclusions:

RB cold-start DvP weights:
ZERO

WR cold-start DvP weights:
ZERO

TE cold-start DvP weights:
ZERO pending substantially more evidence.

RB joint DvP tuning improved same-sample results slightly but failed
leave-one-out validation.

Therefore:

Do NOT reintroduce cold-start DvP weights into production without new
independent validation.


======================================================================
18. R8 ROLE-SHARE CONCLUSIONS
======================================================================

Pure/global role-share replacement was REJECTED.

Held-out validation showed:

- RB carry role-share replacement worsened performance.
- RB target role-share replacement worsened performance.
- WR target role-share replacement did not validate.
- TE target role share was directionally promising but sample size was too
  small for promotion.

Locked conclusions:

RB_CARRIES_ROLE_SHARE = REJECT
RB_TARGETS_ROLE_SHARE = REJECT
WR_TARGETS_ROLE_SHARE = REJECT
TE_TARGETS_ROLE_SHARE = RESEARCH_ONLY
GLOBAL_ROLE_SHARE_REPLACEMENT = REJECT

Do not promote global role-share opportunity simply because it is
structurally feasible.


======================================================================
19. R8 EXACT DEPTH-LANE DISPLACEMENT FINDING
======================================================================

Exact depth-lane displacement is structurally clean and remains the preferred
direction for a future full production cold-start system.

Examples established during audit:

ARI:
Jeremiyah Love RB1
displaced exact lane candidate:
Tyler Allgeier RB2

TEN:
Carnell Tate WR1
displaced exact lane candidate:
Chimere Dike

Future production architecture should start from:

cold-start starter
→ exact depth lane
→ displaced player
→ justified opportunity transfer / redistribution
→ team conservation
→ unified-stat publisher
→ canonical forecast
→ Stat Outlook

Do not redistribute a new starter's workload indiscriminately across all
players at the position or across all non-QB rushers.


======================================================================
20. REJECTED R8V / R8W BEHAVIOR
======================================================================

R8V full-team shadow must NOT be promoted as-is.

Reason:

Its broad scaling could distort unrelated incumbent workloads.

Examples included:

- Scaling QB carries during rookie rushing redistribution.
- Upward scaling incumbent receiving targets merely to fill an independent
  target pool.
- Excessive reduction of unrelated incumbent rushing workloads.

R8W correctly failed closed when absolute rookie opportunity plus protected
QB carries exceeded the projected team rushing environment.

Do NOT weaken that safety gate.

The failure demonstrated that absolute opportunity and team opportunity need
coherent displacement semantics.


======================================================================
21. CANONICAL STAT FORECAST AUTHORITY
======================================================================

Canonical statistical forecast artifact:

data/parquet/current_unified_stat_forecasts.parquet

This remains the primary normal Stat Outlook forecast authority.

Game-day presentation fallbacks do NOT constitute canonical promotion.

Any future cold-start production promotion must occur upstream through a
validated publishing path and produce a coherent full-team statistical
forecast.


======================================================================
22. PUBLIC VS ADMIN CONTRACT
======================================================================

PUBLIC:

- Football information
- DFS information
- Predictions
- Expected Pass Rate
- Stat Outlook
- Player/team context
- Normal football explanations

ADMIN ONLY:

- SHA hashes
- Fingerprints
- Cache details
- Solver internals
- Model diagnostics
- Research gates
- Debug information
- Audit internals
- Internal implementation details
- Competitive modeling logic not intended for public exposure


======================================================================
23. NFL LIVE / POSTGAME DIRECTION
======================================================================

NFL Live tracks:

- Offensive fantasy players
- Team DST

After games complete, the Data Center should update with:

- Final scores
- Final player statistics
- Relevant fantasy results

Completed-game information becomes part of the postgame workflow.

Individual defensive player fantasy tracking is not required.


======================================================================
24. SOLVER SAFETY
======================================================================

The DFS solver remains separate from the current coaching and cold-start
research work.

Do not introduce:

- Coaching adjustment into solver weights.
- Rejected R8 DvP logic into solver weights.
- Research-only cold-start artifacts directly into solver scoring.
- Presentation-only TD allocation into underlying solver projections.

Presentation logic is not solver logic.


======================================================================
25. GAME-DAY CHANGE POLICY
======================================================================

On game day:

Prefer the smallest validated change that solves the operational problem.

Do not perform broad architecture rewrites immediately before kickoff.

Priority:

1. Correct availability.
2. Correct starter identity.
3. Correct Stat Outlook.
4. Correct public presentation.
5. Preserve canonical production artifacts.
6. Preserve solver stability.

Research architecture can resume after the game-day system is stable.


======================================================================
26. CURRENT GAME-DAY VALIDATION
======================================================================

Validated 2026-09-13:

ANALYST_COMPILE = PASS

wfs_ai_analyst.py:

387d04451c362717e4eb6e6ffc835b95a39494cd9f8a32cc8d74f61fe84995c2

current_unified_stat_forecasts.parquet:

6067d8ea4a31ae60c5667c647c13c6bede4667c1369724c7861cf32929af13b9

Cold-start fallback smoke:

- Jeremiyah Love row available.
- Carnell Tate row available.
- Jeremiyah Love OUT test correctly blocked.

TD reconciliation smoke:

- QB displayed passing TD = 2
- Displayed receiving TD total = 2
- TD total match = PASS

Canonical artifact remained unchanged.


======================================================================
27. DO NOT REGRESS
======================================================================

Future development must preserve:

- Whole-number public single-game counting stats.
- Fractional underlying model values.
- Exact identity matching.
- Exact depth-lane starter resolution.
- Hard OUT/INACTIVE authority.
- Next-man-up semantics.
- Starter-aware Stat Outlook.
- Generic QB/receiver displayed TD consistency.
- Expected Pass Rate production contract.
- Coach-neutral production forecasting.
- Coaching intelligence analysis-only status.
- Public/admin separation.
- Canonical forecast integrity.
- Solver isolation from unvalidated research.
- Fail-closed behavior.
- Research/production separation.


======================================================================
28. NEXT LONG-TERM COLD-START WORK
======================================================================

After game-day operations are stable, the long-term replacement for the
temporary analyst cold-start bridge should be:

Cold-Start Starter Identification
→ Exact GSIS
→ Exact Depth Lane
→ Displaced Incumbent
→ Validated Opportunity Transfer
→ Efficiency Projection
→ Team Opportunity Conservation
→ Full-Team Statistical Republish
→ Unified-Stat Publisher
→ Canonical Forecast
→ Availability Gate
→ Stat Outlook

Only after this architecture validates should the temporary analyst
cold-start presentation bridge be retired.
======================================================================
29. GAME-DAY LIVE SCORING — VALIDATED
======================================================================
Public game-day behavior was upgraded without changing the canonical
forecast, analyst model, solver, or production forecast logic.

Current app:
app.py
SHA256:
51b26ec6a76806571cb63cde2059d0eacd6df7e5db938999322b91d6a91d78e9

Validated markers:
- WFS_LIVE_MATCHUP_LOOKUP_V2
- WFS_LIVE_GAME_AI_UI_V1
- WFS_LIVE_PLAYER_STATS_ATTACHED_V1
- WFS_LIVE_STAT_RENDER_V1
- WFS_LIVE_POINTS_FOR_V1
- WFS_PREDICTION_RECORD_UI_V1
- WFS_MOBILE_SPACING_FIX_V2

Live player behavior:
- PREGAME:
  show WFS stat projections.
- LIVE:
  show exact ESPN fantasy points and actual-to-date football stats.
  Do not show pregame Stat Predictions as if they were live results.
- FINAL:
  preserve existing final-result behavior.

Live stat lookup:
- Exact ESPN event ID + exact ESPN athlete ID.
- ESPN summary endpoint is used as a presentation bridge.
- No fuzzy player identity matching.

Public counting stats:
- Whole numbers.
- Underlying model decimals remain unchanged.
- Rate statistics may remain percentages.

NFL matchup cards:
- Exact internal game_id -> nfl.db games.espn -> ESPN summary.
- Allows public matchup cards to remain current even if live_events is stale.

ESPN fantasy team totals:
- Use totalPoints when nonzero.
- If zero, sum active exact-week appliedTotal values.
- Bench and IR are excluded.
- This powers live Matchups and League Snapshot Points For fallback.

IMPORTANT:
The ESPN summary presentation fallback does not weaken or replace the
fail-closed live ingest identity system.

======================================================================
30. LIVE INGEST SAFE_FAIL — DO NOT WEAKEN
======================================================================
Known game-day issue:
data/wfs_live.db live_events may remain stale for selected events because
the live ingest pipeline correctly SAFE_FAILS unresolved or ambiguous
identity cases.

Observed examples included:
- ATL@PIT
- BAL@IND
- CHI@CAR
- NO@DET
- NYJ@TEN

Policy:
- Do not relax exact identity gates to make stale presentation data update.
- Presentation freshness is handled by exact ESPN summary fallback.
- Database integrity remains fail-closed.

Locked live DB schema fingerprint:
1ffa0bcd282c24a05f65696c6a8ae85033f85cb4ad4dca5f86369b7905e21881

Do not casually add tables or fields to wfs_live.db.
Any schema change requires explicit versioned migration.

======================================================================
31. GAME INTELLIGENCE V1 — LIVE PBP ANALYSIS
======================================================================
Primary builder:
scripts/build_live_game_intelligence_v1.py
SHA256:
ff0ce7c9214f29145ab7549c16ea8118b436f838796938a872f7be5ef3cf5d9c

All-active runner:
scripts/run_live_game_intelligence_all_v1.py
SHA256:
2a28f5f1317019eed6fcfbdd3cef88940e70929b1919a2ef97f9a7e96f8115cd

Version:
WFS_GAME_INTELLIGENCE_V1

Architecture:
Pregame Game Paper
→ Expected Game Plan / Tendencies
→ ESPN structured live PBP
→ Situational PBP Interpreter
→ Game-Level Coaching Evidence
→ Live In-Game AI Commentary

Source:
- ESPN raw summary drives and plays.
- wfs_live.db is not mutated.

Structured context used when available:
- offense team
- down
- distance
- yard line
- yards to end zone
- score
- quarter
- clock
- drive context
- play type
- no-play / penalty / turnover state

Core deterministic metrics include:
- scrimmage plays
- pass calls
- rush calls
- live pass rate
- early-down pass rate
- red-zone pass rate
- third-down pass rate
- short-yardage pass rate
- shotgun
- no-huddle
- fourth-down scrimmage calls
- target concentration
- carry concentration

Expected Pass Rate:
- exact game_id + team lookup only.
- Used as paper/baseline context where available.

Deviation labels:
- >= 10 percentage points: material
- >= 5 percentage points: moderate
- otherwise: near expectation

Live sample confidence:
- HIGH: >= 40 scrimmage plays
- MEDIUM: >= 20
- LOW: < 20

Usage parsing:
- structured participant role is preferred.
- deterministic text parsing is fallback.
- no fuzzy identity matching.
- reported-eligible and direct-snap text were specifically guarded against
  false rushing attribution.

Policy:
ANALYSIS ONLY.
No production forecast mutation.
No solver influence.
No persistent coach-prior mutation from live plays.

======================================================================
32. LIVE COACHING INTERPRETATION V1
======================================================================
Builder:
scripts/build_live_coaching_interpretation_v1.py
SHA256:
452dd3a71b4086fc6b957f3011a42327e8dfe6b5116092d9e1a1da79c6916a3b

All-active runner:
scripts/run_live_coaching_interpretation_all_v1.py
SHA256:
7fb00e6149d42257f71b6373b16fe29bb9f1921d63be519f2e1cada12d8dd22a

Version:
WFS_LIVE_COACHING_INTERPRETATION_V1

Behavior:
- Reads live Game Intelligence artifacts.
- Attaches exact current coach identity.
- Compares current-game behavior to the game paper.
- Separately compares current behavior to same-regime historical evidence.
- Excludes current event from historical prior.
- No fuzzy team or coach identity matching.

Identity aliases:
- LAR -> LA
- WSH -> WAS

Historical confidence:
- HIGH: >= 6 games and >= 300 scrimmage plays
- MEDIUM: >= 3 games and >= 140 plays
- LOW: otherwise

Current live sample confidence:
- HIGH: >= 40 plays
- MEDIUM: >= 20
- LOW: < 20

New-regime handling:
- If no historical same-regime games exist, current evidence is explicitly
  labeled new game-level evidence.
- It is not treated as an established coaching tendency.

Public UI:
- concise football-language interpretation only.
- sample/confidence shown.
- internal rate diagnostics stay admin-only.

Policy:
ANALYSIS ONLY.
No solver, production forecast, or coach-prior influence.

======================================================================
33. POSTGAME COACHING EVIDENCE PIPELINE
======================================================================
Final-game assimilation:
scripts/run_postgame_coaching_assimilation_v1.py
SHA256:
2d2d11a8c9f5feef7bb8f14b68cc17fcc5b7fb10a6ef650e2d1694602d59c8dd

Postgame evidence builder:
scripts/build_postgame_coaching_evidence_v1.py
SHA256:
4bb56304e1f7035d920ebdc1a1994cd9e91863a1933a83a2c7ef5f1d76ad794a

Coach attachment:
scripts/build_postgame_coach_attached_evidence_v1.py
SHA256:
a6ed140f5187cab01d6f6fc3879f4b6ce58e3f830743fd693aff4b06e7110bb0

Coach-regime profile:
scripts/build_coach_regime_evidence_profile_v1.py
SHA256:
d873aaca78ab6a330c970b242ae1633d9e8ce70c2c8a297aa9b976053db02b94

Latest artifacts observed:
processed/coaching_intelligence/postgame_coaching_evidence_v1.csv
processed/coaching_intelligence/postgame_coaching_evidence_v1.json
processed/coaching_intelligence/postgame_coach_attached_evidence_v1.csv
processed/coaching_intelligence/postgame_coach_attached_evidence_v1.json
processed/coaching_intelligence/coach_regime_evidence_profile_v1.csv
processed/coaching_intelligence/coach_regime_evidence_profile_v1.json

Latest observed artifact refresh:
2026-09-13 18:05 local file timestamp.

Evidence invariants:
- Final games only.
- Exact game/team/coach identity.
- Duplicate-key gate fails closed.
- pass_calls + rush_calls == scrimmage_plays.
- Scrimmage-play weighted rates.
- Baseline expected/delta metrics only where baseline exists.

Alias support:
- LAR -> LA
- WSH -> WAS

NOTE:
WSH -> WAS live identity lookup was validated during a live game.
Do not claim postgame end-to-end WSH alias validation unless an actual
final-state assimilation run is inspected and confirms it.

Policy:
ANALYSIS ONLY.
Historical coaching evidence is not yet a production model input.

======================================================================
34. COACHING INTELLIGENCE REFRESH + AUTOMATION
======================================================================
Refresh wrapper:
scripts/run_coaching_intelligence_refresh_v1.py
SHA256:
b62105450a18a9a1a163fb5200374a242af02de5fa306c94289b3651e76a479a

Schedule-aware gate:
scripts/run_schedule_aware_coaching_refresh_v1.py
SHA256:
48d93b0d0ccf807d28a475075261f7cb36c54cf822ed90975352f6ec117c6ae7

Schedule policy:
- Timezone: America/New_York
- Reads nfl.db schedule.
- Checks yesterday through tomorrow.
- Active window:
  90 minutes before kickoff through 5 hours 30 minutes after kickoff.
- Malformed schedule rows do not activate.
- Outside active window: clean SKIP / exit 0.

Cron:
CRON_TZ=America/New_York
*/5 * * * * /home/mwynn/nfl_data_engine/scripts/run_schedule_aware_coaching_refresh.sh

IMPORTANT:
All unrelated cron jobs must be preserved.
Never replace the entire crontab to update this block.

Wrapper:
- Uses flock to prevent overlap.
- Lock:
  /tmp/wfs_schedule_aware_coaching_refresh.lock
- Log:
  logs/coaching_schedule_refresh.log

Pipeline stages:
1. live Game Intelligence
2. live coaching interpretation
3. postgame assimilation
4. coach identity attachment
5. coach regime profile

The shell wrapper also invokes WFS prediction accuracy reporting after
successful schedule-aware execution.

Do not claim a specific cron-triggered run occurred unless the log is
inspected.

======================================================================
35. WFS PREDICTION ACCURACY TRACKING
======================================================================
Reporter:
scripts/report_wfs_prediction_accuracy_v1.py
SHA256:
a54df90d8b0bd28caffd020f20e335f92a540ee9ff33e5c4f8c652ec424b13ca

Version:
WFS_PREDICTION_ACCURACY_V1

Source:
processed/forecast_prospective_evaluation_v1.csv

Outputs:
processed/wfs_prediction_accuracy_v1.csv
processed/wfs_prediction_accuracy_v1.json

Latest observed artifact refresh:
2026-09-13 18:05 local file timestamp.

Public Forecast Center:
WFS_PREDICTION_RECORD_UI_V1

Validated public presentation previously showed:
- Week 1: 8-2, 80.0%
- Season: 8-2, 80.0%

IMPORTANT:
This record is derived from the existing prospective evaluation artifact.

Do not claim strict immutable kickoff-time prediction freezing unless a
separate freeze/snapshot mechanism has been verified.

Future audit improvement:
- explicitly freeze game picks before kickoff
- retain versioned prospective snapshot
- grade only against that frozen artifact

======================================================================
36. CURRENT PRODUCTION HASHES — 2026-09-13/14 HANDOFF
======================================================================
app.py:
51b26ec6a76806571cb63cde2059d0eacd6df7e5db938999322b91d6a91d78e9

wfs_ai_analyst.py:
387d04451c362717e4eb6e6ffc835b95a39494cd9f8a32cc8d74f61fe84995c2

data/parquet/current_unified_stat_forecasts.parquet:
6067d8ea4a31ae60c5667c647c13c6bede4667c1369724c7861cf32929af13b9

Expected Pass Rate production artifact:
65f0b307d6dca3b7f9eb7f1dd7c1117f6673b6ac93230d8804ac70b278f02df1

Coach-attached evidence builder:
a6ed140f5187cab01d6f6fc3879f4b6ce58e3f830743fd693aff4b06e7110bb0

Key integrity result:
The analyst and canonical stat forecast hashes remain unchanged from the
validated game-day starter/cold-start bridge baseline.

======================================================================
37. UI STATUS / DEFERRED THEME ISSUE
======================================================================
Mobile top-overlap presentation issue was addressed with:
WFS_MOBILE_SPACING_FIX_V2

Light/Dark theme switching remains visually ineffective in the custom app
presentation.

Investigation found:
- Streamlit Light/Dark selector changes selection.
- Streamlit config contains explicit light and dark theme sections.
- app.py contains many historically hard-coded custom dark/light surfaces.
- Several experimental global theme overrides were removed.
- Theme behavior is still not materially different.

Decision:
DEFER.

Do not continue layering theme CSS during production/game-day work.
Do not redesign the app solely to solve this cosmetic issue.

Preserve:
- current mobile spacing
- branded WFS presentation
- production logic
- public/admin separation

Theme cleanup can be revisited later as an isolated UI refactor.

======================================================================
38. UPDATED DO NOT REGRESS
======================================================================
In addition to all earlier DO NOT REGRESS rules, preserve:

- LIVE player cards use actual-to-date stats, not pregame projections.
- Exact ESPN event + athlete identity for live player stats.
- Exact game_id -> ESPN event lookup for matchup presentation bridge.
- Whole-number public counting stats.
- Live ingest identity gates remain fail-closed.
- No weakening SAFE_FAIL to improve presentation freshness.
- Game Intelligence is structured-evidence-first.
- Free-form AI must not invent PBP facts.
- Coaching intelligence remains analysis-only.
- Current-event evidence must not contaminate historical same-regime prior.
- Postgame evidence is final-only.
- Exact coach/team identity; no fuzzy coach attachment.
- Schedule-aware coaching cron preserves unrelated jobs.
- Prediction accuracy must not be represented as kickoff-frozen unless
  verified.
- Mobile spacing fix remains.
- Light/Dark theme issue is deferred.
- Canonical stat forecast remains untouched by the live presentation
  bridge.
- DFS solver remains isolated from coaching intelligence.

======================================================================
39. NEXT OPERATIONAL CHECKS
======================================================================
High-value follow-up checks:

1. Inspect logs/coaching_schedule_refresh.log after a real cron window and
   verify an actual cron-triggered refresh.

2. Confirm WSH -> WAS alias through a completed-game postgame assimilation
   if not already observed in final-state output.

3. Investigate unresolved live-ingest identity SAFE_FAIL cases without
   weakening exact identity rules.

4. When convenient, implement explicit prospective-pick freezing before
   kickoff for stronger prediction-record auditability.

5. After game-day stability, resume the long-term cold-start opportunity
   transfer architecture described earlier.

6. Keep the Light/Dark theme issue deferred unless it becomes a priority.

======================================================================
END OF EXECUTE SUMMARY
======================================================================

## 40. NFL Live Production Closeout — 2026-09-13

Status: **VALIDATED / FROZEN**

The public NFL Live page was repaired and visually validated on the
production Streamlit service.

### Production deployment authority

- Production service: `wfs.service`
- Streamlit bind: `127.0.0.1:8502`
- Public nginx route proxies to port `8502`.
- Restart production only with:
  `sudo systemctl restart wfs.service`
- Do not manually launch a second Streamlit NFL process.

### Exact LIVE game selection

`wfs_live_view.py` now uses the ESPN `event_id` itself as the Streamlit
Game selectbox value and as the sole downstream render authority.

Marker:

`WFS_LIVE_EXACT_EVENT_SELECTOR_V2`

The selected event ID controls the event header, play-by-play,
athletes, audit data, and associated LIVE presentation reads.

No fuzzy game matching is permitted.

### Manual refresh selection preservation

Marker:

`WFS_LIVE_REFRESH_PRESERVE_SELECTION_V1`

The Refresh Live button no longer invokes an additional `st.rerun()`.
The Streamlit button interaction itself performs the rerun, preserving
the selected exact event.

This fixed the observed behavior where refreshing ATL-PIT could switch
the rendered game to PHI.

### Completed-game state and score authority

Marker:

`WFS_LIVE_COMPLETED_GAME_AUTHORITY_V1`

For an exact ESPN event ID, a completed game in read-only `nfl.db`
overrides stale SAFE_FAIL state/score information from `wfs_live.db`
for presentation only.

Completed games display:

- state = FINAL
- detail = Final
- clock = 0:00
- authoritative final away/home scores from `nfl.db`

This does not modify `wfs_live.db`.

The LIVE ingest identity SAFE_FAIL gate remains unchanged and must not
be weakened.

### Final-period presentation

Marker:

`WFS_LIVE_FINAL_PERIOD_CLEAR_V1`

Completed-game presentation clears the stale LIVE period value rather
than displaying a stale quarter such as Q2.

### Completed-game full play-by-play recovery

Marker:

`WFS_LIVE_COMPLETED_ESPN_PBP_V1`

When an event is completed, the presentation layer may read the ESPN
summary endpoint using the exact ESPN event ID and augment incomplete
stored play-by-play.

The fallback:

- is read-only
- does not persist into `wfs_live.db`
- deduplicates by exact play ID
- preserves the existing WFS play schema
- supports `drives.previous` and `drives.current`
- restores missing completed-game quarters

ATL-PIT validation demonstrated that the persisted LIVE database had
only Q1/Q2 because ingest had stopped under SAFE_FAIL, while ESPN had:

- Q1: 43 plays
- Q2: 62 plays
- Q3: 44 plays
- Q4: 40 plays

After the presentation fallback, the public ATL-PIT page visibly
showed all four quarter tabs and completed-game play-by-play.

### Analyst completed-game authority

The Analyst completed-game correction is independently validated and
frozen.

Markers:

- `WFS_ANALYST_COMPLETED_STATE_AUTHORITY_V2`
- `WFS_ANALYST_POSTGAME_SCORE_AUTHORITY_V2`
- `WFS_ANALYST_RESPONSE_STATE_AUTHORITY_V2`

Validated examples included:

- PIT defeats ATL 20-13
- BAL defeats IND 41-23
- BUF defeats HOU 36-31

Completed `nfl.db` game authority outranks stale SAFE_FAIL LIVE rows.

Do not modify this Analyst behavior without new evidence of an error.

### Validated production hashes

- `wfs_live_view.py`
  `1461dcd3fcc93fd1fc7e16c448dd193da3660499f3d901433735adf37920eae4`

- `app.py`
  `51b26ec6a76806571cb63cde2059d0eacd6df7e5db938999322b91d6a91d78e9`

- `wfs_ai_analyst.py`
  `8642eb7601e1251fa380073f61ef51ceef520d778be18c1e27e258741cf7088b`

- `current_unified_stat_forecasts.parquet`
  `6067d8ea4a31ae60c5667c647c13c6bede4667c1369724c7861cf32929af13b9`

### DO NOT REGRESS

1. Keep exact ESPN event-ID selection throughout NFL Live.
2. Refresh must preserve the selected game.
3. Completed `nfl.db` state/score outranks stale LIVE persistence.
4. Never weaken LIVE identity SAFE_FAIL to solve presentation issues.
5. Completed-game ESPN PBP fallback remains read-only.
6. Do not mutate the canonical stat forecast for LIVE presentation.
7. Do not modify the validated Analyst completed-game authority.
8. Production NFL Streamlit remains systemd-managed on port 8502.
