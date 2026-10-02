# Future Primitive Distribution Shadow V1

## Status

SHADOW_ONLY

This contract defines the requirements for a future stochastic primitive
generator. It does not authorize Monte Carlo execution, production influence,
FanDuel solver influence, or modification of frozen deterministic components.

## Frozen Dependencies

The following components are read-only dependencies:

- simulator_primitive_generator_v1
- simulator_recursive_state_replay_v1

The deterministic Primitive Generator V1 remains the reference/center generator.

## Primitive Output Contract

Every simulated team-game must ultimately produce exactly:

1. points
2. pass_attempts
3. carries
4. passing_yards
5. passing_tds
6. rushing_yards
7. rushing_tds

All seven values must satisfy the existing Primitive Generator / Replay
validation requirements.

## Historical Realized Outcome Authority

Validated historical outcome sources:

- points: data/nfl.db -> games
- offensive primitives: data/nfl.db -> team_game_stats
- validated cross-check artifact:
  processed/offensive_reconciliation_team_history_v1.csv

Validation population:

- 1,742 team-game rows
- 871 games
- zero duplicate game/team rows
- zero primitive nulls
- zero missing point joins

Historical realized outcomes may be used to characterize outcome distributions.

Historical realized outcomes must NOT be treated as pregame evidence for the
game being simulated.

## Historical Temporal Limitation

HISTORICAL_TEMPORAL_PROVENANCE_UNVERIFIED remains active.

Historical realized outcomes are valid postgame outcomes.

Any historical residual calibration requiring a pregame expected value must
use an expectation proven to have been constructed only from evidence
available before that game's cutoff.

Until such expectations are validated, raw outcome distributions may be
studied but must not be represented as leakage-proof forecast residuals.

## Distribution Architecture

The stochastic architecture must follow football dependency order:

shared game environment
-> offensive plays
-> pass/rush allocation
-> pass efficiency / rush efficiency
-> passing yards / rushing yards
-> passing TD / rushing TD realization
-> points reconciliation

The seven primitives must NOT be sampled independently.

## Deterministic Center

Primitive Generator V1 provides the deterministic reference state.

Stochastic layers may vary around or condition upon that state, but they must
not modify the frozen deterministic generator.

The deterministic result must remain reproducible independently of the
stochastic layer.

## Volume Invariants

pass_attempts >= 0
carries >= 0

pass_attempts and carries are integers.

Generated volume must preserve the simulator's explicit offensive-play
relationship.

No independently sampled attempts and carries may create an internally
contradictory play total.

## Yardage Invariants

passing_yards >= 0
rushing_yards >= 0

Yardage generation must remain conditional on generated opportunity volume.

Passing yards must be linked to pass attempts and passing efficiency.

Rushing yards must be linked to carries and rushing efficiency.

## Touchdown Invariants

passing_tds >= 0
rushing_tds >= 0

Touchdowns are nonnegative integer counts.

Touchdown generation must not use unconstrained continuous Normal draws.

TD realization must remain linked to offensive opportunity, yardage/scoring
environment, or another explicitly validated football mechanism.

## Points Invariants

points >= 0
points is an integer.

Points must be reconciled after offensive TD realization.

Generated points may not be lower than the scoring contribution explicitly
required by generated offensive touchdowns under the active scoring
reconciliation policy.

The baseline V1 scoring model remains recognized as incomplete for field goals,
PATs, defensive scores, safeties, and special-teams scoring.

## Dependence Requirements

Historical evidence demonstrates material dependence among primitives.

Observed raw-outcome examples include:

- attempts vs carries: negative dependence
- attempts vs passing yards: positive dependence
- carries vs rushing yards: strong positive dependence
- points vs offensive touchdowns: strong positive dependence

Opponent/team outcomes within the same game also show dependence.

Raw historical correlations must NOT automatically become simulator
correlation parameters because they include game-script effects.

Any explicit covariance/correlation model requires separate validation.

## Distribution Selection

No Normal, Poisson, negative-binomial, bootstrap, copula, or other
distribution family is authorized merely by this contract.

Distribution families must be selected from empirical validation.

Discrete and zero-heavy variables must be treated accordingly.

## Randomness and Reproducibility

Future stochastic execution must use explicit deterministic seeds.

Every stochastic result must record at minimum:

- simulator version
- distribution version
- seed
- scenario_id
- simulation_step
- parent_state_id
- parent_state_hash
- source/version hashes
- evidence cutoff
- simulated_at_utc

Identical validated inputs plus identical seed must produce identical outputs.

## Chronology and Provenance

Every generated primitive result must satisfy the frozen Primitive Generator
chronology and provenance contract.

SIMULATED results require:

completion <= simulated_at_utc <= cutoff_utc

Parent lineage must satisfy the frozen Replay adapter requirements.

No future observation may influence an earlier simulated state.

## Recursive Use

A stochastic result may enter Replay only after it passes the existing
primitive contract and Replay adapter validation.

Recursive State N+1 must be derived from the validated result of State N.

No hidden access to realized future games is permitted.

## Failure Policy

Unknown is not zero.

Missing distribution parameters, missing lineage, invalid chronology,
unsupported population, invalid football state, or unavailable required
evidence must fail closed.

No silent fallback to arbitrary variance is allowed.

## Current Authorization

REALIZED_PRIMITIVE_AUTHORITY=PASS
RAW_OUTCOME_DISTRIBUTION_AUDIT=PASS
DETERMINISTIC_CENTER=VALIDATED_FROZEN
STOCHASTIC_PARAMETERIZATION=NOT_YET_VALIDATED
RESIDUAL_CALIBRATION=NOT_YET_VALIDATED
MONTE_CARLO=NOT_YET_AUTHORIZED
PRODUCTION_INFLUENCE=NONE
FANDUEL_SOLVER_INFLUENCE=NONE
