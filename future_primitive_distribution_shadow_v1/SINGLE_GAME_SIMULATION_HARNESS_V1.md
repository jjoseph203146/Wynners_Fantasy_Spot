# Single-Game Simulation Harness V1

Status: VALIDATED_FROZEN, SHADOW ONLY.
Version: SINGLE_GAME_SIMULATION_HARNESS_V1_001.

`simulate_single_game(data)` obtains both deterministic centers from the actual
frozen Generator, calls the frozen paired sampler exactly once on a valid
invocation, validates the output, and returns six simulated primitives per team.
There is no RNG implementation, batch interface, retry, forecasting API, or
production integration in this harness. Invalid pre-sampling inputs make zero
stochastic calls; a failure after calling the sampler never triggers another call.

## Input

The input is one object with exactly these five fields:

```text
game_identity:
  season: positive integer
  week: positive integer
  game_id: nonempty canonical string
  team_1: nonempty canonical string
  team_2: distinct nonempty canonical string
  home_team / away_team: optional, both supplied together if known
simulation_context:
  kickoff: explicit timezone-aware ISO-8601 timestamp
  completion: explicit timezone-aware ISO-8601 timestamp
  simulated_at: explicit timezone-aware ISO-8601 timestamp
  cutoff: explicit timezone-aware ISO-8601 timestamp
team_1_generator_input: complete frozen Replay output for team_1 versus team_2
team_2_generator_input: complete frozen Replay output for team_2 versus team_1
seed: explicit integer (booleans rejected)
```

The caller supplies the two established Replay states, including their
`state_manifest`, `feature_reconstruction` list of named values, contract hash,
transition validation, and audit hash. This adapter does not reconstruct features.
It requires `TRANSITION_COMPLETE`, no transition issues, satisfied temporal rules,
a matching frozen feature contract, and a valid Replay result hash. Game identity,
team/opponent perspective, and kickoff must match the requested game. Duplicate
feature names and nonfinite values fail closed; the actual Generator enforces its
required feature availability. No missing value is replaced with zero.

The harness uses the frozen Generator timestamp parser. Simulation timestamps
follow the existing Generator SIMULATED contract:
`kickoff < completion <= simulated_at <= cutoff`. The result cutoff is distinct
from each input state's pregame `evidence_cutoff`, which follows the frozen Replay
rule `evidence_cutoff < kickoff`. These are logical simulation timestamps, not a
requirement to wait for an actual future game. No kickoff or completion is inferred.
The adapter does not fabricate a seven-primitive result merely to invoke the full
Generator result validator, because sampled points are deliberately absent.

The focused tests construct both perspectives with frozen Replay `build`, using
its existing synthetic fixture. The fixture game is future relative to its evidence
cutoff, not a live NFL forecast. Replay hashes establish input consistency; they do
not prove historical publication timing or upgrade temporal provenance.

## Generator authority and center mapping

The adapter calls `estimate_team_volume(state)`, then
`estimate_team_yardage(state, volume)`, then
`estimate_team_scoring(state, volume, yardage)` for each team. The complete wrapper
`generate_team_primitives` omits the efficiency mechanisms needed by the sampler,
so the adapter uses its same three validated stages directly. Focused tests also
compare the retained seven deterministic scoring primitives to that wrapper.

| Sampler field | Retained Generator output path |
| --- | --- |
| expected_plays | volume.offensive_plays |
| expected_pass_rate | volume.pass_rate |
| expected_pass_ypa | yardage.passing_yards_per_attempt |
| expected_rush_ypc | yardage.rushing_yards_per_carry |
| expected_passing_tds | scoring.passing_tds |
| expected_rushing_tds | scoring.rushing_tds |
| expected_points | scoring.points |

These are the same mechanism choices used to construct the frozen Residual Bank.
There are no reconstructed efficiency ratios, raw-diagnostic substitutions, or
copied Generator formulas. The frozen seeded sampler's center validator is reused.
The full unmodified volume, yardage, and scoring outputs are retained under each
team's `generator_output`, alongside its `sampler_center` and explicit mapping.

## One stochastic call and output validation

An invocation-local one-shot guard consumes its budget before invoking the paired
sampler, including on failure. Only the paired sampler controls RNG, game selection,
and orientation. The harness passes the seed unchanged and preserves orientation.

After the single call, validation checks version, seed, bank hash, exact output
fields, centers, pair flags, authorization, and both historical identities against
the frozen bank. It verifies each entire vector against its historical row and the
selected orientation. The existing frozen reconciliation function is reused to
check returned mechanisms, primitives, and audit entries exactly; it neither draws
another sample nor repairs output. Six nonnegative integer primitives and volume
conservation are required. Corrupted vectors, audit entries, or primitives fail.

Inputs and Generator outputs are not mutated. Output includes game/context,
authorities, both teams' Generator outputs and centers, historical identities,
residuals, realized mechanisms/primitives, reconciliation, pair integrity,
invariants, and authorization. Deterministic Generator points and expected_points
are diagnostic metadata only. No simulated final score or fantasy points exist.
This six-primitive output is not a complete seven-primitive Replay transition.

Generator `PrimitiveIntelligenceError` and sampler `SamplerError` propagate.
Malformed adapter structure raises `HarnessError`; frozen validation exceptions
also fail closed. No failure substitutes a seed, center, vector, or sampled game.

## Validation and limits

Focused harness tests: 79/79 PASS. Only the fixed seeds 0 and 1 are used.
Frozen deterministic Generator regression: 89/89 PASS.
Frozen Replay regression: 41/41 PASS.
All 50 existing protected files match before and after hashes, including all
non-cache files in both frozen packages and every existing shadow artifact.

The validation JSON contains exact status fields, regression results, test names,
source hashes, protected before/after evidence, and the fixed test seed set.
Reproduce focused tests only with:

```sh
python3 -B -m unittest future_primitive_distribution_shadow_v1.test_single_game_simulation_harness_v1 -q
```

This checkpoint validates single-game integration, not football accuracy,
calibration, production readiness, or fantasy usefulness. Historical temporal
provenance remains UNVERIFIED. Monte Carlo, repeated game simulation, production
influence, and FanDuel solver influence remain unauthorized. No distribution audit
or seed grid was rerun.
