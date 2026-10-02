# Bounded Walk-Forward Validation V1

This is a shadow-only chronological validation of the frozen Generator stages
against the 285 games labelled season 2025 in the training authority. Games are
ordered by parsed `game_date + gametime`, then `game_id`. Each game receives one
SHA256-derived seed and one realization. No points, final score, fantasy score,
FanDuel input, solver input, Monte Carlo run, seed search, retry, or parameter
tuning is performed.

The validator builds the exact feature names required by the frozen Generator
from each target's pregame training row, calls the frozen volume, yardage, and
scoring stages, captures their deterministic centers, calls
`TEMPORAL_RESIDUAL_SELECTOR_V1_001` once, and applies the frozen residual
realization helper to the six permitted mechanisms. It derives attempts, carries,
yards, and touchdowns through the existing validated logic.

Temporal eligibility is strictly `historical_kickoff < target_kickoff`; target,
same-kickoff, and later residual games are excluded. The selector's pair,
orientation, and six-dimensional vectors are retained in each accepted record.
All 1,710 training rows, 855 games, pair identities, kickoff parsing, and unique
realized outcome game/team keys are checked before validation. Frozen Generator
errors would produce an explicit rejection with the exact exception and no
fabricated value.

The run is repeated once to prove deterministic serialization and global Python
and NumPy RNG preservation. The current result is recorded in
`BOUNDED_WALK_FORWARD_VALIDATION_V1.json`, including descriptive primitive error
statistics only; no accuracy threshold is imposed.

`HISTORICAL_TEMPORAL_PROVENANCE=UNVERIFIED`: residual selection is temporally
isolated, but this validation does not prove exact historical as-of provenance of
the pregame feature rows.
