# Temporal Residual Selector V1

Status: validated, SHADOW ONLY. Version: TEMPORAL_RESIDUAL_SELECTOR_V1_001.

```python
from future_primitive_distribution_shadow_v1.temporal_residual_selector_v1 import select_temporal_paired_residual
result = select_temporal_paired_residual(
    target_game_id="2024_08_TEN_DET",
    target_kickoff="2024-10-27T13:00:00",
    seed=7,
)
```

The target must be an explicit training game ID and its explicit kickoff must equal
that game's parsed training kickoff. Kickoffs are naive source wall-clock values:
YYYY-MM-DDTHH:MM[:SS] (a space may replace T). Offset-bearing inputs are rejected;
no timezone is inferred from data that does not specify one. Season/week are
identity consistency checks only, never temporal authority.

Every call validates the frozen bank hash, 839 intact reciprocal pairs, all 1,710
training rows, 855 games, two reciprocal training teams per game, nonempty and
parseable dates/times, identical kickoff within each game, no duplicate
matchup/kickoff under multiple IDs, and complete residual joins including team,
opponent, season and week consistency. The chronology bytes' SHA-256 is returned.
Malformed inputs fail closed before RNG creation. An empty eligible population
also fails before drawing. No population is cached.

Eligibility is strictly historical_kickoff < target_kickoff. Target, simultaneous
and later games cannot enter the eligible population. Population order is frozen
sampler order: game_id, then team. A local random.Random(seed) makes exactly two
random() calls: floor(first * eligible_count) selects the pair; floor(second * 2)
selects orientation. Orientation 0 maps sorted A/B to team_1/team_2, and 1 maps
B/A. Integer seeds are required; booleans fail. Errors propagate without retries.
Both six-dimensional vectors and original row identities are copied intact.

Only the frozen bank loader and paired-population validator are reused. No center
realization, primitive reconciliation, Generator, harness, points sampling,
fantasy calculation, simulation, Monte Carlo or FanDuel integration is called.
No production influence is authorized. No walk-forward simulator was built.

Temporal isolation applies to RESIDUAL SELECTION ONLY.
HISTORICAL_TEMPORAL_PROVENANCE remains UNVERIFIED: this component does not prove
that historical features or fitted expectations were available at their kickoff.

Validation: 13 focused unittest methods passed, using explicit seeds 7 and 11
and controlled draw values for orientation coverage. Tests independently rebuild
the eligible population from the two CSV authorities, inspect the actual complete
population, verify both orientations and exact vectors, count local draws, compare
Python/NumPy global RNG states, inject malformed chronology/pairs and draw failures,
and guard against realization calls. The before manifest protects all preexisting
files in both simulator directories and this shadow directory (ignoring
__pycache__); the validation JSON records every corresponding after hash.

Run: `venv/bin/python -B -m unittest future_primitive_distribution_shadow_v1.test_temporal_residual_selector_v1 -v`

Full audit and example output: TEMPORAL_RESIDUAL_SELECTOR_V1_VALIDATION.json.
