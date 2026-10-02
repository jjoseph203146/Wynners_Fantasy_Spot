# Historical Game Paired Residual Sampler V1

Status: VALIDATED_FROZEN, shadow only.
Version: HISTORICAL_GAME_PAIRED_RESIDUAL_SAMPLER_V1_001.

`historical_game_paired_residual_sampler_v1.sample_paired_residual(center_team_1,
center_team_2, seed=<explicit integer>, bank_path=<frozen bank>)` returns one
structured paired result. There is no batch, repeated game, or integration API.
Both centers use the frozen sampler's seven `expected_*` fields.

The frozen bank hash is checked on every call before parsing or randomness.
All rows are grouped by game_id, requiring exactly 839 games, two distinct
reciprocal opponents per game, matching season/week, and finite complete
six-dimensional vectors. No malformed game is excluded or repaired: the whole
call fails closed. Bank schema, identity, and integer TD residual validation
are reused from the frozen sampler.

Canonical game order is lexicographic game_id order. Within each game, A/B
are lexicographically sorted historical team names, with no football meaning.
A local `random.Random(seed)` supplies exactly two `random()` calls: the first
selects `int(draw * 839)` and the second selects `int(draw * 2)`. Orientation 0
maps A/B to future team_1/team_2; orientation 1 maps B/A. There is no home/away,
favorite, strength, or future identity interpretation. No retry occurs.

Center validation and per-team reconciliation call the unchanged frozen
`_validate_center` and `_realize` functions directly. This intentionally depends
on private frozen V1 helpers; a later version must review that dependency.
Negative sampled plays and arithmetic overflow fail closed. Nonnegative plays
are rounded half-up; pass rate is clamped; attempts are rounded and carries
are the remainder; efficiencies and TDs are floored at zero; yards and TDs
use the frozen rounding semantics. Every changed value carries the frozen
raw/reconciled operation audit. TDs are not additionally capped by opportunities,
matching frozen V1 semantics. expected_points remains center metadata only.

Validation consists of 50 focused paired tests and the unchanged 35-test
seeded sampler regression suite. Tests use only fixed seeds 0/1 and controlled
RNG stubs, with synthetic malformed inputs for rejection tests. No distribution
experiment or Monte Carlo was performed. The 28 protected files (including
all non-cache files in both frozen source directories) match their before hashes.

Reproduce from the repository root with bytecode disabled:

```sh
python3 -B -m unittest future_primitive_distribution_shadow_v1.test_historical_game_paired_residual_sampler_v1 -v
python3 -B -m unittest future_primitive_distribution_shadow_v1.test_seeded_joint_residual_sampler_v1 -v
```

See PAIRED_SAMPLER_V1_VALIDATION.json for status and artifact hashes, and
PAIRED_SAMPLER_V1_FROZEN_HASHES_BEFORE.json / AFTER.json for protected evidence.
No production or FanDuel integration is authorized or present.
