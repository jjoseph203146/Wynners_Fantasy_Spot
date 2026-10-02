# Seeded Joint-Residual Sampler V1

Validated shadow checkpoint: `SEEDED_JOINT_RESIDUAL_SAMPLER_V1_001`.
This checkpoint implements the user's explicit single-vector sampling authorization.
Monte Carlo and production/Replay integration remain unauthorized. The earlier
CONTRACT.md future full-game requirements are not a claim that this six-primitive
team result is a complete Replay result. Points reconciliation is deferred.
Historical temporal provenance remains UNVERIFIED as recorded in the frozen bank.

## API and deterministic center

`sample_joint_residual(center, seed=<integer>, bank_path=<frozen CSV path>)`
returns one JSON-serializable dictionary. The default path is the adjacent frozen
bank. There is no batch API, database access, generator invocation, or solver hook.
The hash is hard-coded and cannot be overridden through the API. The exact bytes
hashed are the bytes parsed, preventing a separate-read race. All rows are checked
before selecting a row. Missing/extra/reordered columns, duplicate canonical
identities, absent/nonfinite residuals, fractional TD residuals and invalid seeds
fail closed. Boolean seeds and numeric strings in the center are rejected.

Supply the same flat center mapping used by the residual-bank builder:

| Center field | Existing Generator V1 output |
| --- | --- |
| expected_plays | volume.offensive_plays |
| expected_pass_rate | volume.pass_rate |
| expected_pass_ypa | yardage.passing_yards_per_attempt |
| expected_rush_ypc | yardage.rushing_yards_per_carry |
| expected_passing_tds | scoring.passing_tds |
| expected_rushing_tds | scoring.rushing_tds |
| expected_points | scoring.points |

All seven fields are mandatory, numeric, finite and nonnegative. Center pass rate
must be in [0,1] and center TDs must be integer-valued. The caller is responsible
for supplying an actual deterministic Generator V1 center; this API does not
certify its provenance. Additional center metadata is ignored. Inputs are copied,
not mutated. expected_points is retained only in center diagnostic metadata.

## Selection and reconciliation

Exactly one local `random.Random(seed).random()` draw selects
`int(draw * row_count)` in frozen CSV order. Python's integer-seeded random()
sequence is reproducible. No global RNG state is read or changed; no dimension
is sampled separately. No distribution family is fitted. This is discrete
empirical row selection (with the standard finite precision of random()).
The result records the hash, version, seed, zero-based CSV data-row index, season,
week, game_id, team, opponent_team, and all six residuals. Canonical uniqueness
is season/week/game_id/team; changing the opponent cannot disguise a duplicate.

1. Add plays residual. Negative raw plays fail closed, even if rounding could
   have hidden the negative value. Nonnegative plays round to nearest integer,
   with exact halves upward.
2. Add pass-rate residual and clamp to [0,1]. Round plays times pass rate half-up
   to attempts; carries are exactly plays minus attempts. The reported pass rate
   is the reconciled continuous mechanism, not attempts/plays after rounding.
3. Add efficiency residuals and explicitly floor negative efficiencies to zero.
   Multiply by the reconciled opportunity counts and round yardage half-up.
4. Add integer TD residuals to integer center TDs and floor negative totals to
   zero. No new TD model or opportunity cap is introduced: this preserves the
   bank's specified TD residual interpretation. TD/yardage/points consistency
   beyond the requested nonnegative-count invariants is a later phase.
5. Validate integer, nonnegative primitives and exact volume conservation.
   Arithmetic nonfinite results fail closed.

Every clamp, floor or rounding operation that changes a numeric value appears
in reconciliation with field, operation, raw and reconciled values, in execution
order. Unchanged numeric casts do not produce audit entries. No points residual,
realized points, or scoring model is present.

## Validation

Run from the repository root:

```sh
python3 -B -m unittest future_primitive_distribution_shadow_v1.test_seeded_joint_residual_sampler_v1 -v
```

35 focused tests pass. Only the fixed seeds 0 and 1 are used. Boundary tests
exercise the transformation with synthetic vectors without RNG draws. Corruption
tests exercise the private structural parser on in-memory CSV copies because the
public hash gate necessarily rejects any modified frozen bank before parsing.
The public hash gate is tested separately; it has no permissive hash parameter.
No distribution simulation or Monte Carlo was run.

SEEDED_SAMPLER_V1_FROZEN_HASHES_BEFORE.json records all 28 protected files before
implementation, including every non-cache file under Generator and Replay and
all four frozen shadow artifacts. SEEDED_SAMPLER_V1_VALIDATION.json records both
hashes per file and their equality after the tests, plus sampler source hashes.
SEEDED_SAMPLER_V1_TEST_RESULTS.txt contains the final focused test output.
