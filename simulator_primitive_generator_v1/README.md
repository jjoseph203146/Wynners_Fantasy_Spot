# Primitive Game-State Generator V1

SHADOW / VALIDATION ONLY.

This package defines the deterministic primitive-result boundary required
before prospective recursive simulation can be connected to
`simulator_recursive_state_replay_v1`.

V1 foundation does **not** generate football outcomes yet.

It validates seven nullable team primitives:

- points
- pass attempts
- carries
- passing yards
- passing TDs
- rushing yards
- rushing TDs

Unknown values remain `null`; they are never converted to zero.

Every result is bound to:

- season/week/game
- scenario
- simulation step
- kickoff/completion/cutoff
- OBSERVED or SIMULATED provenance
- observation or simulation timestamp
- source version/hash
- parent state ID/hash
- transition version
- two canonical team keys
- schedule binding
- coach bindings

The contract is deterministic and content-hashed.

No production forecasts, FanDuel projections, solver inputs, lineup
generation, Monte Carlo, model training, Cold-Start V1, V3.1, or Replay V1
files are modified by this package.

Permanent historical limitation remains:

`HISTORICAL_TEMPORAL_PROVENANCE_UNVERIFIED`
