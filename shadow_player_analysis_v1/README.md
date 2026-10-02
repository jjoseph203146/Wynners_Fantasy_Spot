# Player Analysis Intelligence V1 — offline foundation

This directory implements design steps 1–5 only. It has no production consumer,
database access, external connector, service, schedule, or UI integration.
Only synthetic fixture approvals are accepted. All real sources are disabled.
No access or licensing permission is inferred from an existing source adapter.

Run from the repository root with bytecode disabled:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B -m unittest shadow_player_analysis_v1.test_foundation -v
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B -m shadow_player_analysis_v1
```

The second command publishes only the fictional bundled fixture under
`shadow_player_analysis_v1/artifacts/<content-addressed-run-id>/`. It accepts no
network URL, arbitrary output directory, database, or production input. Publication
stages and validates a complete bundle before rename; an existing run is verified,
never overwritten. SHA-256 detects corruption, not malicious re-signing.

## Contracts

`core.py` defines the explicit safety flags, taxonomy, source-native event identity,
revision history, claim eligibility, and independent-origin consensus. JSON is
canonical UTF-8 with sorted keys and no NaN. Raw fixture evidence text is preserved
without whitespace normalization. All timestamps must include a timezone.

Identity uses the existing `fanduel_injury_ingest` normalization and exact resolver
with supplied WFS identity snapshots. It never calls its network functions, DB
index builder, schema migration, writer, or main. The imported module's whole-file
hash is recorded. Its pandas/requests dependencies must already be installed;
the foundation installs nothing. Synthetic GSIS placeholders exist only in tests
and fixtures; no WFS identities are created. A snapshot label is a fixture contract,
not cryptographic proof of identity authority. Real snapshot provenance validation
must precede any future expansion beyond fixtures.

Manual claim labels and an exact forward-looking evidence span are required. The
foundation intentionally does not infer opportunity from carries, yards, catches,
targets, or other recap words. It is not an automatic language classifier. Human
annotation correctness still requires review. WFS corroboration is represented in
the taxonomy but quarantined until dependency validation is implemented.

Accepted claims retain source, native event ID, origin when known, publication and
retrieval times, revision, evidence quote, signal, direction, confidence, identity
result, target season/week/game, kickoff, phase, temporal eligibility, and consensus
state. Missing identity/game/time context fails closed. Publication after kickoff,
late capture, retrospective evidence, and expired claims are quarantined. The latest
revision known by the cutoff supersedes earlier claims without deleting history.
Event-level structural failures abort publication; claim-level failures enter the
quarantine artifact with reasons. Null confidence means unknown extraction quality,
not a probability that an analyst prediction is true.

Origin IDs must identify an independently reviewed reporting origin. Source count
is never substituted for origin count. Unknown origins cannot establish agreement.
Opposing role/matchup signals use shared comparison families. Different conditions,
metrics, denominators, baselines, horizons, or evidence kinds remain separate groups.
No numerical forecasts, source weights, player rankings, or selection scores exist.

## Artifact contract

The bundle contains `events.json`, `claims.json`, `consensus.json`,
`quarantine.json`, `input.json`, `validation.json`, and `manifest.json`.
JSON is used instead of Parquet for this small offline foundation to keep canonical
byte serialization directly inspectable. Evidence artifacts have safety envelopes;
the exact input fixture is hash-bound by the manifest. The manifest binds code,
identity input, target game, cutoff, policy, approval, file sizes, and SHA-256 hashes.
The run ID hashes the manifest excluding its own run ID. Wall-clock publication
time is deliberately excluded from deterministic bundle bytes. The input retains
source publication/capture timestamps. No production `current` pointer is created.

## Scope and limitations

No step 6 source connector or step 7 outcome evaluation is implemented. Historical
fixtures are fictional and prove mechanics, not analytical accuracy or commercial
value. No automatic source approval or promotion is available. Real feeds, archived
identity provenance, capture completeness, licensing, retention, outcome calibration,
and promotion each require separate future work and authorization.

Existing repository files were not modified. The directory was not a Git worktree
at implementation time. No pre-existing file required a backup.
