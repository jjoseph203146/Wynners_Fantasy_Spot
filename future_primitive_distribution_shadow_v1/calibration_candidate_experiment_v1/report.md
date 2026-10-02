# Calibration Candidate Experiment V1

## Status

**PASS_SHADOW_ONLY**

This checkpoint records the completed calibration-candidate research phase.
It does not authorize production changes or Monte Carlo use.

## Validation design

- TRAIN: Weeks 1-14
- HOLDOUT: Weeks 15-22
- HOLDOUT was not used to fit candidate parameters.
- HOLDOUT has now been repeatedly inspected and is therefore not a blind
  validation set for additional candidate selection.
- No parameter grid search was performed.

## Supported shadow challenger: rushing volume and yardage

The combined challenger uses:

- TRAIN-only pass-rate residual scale: 0.875493246
- TRAIN-only YPC reliability slope: 0.130098342
- TRAIN-only YPC regression intercept: 3.729193578
- TRAIN sampled RYPC residual mean: -0.141141699

Holdout rushing-yard SD ratio improved from 1.203267 to 1.097962.
Holdout rushing-yard bias was +1.332922 yards.
Holdout carry SD ratio improved from 1.161728 to 1.077975.

This remains a shadow challenger only.

## Passing yardage

The pass-YPA reliability challenger improved marginal YPA calibration but
worsened downstream holdout passing-yard underdispersion.

Holdout passing-yard SD ratio:

- baseline: 0.977870
- YPA reliability candidate: 0.929191

The historical joint residual bank contains a persistent negative
pass-rate/pass-YPA relationship. The sampler faithfully transports that
relationship. No sampler implementation failure was demonstrated.

Passing yardage therefore retains the baseline representation.

## Supported shadow challenger: rushing touchdowns

A TRAIN-only unconditional empirical discrete TD model materially improved
holdout rushing-TD behavior.

Holdout:

- mean bias: +0.190260 -> +0.058896
- SD ratio: 1.071737 -> 0.978491
- P0 error: -0.080195 -> -0.041883
- P3+ error: +0.028636 -> -0.005130

Candidate deterministic fingerprint:

1ec78892221d33e848edf9c800611c65f1a4f4045fe491daa9cd105aeabdec0a

This remains a shadow challenger only.

## Passing touchdowns

The unconditional empirical discrete challenger improved mean bias and the
3+ tail but worsened holdout P0 calibration and produced underdispersion.

Passing TD therefore retains the baseline representation.

## TD diagnosis

The additive TD center-plus-residual representation shows conditional
mismatch, especially for rushing touchdowns.

The zero floor materially contributes positive mean shift but is a valid
physical constraint and must not simply be removed.

Residual recentering and zero-floor removal are not authorized.

## Important limitation

Historical as-of provenance remains **UNVERIFIED**.

No demonstrated future leakage was found in the provenance audit, but the
historical source architecture cannot prove strict-before availability for
all required inputs.

The Weeks 15-22 holdout has also been inspected repeatedly. Future candidate
confirmation therefore requires a genuinely new validation period rather
than continued optimization against this holdout.

## Production status

- Frozen simulator changed: NO
- Residual bank changed: NO
- Monte Carlo authorized: NO
- Production change authorized: NO
- Production influence: NONE
