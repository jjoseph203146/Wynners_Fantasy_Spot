# Game Coupling Audit V1

Status: PASS. Evidence: **MATERIAL_CROSS_TEAM_RESIDUAL_DEPENDENCE**. No architecture authorized.

Bank SHA256: `614b079e45c4cda7c09c58b1f6820825f221f6a40596d356c26d93d199064c6b`

Rows: 1678. Unique games: 839. Exactly two rows: 839. Malformed games: 0.

game_id ascending, then team lexicographically ascending. Distinct teams, reciprocal opponents, matching season/week, and finite residuals verified. No 2026 data.

## Cross-team correlations

| Residual | Pearson | Spearman | Symmetric Pearson | Symmetric Spearman |
|---|---:|---:|---:|---:|
| resid_plays | -0.347847 | -0.361812 | -0.350801 | -0.362526 |
| resid_pass_rate | -0.211847 | -0.229586 | -0.212281 | -0.230229 |
| resid_pass_ypa | -0.014474 | +0.005332 | -0.014608 | +0.005827 |
| resid_rush_ypc | -0.043012 | -0.028870 | -0.043857 | -0.028393 |
| resid_passing_tds | +0.105120 | +0.101205 | +0.104142 | +0.100097 |
| resid_rushing_tds | +0.004835 | -0.002480 | +0.004684 | -0.002815 |

Pearson product-moment and Spearman average-tie ranks, one deterministic pair per game. No missing values or exclusions. SciPy results cross-checked against pandas Pearson and ranked Pearson.

## Season stability

| Season | Games | Residual | Pearson | Spearman |
|---|---:|---|---:|---:|
| 2023 | 269 | resid_plays | -0.323345 | -0.360127 |
| 2023 | 269 | resid_pass_rate | -0.164131 | -0.172098 |
| 2023 | 269 | resid_pass_ypa | -0.072707 | -0.040495 |
| 2023 | 269 | resid_rush_ypc | +0.020646 | +0.042155 |
| 2023 | 269 | resid_passing_tds | +0.114090 | +0.097318 |
| 2023 | 269 | resid_rushing_tds | +0.015061 | +0.020189 |
| 2024 | 285 | resid_plays | -0.371230 | -0.373688 |
| 2024 | 285 | resid_pass_rate | -0.275237 | -0.287821 |
| 2024 | 285 | resid_pass_ypa | +0.015526 | +0.027761 |
| 2024 | 285 | resid_rush_ypc | +0.015203 | -0.005421 |
| 2024 | 285 | resid_passing_tds | +0.130560 | +0.117690 |
| 2024 | 285 | resid_rushing_tds | +0.009463 | -0.003455 |
| 2025 | 285 | resid_plays | -0.348037 | -0.351097 |
| 2025 | 285 | resid_pass_rate | -0.188571 | -0.218249 |
| 2025 | 285 | resid_pass_ypa | +0.005370 | +0.023590 |
| 2025 | 285 | resid_rush_ypc | -0.138420 | -0.125978 |
| 2025 | 285 | resid_passing_tds | +0.074069 | +0.086685 |
| 2025 | 285 | resid_rushing_tds | -0.008751 | -0.024909 |

## Transformed distributions

Sample standard deviation ddof=1; linear-interpolated empirical quantiles; team offensive TD uses 1678 team rows, all other quantities use 839 games. TD sums are residual sums, not actual scoring totals.

### All seasons

| Quantity | n | Mean | SD | Min | p01 | p05 | p25 | Median | p75 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sum_resid_plays | 839 | -0.2777 | 10.2599 | -30.0000 | -23.0000 | -17.0000 | -7.0000 | -1.0000 | 6.0000 | 17.1000 | 26.6200 | 36.0000 |
| absolute_difference_resid_plays | 839 | 11.9750 | 8.6866 | 0.0000 | 0.0000 | 1.0000 | 5.0000 | 11.0000 | 17.0000 | 29.0000 | 36.6200 | 47.0000 |
| sum_resid_passing_tds | 839 | -0.0012 | 1.8276 | -5.0000 | -4.0000 | -3.0000 | -1.0000 | 0.0000 | 1.0000 | 3.0000 | 5.0000 | 6.0000 |
| sum_resid_rushing_tds | 839 | -0.0584 | 1.4810 | -4.0000 | -3.0000 | -2.0000 | -1.0000 | 0.0000 | 1.0000 | 2.0000 | 4.0000 | 6.0000 |
| team_offensive_td_residual | 1678 | -0.0298 | 1.5014 | -4.0000 | -3.0000 | -2.0000 | -1.0000 | 0.0000 | 1.0000 | 2.0000 | 4.0000 | 7.0000 |
| combined_game_offensive_td_residual | 839 | -0.0596 | 2.2401 | -7.0000 | -5.0000 | -4.0000 | -2.0000 | 0.0000 | 1.0000 | 4.0000 | 5.0000 | 6.0000 |

### 2023

| Quantity | n | Mean | SD | Min | p01 | p05 | p25 | Median | p75 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sum_resid_plays | 269 | -0.4610 | 10.5913 | -30.0000 | -23.3200 | -20.0000 | -7.0000 | -1.0000 | 6.0000 | 17.8000 | 26.3200 | 36.0000 |
| absolute_difference_resid_plays | 269 | 12.0669 | 8.8610 | 0.0000 | 0.0000 | 1.0000 | 5.0000 | 11.0000 | 18.0000 | 29.0000 | 33.6400 | 41.0000 |
| sum_resid_passing_tds | 269 | 0.0818 | 1.8061 | -5.0000 | -3.0000 | -2.0000 | -1.0000 | 0.0000 | 1.0000 | 3.0000 | 5.0000 | 6.0000 |
| sum_resid_rushing_tds | 269 | -0.0186 | 1.4311 | -3.0000 | -3.0000 | -2.0000 | -1.0000 | 0.0000 | 1.0000 | 2.0000 | 4.0000 | 4.0000 |
| team_offensive_td_residual | 538 | 0.0316 | 1.5137 | -4.0000 | -3.0000 | -2.0000 | -1.0000 | 0.0000 | 1.0000 | 3.0000 | 4.0000 | 7.0000 |
| combined_game_offensive_td_residual | 269 | 0.0632 | 2.2176 | -5.0000 | -4.3200 | -3.0000 | -1.0000 | 0.0000 | 1.0000 | 4.0000 | 6.0000 | 6.0000 |

### 2024

| Quantity | n | Mean | SD | Min | p01 | p05 | p25 | Median | p75 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sum_resid_plays | 285 | 0.0175 | 10.0371 | -26.0000 | -22.1600 | -17.0000 | -7.0000 | 0.0000 | 7.0000 | 16.0000 | 24.4800 | 28.0000 |
| absolute_difference_resid_plays | 285 | 12.0877 | 8.5417 | 0.0000 | 0.0000 | 1.0000 | 6.0000 | 11.0000 | 17.0000 | 28.0000 | 35.9600 | 44.0000 |
| sum_resid_passing_tds | 285 | 0.0281 | 1.8035 | -4.0000 | -4.0000 | -3.0000 | -1.0000 | 0.0000 | 1.0000 | 3.0000 | 5.0000 | 5.0000 |
| sum_resid_rushing_tds | 285 | -0.0211 | 1.5034 | -4.0000 | -3.0000 | -2.0000 | -1.0000 | 0.0000 | 1.0000 | 2.0000 | 4.0000 | 6.0000 |
| team_offensive_td_residual | 570 | 0.0035 | 1.4798 | -4.0000 | -3.0000 | -2.0000 | -1.0000 | 0.0000 | 1.0000 | 2.0000 | 3.3100 | 6.0000 |
| combined_game_offensive_td_residual | 285 | 0.0070 | 2.2123 | -7.0000 | -5.0000 | -3.0000 | -2.0000 | 0.0000 | 1.0000 | 4.0000 | 5.0000 | 6.0000 |

### 2025

| Quantity | n | Mean | SD | Min | p01 | p05 | p25 | Median | p75 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sum_resid_plays | 285 | -0.4000 | 10.1919 | -24.0000 | -22.0000 | -16.0000 | -8.0000 | -1.0000 | 6.0000 | 17.8000 | 24.8000 | 35.0000 |
| absolute_difference_resid_plays | 285 | 11.7754 | 8.6910 | 0.0000 | 0.0000 | 1.0000 | 5.0000 | 10.0000 | 17.0000 | 29.0000 | 37.4800 | 47.0000 |
| sum_resid_passing_tds | 285 | -0.1088 | 1.8724 | -4.0000 | -4.0000 | -3.0000 | -1.0000 | 0.0000 | 1.0000 | 3.0000 | 5.0000 | 5.0000 |
| sum_resid_rushing_tds | 285 | -0.1333 | 1.5069 | -3.0000 | -3.0000 | -2.0000 | -1.0000 | 0.0000 | 1.0000 | 2.8000 | 4.0000 | 4.0000 |
| team_offensive_td_residual | 570 | -0.1211 | 1.5096 | -4.0000 | -3.0000 | -2.0000 | -1.0000 | 0.0000 | 1.0000 | 2.0000 | 3.0000 | 4.0000 |
| combined_game_offensive_td_residual | 285 | -0.2421 | 2.2846 | -5.0000 | -5.0000 | -4.0000 | -2.0000 | 0.0000 | 1.0000 | 4.0000 | 5.0000 | 5.0000 |

## Interpretation

Material cross-team dependence is concentrated in volume and allocation: plays Pearson is -0.348 overall (-0.323, -0.371, -0.348 by season), and pass rate is -0.212 (-0.164, -0.275, -0.189). Spearman agrees on direction and approximate magnitude.

Passing-TD residual dependence is modest positive in every season; pass-YPA and rushing-TD correlations are near zero overall. Rush-YPC dependence varies, becoming more negative in 2025. A near-zero correlation does not establish independence or exclude nonlinear or tail dependence.

The descriptive MATERIAL label reflects stable, appreciable dependence in plays and pass rate; it does not imply all mechanisms are materially dependent. This is an effect-size judgment, not a preregistered significance test or architecture authorization.

A: Independent team-level vectors would remove observed cross-team covariance while preserving within-team vectors. Negative plays covariance means independence would tend to overstate the variance of combined plays and understate the variance of team plays differences, holding marginal distributions fixed.

B: Historical game-paired vectors would retain empirical cross-team and cross-mechanism relationships before reconciliation, but finite historical coverage, arbitrary assignment to future teams, and transfer across seasons/contexts need separate validation.

C: A future shared-game mechanism could model dependence selectively and condition it on context. A simple same-sign shared pace shock would not reproduce negative plays covariance; opposing or constrained effects would need investigation. Added complexity requires separate validation.

These data argue against treating independence as an empirically faithful default for every mechanism. They do not distinguish historical pairing from a designed shared-game mechanism sufficiently to select an architecture. No implementation or simulation is authorized.

Provided raw opponent correlations contain game-script effects and use different outcome scales: points -0.058, attempts -0.116, carries -0.570, passing yards +0.177, passing TDs +0.114, rushing yards -0.278, rushing TDs -0.005. They are context only, not residual covariance targets. Attempts/carries combine plays with pass rate; yards combine volume with efficiency, so direct matching is inappropriate.

The previously measured points-residual correlation +0.008 is supplied context, not recomputed here. Near-zero points dependence does not rule out mechanism-level dependence or cancellation among components. No points residual is sampled.

Lexicographic team ordering is bookkeeping only, with no causal or home/away interpretation. Symmetrized sensitivity reports both orientations to remove arbitrary per-game orientation; its 1678 directed observations are still only 839 unique games. No p-values or independent-observation claim is made.

Results describe eligible frozen-bank games only, with 269 games in 2023 and 285 each in 2024 and 2025. No 2026 data, RNG, resampling, sampler calls, or Monte Carlo were used for this audit. Repeated teams and shared fitted centers limit causal and inferential interpretation.

## Safety verification

All protected before/after SHA256 values match; full per-file evidence is in the JSON. Generator and Replay file inventories also match, excluding __pycache__. Sampler source and tests were not modified. Focused sampler tests: 35/35 PASS after schema normalization. RNG was used only by the existing bounded sampler tests, not the coupling audit. Monte Carlo run/authorized: false. Production/FanDuel influence: NONE.
