# Paired Sampler Distribution Recovery Audit V1

PASS is a descriptive sampler-recovery assessment with exact integrity and reproducibility checks. No statistical equivalence or significance threshold was applied. All discrepancies are reported below. Frequency and orientation counts are diagnostics only.

Each complete run called the unchanged frozen sampler once for every seed 0..9999: 10,000 pairs and 20,000 team vectors. Two runs were performed (20,000 calls total across the reproducibility check). Both teams used the fixed center {"expected_pass_rate": 0.5, "expected_pass_ypa": 7, "expected_passing_tds": 2, "expected_plays": 100, "expected_points": 24, "expected_rush_ypc": 4, "expected_rushing_tds": 1}. Only selected residuals enter the comparisons; realized primitives were neither analyzed nor interpreted.

Standard deviations use ddof=1; quantiles use linear interpolation; Spearman uses average-tie ranks. Source summaries and correlations were reconstructed and checked against GAME_COUPLING_AUDIT_V1 within 1e-12 (floating-point reference-check tolerance, not a recovery threshold).

| Status field | Value |
| --- | --- |
| AUDIT_STATUS | PASS |
| SAMPLER_DISTRIBUTION_RECOVERY | PASS (DESCRIPTIVE) |
| SEEDS | 0..9999 |
| PAIRED_DRAWS | 10000 |
| TEAM_VECTOR_OBSERVATIONS | 20000 |
| MARGINAL_RECOVERY | PASS (DESCRIPTIVE) |
| CROSS_TEAM_DEPENDENCE_RECOVERY | PASS (DESCRIPTIVE) |
| WITHIN_TEAM_DEPENDENCE_RECOVERY | PASS (DESCRIPTIVE) |
| GAME_LEVEL_TRANSFORM_RECOVERY | PASS (DESCRIPTIVE) |
| MAX_CROSS_TEAM_PEARSON_DIFFERENCE | 0.019254010 |
| MAX_CROSS_TEAM_SPEARMAN_DIFFERENCE | 0.018660856 |
| MAX_WITHIN_TEAM_CORRELATION_DIFFERENCE | 0.017823237 |
| MEAN_WITHIN_TEAM_OFFDIAGONAL_DIFFERENCE | 0.004845900 |
| HISTORICAL_GAME_SELECTION_COUNT_MIN | 4 |
| HISTORICAL_GAME_SELECTION_COUNT_MAX | 25 |
| HISTORICAL_GAME_SELECTION_COUNT_MEAN | 11.918951132 |
| HISTORICAL_GAME_SELECTION_COUNT_STD | 3.488557613 |
| ORIENTATION_0_COUNT | 5093 |
| ORIENTATION_1_COUNT | 4907 |
| REPRODUCIBLE | True |
| RESULT_SHA256_RUN_1 | 8b2c8832bab6684e73a3aaeaf962d46f8fb39a7632ceb70178fb18d30bba2355 |
| RESULT_SHA256_RUN_2 | 8b2c8832bab6684e73a3aaeaf962d46f8fb39a7632ceb70178fb18d30bba2355 |
| GLOBAL_RNG_UNTOUCHED | True |
| PAIRED_SAMPLER_UNCHANGED | True |
| SEEDED_SAMPLER_UNCHANGED | True |
| RESIDUAL_BANK_UNCHANGED | True |
| CALIBRATION_SPEC_UNCHANGED | True |
| FROZEN_GENERATOR_UNCHANGED | True |
| FROZEN_REPLAY_UNCHANGED | True |
| HISTORICAL_TEMPORAL_PROVENANCE | UNVERIFIED |
| MONTE_CARLO_RUN | False |
| MONTE_CARLO_AUTHORIZED | False |
| PRODUCTION_INFLUENCE | NONE |
| FANDUEL_SOLVER_INFLUENCE | NONE |

## Marginal recovery

### resid_plays

| Population | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source | -0.138855781 | 9.001379560 | -15.000000000 | -6.000000000 | 0.000000000 | 6.000000000 | 14.000000000 |
| sampled | -0.065050000 | 8.967437719 | -15.000000000 | -6.000000000 | 0.000000000 | 6.000000000 | 14.000000000 |

| Absolute mean difference | Relative std difference | Absolute relative std difference |
| --- | --- | --- |
| 0.073805781 | -0.003770738 | 0.003770738 |

| Quantile | Sample minus source | Absolute difference |
| --- | --- | --- |
| p05 | 0.000000000 | 0.000000000 |
| p25 | 0.000000000 | 0.000000000 |
| p50 | 0.000000000 | 0.000000000 |
| p75 | 0.000000000 | 0.000000000 |
| p95 | 0.000000000 | 0.000000000 |

### resid_pass_rate

| Population | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source | -0.000474087 | 0.109134192 | -0.169806441 | -0.079257097 | 0.000407962 | 0.080417847 | 0.176486706 |
| sampled | -0.000640424 | 0.110255380 | -0.172930484 | -0.080298786 | 0.000000000 | 0.080769231 | 0.177517564 |

| Absolute mean difference | Relative std difference | Absolute relative std difference |
| --- | --- | --- |
| 0.000166338 | 0.010273483 | 0.010273483 |

| Quantile | Sample minus source | Absolute difference |
| --- | --- | --- |
| p05 | -0.003124043 | 0.003124043 |
| p25 | -0.001041689 | 0.001041689 |
| p50 | -0.000407962 | 0.000407962 |
| p75 | 0.000351384 | 0.000351384 |
| p95 | 0.001030858 | 0.001030858 |

### resid_pass_ypa

| Population | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source | 0.010404587 | 2.093809185 | -3.153102578 | -1.449445515 | -0.087474232 | 1.251594487 | 3.657995602 |
| sampled | -0.015384055 | 2.087958228 | -3.211209133 | -1.467987528 | -0.134323432 | 1.231521739 | 3.583213307 |

| Absolute mean difference | Relative std difference | Absolute relative std difference |
| --- | --- | --- |
| 0.025788642 | -0.002794408 | 0.002794408 |

| Quantile | Sample minus source | Absolute difference |
| --- | --- | --- |
| p05 | -0.058106555 | 0.058106555 |
| p25 | -0.018542013 | 0.018542013 |
| p50 | -0.046849201 | 0.046849201 |
| p75 | -0.020072747 | 0.020072747 |
| p95 | -0.074782294 | 0.074782294 |

### resid_rush_ypc

| Population | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source | -0.145090278 | 1.526146654 | -2.572626905 | -1.131101190 | -0.187353103 | 0.798989520 | 2.363015640 |
| sampled | -0.139774044 | 1.541002683 | -2.581955063 | -1.141189675 | -0.166291037 | 0.813299233 | 2.360784314 |

| Absolute mean difference | Relative std difference | Absolute relative std difference |
| --- | --- | --- |
| 0.005316234 | 0.009734339 | 0.009734339 |

| Quantile | Sample minus source | Absolute difference |
| --- | --- | --- |
| p05 | -0.009328158 | 0.009328158 |
| p25 | -0.010088484 | 0.010088484 |
| p50 | 0.021062066 | 0.021062066 |
| p75 | 0.014309713 | 0.014309713 |
| p95 | -0.002231327 | 0.002231327 |

### resid_passing_tds

| Population | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source | -0.000595948 | 1.229482636 | -2.000000000 | -1.000000000 | 0.000000000 | 1.000000000 | 2.000000000 |
| sampled | -0.002950000 | 1.228216888 | -2.000000000 | -1.000000000 | 0.000000000 | 1.000000000 | 2.000000000 |

| Absolute mean difference | Relative std difference | Absolute relative std difference |
| --- | --- | --- |
| 0.002354052 | -0.001029496 | 0.001029496 |

| Quantile | Sample minus source | Absolute difference |
| --- | --- | --- |
| p05 | 0.000000000 | 0.000000000 |
| p25 | 0.000000000 | 0.000000000 |
| p50 | 0.000000000 | 0.000000000 |
| p75 | 0.000000000 | 0.000000000 |
| p95 | 0.000000000 | 0.000000000 |

### resid_rushing_tds

| Population | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source | -0.029201430 | 1.044498644 | -1.000000000 | -1.000000000 | 0.000000000 | 1.000000000 | 2.000000000 |
| sampled | -0.027600000 | 1.045654297 | -1.000000000 | -1.000000000 | 0.000000000 | 1.000000000 | 2.000000000 |

| Absolute mean difference | Relative std difference | Absolute relative std difference |
| --- | --- | --- |
| 0.001601430 | 0.001106419 | 0.001106419 |

| Quantile | Sample minus source | Absolute difference |
| --- | --- | --- |
| p05 | 0.000000000 | 0.000000000 |
| p25 | 0.000000000 | 0.000000000 |
| p50 | 0.000000000 | 0.000000000 |
| p75 | 0.000000000 | 0.000000000 |
| p95 | 0.000000000 | 0.000000000 |

## Cross-team dependence

The requested targets use lexicographic historical A/B ordering. Random orientation targets the symmetrized population instead; both comparisons are shown. This small reference difference is structural, not sampler tuning. Plays, pass rate, and passing TDs are listed explicitly with all mechanisms.

### pearson

| Residual | Canonical target | Sampled | Absolute difference | Symmetrized target | Absolute symmetrized difference |
| --- | --- | --- | --- | --- | --- |
| resid_plays | -0.347847488 | -0.367101498 | 0.019254010 | -0.350800914 | 0.016300585 |
| resid_pass_rate | -0.211846874 | -0.208921389 | 0.002925485 | -0.212281037 | 0.003359648 |
| resid_pass_ypa | -0.014473762 | -0.020383666 | 0.005909904 | -0.014607505 | 0.005776160 |
| resid_rush_ypc | -0.043012208 | -0.047483207 | 0.004470999 | -0.043857042 | 0.003626164 |
| resid_passing_tds | 0.105119769 | 0.115701871 | 0.010582101 | 0.104141801 | 0.011560069 |
| resid_rushing_tds | 0.004834939 | 0.002610691 | 0.002224248 | 0.004683687 | 0.002072996 |

### spearman

| Residual | Canonical target | Sampled | Absolute difference | Symmetrized target | Absolute symmetrized difference |
| --- | --- | --- | --- | --- | --- |
| resid_plays | -0.361811822 | -0.380472678 | 0.018660856 | -0.362526259 | 0.017946419 |
| resid_pass_rate | -0.229586142 | -0.226771032 | 0.002815110 | -0.230229204 | 0.003458171 |
| resid_pass_ypa | 0.005332217 | -0.003536871 | 0.008869088 | 0.005826723 | 0.009363594 |
| resid_rush_ypc | -0.028870174 | -0.032400742 | 0.003530568 | -0.028393497 | 0.004007245 |
| resid_passing_tds | 0.101204648 | 0.107916009 | 0.006711361 | 0.100096848 | 0.007819161 |
| resid_rushing_tds | -0.002479826 | -0.001413985 | 0.001065840 | -0.002815229 | 0.001401244 |

## Within-team dependence

source_matrix

| Residual | resid_plays | resid_pass_rate | resid_pass_ypa | resid_rush_ypc | resid_passing_tds | resid_rushing_tds |
| --- | --- | --- | --- | --- | --- | --- |
| resid_plays | 1.000000000 | -0.055629271 | -0.062131792 | -0.070750253 | 0.111202855 | 0.110305845 |
| resid_pass_rate | -0.055629271 | 1.000000000 | -0.304949981 | 0.046696109 | 0.018710460 | -0.285881088 |
| resid_pass_ypa | -0.062131792 | -0.304949981 | 1.000000000 | -0.089170915 | 0.422135164 | 0.212785309 |
| resid_rush_ypc | -0.070750253 | 0.046696109 | -0.089170915 | 1.000000000 | 0.002756107 | 0.245341164 |
| resid_passing_tds | 0.111202855 | 0.018710460 | 0.422135164 | 0.002756107 | 1.000000000 | -0.135601014 |
| resid_rushing_tds | 0.110305845 | -0.285881088 | 0.212785309 | 0.245341164 | -0.135601014 | 1.000000000 |

sampled_matrix

| Residual | resid_plays | resid_pass_rate | resid_pass_ypa | resid_rush_ypc | resid_passing_tds | resid_rushing_tds |
| --- | --- | --- | --- | --- | --- | --- |
| resid_plays | 1.000000000 | -0.065164482 | -0.058454722 | -0.072370234 | 0.103996704 | 0.112560033 |
| resid_pass_rate | -0.065164482 | 1.000000000 | -0.304227775 | 0.051468005 | 0.016470614 | -0.283287849 |
| resid_pass_ypa | -0.058454722 | -0.304227775 | 1.000000000 | -0.080311254 | 0.424903010 | 0.206383622 |
| resid_rush_ypc | -0.072370234 | 0.051468005 | -0.080311254 | 1.000000000 | 0.002261724 | 0.247063069 |
| resid_passing_tds | 0.103996704 | 0.016470614 | 0.424903010 | 0.002261724 | 1.000000000 | -0.153424251 |
| resid_rushing_tds | 0.112560033 | -0.283287849 | 0.206383622 | 0.247063069 | -0.153424251 | 1.000000000 |

absolute_difference_matrix

| Residual | resid_plays | resid_pass_rate | resid_pass_ypa | resid_rush_ypc | resid_passing_tds | resid_rushing_tds |
| --- | --- | --- | --- | --- | --- | --- |
| resid_plays | 0.000000000 | 0.009535212 | 0.003677070 | 0.001619981 | 0.007206151 | 0.002254188 |
| resid_pass_rate | 0.009535212 | 0.000000000 | 0.000722206 | 0.004771896 | 0.002239846 | 0.002593240 |
| resid_pass_ypa | 0.003677070 | 0.000722206 | 0.000000000 | 0.008859661 | 0.002767845 | 0.006401687 |
| resid_rush_ypc | 0.001619981 | 0.004771896 | 0.008859661 | 0.000000000 | 0.000494383 | 0.001721905 |
| resid_passing_tds | 0.007206151 | 0.002239846 | 0.002767845 | 0.000494383 | 0.000000000 | 0.017823237 |
| resid_rushing_tds | 0.002254188 | 0.002593240 | 0.006401687 | 0.001721905 | 0.017823237 | 0.000000000 |

## Game-level residual transforms

### sum_resid_plays

| Population | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source | -0.277711561 | 10.259886274 | -17.000000000 | -7.000000000 | -1.000000000 | 6.000000000 | 17.100000000 |
| sampled | -0.130100000 | 10.089120528 | -17.000000000 | -7.000000000 | 0.000000000 | 6.000000000 | 17.000000000 |

| Absolute mean difference | Relative std difference |
| --- | --- |
| 0.147611561 | -0.016644019 |

| Quantile | Sample minus source | Absolute difference |
| --- | --- | --- |
| p05 | 0.000000000 | 0.000000000 |
| p25 | 0.000000000 | 0.000000000 |
| p50 | 1.000000000 | 1.000000000 |
| p75 | 0.000000000 | 0.000000000 |
| p95 | -0.100000000 | 0.100000000 |

### absolute_difference_resid_plays

| Population | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source | 11.974970203 | 8.686565106 | 1.000000000 | 5.000000000 | 11.000000000 | 17.000000000 | 29.000000000 |
| sampled | 12.017100000 | 8.686791906 | 1.000000000 | 5.000000000 | 11.000000000 | 17.000000000 | 29.000000000 |

| Absolute mean difference | Relative std difference |
| --- | --- |
| 0.042129797 | 0.000026109 |

| Quantile | Sample minus source | Absolute difference |
| --- | --- | --- |
| p05 | 0.000000000 | 0.000000000 |
| p25 | 0.000000000 | 0.000000000 |
| p50 | 0.000000000 | 0.000000000 |
| p75 | 0.000000000 | 0.000000000 |
| p95 | 0.000000000 | 0.000000000 |

### sum_resid_passing_tds

| Population | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source | -0.001191895 | 1.827592417 | -3.000000000 | -1.000000000 | 0.000000000 | 1.000000000 | 3.000000000 |
| sampled | -0.005900000 | 1.834721181 | -3.000000000 | -1.000000000 | 0.000000000 | 1.000000000 | 3.000000000 |

| Absolute mean difference | Relative std difference |
| --- | --- |
| 0.004708105 | 0.003900631 |

| Quantile | Sample minus source | Absolute difference |
| --- | --- | --- |
| p05 | 0.000000000 | 0.000000000 |
| p25 | 0.000000000 | 0.000000000 |
| p50 | 0.000000000 | 0.000000000 |
| p75 | 0.000000000 | 0.000000000 |
| p95 | 0.000000000 | 0.000000000 |

### sum_resid_rushing_tds

| Population | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source | -0.058402861 | 1.481040989 | -2.000000000 | -1.000000000 | 0.000000000 | 1.000000000 | 2.000000000 |
| sampled | -0.055200000 | 1.480733675 | -2.000000000 | -1.000000000 | 0.000000000 | 1.000000000 | 2.000000000 |

| Absolute mean difference | Relative std difference |
| --- | --- |
| 0.003202861 | -0.000207498 |

| Quantile | Sample minus source | Absolute difference |
| --- | --- | --- |
| p05 | 0.000000000 | 0.000000000 |
| p25 | 0.000000000 | 0.000000000 |
| p50 | 0.000000000 | 0.000000000 |
| p75 | 0.000000000 | 0.000000000 |
| p95 | 0.000000000 | 0.000000000 |

### combined_game_offensive_td_residual

| Population | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source | -0.059594756 | 2.240072307 | -4.000000000 | -2.000000000 | 0.000000000 | 1.000000000 | 4.000000000 |
| sampled | -0.061100000 | 2.215233061 | -4.000000000 | -2.000000000 | 0.000000000 | 1.000000000 | 4.000000000 |

| Absolute mean difference | Relative std difference |
| --- | --- |
| 0.001505244 | -0.011088591 |

| Quantile | Sample minus source | Absolute difference |
| --- | --- | --- |
| p05 | 0.000000000 | 0.000000000 |
| p25 | 0.000000000 | 0.000000000 |
| p50 | 0.000000000 | 0.000000000 |
| p75 | 0.000000000 | 0.000000000 |
| p95 | 0.000000000 | 0.000000000 |

## Frequency diagnostics

All 839 game counts, including any zero counts, are recorded in the JSON primary_results.frequency.count_per_historical_game. Summary and orientation counts appear in the status table above. No uniformity threshold was used.

## Reproducibility and limits

The SHA256 values cover canonical sorted-key, indented UTF-8 JSON of primary_results with a trailing newline. This excludes the enclosing reproducibility metadata to avoid self-referential hashing. Both full runs were independently computed. Python and NumPy global RNG states were unchanged. Protected before/after hashes and all runtime versions are in the JSON.

This audit describes recovery of a frozen empirical residual population. It establishes no forecast accuracy, future-game calibration, production readiness, or FanDuel usefulness. Historical temporal provenance remains UNVERIFIED. MONTE_CARLO_RUN=false and MONTE_CARLO_AUTHORIZED=false denote no forecast/production Monte Carlo; the only repeated draws were this explicitly authorized fixed validation grid.

## Files created or modified

- `future_primitive_distribution_shadow_v1/audit_paired_sampler_distribution_recovery_v1.py`
- `future_primitive_distribution_shadow_v1/PAIRED_SAMPLER_DISTRIBUTION_RECOVERY_V1.json`
- `future_primitive_distribution_shadow_v1/PAIRED_SAMPLER_DISTRIBUTION_RECOVERY_V1.md`
- `future_primitive_distribution_shadow_v1/PAIRED_SAMPLER_DISTRIBUTION_RECOVERY_V1_HASHES_BEFORE.json`

Reproduce only within the same authorized audit scope: `python3 -B -m future_primitive_distribution_shadow_v1.audit_paired_sampler_distribution_recovery_v1`
