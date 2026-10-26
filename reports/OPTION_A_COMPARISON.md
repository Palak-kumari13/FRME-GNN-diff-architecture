# Option A: Three Mask-Learning Strategies

Descriptive paired-seed comparisons; no causal or significance claims. Scores are from untouched test evaluations after validation selection.

| Dataset | Method | Retention | n | AUC (mean±sd) | Learned−random AUC | Learned−degree AUC | Mask-score SD |
|---|---|---:|---:|---:|---:|---:|---:|
| bail | bilevel | 0.3 | 5 | 0.9712 ± 0.0023 | +0.0292 | +0.0438 | 0.00000 |
| bail | bilevel | 0.5 | 5 | 0.9686 ± 0.0024 | +0.0368 | +0.0467 | 0.00000 |
| bail | bilevel | 0.7 | 5 | 0.9356 ± 0.0157 | -0.0002 | +0.0116 | 0.00000 |
| bail | independent | 0.3 | 5 | 0.9721 ± 0.0026 | +0.0287 | +0.0436 | 0.00000 |
| bail | independent | 0.5 | 5 | 0.9708 ± 0.0006 | +0.0350 | +0.0433 | 0.00000 |
| bail | independent | 0.7 | 5 | 0.9382 ± 0.0058 | -0.0002 | +0.0103 | 0.00000 |
| bail | perturbation | 0.3 | 5 | 0.9727 ± 0.0011 | +0.0305 | +0.0456 | 0.00000 |
| bail | perturbation | 0.5 | 5 | 0.9693 ± 0.0028 | +0.0383 | +0.0482 | 0.00000 |
| bail | perturbation | 0.7 | 5 | 0.9348 ± 0.0151 | -0.0000 | +0.0113 | 0.00000 |
| credit | bilevel | 0.3 | 5 | 0.6835 ± 0.0218 | -0.0040 | +0.0495 | 0.00915 |
| credit | bilevel | 0.5 | 5 | 0.7109 ± 0.0087 | -0.0039 | +0.0581 | 0.00019 |
| credit | bilevel | 0.7 | 5 | 0.7092 ± 0.0165 | -0.0037 | +0.0038 | 0.00060 |
| credit | independent | 0.3 | 5 | 0.6933 ± 0.0287 | -0.0049 | +0.0507 | 0.00001 |
| credit | independent | 0.5 | 5 | 0.7142 ± 0.0143 | -0.0042 | +0.0613 | 0.00006 |
| credit | independent | 0.7 | 5 | 0.7221 ± 0.0089 | -0.0040 | +0.0041 | 0.00002 |
| credit | perturbation | 0.3 | 5 | 0.6847 ± 0.0207 | -0.0033 | +0.0470 | 0.00010 |
| credit | perturbation | 0.5 | 5 | 0.7091 ± 0.0085 | -0.0044 | +0.0497 | 0.00000 |
| credit | perturbation | 0.7 | 5 | 0.7106 ± 0.0073 | -0.0042 | +0.0023 | 0.00003 |

Interpretation requires matched-seed uncertainty, several random control draws, and non-degenerate masks. Comparison here evaluates masks on fixed trained models; retrained controls and perturbation validation are additional evidence.
