# FRME-GNN repaired evaluation / Plan B

Primary metric: **auc**, practical delta threshold: **0.01**, minimum matched seeds: **5**. These are user-adjustable screening rules, **not** statistical significance.

No historical results used. A missing experiment is INCOMPLETE, never a failure. **Do not use test-set results for model selection.**

## Bail
- Graph: **INCOMPLETE_OR_MIXED**; GCN−MLP: `{'n': 5, 'mean_delta': -0.060792, 'median_delta': -0.064762, 'positive': 0, 'differences': [-0.065569, -0.068761, -0.060253, -0.064762, -0.044613]}`; GCN−rewired: `{'n': 5, 'mean_delta': 0.273552, 'median_delta': 0.28242, 'positive': 5, 'differences': [0.28242, 0.278022, 0.203744, 0.299996, 0.303579]}`.
- Mask: `{'0.5': 'SUPPORTED_PRELIMINARY'}`; equal-budget contrasts: `{'0.5': {'vs_random': {'n': 5, 'mean_delta': 0.031456, 'median_delta': 0.032409, 'positive': 5, 'differences': [0.032971, 0.032438, 0.029626, 0.032409, 0.029833]}, 'vs_degree': {'n': 5, 'mean_delta': 0.040438, 'median_delta': 0.040572, 'positive': 5, 'differences': [0.038504, 0.043258, 0.036877, 0.042981, 0.040572]}}}`.
- Fairness: **INCOMPLETE**; A3−A2 parity: `None`; utility: `None`.
- Robustness: **INCOMPLETE**; A4−A3 drift: `None`.
- **Recommended Plan B:** INCOMPLETE/MIXED: do not claim success or total failure. Finish missing checkpoints/controls, then investigate whichever gate fails.

## Credit
- Graph: **SUPPORTED_PRELIMINARY**; GCN−MLP: `{'n': 5, 'mean_delta': 0.060398, 'median_delta': 0.075039, 'positive': 4, 'differences': [0.068763, -0.010002, 0.086057, 0.075039, 0.082135]}`; GCN−rewired: `{'n': 5, 'mean_delta': 0.134198, 'median_delta': 0.141634, 'positive': 5, 'differences': [0.161228, 0.091892, 0.145082, 0.131155, 0.141634]}`.
- Mask: `{'0.5': 'INCOMPLETE_OR_MIXED'}`; equal-budget contrasts: `{'0.5': {'vs_random': {'n': 5, 'mean_delta': -0.003405, 'median_delta': -0.002131, 'positive': 0, 'differences': [-0.001894, -0.005969, -0.005715, -0.002131, -0.001317]}, 'vs_degree': {'n': 5, 'mean_delta': 0.046135, 'median_delta': 0.059461, 'positive': 5, 'differences': [0.064064, 0.02763, 0.059461, 0.008015, 0.071506]}}}`.
- Fairness: **INCOMPLETE**; A3−A2 parity: `None`; utility: `None`.
- Robustness: **INCOMPLETE**; A4−A3 drift: `None`.
- **Recommended Plan B:** INCOMPLETE/MIXED: do not claim success or total failure. Finish missing checkpoints/controls, then investigate whichever gate fails.

## Five-outcome fallback matrix

| Outcome | Interpretation | Next experiment |
|---|---|---|
| GCN ≈ MLP | No demonstrated structural gain | Audit graph/homophily, degree-matched rewiring, new structural dataset |
| GCN > MLP; learned ≈ random | Useful graph, ineffective selection | Independent edge scoring / alternative mask optimization and deletion tests |
| Mask helps; fairness fails | Mask useful, fairness penalty weak | Isolate DP/EO, explore selective geometric correction |
| Fairness helps; robustness fails | Intervention sensitivity persists | Stronger perturbation budgets, adversarial explanation robustness |
| All objectives fail | Design assumptions not supported | Diagnostic/benchmark paper with CSBM and more real datasets |

## Restrictions
- Outcomes require consistent splits and paired seeds; nested train/validation/test separation.
- Mask controls are fixed-model replacements; retrained-random-mask controls and deletion metrics should precede explanation claims.
- Lack of drift can also mean insensitive or degenerate predictors, not robustness.
- Stage 4/distillation needs a separate teacher-versus-student assessment if enabled.
- Screening labels are not confidence intervals or proof of causality.
