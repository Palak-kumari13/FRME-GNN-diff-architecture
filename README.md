# FRME-GNN — Clean Mask Repair + Plan B Experiments

**New, independent repository. No previous or fabricated results included.** Prior FRME-GNN and AngleFair-GNN remain unchanged. This is corrected *candidate* code, not a claim of real-data improvement.

## Install and data

`pip install -r requirements.txt`

Add `data/bail.npz` and `data/credit.npz`, each with `x [N,F]`, `edge_index [2,E]`, `y [N]`, `sensitive [N]`, and fixed, disjoint `train_idx`, `val_idx`, `test_idx`. Do not compare against earlier scores if you change splits or graph construction. No datasets are bundled.

## What was repaired

- Non-identical initial edge scores (randomized final mask layer; verify with `sanity.py`).
- Hard exact-budget top-k edge selection with straight-through gradient surrogate; fixed edge count at 30/50/70% but faithfulness is NOT guaranteed.
- Mask forwarded into weighted GCN normalization and evaluated with zero/full masks.
- Repaired Stage 1 A0/A1/A2 and Stage 2 A3; Stage 3 A4 and Stage 4 A5 retained as optional code.
- Frozen A0 teacher fidelity on train nodes, validation-based threshold, seeds 0–4.
- Stage 0 E0 always positive, E1 MLP, E2 real-edge GCN, E3 degree-preserving rewired GCN, E4 full-edge GCN parity check (E4 is NOT an independent model).
- E5 random top-k, E6 degree top-k and E7 learned top-k on identical trained checkpoint, retention and threshold.
- E8 A2 loss ablations: both, no fidelity, no budget, neither. **In hard-topk mode the mean-mask budget term is inactive by design, so no-budget should match both absent other stochastic effects; this is a sanity check, not independent discovery.**

## Run in order

```bash
python sanity.py --data data/bail.npz
python stage0.py --dataset bail --data-dir data --output results/stage0/bail.csv
python stage0.py --dataset credit --data-dir data --output results/stage0/credit.csv
python train.py --stage 1 --dataset bail --data-dir data --retention .5 --output results/repaired_stage1/bail
python train.py --stage 1 --dataset credit --data-dir data --retention .5 --output results/repaired_stage1/credit
python run_controls.py --data-dir data --checkpoint-dir results/repaired_stage1 --retention .5
python run_ablation.py --dataset bail --data-dir data
python run_ablation.py --dataset credit --data-dir data
python plan_b.py
```

`bash run_repaired.sh data` runs the sequence. Outputs refuse to overwrite; use a clean results folder for new attempts. For additional budgets, train in separate output folders and pass the matching retention to controls.

**After Stage 1 passes**:

```bash
python train.py --stage 2 --dataset bail --data-dir data --output results/repaired_stage2/bail
python train.py --stage 2 --dataset credit --data-dir data --output results/repaired_stage2/credit
python train.py --stage 3 --dataset bail --data-dir data --output results/repaired_stage3/bail
python train.py --stage 3 --dataset credit --data-dir data --output results/repaired_stage3/credit
python plan_b.py
```

`train.py` retrains all earlier models within the selected stage; outputs must stay in unique folders. Stage 3/4 should be interpreted only after preceding checks pass. `plan_b.py` auto-detects available repaired result CSVs.

## Plan B is an executable outcome analysis, not a promise

`python plan_b.py --metric auc --margin 0.01 --min-seeds 5` writes `reports/PLAN_B.md` and `.json` for **both datasets** and marks unavailable comparisons `INCOMPLETE`.

| Observed pattern | Interpretation | Action |
|---|---|---|
| GCN ≈ MLP | Graph not demonstrably beneficial | Audit topology, homophily; test CSBM/structural datasets |
| GCN > MLP, learned ≈ random | Graph helps but masks fail | Redesign scoring and faithfulness loss |
| Mask works, fairness fails | Group-regularizer ineffective | Compare prediction losses, then AngleFair selective geometry |
| Fairness works, robustness fails | Sensitivity remains | Better perturbation audit and robust explanation learning |
| All fail | Experimental assumptions inadequate | Benchmark/diagnostic research, no forced success claim |

Decisions are *screening* only: five seeds, thresholded mean differences and majority sign are insufficient for scientific significance. Bootstrap confidence intervals, multiple random-mask draws, uncertainty and validation-only hyperparameter selection are still needed. A top-k mask can be nontrivial by construction yet uninformative. Use balanced accuracy, MCC, AUPRC and class prevalence alongside F1 for Credit.

## Code map

`train.py` repaired A0–A5; `stage0.py` E0–E4; `mask_controls.py` E5–E7; `run_ablation.py` E8; `sanity.py` mask correctness; `plan_b.py` branch logic; `run_repaired.sh` sequence; `prepare_splits.py` optional split creation. No historical metrics, checkpoints or synthetic scores are packaged.


## Option A — final three-way mask-method comparison (ADDED)

One additional, optional full-grid command now compares THREE distinct mechanisms on Bail and Credit under equal retained-edge fractions (0.3, 0.5, 0.7), five paired seeds and common controls:

1. `independent`: separate edge-scoring feature encoder and GCN propagation, trained jointly with exact top-k straight-through selection.
2. `bilevel`: **first-order alternating bilevel approximation**; classifier trained on an inner subset of TRAIN nodes; mask scorer optimized on a disjoint outer subset of TRAIN nodes. Does not use true val/test labels for outer training; it is **not exact differentiable hypergradient bilevel optimization**.
3. `perturbation`: task-supervised retention and complementary edge-deletion margin, with feature-noise mask-score consistency. The complement objective can be gamed, so fidelity and control tests are mandatory; this is **not a complete perturbation explainer benchmark**.

Each learned mask is compared on the SAME trained checkpoint against random and degree top-k, full graph, and complement, with validation-selected threshold. AUROC/AUPRC, F1, MCC, balanced accuracy, parity/equality, score standard deviation, sufficiency and complement KL are written to fresh CSVs. The same budgets and seeds permit paired contrasts, but current random control uses one draw per seed; strengthen with multiple draws for publication.

```bash
python option_a.py --dataset bail --data-dir data --retentions .3 .5 .7 --seeds 0 1 2 3 4
python option_a.py --dataset credit --data-dir data --retentions .3 .5 .7 --seeds 0 1 2 3 4
python compare_option_a.py
```

For **one-run end-to-end** execution (Stage 0, repaired Stage 1, E8 ablations, Option A, existing controls and Plan B):

```bash
bash run_everything.sh data
```

This is a large experiment grid: 2 datasets × 3 methods × 3 budgets × 5 seeds = **90 Option A trainings** in addition to Stage 0 and repaired Stage 1. GPU strongly recommended. `run_everything.sh` will stop if output paths already exist rather than silently reuse results. Stage 2–4 training is kept optional until structural and mask controls pass. **No historical CSV results are included, and no real-data scores are claimed.**

`plan_b.py` summarizes the baseline repaired-model controls; `compare_option_a.py` summarizes the three alternative methods separately. If needed, use the Option A results to decide whether to replace the repaired mask trainer before Stage 2 fairness.
