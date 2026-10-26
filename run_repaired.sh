#!/usr/bin/env bash
set -euo pipefail
DATA_DIR=${1:-data}
python sanity.py --data "$DATA_DIR/bail.npz"
for ds in bail credit; do
  python stage0.py --dataset "$ds" --data-dir "$DATA_DIR" --output "results/stage0/${ds}.csv"
  python train.py --stage 1 --dataset "$ds" --data-dir "$DATA_DIR" --retention 0.5 --output "results/repaired_stage1/${ds}"
done
python run_controls.py --data-dir "$DATA_DIR" --checkpoint-dir results/repaired_stage1 --retention 0.5
python run_ablation.py --data-dir "$DATA_DIR" --dataset bail
python run_ablation.py --data-dir "$DATA_DIR" --dataset credit
python plan_b.py
