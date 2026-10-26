#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${1:-data}"
for d in bail credit; do
  python sanity.py --data "$DATA_DIR/$d.npz"
  python stage0.py --dataset "$d" --data-dir "$DATA_DIR" --output "results/stage0/$d.csv"
  python train.py --stage 1 --dataset "$d" --data-dir "$DATA_DIR" --retention .5 --output "results/repaired_stage1/$d"
  python run_ablation.py --dataset "$d" --data-dir "$DATA_DIR"
  python option_a.py --dataset "$d" --data-dir "$DATA_DIR" --output "results/option_a/$d"
done
python run_controls.py --data-dir "$DATA_DIR" --checkpoint-dir results/repaired_stage1 --retention .5
python compare_option_a.py
python plan_b.py
