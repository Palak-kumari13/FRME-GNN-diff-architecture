"""Aggregate paired matched-seed method and equal-budget random/degree controls."""
import csv, math, statistics
from collections import defaultdict
from pathlib import Path

def main():
 out=Path('reports/OPTION_A_COMPARISON.md');out.parent.mkdir(parents=True,exist_ok=True)
 lines=['# Option A: Three Mask-Learning Strategies','','Descriptive paired-seed comparisons; no causal or significance claims. Scores are from untouched test evaluations after validation selection.','','| Dataset | Method | Retention | n | AUC (mean±sd) | Learned−random AUC | Learned−degree AUC | Mask-score SD |','|---|---|---:|---:|---:|---:|---:|---:|']
 for dataset in ['bail','credit']:
  path=Path(f'results/option_a/{dataset}/results.csv')
  if not path.exists():lines.append(f'\n{dataset}: NOT RUN');continue
  groups=defaultdict(list)
  for r in csv.DictReader(path.open()):
   groups[(r['model'],r['retention'])].append(r)
  for (method,rho),rows in sorted(groups.items()):
   def values(key):
    vals=[]
    for r in rows:
     try:
      x=float(r[key]);
      if math.isfinite(x):vals.append(x)
     except (KeyError,ValueError):pass
    return vals
   auc=values('learned_auc');diff=[float(r['learned_auc'])-float(r['random_auc']) for r in rows];deg=[float(r['learned_auc'])-float(r['degree_auc']) for r in rows]
   if not auc:continue
   sd=statistics.stdev(auc) if len(auc)>1 else 0
   lines.append(f'| {dataset} | {method} | {rho} | {len(auc)} | {statistics.mean(auc):.4f} ± {sd:.4f} | {statistics.mean(diff):+.4f} | {statistics.mean(deg):+.4f} | {statistics.mean(values("mask_score_std")):.5f} |')
 lines.extend(['','Interpretation requires matched-seed uncertainty, several random control draws, and non-degenerate masks. Comparison here evaluates masks on fixed trained models; retrained controls and perturbation validation are additional evidence.'])
 out.write_text('\n'.join(lines)+'\n');print(out)
if __name__=='__main__':main()
