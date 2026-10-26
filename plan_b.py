"""Pre-specified, conservative Plan B triage from ACTUAL new CSVs.
Missing experiments produce INCOMPLETE not failure. Differences are descriptive paired-seed effects;
5 seeds alone do not support generalization/significance.
"""
import argparse,csv,json,math,statistics
from collections import defaultdict
from pathlib import Path

def read_rows(paths):
 out=[]
 for p in paths:
  for r in csv.DictReader(p.open(newline='')):
   r['_file']=str(p);out.append(r)
 return out

def paired(rows,a,b,metric='auc'):
 by=defaultdict(dict)
 for r in rows:
  try:
   seed=int(r['seed']); val=float(r[metric]); model=r.get('experiment',r.get('model'))
   if math.isfinite(val):by[seed][model]=val
  except (KeyError,ValueError,TypeError):pass
 ds=[v[b]-v[a] for v in by.values() if a in v and b in v]
 if not ds:return None
 return {'n':len(ds),'mean_delta':round(statistics.mean(ds),6),'median_delta':round(statistics.median(ds),6),'positive':sum(d>0 for d in ds),'differences':[round(v,6) for v in ds]}

def summarize_data(root,dataset,metric,margin,min_seeds):
 stage=read_rows([root/'stage0'/f'{dataset}.csv']) if (root/'stage0'/f'{dataset}.csv').exists() else []
 controls=read_rows(sorted((root/'controls').glob(f'{dataset}_seed*_r*.csv')))
 # Prevent double-counting 30%,50%,70%: group by budget.
 bybudget=defaultdict(list)
 for r in controls:
  bybudget[r.get('retention','?')].append(r)
 gr=paired(stage,'E1_mlp','E2_gcn',metric)
 rew=paired(stage,'E3_rewired_gcn','E2_gcn',metric)
 mask={key:{'vs_random':paired(rs,'E5_random_topk','E7_learned_topk',metric),'vs_degree':paired(rs,'E6_degree_topk','E7_learned_topk',metric)} for key,rs in bybudget.items()}
 def classify(p):
  if not p or p['n']<min_seeds:return 'INCOMPLETE'
  if p['mean_delta']>margin and p['positive']>=min_seeds-1:return 'POSITIVE_SCREEN'
  if p['mean_delta'] < -margin and (p['n']-p['positive'])>=min_seeds-1:return 'NEGATIVE_SCREEN'
  return 'INCONCLUSIVE'
 graph_ml=classify(gr);graph_rw=classify(rew)
 graph='SUPPORTED_PRELIMINARY' if graph_ml=='POSITIVE_SCREEN' and graph_rw=='POSITIVE_SCREEN' else ('UNSUPPORTED_PRELIMINARY' if graph_ml in ('NEGATIVE_SCREEN','INCONCLUSIVE') and graph_rw in ('NEGATIVE_SCREEN','INCONCLUSIVE') else 'INCOMPLETE_OR_MIXED')
 masks={k:('SUPPORTED_PRELIMINARY' if classify(v['vs_random'])=='POSITIVE_SCREEN' and classify(v['vs_degree'])=='POSITIVE_SCREEN' else 'UNSUPPORTED_PRELIMINARY' if classify(v['vs_random']) in ('NEGATIVE_SCREEN','INCONCLUSIVE') and classify(v['vs_degree']) in ('NEGATIVE_SCREEN','INCONCLUSIVE') else 'INCOMPLETE_OR_MIXED') for k,v in mask.items()}
 maskok=any(v=='SUPPORTED_PRELIMINARY' for v in masks.values())
 stage2path=root/'repaired_stage2'/dataset/'results.csv'
 stage2=read_rows([stage2path]) if stage2path.exists() else []
 # fairness improvement defined as lower DP gap; utility AUC change as separate paired control
 fair=paired(stage2,'A2','A3','parity');utility=paired(stage2,'A2','A3',metric)
 fairness='INCOMPLETE'
 if fair and utility and fair['n']>=min_seeds and utility['n']>=min_seeds:
  fairness='SUPPORTED_PRELIMINARY' if fair['mean_delta'] < -margin and utility['mean_delta']>=-margin else 'UNSUPPORTED_OR_MIXED'
 stage3path=root/'repaired_stage3'/dataset/'results.csv'
 stage3=read_rows([stage3path]) if stage3path.exists() else []
 # decrease robust drift (less predicted difference) with clean utility preserved
 rob=paired(stage3,'A3','A4','drift_pred_change_final');ru=paired(stage3,'A3','A4',metric)
 robustness='INCOMPLETE'
 if rob and ru and rob['n']>=min_seeds and ru['n']>=min_seeds:
  robustness='SUPPORTED_PRELIMINARY' if rob['mean_delta'] < -margin and ru['mean_delta']>=-margin else 'UNSUPPORTED_OR_MIXED'
 if graph=='UNSUPPORTED_PRELIMINARY': branch='GCN ≈ MLP / rewired: inspect graph construction, label homophily, test structurally informative datasets and CSBM.'
 elif graph=='SUPPORTED_PRELIMINARY' and masks and not maskok and all(v=='UNSUPPORTED_PRELIMINARY' for v in masks.values()):branch='Graph helps, mask ≈ random/degree: redesign edge scorer/objective; examine fidelity, comprehensiveness and alternative explainers.'
 elif graph=='SUPPORTED_PRELIMINARY' and maskok and fairness=='UNSUPPORTED_OR_MIXED':branch='Mask helps but fairness fails: isolate parity/equality objectives and test selective geometry only after mask integrity gates.'
 elif graph=='SUPPORTED_PRELIMINARY' and maskok and fairness=='SUPPORTED_PRELIMINARY' and robustness=='UNSUPPORTED_OR_MIXED':branch='Fairness works, robustness fails: run stronger standardized perturbations and robust explanation objectives.'
 elif graph=='SUPPORTED_PRELIMINARY' and maskok and fairness=='SUPPORTED_PRELIMINARY' and robustness=='SUPPORTED_PRELIMINARY':branch='Candidate success: confirm with repeated draws, confidence intervals, new split and explanation ground truth.'
 else:branch='INCOMPLETE/MIXED: do not claim success or total failure. Finish missing checkpoints/controls, then investigate whichever gate fails.'
 return {'dataset':dataset,'metric':metric,'graph_vs_mlp':gr,'graph_vs_rewired':rew,'graph_status':graph,'mask_comparisons':mask,'mask_status_by_budget':masks,'fairness_delta_A3_minus_A2':fair,'utility_delta_A3_minus_A2':utility,'fairness_status':fairness,'robustness_drift_delta_A4_minus_A3':rob,'robustness_status':robustness,'plan_b':branch}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='results');ap.add_argument('--out',default='reports/PLAN_B.md');ap.add_argument('--metric',choices=['auc','balanced_accuracy','mcc','auprc','f1'],default='auc');ap.add_argument('--margin',type=float,default=.01,help='PRE-SPECIFY absolute practical-effect threshold, not statistical significance');ap.add_argument('--min-seeds',type=int,default=5);args=ap.parse_args()
 root=Path(args.root);reports=[summarize_data(root,d,args.metric,args.margin,args.min_seeds) for d in ['bail','credit']]
 out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
 lines=['# FRME-GNN repaired evaluation / Plan B','',f'Primary metric: **{args.metric}**, practical delta threshold: **{args.margin}**, minimum matched seeds: **{args.min_seeds}**. These are user-adjustable screening rules, **not** statistical significance.','','No historical results used. A missing experiment is INCOMPLETE, never a failure. **Do not use test-set results for model selection.**','']
 for r in reports:
  lines.extend([f"## {r['dataset'].title()}",f"- Graph: **{r['graph_status']}**; GCN−MLP: `{r['graph_vs_mlp']}`; GCN−rewired: `{r['graph_vs_rewired']}`.",f"- Mask: `{r['mask_status_by_budget']}`; equal-budget contrasts: `{r['mask_comparisons']}`.",f"- Fairness: **{r['fairness_status']}**; A3−A2 parity: `{r['fairness_delta_A3_minus_A2']}`; utility: `{r['utility_delta_A3_minus_A2']}`.",f"- Robustness: **{r['robustness_status']}**; A4−A3 drift: `{r['robustness_drift_delta_A4_minus_A3']}`.",f"- **Recommended Plan B:** {r['plan_b']}",''])
 lines += ['## Five-outcome fallback matrix','','| Outcome | Interpretation | Next experiment |','|---|---|---|','| GCN ≈ MLP | No demonstrated structural gain | Audit graph/homophily, degree-matched rewiring, new structural dataset |','| GCN > MLP; learned ≈ random | Useful graph, ineffective selection | Independent edge scoring / alternative mask optimization and deletion tests |','| Mask helps; fairness fails | Mask useful, fairness penalty weak | Isolate DP/EO, explore selective geometric correction |','| Fairness helps; robustness fails | Intervention sensitivity persists | Stronger perturbation budgets, adversarial explanation robustness |','| All objectives fail | Design assumptions not supported | Diagnostic/benchmark paper with CSBM and more real datasets |','','## Restrictions','- Outcomes require consistent splits and paired seeds; nested train/validation/test separation.','- Mask controls are fixed-model replacements; retrained-random-mask controls and deletion metrics should precede explanation claims.','- Lack of drift can also mean insensitive or degenerate predictors, not robustness.','- Stage 4/distillation needs a separate teacher-versus-student assessment if enabled.','- Screening labels are not confidence intervals or proof of causality.']
 out.write_text('\n'.join(lines)+'\n');(out.with_suffix('.json')).write_text(json.dumps(reports,indent=2));print('Saved',out)
if __name__=='__main__':main()
