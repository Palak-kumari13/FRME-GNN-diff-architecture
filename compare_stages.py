"""Summarize available Stage 1–4 CSVs without inventing missing experiments."""
import argparse, csv, math, statistics
from pathlib import Path

METRICS=['f1','auc','parity','equality','positive_prediction_rate','mask_mean','mask_entropy','mask_lt_01','mask_gt_09','degenerate_mask','random_mask_f1','drift_pred_change_final','drift_mask_l1_final']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',default='results')
    p.add_argument('--output',default='results/all_stages_summary.csv')
    args=p.parse_args()
    rows=[]
    for f in sorted(Path(args.results).glob('stage[1-4]/*/results.csv')):
        with f.open(newline='') as h: data=list(csv.DictReader(h))
        groups={}
        for r in data: groups.setdefault((r['dataset'],r['model']),[]).append(r)
        for (dataset,model), group in groups.items():
            for metric in METRICS:
                vals=[]
                for r in group:
                    try:v=float(r.get(metric,''))
                    except(ValueError,TypeError):continue
                    if math.isfinite(v):vals.append(v)
                if vals:
                    rows.append({'stage':f.parent.parent.name,'dataset':dataset,'model':model,'metric':metric,'n_seeds':len(vals),'mean':statistics.mean(vals),'sd':statistics.stdev(vals) if len(vals)>1 else 0})
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        print('No new stage results found. Run train.py first.')
        return
    with target.open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Wrote {len(rows)} summary rows to {target}')
if __name__=='__main__':main()
