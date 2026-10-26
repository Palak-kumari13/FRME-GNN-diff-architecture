"""Run mask controls for Bail and Credit if checkpoints are available."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--checkpoint-dir',default='results/repaired_stage1');p.add_argument('--output-dir',default='results/controls');p.add_argument('--retention',type=float,default=.5);p.add_argument('--seeds',type=int,nargs='+',default=list(range(5)));args=p.parse_args()
for ds in ('bail','credit'):
 for seed in args.seeds:
  checkpoints=list(Path(args.checkpoint_dir).rglob(f'{ds}_A2_seed{seed}.pt'))
  if len(checkpoints)!=1:print('SKIP',ds,seed,'expected exactly one A2 checkpoint; found',len(checkpoints));continue
  target=Path(args.output_dir)/f'{ds}_seed{seed}_r{int(args.retention*100)}.csv'
  subprocess.run([sys.executable,'mask_controls.py','--dataset',ds,'--data-dir',args.data_dir,'--checkpoint',str(checkpoints[0]),'--seed',str(seed),'--retention',str(args.retention),'--output',str(target)],check=True)
