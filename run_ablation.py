"""E8: remove A2 fidelity/budget terms independently. Result directories never overwritten."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--dataset',choices=['bail','credit'],required=True);p.add_argument('--root',default='results/ablations');p.add_argument('--retention',type=float,default=.5);p.add_argument('--seeds',nargs='+',default=['0','1','2','3','4']);p.add_argument('--epochs',default='300');args=p.parse_args()
for name,lam_fid,lam_budget in [('both',1,1),('no_fidelity',0,1),('no_budget',1,0),('neither',0,0)]:
 target=Path(args.root)/args.dataset/name
 cmd=[sys.executable,'train.py','--dataset',args.dataset,'--data-dir',args.data_dir,'--stage','1','--models','A0','A2','--seeds',*args.seeds,'--epochs',args.epochs,'--retention',str(args.retention),'--lambda-fid',str(lam_fid),'--lambda-budget',str(lam_budget),'--output',str(target)]
 print('RUN',name,flush=True);subprocess.run(cmd,check=True)
