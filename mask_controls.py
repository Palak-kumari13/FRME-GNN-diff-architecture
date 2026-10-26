"""E5/E6/E7 held-out mask comparisons from a repaired Stage 1 checkpoint.
E5 random exact-top-k, E6 degree-top-k, E7 learned exact-top-k.
Only evaluates trained checkpoints; it does NOT train on test outcomes.
"""
import argparse,csv,hashlib
from pathlib import Path
import numpy as np
import torch
from train import MaskGCN,load_npz,undirected_edges
from stage0 import evaluate

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--dataset',choices=['bail','credit','german'],required=True)
 p.add_argument('--data-dir',default='data');p.add_argument('--checkpoint',required=True)
 p.add_argument('--seed',type=int,default=0);p.add_argument('--hidden',type=int,default=64)
 p.add_argument('--dropout',type=float,default=.3);p.add_argument('--retention',type=float,default=.5)
 p.add_argument('--output',required=True);p.add_argument('--threshold',type=float,default=None)
 args=p.parse_args();out=Path(args.output)
 if out.exists():raise FileExistsError(out)
 if not (0<args.retention<1):raise ValueError('retention must be between 0 and 1')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 path=Path(args.data_dir)/f'{args.dataset}_seed{args.seed}.npz'
 if not path.exists():path=Path(args.data_dir)/f'{args.dataset}.npz'
 x,eraw,y,s,idx=load_npz(path,device);edges=undirected_edges(eraw,len(x))
 model=MaskGCN(x.size(1),args.hidden,args.dropout,masked=True).to(device)
 model.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=True))
 model.retention=args.retention;model.eval()
 with torch.no_grad():
  score=model.get_mask(x,edges); k=max(1,min(score.numel(),round(args.retention*score.numel())))
  learned=torch.zeros_like(score).scatter_(0,torch.topk(score,k).indices,1.)
  degree=torch.zeros(len(x),device=device).index_add_(0,edges[0],torch.ones_like(score)).index_add_(0,edges[1],torch.ones_like(score))
  degree_score=degree[edges[0]]+degree[edges[1]]
  degree_mask=torch.zeros_like(score).scatter_(0,torch.topk(degree_score,k).indices,1.)
  rng=np.random.default_rng(args.seed+12463)
  rnd=torch.zeros_like(score); sel=torch.as_tensor(rng.choice(score.numel(),k,replace=False),device=device,dtype=torch.long);rnd[sel]=1
  masks={'E5_random_topk':rnd,'E6_degree_topk':degree_mask,'E7_learned_topk':learned,'full_edges':torch.ones_like(score),'zero_edges':torch.zeros_like(score)}
  threshold=args.threshold
  if threshold is None:
   from sklearn.metrics import f1_score
   logits,_=model(x,edges,mask_override=learned)
   valp=logits.softmax(-1)[idx['val_idx'],1].cpu().numpy();valy=y[idx['val_idx']].cpu().numpy()
   threshold=float(max(np.linspace(.1,.9,81),key=lambda t:(f1_score(valy,valp>=t,zero_division=0),-abs(t-.5))))
  rows=[]
  for name,mask in masks.items():
   logits,_=model(x,edges,mask_override=mask)
   met=evaluate(logits,y,s,idx['test_idx'],threshold)
   rows.append({'dataset':args.dataset,'seed':args.seed,'experiment':name,'retention':args.retention,'k':k,'num_edges':score.numel(),'mask_score_std':float(score.std()),**met})
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('Wrote',out,'; learned F1',next(r['f1'] for r in rows if r['experiment']=='E7_learned_topk'))
if __name__=='__main__':main()
