"""E0/E1/E2/E3/E4 diagnostics: always-positive, MLP, GCN, degree-preserving rewired GCN, uniform-edge GCN.
Uses supplied fixed splits, validation-derived thresholds, and paired seeds. No test labels for tuning.
"""
import argparse,csv,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score,roc_auc_score,balanced_accuracy_score,matthews_corrcoef,average_precision_score,confusion_matrix
from train import load_npz,undirected_edges,MaskGCN,set_seed

class MLP(nn.Module):
    def __init__(self,n,h,dropout):
        super().__init__();self.dropout=dropout;self.net=nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Dropout(dropout),nn.Linear(h,2))
    def forward(self,x,e):return self.net(x),torch.ones(e.shape[1],device=x.device)

def edge_homophily(e,a):
    u,v=e
    return float((a[u]==a[v]).float().mean())

def rewire_degree_preserving(edges,seed,swaps_factor=10):
    """Undirected double-edge swaps; preserves each node's degree and edge count."""
    rng=random.Random(seed)
    pairs=[tuple(sorted(p)) for p in edges.T.cpu().tolist()]
    uniq=set(pairs)
    if len(uniq)!=len(pairs):raise ValueError('Edges must be unique undirected pairs')
    if len(pairs)<2:return edges
    for _ in range(swaps_factor*len(pairs)):
        i,j=rng.sample(range(len(pairs)),2)
        a,b=pairs[i]; c,d=pairs[j]
        if len({a,b,c,d})<4:continue
        cand1=tuple(sorted((a,d)));cand2=tuple(sorted((c,b)))
        if cand1==cand2 or cand1 in uniq or cand2 in uniq:continue
        uniq.remove(pairs[i]);uniq.remove(pairs[j]);uniq.update((cand1,cand2))
        pairs[i],pairs[j]=cand1,cand2
    return torch.tensor(pairs,device=edges.device,dtype=torch.long).T.contiguous()

def evaluate(logits,y,s,idx,threshold):
    ids=idx.cpu().numpy();labels=y[ids].cpu().numpy();sensitive=s[ids].cpu().numpy()
    probs=logits.softmax(-1)[ids,1].detach().cpu().numpy()
    pred=(probs>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
    stats=dict(f1=f1_score(labels,pred,zero_division=0),balanced_accuracy=balanced_accuracy_score(labels,pred),mcc=matthews_corrcoef(labels,pred),auc=roc_auc_score(labels,probs) if len(set(labels))>1 else float('nan'),auprc=average_precision_score(labels,probs),positive_rate=pred.mean(),prevalence=labels.mean(),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),threshold=threshold)
    r=[];t=[]
    for g in [0,1]:
        select=sensitive==g
        r.append(float(pred[select].mean()) if select.any() else float('nan'))
        pos=select&(labels==1)
        t.append(float(pred[pos].mean()) if pos.any() else float('nan'))
    stats.update(parity=abs(r[0]-r[1]),equality=abs(t[0]-t[1]))
    return stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--dataset',choices=['bail','credit','german'],required=True);p.add_argument('--output',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]);p.add_argument('--epochs',type=int,default=160);p.add_argument('--patience',type=int,default=30);p.add_argument('--hidden',type=int,default=64);p.add_argument('--lr',type=float,default=.005);p.add_argument('--dropout',type=float,default=.3)
    args=p.parse_args();target=Path(args.output)
    if target.exists():raise FileExistsError('Refusing overwrite: '+str(target))
    target.parent.mkdir(parents=True,exist_ok=True)
    rows=[];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for seed in args.seeds:
        path=Path(args.data_dir)/f'{args.dataset}_seed{seed}.npz'
        if not path.exists():path=Path(args.data_dir)/f'{args.dataset}.npz'
        x,e_raw,y,s,idx=load_npz(path,device);e=undirected_edges(e_raw,len(x))
        print(f'{args.dataset} seed={seed}: label homophily {edge_homophily(e,y):.4f}, sensitive homophily {edge_homophily(e,s):.4f}',flush=True)
        set_seed(seed)
        # always positive has no trainable parameters
        dummy=torch.zeros(len(x),2,device=device);dummy[:,1]=1
        rows.append(dict(dataset=args.dataset,seed=seed,experiment='E0_always_positive',label_homophily=edge_homophily(e,y),sensitive_homophily=edge_homophily(e,s),**evaluate(dummy,y,s,idx['test_idx'],0.5)))
        for kind in ['E1_mlp','E2_gcn','E3_rewired_gcn','E4_uniform_edges']:
            set_seed(seed)
            model=MLP(x.shape[1],args.hidden,args.dropout) if kind=='E1_mlp' else MaskGCN(x.shape[1],args.hidden,args.dropout,False)
            model=model.to(device)
            edges=rewire_degree_preserving(e,seed+550) if kind=='E3_rewired_gcn' else e
            optim=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=.0005)
            best=-1;best_weights=None;bad=0
            for epoch in range(args.epochs):
                model.train();optim.zero_grad();lg,_=model(x,edges)
                loss=F.cross_entropy(lg[idx['train_idx']],y[idx['train_idx']]);loss.backward();optim.step()
                model.eval()
                with torch.no_grad():lg,_=model(x,edges)
                probs=lg.softmax(-1)[idx['val_idx'],1].detach().cpu().numpy();labels=y[idx['val_idx']].cpu().numpy()
                auc=roc_auc_score(labels,probs) if len(set(labels))>1 else 0.5
                if auc>best+1e-7:best=auc;bad=0;best_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
                else:
                    bad+=1
                    if bad>=args.patience:break
            model.load_state_dict(best_weights);model.eval()
            with torch.no_grad():lg,_=model(x,edges)
            vp=lg.softmax(-1)[idx['val_idx'],1].cpu().numpy();vy=y[idx['val_idx']].cpu().numpy()
            threshold=max(np.linspace(.1,.9,81),key=lambda t:(f1_score(vy,vp>=t,zero_division=0),-abs(t-.5)))
            rows.append(dict(dataset=args.dataset,seed=seed,experiment=kind,label_homophily=edge_homophily(e,y),sensitive_homophily=edge_homophily(e,s),**evaluate(lg,y,s,idx['test_idx'],threshold)))
    with target.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Saved',target)
if __name__=='__main__':main()
