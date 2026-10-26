"""Option A: three mask-training strategies, paired seeds and common evaluation.
All labels used for optimization are drawn exclusively from train_idx.
Held-out test is evaluated once after validation checkpoint/threshold selection.
Experimental implementations, not established bilevel or perturbation-explainer SOTA.
"""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from train import MaskGCN, load_npz, undirected_edges, set_seed
from stage0 import evaluate


def split_training(ids, seed):
    rng=np.random.default_rng(seed+814)
    a=ids.detach().cpu().numpy().copy();rng.shuffle(a)
    n=max(1,min(len(a)-1,int(.20*len(a))))
    return (torch.as_tensor(a[n:],device=ids.device),torch.as_tensor(a[:n],device=ids.device))


def hardmask(score, ratio):
    k=max(1,min(len(score)-1,round(ratio*len(score))))
    top=torch.topk(score,k,sorted=False).indices
    hard=torch.zeros_like(score).scatter_(0,top,1.)
    return hard-score.detach()+score


def evaluate_mask_quality(model,x,ei,y,s,idx,threshold,retention,seed):
    model.eval()
    with torch.no_grad():
        score=model.get_mask(x,ei); k=max(1,min(len(score)-1,round(retention*len(score))))
        learned=torch.zeros_like(score).scatter_(0,torch.topk(score,k).indices,1.)
        deg=torch.bincount(ei.flatten(),minlength=len(x)).float()
        degree_score=deg[ei[0]]+deg[ei[1]]
        degree=torch.zeros_like(score).scatter_(0,torch.topk(degree_score,k).indices,1.)
        rng=np.random.default_rng(seed+889)
        rand=torch.zeros_like(score);rand[torch.tensor(rng.choice(len(score),k,replace=False),device=score.device)]=1.
        full=torch.ones_like(score)
        masks={'learned':learned,'random':rand,'degree':degree,'full':full,'complement':1-learned}
        res={}
        for name,m in masks.items():
            logits,_=model(x,ei,mask_override=m)
            z=evaluate(logits,y,s,idx['test_idx'],threshold)
            for metric in ['f1','auc','balanced_accuracy','mcc','auprc','parity','equality','positive_prediction_rate']:
                if metric in z:res[name+'_'+metric]=z[metric]
        ref,_=model(x,ei,mask_override=full)
        pred,_=model(x,ei,mask_override=learned)
        comp,_=model(x,ei,mask_override=1-learned)
        index=idx['test_idx']
        res.update({'mask_score_std':float(score.std(unbiased=False)), 'mask_entropy':float((-(score.clamp(1e-6,1-1e-6)*score.clamp(1e-6,1-1e-6).log()+(1-score.clamp(1e-6,1-1e-6))*(1-score.clamp(1e-6,1-1e-6)).log())).mean()),'mask_frac_lt_01':float((score<.1).float().mean()),'mask_frac_gt_09':float((score>.9).float().mean()),'sufficiency_kl':float(F.kl_div(F.log_softmax(pred[index],-1),F.softmax(ref[index],-1),reduction='batchmean')),'complement_kl':float(F.kl_div(F.log_softmax(comp[index],-1),F.softmax(ref[index],-1),reduction='batchmean'))})
    return res


def fit(x,ei,y,s,idx,method,retention,seed,args,outdir):
    set_seed(seed)
    model=MaskGCN(x.size(1),args.hidden,args.dropout,masked=True).to(x.device)
    model.retention=retention;model.hard_topk=True
    # Shared encoder for task message passing, separate pre+MLP scorer for edge weights.
    scorer=list(model.pre.parameters())+list(model.mask_mlp.parameters())
    encoder=list(model.g1.parameters())+list(model.g2.parameters())
    opt_main=torch.optim.Adam(encoder,lr=args.lr,weight_decay=args.weight_decay)
    opt_mask=torch.optim.Adam(scorer,lr=args.mask_lr,weight_decay=args.weight_decay)
    opt_joint=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    inner,outer=split_training(idx['train_idx'],seed)
    best=-float('inf'); state=None; patience=0; peak_grad=0.
    for epoch in range(args.epochs):
        model.train();warm=epoch<args.warmup
        if method=='independent':
            opt_joint.zero_grad()
            logits,score=model(x,ei,force_full=warm)
            loss=F.cross_entropy(logits[idx['train_idx']],y[idx['train_idx']])
            loss.backward()
            if not warm:
                peak_grad=max(peak_grad,float(model.mask_mlp[-1].weight.grad.norm()) if model.mask_mlp[-1].weight.grad is not None else 0.)
            opt_joint.step()
        elif method=='bilevel':
            # Alternating inner training and outer mask optimization on DISJOINT train-only subsets.
            # This is first-order alternating bilevel approximation; NOT exact hypergradient.
            opt_main.zero_grad();logits,_=model(x,ei,force_full=warm)
            F.cross_entropy(logits[inner],y[inner]).backward();opt_main.step()
            if not warm:
                opt_mask.zero_grad()
                # Freeze classifier weights for outer update, but allow gradient through masked propagation.
                for p in encoder:p.requires_grad_(False)
                logits,_=model(x,ei)
                loss=F.cross_entropy(logits[outer],y[outer]);loss.backward()
                if model.mask_mlp[-1].weight.grad is not None:peak_grad=max(peak_grad,float(model.mask_mlp[-1].weight.grad.norm()))
                opt_mask.step()
                for p in encoder:p.requires_grad_(True)
        elif method=='perturbation':
            opt_joint.zero_grad()
            logits,score=model(x,ei,force_full=warm)
            loss=F.cross_entropy(logits[inner],y[inner])
            if not warm:
                # Perturbation-based learning: selected edges must preserve task signal and
                # complement edges should not predict as confidently as selected edges.
                k=max(1,min(len(score)-1,round(retention*len(score))))
                hard=torch.zeros_like(score).scatter_(0,torch.topk(score,k).indices,1.)
                selected=hard-score.detach()+score
                opposite=1-selected
                complement_logits,_=model(x,ei,mask_override=opposite)
                # Uniform complement prediction is NOT demanded; instead margin on supervised task CE.
                ce_selected=F.cross_entropy(logits[inner],y[inner])
                ce_complement=F.cross_entropy(complement_logits[inner],y[inner])
                margin=F.relu(args.explanation_margin+ce_selected-ce_complement)
                # Smooth mask-score scores under feature noise (train-only supervision).
                generator=torch.Generator(device=x.device).manual_seed(seed*100000+epoch)
                xn=x+args.feature_noise*torch.randn(x.shape,device=x.device,generator=generator)
                score_noise=model.get_mask(xn,ei)
                stability=(score-score_noise).square().mean()
                loss=loss+args.lambda_explain*margin+args.lambda_stability*stability
            loss.backward()
            if not warm and model.mask_mlp[-1].weight.grad is not None:peak_grad=max(peak_grad,float(model.mask_mlp[-1].weight.grad.norm()))
            opt_joint.step()
        model.eval()
        with torch.no_grad():
            v,_=model(x,ei);val=evaluate(v,y,s,idx['val_idx'],.5)
            auc=val.get('auc',float('nan'));scoreval=(auc+val['f1'])/2 if np.isfinite(auc) else val['f1']
        if warm:continue
        if scoreval>best+1e-7:
            best=scoreval;patience=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience+=1
            if patience>=args.patience:break
    if state is None:raise RuntimeError('Missing non-warmup checkpoint')
    model.load_state_dict(state);model.eval()
    with torch.no_grad():
        logits,_=model(x,ei)
        prob=logits.softmax(-1)[idx['val_idx'],1].cpu().numpy();truth=y[idx['val_idx']].cpu().numpy()
        thr=float(max(np.linspace(.1,.9,81),key=lambda t:(f1_score(truth,prob>=t,zero_division=0),-abs(t-.5))))
    metrics=evaluate_mask_quality(model,x,ei,y,s,idx,thr,retention,seed)
    metrics.update({'model':method,'seed':seed,'retention':retention,'validation_threshold':thr,'validation_selection_score':best,'epochs_trained':epoch+1,'mask_grad_max':peak_grad,'num_edges':ei.size(1)})
    outdir.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),outdir/f'{method}_r{retention:g}_seed{seed}.pt')
    return metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['bail','credit'],required=True);p.add_argument('--data-dir',default='data')
    p.add_argument('--methods',nargs='+',choices=['independent','bilevel','perturbation'],default=['independent','bilevel','perturbation'])
    p.add_argument('--retentions',nargs='+',type=float,default=[.3,.5,.7]);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--epochs',type=int,default=300);p.add_argument('--warmup',type=int,default=15);p.add_argument('--patience',type=int,default=40)
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--dropout',type=float,default=.3)
    p.add_argument('--lr',type=float,default=.005);p.add_argument('--mask-lr',type=float,default=.002);p.add_argument('--weight-decay',type=float,default=.0005)
    p.add_argument('--feature-noise',type=float,default=.02);p.add_argument('--lambda-explain',type=float,default=.3)
    p.add_argument('--lambda-stability',type=float,default=.1);p.add_argument('--explanation-margin',type=float,default=.1)
    p.add_argument('--output',default=None);a=p.parse_args()
    if not all(0<v<1 for v in a.retentions):raise ValueError('Retention must be in (0,1)')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=Path(a.output or f'results/option_a/{a.dataset}');out.mkdir(parents=True,exist_ok=True)
    target=out/'results.csv'
    if target.exists():raise FileExistsError(f'{target} exists; outputs never overwritten')
    with target.open('w',newline='') as f:
        for seed in a.seeds:
            file=Path(a.data_dir)/f'{a.dataset}_seed{seed}.npz'
            if not file.exists():file=Path(a.data_dir)/f'{a.dataset}.npz'
            data=load_npz(file,device);x,raw,y,s,idx=data;ei=undirected_edges(raw,len(x))
            for rho in a.retentions:
                for method in a.methods:
                    row=fit(x,ei,y,s,idx,method,rho,seed,a,out/'checkpoints')
                    row['dataset']=a.dataset
                    writer=csv.DictWriter(f,fieldnames=list(row.keys()))
                    if f.tell()==0:writer.writeheader()
                    writer.writerow(row);f.flush()
                    print(json.dumps({'dataset':a.dataset,'method':method,'seed':seed,'rho':rho,'auc':row.get('learned_auc'),'random_auc':row.get('random_auc')},allow_nan=True),flush=True)
    print('Results:',target)
if __name__=='__main__':main()
