"""No training needed: verifies mask weights influence graph messages; init edge scores not identical."""
import argparse,torch
from pathlib import Path
from train import MaskGCN,load_npz,undirected_edges,set_seed
p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args()
set_seed(0)
X,E,Y,S,I=load_npz(Path(a.data),torch.device('cpu'));E=undirected_edges(E,len(X))
m=MaskGCN(X.shape[1],32,dropout=0,masked=True).eval();m.retention=.5
with torch.no_grad():
    default,scores=m(X,E)
    none,_=m(X,E,mask_override=torch.zeros_like(scores))
    full,_=m(X,E,force_full=True)
print({'init_mask_std':float(scores.std(unbiased=False)),'initial_edge_count':scores.numel(),'max_difference_zero_vs_full':float((none-full).abs().max()),'max_difference_topk_vs_full':float((default-full).abs().max())})
assert float((none-full).abs().max())>1e-8,'ERROR: graph edge messages are not changing outputs'
assert float(scores.std(unbiased=False))>1e-8,'ERROR: edge scores start identical'
