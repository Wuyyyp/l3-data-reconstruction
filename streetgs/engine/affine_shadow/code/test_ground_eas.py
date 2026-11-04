import torch
from ground_eas import select_candidates
ids=torch.arange(6);grad=torch.arange(6,dtype=torch.float32);weight=torch.tensor([0.,0.,2.,1.,0.,0.])
a,n=select_candidates(ids,grad,weight,3,100)
assert n==2 and set(a.tolist())=={2,3,5}
b,_=select_candidates(ids,grad,weight,3,100);assert torch.equal(a,b)
c,n=select_candidates(ids,grad,weight*0,2,100);assert n==0 and c.tolist()==[5,4]
print('GROUND_EAS_SELECTION_PASS')
