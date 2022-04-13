# number of iterations is the same number of layers in a model
import copy
import torch
from torch import nn


def FedMA(w,numLayer):
  

    w_base = copy.deepcopy(w[0])
    f1bias=w.get('fc1.bias')
    f2bias=w.get('fc2.bias')
    f1weight=w.get('fc1.weight')
    fc2.weight==w.get('fc2.weight')
    del[w_base['fc1.bias']]
    del[w_base['fc1.weight']]
    del[w_base['fc2.bias']]
    del[w_base['fc2.weight']]

    for k in w_base.keys():
        for i in range(1, len(w_base)):
            w_base[k] += w_base[i][k]
        w_base[k] = torch.div(w_base[k], len(w_base))

    w_base['fc1.weight']=f1weight
  
    w_base['fc1.bias']=f1bias

    w_base['fc2.weight']=f2weight

    w_base['fc2.bias']=f2bias
    
    return w_base
