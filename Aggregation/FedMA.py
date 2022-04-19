# number of iterations is the same number of layers in a model
import copy
import torch
from torch import nn


def FedMA(clients):
  

    w_base = copy.deepcopy(w[0])
   
    for k in w_base.keys():
        for i in range(1, len(w_base)):
            w_base[k] += w[i][k]
        w_base[k] = torch.div(w_base[k], len(w_base))

    
    return w_base
