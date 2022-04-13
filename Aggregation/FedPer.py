

import copy
import torch
from torch import nn


def FedPer(w):
  
    w_base = copy.deepcopy(w[0])

  

    for k in w_base.keys():
        for i in range(1, len(w)):
            w_base[k] += w_base[i][k]
        w_base[k] = torch.div(w_base[k], len(w))

    

    return w_base
