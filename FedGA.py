
import copy
import torch
from torch import nn


def FedGA(w,global_M):
    initial_popluation= copy.deepcopy(w)
    previous_M=global_M

    
    w_avg = copy.deepcopy(w[0])  #Renvoie une copie récursive de x.
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg