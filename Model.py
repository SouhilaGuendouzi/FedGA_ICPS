

import torch
from torch import nn, autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

class Model(nn.Module):

    
     def __init__(self, args):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  #(in , out, kernel_size) #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
       
        self.pool = nn.MaxPool2d(2, 2) # pool of square window of size=2, stride=2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

     def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5) # the size -1 is inferred from other dimensions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

   




