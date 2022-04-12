

import torch
from torch import nn, autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5) #for gray images ==> args.num_channels===1 
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        #self.flatten= torch.nn.Flatten()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def test_img(self,net_g, datatest, args):
        net_g.eval()
        # testing
        test_loss = 0
        correct = 0
        for idx, (data, target) in enumerate(datatest):
           log_probs = net_g(data)
           # sum up batch loss
           test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
           # get the index of the max log-probability
           y_pred = log_probs.data.max(1, keepdim=True)[1]
           correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(datatest.dataset)
        accuracy = 100.00 * correct / len(datatest.dataset)
        if args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(datatest.dataset), accuracy))
        return accuracy, test_loss




