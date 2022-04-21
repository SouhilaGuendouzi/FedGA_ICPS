#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# https://pytorch.org/vision/stable/datasets.html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Dataset
from utils.Options import args_parser
from Model import Model_MNIST, CNNCifar, Model_Fashion 
from Client import Client 
from Aggregation.FedAVG import FedAvg
from Aggregation.FedGA import FedGA
from Aggregation.FedPer import FedPer
from Aggregation.FedMA import FedMA
from utils.Split import DatasetSplit

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    