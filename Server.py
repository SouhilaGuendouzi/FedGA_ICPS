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
from Options import args_parser
from Model import CNNMnist 
from Client import Client , LocalUpdate 
from FedAVG import FedAvg
from FedGA import FedGA

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    test_subset, val_subset = torch.utils.data.random_split(dataset_test, [8000, 2000], generator=torch.Generator().manual_seed(1))
    dataset_test = DataLoader(dataset=test_subset, shuffle=True)
    dataset_validate = DataLoader(dataset=val_subset, shuffle=False)



    num_items = int(len(dataset_train)/args.num_users)  # dataset size is equal  for all users 
    net_glob = CNNMnist(args=args).to(args.device)
    net_glob.train() # au debut tous les clients ont le meme model
    # split dataset with iid 
    dict_users, all_idxs = {}, [i for i in range(len(dataset_train))]
    for i in range(args.num_users):
        dataset_index_client= set(np.random.choice(all_idxs, num_items, replace=False)) #la liste des index des itemes dans une dataset
        client=Client(i,net_glob, dataset_index_client)
        # for each client_i, choose different samples without replacement
        dict_users[i] = client
        #print(dict_users[i].dataset)
        all_idxs = list(set(all_idxs) - dict_users[i].dataset) # Update the list of sample indexes

    

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
       print("Aggregation over all clients")
       w_locals = [w_glob for i in range(args.num_users)]

    for iter in range(args.epochs): 
        i=0 
        print('iteration',iter) 
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        ids_users = np.random.choice(range(args.num_users), m, replace=False) # choose m users from num_users
        for id in ids_users: #idx of a user
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[id].dataset)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[id] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
       # w_glob = FedAvg(w_locals)
     
        w_locals= np.array(w_locals)
        initial_population=w_locals
       
        for d in w_locals: # for each user
          print(type(d))
          print('user ',i+1)
          weight=[]
          if isinstance(d, dict):
            for x in d.items():  #get weights of each layer
                 array = np.array(x[1], dtype='f')#1 is a tensor
                 array= array.flatten()
                 weight= np.concatenate((weight, array), axis=0)
            initial_population[i]= np.array(weight,dtype='f')
           
         
        

         #print(weight) 
         
          
          
          i=i+1
       
        w_glob = FedGA(initial_population,net_glob,dataset_validate)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = net_glob.test_img(net_glob, dataset_train, args)
    acc_test, loss_test = net_glob.test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

