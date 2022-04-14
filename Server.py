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
from Model import Model 
from Client import Client 
from Aggregation.FedAVG import FedAvg
from Aggregation.FedGA import FedGA
from Aggregation.FedPer import FedPer
from utils.Split import DatasetSplit


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    test_subset, val_subset = torch.utils.data.random_split(dataset_test, [8000, 2000], generator=torch.Generator().manual_seed(1))
    #dataset_train= DataLoader(dataset= dataset_train, shuffle=True)
    #dataset_test = DataLoader(dataset=test_subset, shuffle=True)
    dataset_validate = DataLoader(dataset=val_subset, shuffle=True)


    # split dataset with iid 
    num_items_train = int(len(dataset_train)/args.num_users)  # dataset size is equal  for all users 
    num_items_test= int(len(dataset_test)/args.num_users)
    net_glob = Model(args=args).to(args.device)
    net_glob.train() # au debut tous les clients ont le meme model 
    dict_users, all_idxs_train, all_idxs_test = {}, [i for i in range(len(dataset_train))], [i for i in range(len(dataset_test))]



    for i in range(args.num_users):
        
        dataset_train_index_client= set(np.random.choice(all_idxs_train, num_items_train, replace=False)) #la liste des index des itemes dans une dataset
        dataset_test_index_client= set(np.random.choice(all_idxs_test, num_items_test, replace=False)) #la liste des index des itemes dans une dataset

        train_dataset_client=DatasetSplit( dataset_train,  idxs=dataset_train_index_client)
        test_dataset_client=DatasetSplit( dataset_test, idxs= dataset_test_index_client)
        
        #
        # train_dataset_client=DataLoader(dataset=DatasetSplit( dataset_train,  idxs=dataset_train_index_client),shuffle=True,batch_size=args.local_bs) #,batch_size=args.local_bs
        test_dataset_client=DataLoader(dataset=test_dataset_client, shuffle=True)
     


        client=Client(i,net_glob, train_dataset_client,test_dataset_client,args)  
        dict_users[i] = client 
      
        all_idxs_train = list(set(all_idxs_train) - dataset_train_index_client) # Update the list of sample indexes
        all_idxs_test = list(set(all_idxs_test) - dataset_test_index_client) # Update the list of sample indexes

    

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    loss_locals = []
    w_locals = [w_glob for i in range(args.num_users)]

    # personnalized layers
    f1bias=[]
    f2bias=[]
    f2weight=[]
    f1weight=[]

    m = max(int(args.frac * args.num_users), 1)
    ids_users = np.random.choice(range(args.num_users), m, replace=False) # choose m users from num_users

    
    for id in ids_users: #idx of a user
            w, loss =  dict_users[id].local_update(w_glob)

            if args.all_clients:
                w_locals[id] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Before aggregation')
    print(' Average loss {:.3f}'.format(loss_avg))

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
       
        if  (args.aggr=='fedPer'):
              for id in ids_users: #idx of a user
                   base=w_glob
                   base.update(w_locals[id]['fc1.weight'])
                   base.update(w_locals[id]['fc1.bias'])
                   base.update(w_locals[id]['fc2.weight'])
                   base.update(w_locals[id]['fc2.bias'])
                   w, loss = dict_users[id].local_update(w_glob)

                   if args.all_clients:
                     w_locals[id] = copy.deepcopy(w)
                   else:
                     w_locals.append(copy.deepcopy(w))
                   loss_locals.append(copy.deepcopy(loss))
              loss_avg = sum(loss_locals) / len(loss_locals)

                  
        else :    
            for id in ids_users: #idx of a user

              w, loss = dict_users[i].local_update(w_glob)

              if args.all_clients:
                w_locals[id] = copy.deepcopy(w)
              else:
                w_locals.append(copy.deepcopy(w))
              loss_locals.append(copy.deepcopy(loss))

            loss_avg = sum(loss_locals) / len(loss_locals)


        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        w_locals= np.array(w_locals)
        if (args.aggr=='fedAVG'):
            # update global weights
            w_glob = FedAvg(w_locals)
        elif (args.aggr=='fedGA'):
             initial_population=w_locals
       
             for d in w_locals: # for each user
                weight=[]
                if isinstance(d, dict):
                     for x in d.items():  #get weights of each layer
                         array = np.array(x[1], dtype='f')#1 is a tensor
                         array= array.flatten()
                         weight= np.concatenate((weight, array), axis=0)
                         initial_population[i]= np.array(weight,dtype='f')
                i=i+1 # next weight vector (user)
       
             w_glob = FedGA(initial_population,net_glob,dataset_validate)
        elif (args.aggr=='fedPer'):
           
           
            for i in range(len(w_locals)):

             f1bias[i]=w_locals[i].get('fc1.bias')
             f2bias[i]=w_locals[i].get('fc2.bias')
             f1weight[i]=w_locals[i].get('fc1.weight')
             f2weight[i]=w_locals[i].get('fc2.weight')
             del[w_locals[i]['fc1.bias']]
             del[w_locals[i]['fc1.weight']]
             del[w_locals[i]['fc2.bias']]
             del[w_locals[i]['fc2.weight']]


            w_glob = FedPer(w_locals)

            for i in range(len(w_locals)):
               w_locals[i]['fc1.weight']= f1weight[i]
               w_locals[i]['fc1.bias']= f2bias[i]
               w_locals[i]['fc2.weight']=f2weight[i]
               w_locals[i]['fc2.bias']=f2bias


        net_glob.load_state_dict(w_glob)

        # print loss
        

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_.png'.format(args.aggre, args.iid))

    # testing
    net_glob.eval()
    print('train Test')
    dataset_train=  DataLoader(dataset_train)
    acc_train, loss_train = net_glob.test_img(net_glob, dataset_train, args)
    print('test Test')
    acc_test, loss_test = net_glob.test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))