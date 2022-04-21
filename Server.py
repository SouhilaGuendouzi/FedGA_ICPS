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
from create_MNIST_datasets import get_MNIST, plot_samples
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    net_glob = Model_MNIST(args=args).to(args.device)
    net_glob.train() 
###################################################################################################################
    mnist_non_iid_train_dls, mnist_non_iid_test_dls = get_MNIST("non_iid",
    n_samples_train =2000, n_samples_test=1000, n_clients =4, 
    batch_size =25, shuffle =True)
    dict_users={}
    client=Client(0,net_glob, mnist_non_iid_train_dls[0],mnist_non_iid_test_dls[0],args)
    dict_users[0] = client 
    client=Client(1,net_glob, mnist_non_iid_train_dls[1],mnist_non_iid_test_dls[1],args)
    dict_users[1] = client    
    client=Client(2,net_glob, mnist_non_iid_train_dls[2],mnist_non_iid_test_dls[2],args)
    dict_users[2] = client  
    plot_samples(next(iter(mnist_non_iid_train_dls[0])), 0, "Client 1")
    plot_samples(next(iter(mnist_non_iid_train_dls[1])), 0, "Client 2")
    plot_samples(next(iter(mnist_non_iid_train_dls[2])), 0, "Client 3")
    dataset_validate=mnist_non_iid_train_dls[3]
    
   


    # copy weights
    w_glob = net_glob.state_dict()


    # training
    loss_train = []
    loss_locals = []
    w_locals = [w_glob for i in range(args.num_users)]

    # personnalized layers

    #m = max(int(args.frac * args.num_users), 1)
    ids_users = range(args.num_users)#np.random.choice(range(args.num_users), m, replace=False) # choose m users from num_users

    
   

    if args.all_clients: 
       print("Aggregation over all clients")
       w_locals = [w_glob for i in range(args.num_users)]

    if (args.aggr=='fedMA'):
           clients=FedMA(dict_users)
    else :         
     for iter in range(args.epochs): 
        i=0 
        print('iteration',iter) 
        loss_locals = []
        if not args.all_clients:
            w_locals = []
       
       
        '''
        For Local updates
        
        '''
        if  (args.aggr=='fedPer'):
              for id in ids_users: #idx of a user
                
                   w, loss = dict_users[id].local_updatePer(w_glob)

                   if args.all_clients:
                     w_locals[id] = copy.deepcopy(w)
                   else:
                     w_locals.append(copy.deepcopy(w))
                   loss_locals.append(copy.deepcopy(loss))
              loss_avg = sum(loss_locals) / len(loss_locals)

                  
        else :    
            for id in ids_users: #idx of a user
              
              w, loss = dict_users[id].local_update(w_glob)
              if args.all_clients:
                w_locals[id] = copy.deepcopy(w)
              else:
                w_locals.append(copy.deepcopy(w))
              loss_locals.append(copy.deepcopy(loss))
            loss_avg = sum(loss_locals) / len(loss_locals)
            print(loss_avg)


        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        w_locals= np.array(w_locals)



        '''
        For aggregation
        
        '''

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
            w_glob = FedPer(w_locals)

        net_glob.load_state_dict(w_glob)

        # print loss
        

    # plot loss curve
     for id in ids_users: #idx of a user
            w, loss =  dict_users[id].local_update(w_glob)

            if args.all_clients:
                w_locals[id] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

    loss_avg = sum(loss_locals) / len(loss_locals)
    print('After all itrations')
    print(' Average loss {:.3f}'.format(loss_avg))
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_.png'.format(args.aggr, args.iid))

    # testing
    net_glob.eval()
    print('train Test')
    
    acc_train, loss_train= [],[]
    acc_test, loss_test =[], []
    for id in ids_users:
         acc, loss = dict_users[id].test_img(dict_users[id].datasetTrain)
         print("Training accuracy for client {} is : {:.2f}".format(id,acc))
         acc_train.append(acc)
         loss_train.append(loss)

         acc, loss = dict_users[id].test_img(dict_users[id].datasetTest)
         print("Testing accuracy for client {} is : {:.2f}".format(id,acc))
         acc_test.append(acc)
         loss_test.append(loss)
        
    acc_train_avg= sum(acc_train) / len(ids_users)
    acc_test_avg= sum(acc_test) / len(ids_users)


   
    print("Average Testing accuracy: {:.2f}".format(acc_test_avg))
    print("Average Ttraining accuracy: {:.2f}".format(acc_train_avg))