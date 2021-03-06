#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10
# https://pytorch.org/vision/stable/datasets.html

import matplotlib
matplotlib.use('Agg')
import torch
from utils.Options import args_parser
from Entities.Model import Model_B,Model_A,Model_C,Model_D,Model_Fashion,Net
from Entities.Edge import Edge
from utils.create_MNIST_datasets import get_FashionMNIST, get_MNIST
from Entities.Cloud import Cloud
#from utils.Plot import accuracy_test, accuracy_train, loss_test, loss_train
from utils.Graph import Plot_Graphes
from tabulate import tabulate
import time



if __name__ == '__main__':
    # parse args
    args = args_parser()
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(torch.cuda.is_available())
    net_glob = Model_Fashion().to(args.device)

    net_glob.train() 
    w=net_glob.state_dict()
    
############################ Prepare Clients#######################################################################################

  

    weights_locals=[]
    mnist_non_iid_train_dls, mnist_non_iid_test_dls = get_FashionMNIST(args.iid,
    n_samples_train =1500, n_samples_test=250, n_clients =4,  # i have calculated because there are 60000/ 1000
    batch_size =50, shuffle =True)                            #(1500+250) samples for each client / 50 batch size ==num of epochs / and 30 number of batch 

    dict_users={}
 

    model_A=Model_D().to(args.device)
    model_B=Model_D().to(args.device)
    model_C=Model_D().to(args.device)
    model_D=Model_D().to(args.device)


    dict_users[0] = Edge (0,model_A, mnist_non_iid_train_dls[0], mnist_non_iid_test_dls[0],args)    #A
    dict_users[1] = Edge (1,model_C, mnist_non_iid_train_dls[1], mnist_non_iid_test_dls[1],args)    #B
    dict_users[2] = Edge (2,model_B, mnist_non_iid_train_dls[2],mnist_non_iid_test_dls[2],args)     #C  
    dict_users[3] = Edge (3,model_D, mnist_non_iid_train_dls[3],mnist_non_iid_test_dls[3],args)     #D



    print('Train length',len(mnist_non_iid_train_dls[0]),len(mnist_non_iid_train_dls[1]),len(mnist_non_iid_train_dls[2]),len(mnist_non_iid_train_dls[3]))
    print('Test length',len(mnist_non_iid_test_dls[0]),len(mnist_non_iid_test_dls[1]),len(mnist_non_iid_test_dls[2]),len(mnist_non_iid_test_dls[3]))


   
########################## Prapare Cloud #########################################################################################
 
     
    train, test = get_FashionMNIST('iid',
    n_samples_train =1500, n_samples_test=250, n_clients =2,  
    batch_size =50, shuffle =True)
    print('Cloud length',len(test[0]))
    cloud=Cloud(dict_users,net_glob,test[0],args)

    


########################## Begin process #########################################################################################

    accloss=[[0 for _ in range(len(dict_users)+1)] for _ in range(2)]
    for iter in range(args.epochs):
        

      weights_locals,loss_locals_train,loss_locals_test, accuracy_locals_train,accuracy_locals_test=cloud.Launch_local_updates(iter)
      if (iter==0):
          for i in range(len(dict_users)):
   
            accloss[0][i]=accuracy_locals_train[0][i]
            accloss[1][i]=accuracy_locals_test[0][i]

          row=accloss
          col=['Client {}'.format(j) for j in range(len(dict_users))]
          print(tabulate(row, headers=col, tablefmt="fancy_grid"))
        

      net_glob=cloud.aggregate(weights_locals,args.aggr)

    print("After Last Aggregation",f" Round  {iter+1}")
    weights_locals, loss_locals_train,loss_locals_test,accuracy_locals_train,accuracy_locals_test=cloud.Launch_local_updates(iter+1)


    
########################## Evaluation #########################################################################################
    aclo=[[0 for _ in range(len(dict_users))] for _ in range(2)]

    for i in range(len(dict_users)):
          
            aclo[0][i]=accuracy_locals_train[iter+1][i]
            aclo[1][i]=accuracy_locals_test[iter+1][i]

    
    
    row=aclo
    col=['Client {}'.format(j) for j in range(len(dict_users))]
    print(tabulate(row, headers=col, tablefmt="fancy_grid"))

    Plot_Graphes(args,accuracy_locals_train,accuracy_locals_test,loss_locals_train,loss_locals_test)

