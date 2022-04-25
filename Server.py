#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# https://pytorch.org/vision/stable/datasets.html
from tkinter import Place
import matplotlib
matplotlib.use('Agg')
import torch
from utils.Options import args_parser
from Entities.Model import Model_MNIST, CNNCifar, Model_Fashion 
from Entities.Edge import Edge
from utils.create_MNIST_datasets import get_MNIST
from Entities.Cloud import Cloud
from utils.Plot import Plot

if __name__ == '__main__':
    # parse args
    args = args_parser()
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    net_glob = Model_MNIST(args=args).to(args.device)
    net_glob.train() 
    weights_locals=[]
############################ Prepare Clients#######################################################################################

    mnist_non_iid_train_dls, mnist_non_iid_test_dls = get_MNIST("non_iid",
    n_samples_train =1500, n_samples_test=250, n_clients =4,  # i have calculated because there are 60000/ 1000
    batch_size =50, shuffle =True) #(1500+250) samples for each client / 50 batch size ==num of epochs / and 30 number of batch 
    dict_users={}
    client=Edge (0,net_glob, mnist_non_iid_train_dls[0],mnist_non_iid_test_dls[0],args)
    dict_users[0] = client 
    client=Edge (1,net_glob, mnist_non_iid_train_dls[1],mnist_non_iid_test_dls[1],args)
    dict_users[1] = client    
    client=Edge (2,net_glob, mnist_non_iid_train_dls[2],mnist_non_iid_test_dls[2],args)
    dict_users[2] = client  
    client=Edge (3,net_glob, mnist_non_iid_train_dls[3],mnist_non_iid_test_dls[3],args)
    dict_users[3] = client


    
    print('Train length',len(mnist_non_iid_train_dls[0]),len(mnist_non_iid_train_dls[1]),len(mnist_non_iid_train_dls[2]),len(mnist_non_iid_train_dls[3]))
    print('Test length',len(mnist_non_iid_test_dls[0]),len(mnist_non_iid_test_dls[1]),len(mnist_non_iid_test_dls[2]),len(mnist_non_iid_test_dls[3]))

########################## Prapare Cloud #########################################################################################
 
    train_Cloud, test_Cloud = get_MNIST("server",n_samples_train =20000, n_samples_test=10000)

    cloud=Cloud(dict_users,net_glob,test_Cloud,args)


########################## Begin process #########################################################################################
    cloud.loss_locals=[]
    cloud.accuracy_locals=[]
    for iter in range(args.epochs):
      weights_locals=cloud.Launch_local_updates(iter)
      cloud.aggregate(weights_locals,args.aggr)

    weights_locals=cloud.Launch_local_updates(iter+1)

########################## Evaluation process #########################################################################################
    print('Evaluation after Federation')
    net_glob.eval()
    acc_train, loss_train= [],[]
    acc_test, loss_test =[], []
    for id in range(len(dict_users)): #ids_users
          acc, loss = dict_users[id].test_img(dict_users[id].datasetTrain)
          #print("Training accuracy for client {} is : {:.2f}".format(id,acc))
          acc_train.append(acc)
          loss_train.append(loss)
          acc, loss = dict_users[id].test_img(dict_users[id].datasetTest)
          print("Testing accuracy for client {} is : {:.2f}".format(id,acc))
          acc_test.append(acc)
          loss_test.append(loss)

    acc_train_avg= sum(acc_train) / len(dict_users)
    acc_test_avg= sum(acc_test) / len(dict_users)
    plt = Plot(args,acc_train,acc_test, loss_train, loss_test )
    plt.get_graph_train('accuracy',args.aggr)
    plt.get_graph_train('loss',args.aggr)
    plt.get_graph_test('accuracy',args.aggr)
    plt.get_graph_test('loss',args.aggr)

    

    print("Average Testing accuracy: {:.2f}".format(acc_test_avg))
    print("Average Ttraining accuracy: {:.2f}".format(acc_train_avg))
