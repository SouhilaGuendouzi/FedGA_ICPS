import matplotlib.pyplot as plt 
import numpy as np 
import math 




def Plot_Graphes (args, accTrain,accTest,lossTrain,lossTest):
    clients_accuracy_train=[[0 for _ in range(args.epochs)] for _ in range(args.num_users)]
    clients_accuracy_test=[[0 for _ in range(args.epochs)] for _ in range(args.num_users)] 
    clients_loss_train=[[0 for _ in range(args.epochs)] for _ in range(args.num_users)]
    clients_loss_test=[[0 for _ in range(args.epochs)] for _ in range(args.num_users)]

    figure, axis = plt.subplots(2, 2) 
    epochs = np.arange(0, args.epochs)
    for i in range(args.epochs):
      for j in range(args.num_users):
        clients_accuracy_train[j][i]= accTrain[i][j] 
        clients_accuracy_test[j][i]= accTest[i][j] 
        clients_loss_train[j][i]= lossTrain[i][j] 
        clients_loss_test[j][i]= lossTest[i][j] 

    for i in range(args.num_users):

  
      axis[0, 0].plot( epochs,clients_accuracy_train[i] ) 
      axis[0, 0].set_title("Train Accuracy") 
      axis[0, 0].legend(['Client {}'.format(j) for j in range(args.num_users)])
  
      axis[0, 1].plot(epochs,  clients_loss_train[i]) 
      axis[0, 1].set_title("Train Loss") 
      axis[0, 1].legend(['Client {}'.format(j) for j in range(args.num_users)])
  
      axis[1, 0].plot(epochs, clients_accuracy_test[i]) 
      axis[1, 0].set_title("Test Accuracy") 
      axis[1, 0].legend(['Client {}'.format(j) for j in range(args.num_users)])
   
      axis[1, 1].plot(epochs,  clients_loss_test[i]) 
      axis[1, 1].set_title("Test Loss") 
      axis[1, 1].legend(['Client {}'.format(j) for j in range(args.num_users)])


    plt.savefig('./save/FedGAPer_30.png')
    plt.show() 