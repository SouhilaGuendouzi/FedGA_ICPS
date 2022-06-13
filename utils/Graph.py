import matplotlib.pyplot as plt 
import numpy as np 


def Plot_Graphes (args, accTrain,accTest,lossTrain,lossTest):
    methode=args.aggr
    if (args.aggr=='fedPerGA'): methode="FedGA"
    clients_accuracy_train=[[0 for _ in range(args.epochs+1)] for _ in range(args.num_users)]
    clients_accuracy_test=[[0 for _ in range(args.epochs+1)] for _ in range(args.num_users)] 
    clients_loss_train=[[0 for _ in range(args.epochs+1)] for _ in range(args.num_users)]
    clients_loss_test=[[0 for _ in range(args.epochs+1)] for _ in range(args.num_users)]

    figure, axis = plt.subplots(2, 2,figsize=(10,8)) 
    epochs = np.arange(0, args.epochs+1)
    for i in range(args.epochs+1):
      for j in range(args.num_users):
        clients_accuracy_train[j][i]= accTrain[i][j] 
        clients_accuracy_test[j][i]= accTest[i][j] 
        clients_loss_train[j][i]= lossTrain[i][j] 
        clients_loss_test[j][i]= lossTest[i][j] 

    for i in range(args.num_users):
        
     
      axis[0, 0].plot( epochs,clients_accuracy_train[i] ) 
      axis[0, 0].set_title("{} Train Accuracy".format( methode),fontsize=10) 
      axis[0, 0].legend(['Edge {}'.format(j+1) for j in range(args.num_users)],fontsize=10)
      
      axis[0, 1].plot(epochs,  clients_loss_train[i]) 
      axis[0, 1].set_title("{} Train Loss".format( methode),fontsize=10) 
      axis[0, 1].legend(['Edge {}'.format(j+1)  for j in range(args.num_users)],fontsize=10)
      
  
      axis[1, 0].plot(epochs, clients_accuracy_test[i]) 
      axis[1, 0].set_title("{} Test Accuracy".format( methode),fontsize=10) 
      axis[1, 0].legend(['Edge {}'.format(j+1)  for j in range(args.num_users)],fontsize=10)
     
   
      axis[1, 1].plot(epochs,  clients_loss_test[i]) 
      axis[1, 1].set_title("{} Test Loss".format( methode),fontsize=10) 
      axis[1, 1].legend(['Edge {}'.format(j+1) for j in range(args.num_users)],fontsize=10)
      

    plt.xlabel("FL rounds ")
    plt.savefig('./save/{}.png'.format(methode))
    plt.show() 