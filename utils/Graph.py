import matplotlib.pyplot as plt 
import matplotlib
import numpy as np 


def Plot_Graphes (args, accTrain,accTest,lossTrain,lossTest):
    matplotlib.use('TkAgg')
    methode=args.aggr
    print(accTrain)
    print(accTest)
    print(lossTrain)
    print(lossTest)
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


def Plot_Graphes_for_fog (id,epochs,num_users,accTrain,accTest,lossTrain,lossTest):
    matplotlib.use('TkAgg')
    methode=f"Fog {id}"
    print(methode)
    print(num_users)
    print(epochs)
    print(accTrain)
    print(accTest)
    print(lossTrain)
    print(lossTest)


    clients_accuracy_train=[[0 for _ in range(epochs)] for _ in range(num_users)]
    clients_accuracy_test=[[0 for _ in range(epochs)] for _ in range(num_users)] 
    clients_loss_train=[[0 for _ in range(epochs)] for _ in range(num_users)]
    clients_loss_test=[[0 for _ in range(epochs)] for _ in range(num_users)]

    figure, axis = plt.subplots(2, 2,figsize=(10,8)) 
  
    for i in range(epochs):
     
      for j in range(num_users):
        print(i,j)
        clients_accuracy_train[j][i]= accTrain[j][i]
        clients_accuracy_test[j][i]= accTest[j][i] 
        clients_loss_train[j][i]= lossTrain[j][i]
        clients_loss_test[j][i]= lossTest[j][i]

    if (id==1):
     for i in range(num_users):
        
      print(f'Client {i}',range(epochs),clients_accuracy_train[i])
      axis[0, 0].plot(range(epochs),clients_accuracy_train[i] ) 
      axis[0, 0].set_title("{} Train Accuracy".format( methode),fontsize=10) 
      axis[0, 0].legend(['Edge {}'.format(j+1) for j in range(num_users)],fontsize=10)
      
      axis[0, 1].plot(range(epochs),  clients_loss_train[i]) 
      axis[0, 1].set_title("{} Train Loss".format( methode),fontsize=10) 
      axis[0, 1].legend(['Edge {}'.format(j+1)  for j in range(num_users)],fontsize=10)
      
  
      axis[1, 0].plot(range(epochs), clients_accuracy_test[i]) 
      axis[1, 0].set_title("{} Test Accuracy".format( methode),fontsize=10) 
      axis[1, 0].legend(['Edge {}'.format(j+1)  for j in range(num_users)],fontsize=10)
     
   
      axis[1, 1].plot(range(epochs),  clients_loss_test[i]) 
      axis[1, 1].set_title("{} Test Loss".format( methode),fontsize=10) 
      axis[1, 1].legend(['Edge {}'.format(j+1) for j in range(num_users)],fontsize=10)
      
    elif(id==2):
     for i in range(num_users):
      print('iiiii',i,range(num_users))  
      print(f'Client {i}',range(epochs),clients_accuracy_train[i])
      axis[0, 0].plot(range(epochs),clients_accuracy_train[i] ) 
      axis[0, 0].set_title("{} Train Accuracy".format( methode),fontsize=10) 
      axis[0, 0].legend(['Edge {}'.format(id+j+1) for j in range(num_users)],fontsize=10)
      
      axis[0, 1].plot(range(epochs),  clients_loss_train[i]) 
      axis[0, 1].set_title("{} Train Loss".format( methode),fontsize=10) 
      axis[0, 1].legend(['Edge {}'.format(id+j+1) for j in range(num_users)],fontsize=10)
      
  
      axis[1, 0].plot(range(epochs), clients_accuracy_test[i]) 
      axis[1, 0].set_title("{} Test Accuracy".format( methode),fontsize=10) 
      axis[1, 0].legend(['Edge {}'.format(id+j+1)  for j in range(num_users)],fontsize=10)
     
   
      axis[1, 1].plot(range(epochs),  clients_loss_test[i]) 
      axis[1, 1].set_title("{} Test Loss".format( methode),fontsize=10) 
      axis[1, 1].legend(['Edge {}'.format(id+j+1) for j in range(num_users)],fontsize=10)
    else :
      for i in range(num_users):
       print(f'Client {i}',range(epochs),clients_accuracy_train[i])
       axis[0, 0].plot(range(epochs),clients_accuracy_train[i] ) 
       axis[0, 0].set_title("{} Train Accuracy".format( methode),fontsize=10) 
       axis[0, 0].legend(['Edge {}'.format(id+j+2) for j in range(num_users)],fontsize=10)
      
       axis[0, 1].plot(range(epochs),  clients_loss_train[i]) 
       axis[0, 1].set_title("{} Train Loss".format( methode),fontsize=10) 
       axis[0, 1].legend(['Edge {}'.format(id+j+2) for j in range(num_users)],fontsize=10)
       
  
       axis[1, 0].plot(range(epochs), clients_accuracy_test[i]) 
       axis[1, 0].set_title("{} Test Accuracy".format( methode),fontsize=10) 
       axis[1, 0].legend(['Edge {}'.format(id+j+2)  for j in range(num_users)],fontsize=10)
     
   
       axis[1, 1].plot(range(epochs),  clients_loss_test[i]) 
       axis[1, 1].set_title("{} Test Loss".format( methode),fontsize=10) 
       axis[1, 1].legend(['Edge {}'.format(id+j+2) for j in range(num_users)],fontsize=10)


    plt.xlabel("FL rounds ")
    plt.savefig(f'save/fog{id}{methode}.png')
    plt.show() 