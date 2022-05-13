import numpy as np
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4




def accuracy_train (args, accTRAIN):
    clients=[[0 for _ in range(args.epochs)] for _ in range(args.num_users)]

    x=range(args.epochs)


    for i in range(args.epochs):
      for j in range(args.num_users):
         clients[j][i]= accTRAIN[i][j]
              
      
   #plt.ylim([0, 100])
    plt1.legend(['Client {}'.format(j) for j in range(args.num_users)])
    plt1.ylabel('{} Train Accuracy'.format(args.aggr))
    plt1.xlabel('Communication rounds')
    plt1.savefig('./save/{}_train_accuracy_{}.png'.format(args.aggr,args.iid))
    for j in range(args.num_users):
           plt1.plot(x,clients[j])

    #plt.figure()


def loss_train (args, lossTRAIN):
    clients=[[0 for _ in range(args.epochs)] for _ in range(args.num_users)]

    x=range(args.epochs)


    for i in range(args.epochs):
      for j in range(args.num_users):
         clients[j][i]= lossTRAIN[i][j]
              
      
    #plt.ylim([0, 100])
    plt2.legend(['Client {}'.format(j) for j in range(args.num_users)])
    plt2.ylabel('{} Train Loss'.format(args.aggr))
    plt2.xlabel('Communication rounds')
    plt2.savefig('./save/{}_train_loss_{}.png'.format(args.aggr,args.iid))
    for j in range(args.num_users):
            plt2.plot(x,clients[j])

    #plt.figure()
def accuracy_test (args, accTest):
    clients=[[0 for _ in range(args.epochs)] for _ in range(args.num_users)]

    x=range(args.epochs)


    for i in range(args.epochs):
      for j in range(args.num_users):
         clients[j][i]= accTest[i][j]
              
      
   #plt.ylim([0, 100])
    plt3.legend(['Client {}'.format(j) for j in range(args.num_users)])
    plt3.ylabel('{} Test Accuracy'.format(args.aggr))
    plt3.xlabel('Communication rounds')
    plt3.savefig('./save/{}_test_accuracy_{}.png'.format(args.aggr,args.iid))
    for j in range(args.num_users):
           plt3.plot(x,clients[j])
    #plt.figure()

def loss_test (args, lossTest):
    clients=[[0 for _ in range(args.epochs)] for _ in range(args.num_users)]

    x=range(args.epochs)


    for i in range(args.epochs):
      for j in range(args.num_users):
         clients[j][i]= lossTest[i][j]
              
      
    #plt.ylim([0, 100])
    plt4.legend(['Client {}'.format(j) for j in range(args.num_users)])
    plt4.ylabel('{} Test Loss'.format(args.aggr))
    plt4.xlabel('Communication rounds')
    plt4.savefig('./save/{}_test_loss_{}.png'.format(args.aggr,args.iid))
    
        
    for j in range(args.num_users):
           plt4.plot(x,clients[j])

    #plt.figure()




def Plot_table(data,col):
        
      fig, ax =plt.subplots(1,1)
      data=data
      column_labels=col
      ax.axis('tight')
      ax.axis('off')
      ax.table(cellText=data,colLabels=column_labels,loc="center")
      plt.savefig('./tables/results')

      plt.figure()