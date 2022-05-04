import numpy as np
import matplotlib.pyplot as plt


class Plot(object):

    def __init__(self,args,accTRAIN,accTEST, lossTRAIN, lossTEST):#,device
      self.args=args
      self.test_acc=accTEST
      self.test_loss=lossTEST
      self.train_acc=accTRAIN
      self.train_loss=lossTRAIN 
      self.clients=[[0 for _ in range(self.args.epochs)] for _ in range(self.args.num_users)]


    def get_graph_train(self,eval,method):
        
    

        x=range(self.args.epochs)
        if (eval=='accuracy'):

           for i in range(self.args.epochs):
              for j in range(self.args.num_users):
                 self.clients[j][i]= self.train_acc[i][j]
              
      
           #plt.ylim([0, 100])
           plt.legend(['Client {}'.format(j) for j in range(self.args.num_users)])
           plt.ylabel('{} Train Accuracy'.format(method))
           plt.xlabel('Communication rounds')
           plt.savefig('./save/{}_train_accuracy_{}.png'.format(method,self.args.iid))

        elif (eval=='loss'):
           for  i in range(self.args.epochs):
             for j in range(self.args.num_users):
                 self.clients[j][i]= self.train_acc[i][j]
           plt.legend(['Client {}'.format(j) for j in range(self.args.num_users)])
           plt.ylabel('{} Train Loss'.format(method))
           plt.xlabel('Communication rounds')
           plt.savefig('./save/{}_train_loss_{}.png'.format(method,self.args.iid))
        plt.figure()
        for j in range(self.args.num_users):
           plt.plot(x,self.clients[j])
       


    def get_graph_test(self,eval,method):
   
        
        x=range(self.args.epochs)
        if (eval=='accuracy'):
            for i in range(self.args.epochs):
              for j in range(self.args.num_users):
                 self.clients[j][i]= self.train_acc[i][j]
            plt.legend(['Client {}'.format(j) for j in range(self.args.num_users)])
            plt.ylabel('{} test Accuracy'.format(method))
            plt.xlabel('Communication rounds')
            #plt.ylim([0, 100])
            plt.savefig('./save/{}_test_accuracy_{}.png'.format(method,self.args.iid))
        elif (eval=='loss'):
           for i in range(self.args.epochs):
              for j in range(self.args.num_users):
                 self.clients[j][i]= self.train_acc[i][j]
           plt.legend(['Client {}'.format(j) for j in range(self.args.num_users)])
           plt.ylabel('{} test Loss'.format(method))
           plt.xlabel('Communication rounds')
           plt.savefig('./save/{}_test_loss_{}.png'.format(method,self.args.iid))
        plt.figure()
        
        for j in range(self.args.num_users):
           plt.plot(x,self.clients[j])
        
    
    def Plot_table(data,col):
        
      fig, ax =plt.subplots(1,1)
      data=data
      column_labels=col
      ax.axis('tight')
      ax.axis('off')
      ax.table(cellText=data,colLabels=column_labels,loc="center")
      plt.savefig('./tables/results')

      plt.figure()