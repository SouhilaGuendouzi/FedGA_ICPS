import numpy as np
import matplotlib.pyplot as plt


class Plot(object):

    def __init__(self,args,accTRAIN,accTEST, lossTRAIN, lossTEST):#,device
      self.args=args
      self.test_acc=accTEST
      self.test_loss=lossTEST
      self.train_acc=accTRAIN
      self.train_loss=lossTRAIN 


    def get_graph_train(self,eval,method):
        client1, client2, client3, client4=[0 for _ in range(self.args.epochs)], [0 for _ in range(self.args.epochs)], [0 for _ in range(self.args.epochs)], [0 for _ in range(self.args.epochs)]
    

        x=range(self.args.epochs)
        if (eval=='accuracy'):
           for i in range(self.args.epochs):
    
              client1[i]= self.train_acc[i][0]
              client2[i]= self.train_acc[i][1]
              client3[i]= self.train_acc[i][2]
              client4[i]= self.train_acc[i][3]

           plt.legend(["Client 1", "Client 2", "Client 3", "Client 4"])
           plt.ylabel('{} Train Accuracy'.format(method))
           plt.xlabel('Communication rounds')
           plt.savefig('./save/{}_train_accuracy.png'.format(method))

        elif (eval=='loss'):
           for  i in range(self.args.epochs):
              client1[i]= self.train_loss[i][0]
              client2[i]= self.train_loss[i][1]
              client3[i]= self.train_loss[i][2]
              client4[i]= self.train_loss[i][3]
           plt.legend(["Client 1", "Client 2", "Client 3", "Client 4"])
           plt.ylabel('{} Train Loss'.format(method))
           plt.xlabel('Communication rounds')
           plt.savefig('./save/{}_train_loss.png'.format(method))
        plt.figure()
        plt.plot(x,client1)
        plt.plot(x,client2)
        plt.plot(x,client3)
        plt.plot(x,client4)


    def get_graph_test(self,eval,method):
   
        client1, client2, client3, client4=[0 for _ in range(self.args.epochs)], [0 for _ in range(self.args.epochs)], [0 for _ in range(self.args.epochs)], [0 for _ in range(self.args.epochs)]
        x=range(self.args.epochs)
        if (eval=='accuracy'):
            for i in range(self.args.epochs):
              client1[i]= self.test_acc[i][0]
              client2[i]= self.test_acc[i][1]
              client3[i]= self.test_acc[i][2]
              client4[i]= self.test_acc[i][3]
            plt.legend(["Client 1", "Client 2", "Client 3", "Client 4"])
            plt.ylabel('{} test Accuracy'.format(method))
            plt.xlabel('Communication rounds')
            plt.savefig('./save/{}_test_accuracy.png'.format(method))
        elif (eval=='loss'):
           for i in range(self.args.epochs):
              client1[i]= self.test_loss[i][0]
              client2[i]= self.test_loss[i][1]
              client3[i]= self.test_loss[i][2]
              client4[i]= self.test_loss[i][3]
           plt.legend(["Client 1", "Client 2", "Client 3", "Client 4"])
           plt.ylabel('{} test Loss'.format(method))
           plt.xlabel('Communication rounds')
           plt.savefig('./save/{}_test_loss.png'.format(method))
        plt.figure()
        plt.plot(x,client1)
        plt.plot(x,client2)
        plt.plot(x,client3)
        plt.plot(x,client4)
        