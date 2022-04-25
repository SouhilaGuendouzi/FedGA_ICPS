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

        x=range(self.args.epochs)
        if (eval=='accuracy'):
           client1= self.train_acc[0]
           client2=self.train_acc[1]
           client3=self.train_acc[2]
           client4=self.train_acc[3]
           plt.legend(["Client 1", "Client 2", "Client 3", "Client 4"])
           plt.ylabel('{} Train Accuracy'.format(method))
           plt.xlabel('Communication rounds')
           plt.savefig('./save/{}_train_accuracy.png'.format(method))
        elif (eval=='loss'):
           client1= self.train_loss[0]
           client2=self.train_loss[1]
           client3=self.train_loss[2]
           client4=self.train_loss[3]
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

        x=range(self.args.epochs)
        if (eval=='accuracy'):
           client1= self.test_acc[0]
           client2=self.test_acc[1]
           client3=self.test_acc[2]
           client4=self.test_acc[3]
           plt.legend(["Client 1", "Client 2", "Client 3", "Client 4"])
           plt.ylabel('{} test Accuracy'.format(method))
           plt.xlabel('Communication rounds')
           plt.savefig('./save/{}_test_accuracy.png'.format(method))
        elif (eval=='loss'):
           client1= self.test_loss[0]
           client2=self.test_loss[1]
           client3=self.test_loss[2]
           client4=self.test_loss[3]
           plt.legend(["Client 1", "Client 2", "Client 3", "Client 4"])
           plt.ylabel('{} test Loss'.format(method))
           plt.xlabel('Communication rounds')
           plt.savefig('./save/{}_test_loss.png'.format(method))
        plt.figure()
        plt.plot(x,client1)
        plt.plot(x,client2)
        plt.plot(x,client3)
        plt.plot(x,client4)
        