import numpy as np
import matplotlib.pyplot as plt


class Plot(object):

    def __init__(self,list_accuracy):#,device
        self.list=list_accuracy


    def get_graph(self):
        plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_.png'.format(args.aggr, args.iid))  
