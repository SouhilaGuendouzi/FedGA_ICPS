
import matplotlib
matplotlib.use('Agg')
import copy
import numpy as np
from Aggregation.FedAVG import FedAvg
from Aggregation.FedGA import FedGA
from Aggregation.FedPer import FedPer



class Cloud(object):

    def __init__(self, clients, global_model,dataset,args):
        self.clients_list=clients
        self.global_model=global_model
        self.weights_global=self.global_model.state_dict()
        self.dataset=dataset      #used for fedGA
        self.args=args
        self.method_name='fedAVG' #by default
        self.loss_locals_train=[[0 for _ in range(self.args.num_users)] for _ in range(self.args.epochs)]
        self.accuracy_locals_train=[[0 for _ in range(self.args.num_users)] for _ in range(self.args.epochs)]
        self.loss_locals_test=[[0 for _ in range(self.args.num_users)] for _ in range(self.args.epochs)]
        self.accuracy_locals_test=[[0 for _ in range(self.args.num_users)] for _ in range(self.args.epochs)]
       


        if self.args.all_clients: 
              print("Aggregation over all clients")
              self.weights_locals = [self.weights_global for i in range(args.num_users)]
        else :
               self.weights_locals=[]   #vector in vector
        self.loss_train = []
        self.loss_locals = []
       
    def aggregate(self,weights_clients,method_name):
        self.method_name=method_name
        self.weights_locals=weights_clients
       
        if (self.method_name=='fedAVG'):
              self.weights_global=FedAvg(self.weights_locals)
        elif (self.method_name=='fedGA'):
             initial_population=self.weights_locals
       
             for d in self.weights_locals: # for each user
                weight=[]
                if isinstance(d, dict):
                     for x in d.items():  #get weights of each layer
                         array = np.array(x[1], dtype='f')#1 is a tensor
                         array= array.flatten()
                         weight= np.concatenate((weight, array), axis=0)
                         initial_population[i]= np.array(weight,dtype='f')
                i=i+1 # next weight vector (user)
             self.weights_global = FedGA(initial_population,self.global_model,self.dataset)

        elif (self.method_name=='fedPer'):
             self.weights_global=FedPer(self.weights_locals)

        self.global_model.load_state_dict( self.weights_global)

        return self.global_model
    
    def Launch_local_updates(self,iter):
        self.global_model.train() 
        

        for id in range(len(self.clients_list)):
            if (self.method_name=='fedPer'):
                w, loss =  self.clients_list[id].local_updatePer( self.weights_global)
            else :
                w, loss =  self.clients_list[id].local_update( self.weights_global)

            acc, loss = self.clients_list[id].test_img(self.clients_list[id].datasetTrain)
            self.loss_locals_train[iter][id]=loss
            self.accuracy_locals_train[iter][id]=acc
            #print("Training accuracy for client {} is : {:.2f}".format(id,acc))
            acc, loss = self.clients_list[id].test_img(self.clients_list[id].datasetTest)
            self.loss_locals_test[iter][id]=loss
            self.accuracy_locals_test[iter][id]=acc
            #print("Testing accuracy for client {} is : {:.2f}".format(id,acc))
            if self.args.all_clients:
                self.weights_locals[id]=copy.deepcopy(w)
            else:
                self.weights_locals.append(copy.deepcopy(w))

            self.loss_locals.append(copy.deepcopy(loss))

        loss_avg = sum(self.loss_locals) / len(self.loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        self.loss_train.append(loss_avg)
        self.weights_locals= np.array(self.weights_locals)

        return self.weights_locals, self.loss_locals_train, self.loss_locals_test,self.accuracy_locals_train,self.accuracy_locals_test





                   


           


