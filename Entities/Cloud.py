
import matplotlib
matplotlib.use('Agg')
import copy
import numpy as np
from Aggregation.FedAVG import FedAvg
from Aggregation.FedGA import FedGA
from Aggregation.FedPer import FedPer
from Aggregation.FedPerGA import FedPerGA



class Cloud(object):

    def __init__(self, clients, global_model,dataset,args):
        self.clients_list=clients
        self.global_model=global_model
        self.weights_global=self.global_model.state_dict()
        self.weights_previous=self.global_model.state_dict() # we need it for FedPer
        self.dataset=dataset      #used for fedGA
        self.args=args
        self.method_name=args.aggr 
        self.loss_locals_train=[[0 for _ in range(self.args.num_users)] for _ in range(self.args.epochs+1)]
        self.accuracy_locals_train=[[0 for _ in range(self.args.num_users)] for _ in range(self.args.epochs+1)]
        self.loss_locals_test=[[0 for _ in range(self.args.num_users)] for _ in range(self.args.epochs+1)]
        self.accuracy_locals_test=[[0 for _ in range(self.args.num_users)] for _ in range(self.args.epochs+1)]
              
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


        self.i=0
        if (self.method_name=='fedAVG'):
              self.weights_global=FedAvg(self.weights_locals)
              print('Global Length',len(self.weights_global))
        elif (self.method_name=='fedGA'):
             initial_population=self.weights_locals
        
             for d in self.weights_locals: # for each user
                weight=[]
                if isinstance(d, dict):
                     for x in d.items():  #get weights of each layer
                         array = np.array(x[1], dtype='f')#1 is a tensor
                         array= array.flatten()
                         weight= np.concatenate((weight, array), axis=0)
                         initial_population[ self.i]= np.array(weight,dtype='f')
                self.i= self.i+1 # next weight vector (user)
             self.weights_global = FedGA(initial_population,self.global_model,self.dataset)

        elif (self.method_name=='fedPer'):

             self.weights_global=FedPer(self.weights_locals, self.global_model)

             #self.weights_previous=self.global_model.state_dict()
             #self.weights_previous.update(self.weights_global)
             #self.weights_global= self.weights_previous

        elif (self.method_name=='fedPerGA'):
            
              #for i in range(len(self.weights_locals)):
                  #self.weights_previous=copy.deepcopy(self.global_model.state_dict())
                  #self.weights_previous.update( self.weights_locals[i])
                  #self.weights_locals[i]=copy.deepcopy( self.weights_previous)
               
              initial_population=self.weights_locals #machi kamline
             
           
       
              for d in self.weights_locals: # for each user
                print(len(d))
                weight=[]
                if isinstance(d, dict):
                  try:
                     for x in d.items():  #get weights of each layer
                         array = np.array(x[1], dtype='f')#1 is a tensor
                         array= array.flatten()
                         weight= np.concatenate((weight, array), axis=0)
                         initial_population[ self.i]= np.array(weight,dtype='f')

                  except:
                      
                      for x in d.items():  #get weights of each layer                                       
                         array = np.array(x[1].cpu(), dtype='f')#1 is a tensor           
                         array= array.flatten()      
                         weight= np.concatenate((weight, array), axis=0)
                         initial_population[ self.i]= np.array(weight,dtype='f')
                self.i= self.i+1 # next weight vector (user)
              

              self.weights_global = FedPerGA(initial_population,self.global_model.classification,self.dataset)

        if (self.args.aggr=='fedAVG'):
            print('hh')
            #self.global_model.load_state_dict(self.weights_global)
        else :

            self.global_model.classification.load_state_dict(self.weights_global)

        return self.global_model
    
    def Launch_local_updates(self,iter):
        self.global_model.train() 
        self.Per_weights=[]

        if (self.args.aggr=='fedAVG'):

            self.net=copy.deepcopy(self.global_model.state_dict())

        else:
            self.net=copy.deepcopy(self.global_model.classification.state_dict())        
      
        for id in range(len(self.clients_list)):
            
            if (iter==0):
                  w, loss =  self.clients_list[id].local_updateFirst()

            else :
            
            
             if (self.method_name=='fedPer'):
                
                  w, loss =  self.clients_list[id].local_updatePer(self.net)  #self.net== classification layer

             elif (self.method_name=='fedPerGA'): 

                # self.net=copy.deepcopy(self.global_model.state_dict())
                 w, loss =  self.clients_list[id].local_updatePer(self.net) #self.net== classification layer

                 #self.net.update(w)
                # w=self.net

             else : # for fedAVG
                 w, loss =  self.clients_list[id].local_update(self.weights_global)

            acc, loss = self.clients_list[id].test_img('train')
            #acc = self.clients_list[id].Trainaccuracy
            #print('loss {} and accuracy {}'.format(loss,acc))
            self.loss_locals_train[iter][id]=loss
            self.accuracy_locals_train[iter][id]=acc
            #print("Training accuracy for client {} is : {:.2f}".format(id,acc))
            acc, loss = self.clients_list[id].test_img('test')
            #acc =self.clients_list[id].Testaccuracy
            self.loss_locals_test[iter][id]=loss
            self.accuracy_locals_test[iter][id]=acc
           # print("Testing accuracy for client {} is : {:.2f}".format(id,acc))
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







    def Launch_local_updatesClassic(self,iter):
        

        for id in range(len(self.clients_list)):
            
            
            w, loss =  self.clients_list[id].local_updateFirst()
            acc, loss = self.clients_list[id].test_img('train')
            #acc = self.clients_list[id].Trainaccuracy
            #print('loss {} and accuracy {}'.format(loss,acc))
            self.loss_locals_train[iter][id]=loss
            self.accuracy_locals_train[iter][id]=acc
            #print("Training accuracy for client {} is : {:.2f}".format(id,acc))
            acc, loss = self.clients_list[id].test_img('test')
            #acc =self.clients_list[id].Testaccuracy
            self.loss_locals_test[iter][id]=loss
            self.accuracy_locals_test[iter][id]=acc
        

            self.loss_locals.append(copy.deepcopy(loss))

        loss_avg = sum(self.loss_locals) / len(self.loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        self.loss_train.append(loss_avg)

        return self.weights_locals, self.loss_locals_train, self.loss_locals_test,self.accuracy_locals_train,self.accuracy_locals_test
        



                   


           


