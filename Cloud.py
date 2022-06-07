# Import required modules


import socket
import threading
import pickle
import tkinter as tk
from utils.CloudOptions import args_parser
import torch
from utils.Empty import Empty
from Aggregation.FedAVG import *
from Aggregation.FedPer import *
from Aggregation.FedPerGA import *
from Aggregation.FedGA import *
from Entities.Model import Model_Fashion
from utils.create_MNIST_datasets import get_FashionMNIST
import numpy as np

HOST = '127.0.0.1'
PORT = 12345 # You can use any port between 0 to 65535
LISTENER_LIMIT = 5



# Function to listen for upcoming messages from a client
class Cloud:
    def __init__(self,Host,args,model,dataset):
        self.args=args

        self.HOST=Host
        self.PORT=args.myport
        self.LISTENER_LIMIT=args.LISTENER_LIMIT
        self.active_fogs = []
        self.numberFogsreceived=0
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # IPv4 addresses, TCP



        self.dataset=dataset      #used for fedGA
        self.global_model= model

        self.registry=[]

        self.localmodels=[]
        self.FLrounds=args.epochs
        self.aggregation=args.aggr
        self.weights_global=[]
        
          
        self.Actuator=False
      
        
        self.scoring=0
        self.pyhical_attributes={}      
        
       



        ###################### UI #####################################################
        self.root =tk.Tk()
        self.root.geometry("600x600")
        self.root.title(" Cloud Interface ") 
        self.l = tk.Label(text = "Cloud Server")
        self.l.pack()
        self.inputtxt = tk.Text(self.root, height = 25,
                width = 60,
                bg = "light yellow")

        self.inputtxt.pack()
        self.inputtxt.configure(state='disabled')
       

        self.Start_FL = tk.Button(self.root, height = 2,
                 width = 20,
                 text ="Start FedGA-ICPS",
                 command = lambda:self.start_FL()
                 )
        self.Start_FL.pack(padx=200, pady=10, side=tk.LEFT)
        try:
          self.server.bind((HOST, PORT))
          print(f"Running the server on {HOST} {PORT}")
          self.add_message(f"Running the server on {HOST} {PORT} \n")
        except Exception as e:
           print(f"Unable to bind to host {HOST} and port {PORT} because of {e}")

        
    #*****************************************************************************************#
    def add_message(self,message):
     
       self.inputtxt.config(state=tk.NORMAL)
       self.inputtxt.insert(tk.END, message )
       self.inputtxt.config(state=tk.DISABLED)
#*****************************************************************************************#
    def send_message_to_fog(self,fog,message,subject):
       try :
   
        if message != '':
           obj= Empty()
           obj.data=message
           obj.subject=subject
           message=obj
           message = pickle.dumps(message)
           fog.send(message)
      
        else:
           print("Empty message", "Message cannot be empty")
       except Exception as e : 
           print(e)
    #*****************************************************************************************#
    def send_messages_to_all(self,message,subject):  
    
       for user in self.active_fogs:

          self.send_message_to_fog(user[1], message,subject)
    #*****************************************************************************************#
    def listen_for_messages_from_fog(self,fog, id):
    
      while 1:
        
        message = fog.recv(1000000)#.decode('utf-8')
        message=pickle.loads(message)
        i=0
        Find=False
        
        if message != '':
          self.add_message(' Message received From Fog'+ str(id)+' About '+message.subject  + ' \n')

          while (i<len(self.active_fogs) and Find==False):
              username="Fog "+str(id)
              if (self.active_fogs[i][0]==id): ### search about Fog
                
                try:
                 Find=True
                 print(message.subject)
                 if (message.subject=="LocalModels"):
                    
                    self.numberFogsreceived+=1
                    self.active_fogs[i][2]=message.data
                    
                    threading.Thread(target=self.registryUpdate, args=()).start()
                    threading.Thread(target=self.FLAggregation, args=()).start()


                 elif (message.subject=="RequestTLModel"):

                  threading.Thread(target= self.SearchANdRequestTlModel(), args=()).start()
                  
                 elif (message.subject=="TLModel"):
                    threading.Thread(target=  self.TransferTLModelToFog(), args=()).start()

                  
                except Exception as e:
                  print('Exception from listen_for_messages',e)
              i=i+1
          self.send_message_to_fog(fog,"FOG  ~~ Successful Demand Received ","ACK")


        else:
            print(f"The message send from Fog {username} is empty")
    #*****************************************************************************************#   

    def main(self):

        self.root.mainloop()
    
    #*****************************************************************************************#   
    def receive_fogs(self):

      
       while 1:
          fog, address = self.server.accept()
          threading.Thread(target=self.Fog_handler, args=(fog, address)).start()

#*****************************************************************************************#    
    def Fog_handler(self,fog,address):  
  
      while 1:
          existedFog=False
          i=0
          msg= fog.recv(1000000)#.decode('utf-8')
          msg=pickle.loads(msg)
          id=msg.data
          print(id)
          if str(id) != '':
            while (i<len(self.active_fogs) and existedFog==False):
               if (self.active_fogs[i][0]==id): existedFog=True
               else: i=i+1
            if (existedFog==False):
               self.active_fogs.append([id, fog,address, []])  #id, socket, address, (list of ==> [idEdge,address,accuracy,persoModel,domain,task])
               print( "" + f" Fog {id} added to the System")
               self.add_message(f"Fog {id} added to the System \n")
               #self.send_message_to_fog(fog,"Server ~~ Successfully connected to Cloud Server ")
               break
            else:
               self.active_fogs[i][1]= fog
               self.active_fogs[i][2]=address
  
               print( "" + f" Fog {id} reconnected to the System")
               self.add_message(f"Fog {id}  reconnected to the System \n")
          else:
            print("Fog username is empty")
     
      threading.Thread(target=self.listen_for_messages_from_fog, args=(fog, id, )).start()
#*****************************************************************************************# 
    def registryUpdate(self):
      if ( self.numberFogsreceived==self.args.num_fogs):
          
          if (len(self.registry)==0):
            #print(self.active_fogs[0][2][0][0])
            #print(self.active_fogs[0][2][0][1])
            #print(self.active_fogs[0][2][0][2])
            #print(self.active_fogs[0][2][0][4])
            #print(self.active_fogs[0][2][0][5])
            self.registry.append([self.active_fogs[0][2][0][0],self.active_fogs[0][2][0][1],self.active_fogs[0][2][0][2],self.active_fogs[0][2][0][4],self.active_fogs[0][2][0][5]]) #id, address, avgaccuracy,domain , task
          for fog in self.active_fogs:
              print(f'Fog {fog[0]}')
              for usr in fog[2]:
                print(f'user {usr[0]}')
                i=0
                while (i< len(self.registry)):
                  if (usr[4]==self.registry[i][3] and usr[5]==self.registry[i][4] and usr[2]>self.registry[i][2] ): #same domain and task but accuracy >>
                    self.registry[i]=[usr[0],usr[1],usr[2],usr[4],usr[5]] #id, address, avgaccuracy, domain, task
                  else: i+=1
                if (i==len(self.registry)):  self.registry.append([usr[0],usr[1],usr[2],usr[4],usr[5]])
      print(self.registry)
#*****************************************************************************************# 

    def start_FL(self):
      self.FLrounds=self.args.epochs
      self.receive_fogs=0
      self.add_message("Starting FL \n")
      self.send_messages_to_all(self.args.aggr,"FLstart")
    
#*****************************************************************************************# 
    def FLAggregation(self):
       self.FLrounds-=1
       if ( self.numberFogsreceived==self.args.num_fogs):
         local_weights=[]
         for fog in self.active_fogs:
              print(f'Fog {fog[0]}')
              for usr in fog[2]:
                print(f'user {usr[0]}') 
                local_weights.append(usr[3]) #local models weights

         self.aggregate(local_weights,self.args.aggr)
         print(self.weights_global)
         if (self.FLrounds==0): self.send_messages_to_all(self.weights_global,"FLEnd")
         
         else : self.send_messages_to_all(self.weights_global,"FL")
          


#*****************************************************************************************# 

    def aggregate(self,weights_clients,method_name):
        self.method_name=method_name
        self.weights_locals=weights_clients


        self.i=0
        if (self.method_name=='FedAVG'):
              self.weights_global=FedAvg(self.weights_locals)
        elif (self.method_name=='FedGA'):
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

        elif (self.method_name=='FedPer'):

             self.weights_global=FedPer(self.weights_locals, self.global_model)

        elif (self.method_name=='FedPerGA'):
            
              initial_population=self.weights_locals #machi kamline
              for d in self.weights_locals: # for each user
       
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

        if (self.args.aggr=='FedAVG'):
             self.global_model.load_state_dict(self.weights_global)
        else :

            self.global_model.classification.load_state_dict(self.weights_global)

        return self.global_model




if __name__ == '__main__':
  
    args = args_parser()   # ajoute id 
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #print(torch.cuda.is_available())
    train, test = get_FashionMNIST('iid',
    n_samples_train =1500, n_samples_test=250, n_clients =2,  
    batch_size =50, shuffle =True)


    cloud =Cloud(Host=HOST,args=args,model=Model_Fashion(),dataset=test[0])

    cloud.server.listen(LISTENER_LIMIT)
    threading.Thread(target=cloud.receive_fogs, args=()).start()
    cloud.main()
    