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
        self.server.setblocking(True)



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
  
                    self.active_fogs[i][3]=message.data
                    
                    #threading.Thread(target=#, args=()).start()
                    self.registryUpdate()
                    self.FLAggregation()


                 elif (message.subject=="RequestTLModel"):
                  print(message.data)
                  threading.Thread(target= self.SearchANdRequestTlModel, args=(message.data,)).start()
                  
                 elif (message.subject=="TLModel"):
                    print(message.data)
                    threading.Thread(target=  self.TransferTLModelToFog, args=(message.data,)).start()

                  
                except Exception as e:
                  print('Exception from listen_for_messages',e)
              i=i+1
          self.send_message_to_fog(fog,"FOG  ~~ Successful Demand Received ","ACK")


        else:
            print(f"The message send from Fog {username} is empty")
#*****************************************************************************************#   

    def receive_fogs(self):

      
       while 1:
          fog, address = self.server.accept()
          threading.Thread(target=self.Fog_handler, args=(fog, address)).start()

#*****************************************************************************************#   
 
    def SearchANdRequestTlModel(self,message) :
      print('I am searching ...')
      stop =False
      if (message !=''): #message=[idEdge,addressEdge,domain, task]
        i =0
        while (i< len(self.registry) and stop==False):
          if (self.registry[i][4]==message[2] and self.registry[i][5] ==message[3]):  #same domain and task
            stop=True
            print('I find same domain and task  ...')
            obj = [message[0],self.registry[i][0]] #IdEdge (that need the model), IdEdge_cible( that contains the complete model)
            self.send_message_to_fog(self.registry[i][1],obj,"RequestTLModel")  #socketFog\message.subject="RequestTLModel" \message.data=idEdge that contains the model
          else :  #the case that it doesn't find the appropriate model
              i+=1
        if (stop==False):  
            print('we didnot find the appropriate model')

#*****************************************************************************************#   
    def TransferTLModelToFog(self,message):  
      id_user=message[0]
      model = message[1]   
      stop =False 
      i,j =0,0
      while (i< len(self.active_fogs) and stop==False):
        j=0
        while (j< len(self.active_fogs[i][3]) and stop==False):
          if (self.active_fogs[i][3][j][0]==id_user):
            fogSocket=self.active_fogs[i][1]
            print('I am transferring the model')
            self.send_message_to_fog(fogSocket,message,"TLModel")
            stop=True
          j+=j
        i+=i


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
      if ( self.numberFogsreceived==len(self.active_fogs)):
          #print('Registry !!!!!!!!!',self.registry[0])
          print('registry update')
          if (len(self.registry)==0):
            print(self.active_fogs[0][3][0][0])
            print(self.active_fogs[0][1])
            print(self.active_fogs[0][3][0][1])
            print(self.active_fogs[0][3][0][2])
            print(self.active_fogs[0][3][0][4])
            print(self.active_fogs[0][3][0][5])
            self.registry.append([self.active_fogs[0][3][0][0],self.active_fogs[0][1],self.active_fogs[0][3][0][1],self.active_fogs[0][3][0][2],self.active_fogs[0][3][0][4],self.active_fogs[0][3][0][5]]) #idedge,socketFog, addressEdge, avgaccuracy,domain , task
          #print(len(self.active_fogs[0][3]),len(self.active_fogs[0][3]))
          for fog in self.active_fogs:
              #print(f'Fog {fog[0]}')
              for usr in fog[3]:
                #print(f'user {usr[0]}')
                i=0
                stop=False
                while (i< len(self.registry) and stop==False):
                  #print(f'user{usr[0]} comparing with user {[self.registry[i][0]]} ',usr[4],self.registry[i][4],usr[5],self.registry[i][5], usr[2],self.registry[i][3])
                  #print(usr[4]==self.registry[i][3] , usr[5]==self.registry[i][4])
                  if (usr[4]==self.registry[i][4] and usr[4]==self.registry[i][4] and usr[2]>self.registry[i][3] ): #same domain and task but accuracy >>
                    #print(f'Replace {fog  [0]},{usr[0]}')
                    stop=True
                    self.registry[i][0]=usr[0]
                    self.registry[i][1]=fog[1]  #socket
                    self.registry[i][2]=usr[1]
                    self.registry[i][3]=usr[2]
                    self.registry[i][4]=usr[4]
                    self.registry[i][5]=usr[5]
                  elif (usr[4]==self.registry[i][4] and usr[4]==self.registry[i][4] and usr[2]==self.registry[i][3] ): #same domain and task and accuracy 
                    #print('same')
                    stop=True
                  i+=1
                  
                if (i==len(self.registry) and stop==False):  
                  #print(f'Append {fog [0]},{usr[0]}')
                  self.registry.append([usr[0],fog[1],usr[1],usr[2],usr[4],usr[5]])
             
      
#*****************************************************************************************# 

    def start_FL(self):
      self.FLrounds=self.args.epochs
      self.numberFogsreceived=0
      self.add_message("Starting FL \n")
      self.send_messages_to_all(self.args.aggr,"FLstart")
    
#*****************************************************************************************# 
    def FLAggregation(self):
      
       if ( self.numberFogsreceived==len(self.active_fogs)):
        
         self.FLrounds-=1
         self.add_message("Aggregation Round"+ str( self.args.epochs-self.FLrounds)+"\n")
         local_weights=[]
         for fog in self.active_fogs:
              #print(f'Fog {fog[0]}')
              for usr in fog[3]:
                #print(f'user {usr[0]}') 
                local_weights.append(usr[3]) #local models weights

         self.aggregate(local_weights,self.args.aggr)
         #print(self.weights_global)
         if (self.FLrounds==0):
            self.send_messages_to_all(self.weights_global,"FLEnd")
            self.numberFogsreceived=0    
            self.add_message("Sending Last Global Model ")
         
         else : self.send_messages_to_all(self.weights_global,"FL")
         self.numberFogsreceived=0          


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


#*****************************************************************************************# 

    def main(self):

        self.root.mainloop()
#*****************************************************************************************#   
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
    