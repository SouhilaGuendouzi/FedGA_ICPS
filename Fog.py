# Import required modules


from queue import Empty
import socket
import threading
import pickle
from matplotlib.style import use
import numpy as np
import random
import torch
import time
import tkinter as tk
from Entities.Model import Model_Fashion
from utils.FogOptions import args_parser
from Aggregation.FedAVG import *
from Aggregation.FedPer import *
from Aggregation.FedPerGA import *
from Aggregation.FedGA import *
from utils.Plot import loss_test
from utils.create_MNIST_datasets import get_FashionMNIST
from utils.Graph import Plot_Graphes_for_fog
# Function to listen for upcoming messages from a client
class Fog:
    def __init__(self,args,HOST,HostCloud):

       
        self.HOST=HOST
        self.PORT=args.myport
        self.BackupPort=30303
        self.LISTENER_LIMIT=args.LISTENER_LIMIT
        self.HostCloud=HostCloud
        self.PortCloud=args.portCloud

        
        self.FLrounds=args.epochs


        self.HostAggregator=HostCloud
        self.PortAggregator=args.portCloud

        self.active_clients = []
        self.active_fogs=[]
        self.numberFogsreceived=0
       

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  #for clients
        self.cloud=socket.socket(socket.AF_INET, socket.SOCK_STREAM)   #for cloud
        self.aggregatorSocket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)   #for fog aggregator
        self.serverForFogs =socket.socket(socket.AF_INET, socket.SOCK_STREAM) # to get Fogs connections

        self.receivedLocalModels=0

        self.Actuator="Free"
        self.aggregator="Cloud"
        self.method_name="FedAVG"
 
        self.capacity=random.uniform(0,100) 
        self.priority=random.uniform(0,100)    

        self.id=args.id
        self.args=args

        self.accuracy_locals_train=[]
        self.accuracy_locals_test=[]
        self.loss_locals_train=[]
        self.loss_locals_test=[]
        self.roundGraphes=0


       
         

        #****************** Aggregation **********************#
        self.global_model=Model_Fashion()
        train, test = get_FashionMNIST('iid',
        n_samples_train =1500, n_samples_test=250, n_clients =4,  
        batch_size =50, shuffle =True)
        self.dataset=test[self.id]

        
        self.root =tk.Tk()
        self.root.geometry("600x600")
        self.root.title(" Fog Interface ") 
        self.l = tk.Label(text = f"Fog Server {self.id} ")
        self.l.pack()
        self.inputtxt = tk.Text(self.root, height = 25,
                width = 60,
                bg = "light yellow")

        self.inputtxt.pack()
        self.inputtxt.configure(state='disabled')
       
        self.Connect =tk.Button(self.root, height = 2,
                 width = 20,
                 text ="Connect",
                 #command=self.connect
                 command=self.connect
                 )

        self.Connect.pack(padx=100, pady=10, side=tk.LEFT)
        self.Upload = tk.Button(self.root, height = 2,
                 width = 20,
                 text ="Upload",
                 command = lambda:self.sendLocalModels("Cloud")
                 )
        self.Upload.pack(padx=5, pady=20, side=tk.LEFT)
        
       
        self.Start = tk.Button(self.root, height = 2,
                 width = 20,
                 text ="Start FL",
                 command = self.start_FL)
             
         
        self.Start.pack(padx=5, pady=20, side=tk.LEFT)
        try:
          self.server.bind((self.HOST, self.PORT))
          print(f"Running the server on {self.HOST} {self.PORT}")
          self.server.listen(self.LISTENER_LIMIT)
          self.add_message(f"Running the server on {self.HOST} {self.PORT} \n")
        except:
           print(f"Unable to bind to host {self.HOST} and port {self.PortCloud}")
           self.add_message(f"Unable to bind to host {self.HOST} and port {self.PortCloud} \n")
    

#*************************Aggregator setup *************************#

    def connect_to_aggregator(self,adr,port):
       Var= False
   
       try:
           self.aggregatorSocket.connect((adr, port))
           self.add_message(f"Successfully connected to Aggregator server {adr}:{port}\n")
           self.send_message_to_aggregator(self.id,'Connection')
           Var= True

       except Exception as e:
        print(e)
        
    
       threading.Thread(target=self.listen_for_messages_from_aggregator, args=(self.aggregatorSocket, )).start()
    
       return Var
   
    
    def listen_for_messages_from_aggregator(self,socket):

       while 1:
         
        message = socket.recv(1000000)#.decode('utf-8')
        message=pickle.loads(message)
        if message != '':
           try:
             
                  self.add_message('Aggregator Server is requesting For: '+message.subject+"\n")
                
                  if (message.subject=='FLstart'):
                     self.receivedLocalModels=0
              
                     self.send_messages_to_all(message.data, "FLstart")

                  elif  (message.subject=='FL'):
                   
                      self.receivedLocalModels=0
                      self.add_message('Cloud server is requesting for an Other  FL Round \n')
                      self.send_messages_to_all(message.data, "FL")
                  elif  (message.subject=='FLEnd'):
                     
                      self.receivedLocalModels=0
                      self.add_message('Cloud server Compeleted  the Last FL Round \n')
                      self.send_messages_to_all(message.data, "FLEnd")
                      self.sendLocalModels("Cloud")

                  else :
                       self.add_message(message.subject+"\n")

             
           except Exception as e: 
                print('Error from listen_for_messages_from_server', e)       
        else:
            print("Error", "Message recevied from Server is empty")
  

    def send_message_to_aggregator(self,message,subject):
       try :
        if message != '':
                  objectToSend=Empty()
                  objectToSend.data=message
                  objectToSend.subject=subject
                  message=objectToSend
                  message = pickle.dumps(message)
                  self.aggregatorSocket.send(message)
                  print('message sent to aggregator server')                
        else:    
           print("Empty message", "Message cannot be empty")
       except Exception as e : 
           print(e)


 #***********Foge Setup*****************************************#
    def receive_fogs(self):

      
       while 1:
        try :
          fog, address = self.serverForFogs.accept()
          print(fog,address)
          threading.Thread(target=self.Fog_handler, args=(fog, address)).start()
        except Exception as e :
          print("Expecttttttttttt",e)

#*****************************************************************************************# 
    def Fog_handler(self,fog,address):  

  
      while 1:
          existedFog=False
          i=0
          msg= fog.recv(1000000)#.decode('utf-8')
          msg=pickle.loads(msg)
          try: 
            id=msg.data
            if str(id) != '':
              while (i<len(self.active_fogs) and existedFog==False):
               if (self.active_fogs[i][0]==id): existedFog=True
               else: i=i+1
              if (existedFog==False):
               self.active_fogs.append([id, fog,address, [],None,None])  #id, socket, address, (list of ==> [idEdge,address,accuracy,persoModel,domain,task])
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
          except  Exception as e: 
             print('checking as fog',msg)
             #target=self.check_for_message_from_fog, args=(fog, msg, )).start()
             threading.Thread(target=self.check_for_message_from_fog, args=(fog, msg, )).start()
            

     
      threading.Thread(target=self.listen_for_messages_from_fog, args=(fog,id, )).start() 

#************************************************************#

    def check_for_message_from_fog(self,fog,message):
        i=0
        Find=False
        id=-99
        for usr in self.active_fogs:
          if (fog==usr[1]):
            id =usr[0]
            print('hani lkiteh')
        if message != '':
            #msg=username+"  "+message
            self.add_message(' Message received From Fog'+ str(id)+' About '+message.subject  + ' \n')
           
            while (i<len(self.active_clients) and Find==False):
              username="Fog "+str(id)
              if (self.active_fogs[i][0]==id): ### search about user 
                
                try:
                 
                 Find=True
                 if (message.subject=="LocalModel"):
                    self.numberFogsreceived+=1
                    print("choufi souhila from fog ", username,self.numberFogsreceived)
                    self.active_fogs[i][3]=message.data[0]   #list of edge nodes 

                    self.FLAggregation()
                except Exception as e:
                  print('Exception from listen_for_messages',e)
              i=i+1
            self.send_message_to_fog(fog,"FOG  ~~ Successful Demand Received ","ACK")


        else:
            print(f"The message send from Fog {username} is empty") 
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
    def send_messages_to_all_fogs(self,message,subject):  
    
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
          print('emmmmmmmmmmmmmm',message.data)

          while (i<len(self.active_fogs) and Find==False):
              username="Fog "+str(id)
              print(username)
              if (self.active_fogs[i][0]==id): ### search about Fog
                
                try:
                 Find=True
                 print(Find)
                 if (message.subject=="LocalModels"):
                    print("choufi souhila",self.numberFogsreceived)
                    self.numberFogsreceived+=1
  
                    self.active_fogs[i][3]=message.data[0]   #list of edge nodes 

                    self.FLAggregation()

                except Exception as e:
                  print('Exception from listen_for_messages',e)
              i=i+1
          self.send_message_to_fog(fog,"FOG  ~~ Successful Demand Received ","ACK")


        else:
            print(f"The message send from Fog {username} is empty")
#*****************************************************************************************#   

  
#*********************************Cloud Setup******************#
    def connect(self):
       Var= False
   
       try:
           self.cloud.connect((self.HostCloud, self.PortCloud))
           self.add_message("Successfully connected to Cloud server \n")
           data=[self.id,self.BackupPort]
           self.send_message_to_cloud(data,'Connection')
           Var= True

       except Exception as e:
        print(e)
        
    
       threading.Thread(target=self.listen_for_messages_from_Cloud, args=(self.cloud, )).start()
    
       return Var
   
#*****************************************************************************************#    
    def listen_for_messages_from_Cloud(self,socket):

       while 1:
         
        message = socket.recv(1000000)#.decode('utf-8')
        message=pickle.loads(message)
        if message != '':
           try:
             
                  self.add_message('Cloud Server is requesting For: '+message.subject+"\n")
                
                  if (message.subject=='FLstart'):
                     self.receivedLocalModels=0
                     
                     
                     self.send_messages_to_all(message.data, "FLstart")

                  elif  (message.subject=='FL'):
                      self.receivedLocalModels=0
                      self.add_message('Cloud server is requesting for an Other  FL Round \n')
                      self.send_messages_to_all(message.data, "FL")
                  elif  (message.subject=='FLEnd'):
                      self.receivedLocalModels=0
                      self.add_message('Cloud server Compeleted  the Last FL Round \n')
                      self.send_messages_to_all(message.data, "FLEnd")

                  elif  (message.subject=='TLModel'):
                      self.add_message('I am receiving TL model from the server \n')
                      self.TransferModelToClient(message.data)

                  elif  (message.subject=='RequestTLModel'):
                      self.TransferModelToCloud(message)
                  elif (message.subject=='Election'):
                        self.capacity=random.uniform(0,100) 
                        data=[self.capacity, self.priority]
                        self.send_message_to_cloud(data,'Election')
                  elif (message.subject=='ElectedAggregator'):
                        self.Actuator="Aggregator"
                        self.aggregator="Me"
                        self.method_name=message.data
                        try:
                            self.serverForFogs.bind((self.HOST, self.BackupPort))
                            print(f"Running as aggregator server on {self.HOST} {self.BackupPort}")
                            self.serverForFogs.listen(self.LISTENER_LIMIT)
                            self.add_message(f"Running as aggregator server on {self.HOST} {self.BackupPort} \n")
                            threading.Thread(target=self.fog.receive_fogs, args=()).start()
                        except Exception as e:
                            print(f"Running as aggregator server on {self.HOST} {self.BackupPort}")
                            self.add_message(f"Running as aggregator server on {self.HOST} {self.BackupPort} \n")
                            #print(f"Unable to bind to host {self.HOST} and port {self.BackupPort}", e)
                            #self.add_message(f"Unable to bind to host {self.HOST} and port {self.BackupPort} \n")
                  elif (message.subject=="Aggregator"):
                       address=message.data[0]
                       port=message.data[1]

                       print(f'my new aggregator is  {address} and   {port}')
                     

                       if (address!=self.HostCloud or port!= self.PortCloud) : # dans le cas ou un autre fog est selectionné comme aggregateur
                        
                        print('before sleeping')
                        
                        time.sleep(10)

                        print('after sleeping')

                        self.aggregator="Fog"

                        self.connect_to_aggregator(address,port)

                       else :

                          print('The aggregator is always the Cloud')



                  else :
                       self.add_message(message.subject+"\n")

             
           except Exception as e: 
                print('Error from listen_for_messages_from_server', e)       
        else:
            print("Error", "Message recevied from Server is empty")

#*****************************************************************************************#
    def send_message_to_cloud(self,message,subject):
       try :
        if message != '':
                  objectToSend=Empty()
                  objectToSend.data=message
                  objectToSend.subject=subject
                  message=objectToSend
                  message = pickle.dumps(message)
                  self.cloud.send(message)
                  print('message sent to cloud server')                
        else:    
           print("Empty message", "Message cannot be empty")
       except Exception as e : 
           print(e)
       

# *************************************************************************#
    def TransferModelToCloud(self,message):

      id_user=message.data[1]  #id
      for usr in self.active_clients:
        if (usr[0]==id_user): 
          data=[message.data[0],usr[3][5]]  #idCible, #complete model architecture
          self.send_message_to_cloud( data, "TLModel")
#**********************************************************************#  
       
    def add_message(self,message):
       self.inputtxt.config(state=tk.NORMAL)
       self.inputtxt.insert(tk.END, message )
       self.inputtxt.config(state=tk.DISABLED)


#***********Edge Setup*****************************************#
   
    def client_handler(self,client,address):  
      while 1:
          ExistedClient=False
          i=0
          msg = client.recv(1000000)#.decode('utf-8')
          msg=pickle.loads(msg)
          id=msg.id

          try :
            subject= msg.subject
            if (subject=="Connection"):
              if str(id) != '':
               while(i<len(self.active_clients) and  ExistedClient==False ):
                 if (self.active_clients[i][0]==id): ExistedClient=True
                 else : i=i+1
               if (ExistedClient==False):
                 self.active_clients.append([id, client,address,[None,None,None,None,None,None,None]])  #username, adr, data object     
                 print( "SERVER~" + f"Client {id} added to the System")
                 self.add_message(f"Client {id} added to the System \n")
                 
                 
               else :
                 self.active_clients[i][1]=client
                 self.active_clients[i][2]=address
                 print( "SERVER~" + f"Client {id} is reconnected to the System")
                 self.add_message(f"Client {id} reconnected to the System \n")
              
              else:
               print("Client username is empty")
            elif (subject=='RequestTLModel'):
               stop = False
               i =0
               while(i< len(self.active_clients) and stop== False):
                if (client==self.active_clients[i][1]):
                   stop=True
                   msg=[self.active_clients[i][0],self.active_clients[i][2],msg.data[0],msg.data[1]] #id, adress, domain, task
                i+=1

               if (stop==True):  self.send_message_to_cloud(msg,"RequestTLModel" )

            else: self.check_for_message_from_client(client,msg)

              
          except Exception as e: 
             print('checking as edge',msg)
             threading.Thread(target=self.check_for_message_from_client, args=(client, msg, )).start()
          
      
  
          threading.Thread(target=self.listen_for_messages_from_Client, args=(client, id, )).start()
   
#**********************************************************************# 
    
    def receive_clients(self):

      while 1:      
        client, address = self.server.accept()
    
      
        threading.Thread(target=self.client_handler, args=(client,address)).start()

#**********************************************************************# 
    def send_message_to_client(self,client, message,subject): 
          
          obj =Empty()
          obj.data=message
          obj.subject=subject
          message=obj
          
          client.send(pickle.dumps(message)) #sendall    message.ffffffffffhhfgode()

#**********************************************************************# 

    def send_messages_to_all(self,message,subject):
    
       for user in self.active_clients:

          self.send_message_to_client(user[1], message,subject)

#**********************************************************************# 
 
    def check_for_message_from_client(self,client,message):

        i=0
        Find=False
        id=-99
        for usr in self.active_clients:
          if (client==usr[1]):
            id =usr[0]
        if message != '':
            #msg=username+"  "+message
            self.add_message(' Message received From Client'+ str(id)+' About '+message.subject  + ' \n')
           
            while (i<len(self.active_clients) and Find==False):
              username="Client "+str(id)
              if (self.active_clients[i][0]==id): ### search about user 
                
                try:
                 
                 Find=True
                 if (message.subject=="LocalModel"):
                   self.receivedLocalModels+=1
                   print("choufi souhila from edge", username,self.receivedLocalModels)
                   self.active_clients[i][3][0]=message.personnalizedModel ### update model parameters
                   self.active_clients[i][3][1]=message.completeModel
                   self.active_clients[i][3][2]=message.accuracy
                   self.active_clients[i][3][3]=message.domain
                   self.active_clients[i][3][4]=message.task
                   self.active_clients[i][3][5]=message.architecture
                   self.active_clients[i][3][6]=message.loss
                   if (self.aggregator=="Me"): self.FLAggregation()
                   if (self.receivedLocalModels==len(self.active_clients)): 
                     self.sendLocalModels(self.aggregator)
                 elif (message.subject=="FinalLocalModel"):
                  
                   self.receivedLocalModels+=1
                   print('hello', self.receivedLocalModels)
                   self.active_clients[i][3][0]=message.personnalizedModel ### update model parameters
                   self.active_clients[i][3][1]=message.completeModel
                   self.active_clients[i][3][2]=message.accuracy
                   self.active_clients[i][3][3]=message.domain
                   self.active_clients[i][3][4]=message.task
                   self.active_clients[i][3][5]=message.architecture
                   self.accuracy_locals_test.append(message.measures[1])
                   self.accuracy_locals_train.append(message.measures[0])
                   self.loss_locals_test.append(message.measures[3])
                   self.loss_locals_train.append(message.measures[2])

                   self.display()
                   if (self.receivedLocalModels==len(self.active_clients) ):
                    if (self.aggregator!="Me"):   self.sendLocalModels(self.aggregator)
                 elif (message.subject=="RequestTLModel"):
                   #print(self.active_clients[i][0],self.active_clients[i][2],message[0],message[1])
                   msg=[self.active_clients[i][0],self.active_clients[i][2],message[0],message[1]] #id, adress, domain, task
                   self.send_message_to_cloud(msg,"RequestTLModel" )
                except Exception as e:
                  print('Exception from listen_for_messages',e)
              i=i+1
            self.send_message_to_client(client,"FOG  ~~ Successful Demand Received ","ACK")
          

        else:
            print(f"The message send from client {username} is empty")

#********************************************************************#

    def listen_for_messages_from_Client(self,client, id):

      while 1:


        message = client.recv(1000000)#.decode('utf-8')
        message=pickle.loads(message)
        i=0
        Find=False
        
        if message != '':
            #msg=username+"  "+message
            self.add_message(' Message received From Client'+ str(id)+' About '+message.subject  + ' \n')
           
            while (i<len(self.active_clients) and Find==False):
              username="Client "+str(id)
              if (self.active_clients[i][0]==id): ### search about user 
                
                try:
                 
                 Find=True
                 if (message.subject=="LocalModel"):
                   self.receivedLocalModels+=1
                   print("choufi souhila from edge", username,self.receivedLocalModels)
                   self.active_clients[i][3][0]=message.personnalizedModel ### update model parameters
                   self.active_clients[i][3][1]=message.completeModel
                   self.active_clients[i][3][2]=message.accuracy
                   self.active_clients[i][3][3]=message.domain
                   self.active_clients[i][3][4]=message.task
                   self.active_clients[i][3][5]=message.architecture
                   self.active_clients[i][3][6]=message.loss
                   if (self.aggregator=="Me"): self.FLAggregation()
                   if (self.receivedLocalModels==len(self.active_clients)): 
                     self.sendLocalModels(self.aggregator)
                 elif (message.subject=="FinalLocalModel"):

                   self.receivedLocalModels+=1
                   print('hello', self.receivedLocalModels)
                   self.active_clients[i][3][0]=message.personnalizedModel ### update model parameters
                   self.active_clients[i][3][1]=message.completeModel
                   self.active_clients[i][3][2]=message.accuracy
                   self.active_clients[i][3][3]=message.domain
                   self.active_clients[i][3][4]=message.task
                   self.active_clients[i][3][5]=message.architecture
                   self.accuracy_locals_test.append(message.measures[1])
                   self.accuracy_locals_train.append(message.measures[0])
                   self.loss_locals_test.append(message.measures[3])
                   self.loss_locals_train.append(message.measures[2])

                   self.display()

              
                 elif (message.subject=="RequestTLModel"):
                   print(username, message.data[0],message.data[1])
                   msg=[self.active_clients[i][0],self.active_clients[i][2],message.data[0],message.data[1]] #id, adress, domain, task
                   self.send_message_to_cloud(msg,"RequestTLModel" )
                   self.add_message('I am requesting the server about TL model \n')
                except Exception as e:
                  print('Exception from listen_for_messages',e)
              i=i+1
            self.send_message_to_client(client,"FOG  ~~ Successful Demand Received ","ACK")
          

        else:
            print(f"The message send from client {username} is empty")

#**********************************************************************#

    def sendLocalModels(self,qui):
      list= []
      
      for usr in self.active_clients:
        avgAccuracy=(usr[3][2][0]+usr[3][2][1])/2
        #[usr[3][2][0],usr[3][2][1]]
        #
        list.append([usr[0],usr[2],avgAccuracy,usr[3][0],usr[3][3],usr[3][4]]) #id, address, accuracy, Personnalized Model, domain, task
      data=[list, self.capacity, self.priority]
      if (qui=="Cloud"):
        self.send_message_to_cloud(data,"LocalModels")
      else: self.send_message_to_aggregator(data,"LocalModels")
  

#**********************************************************************#

    def TransferModelToClient(self,message):
      id_user=message[0]
      model = message[1] 
      for usr in self.active_clients:
        #print (f'It is my client {usr[0]} and {id_user}')
        if (usr[0]==id_user):  #id
          model = message[1]  # the model and not wights
      self.add_message('I am sending the model to my client \n ')
      self.send_message_to_client(usr[1],model, "TLModel")

#**********************************************************************#

  #*****************************************************************************************# 

    def start_FL(self):

        self.FLrounds=self.args.epochs
        self.numberFogsreceived=0
        self.receivedLocalModels=0
        self.add_message("Starting FL \n")
        self.send_messages_to_all(self.method_name,"FLstart")
        self.send_messages_to_all_fogs(self.method_name,"FLstart")

       
            
#*****************************************************************************************# 
    def FLAggregation(self):
       print(f"{self.numberFogsreceived} and  {self.receivedLocalModels}")
       receiv =self.numberFogsreceived+  self.receivedLocalModels
       total =len(self.active_clients)+ len(self.active_fogs)

       print(receiv, total)

       if (receiv==total):
        
         self.FLrounds-=1
         self.add_message("Aggregation Round"+ str( self.args.epochs-self.FLrounds)+"\n")
         local_weights=[]
         for fog in self.active_fogs:          
              for usr in fog[3]:   
                if (self.method_name=='FedAVG'):    
                   local_weights.append(usr[4]) #local models weights
                else :
                  local_weights.append(usr[3])
         for usr in self.active_clients:
            if (self.method_name=='FedAVG'):
              local_weights.append(usr[3][1])
            else : local_weights.append(usr[3][0])
         self.aggregate(local_weights,self.method_name)
  
         if (self.FLrounds==0):
            self.send_messages_to_all_fogs(self.weights_global,"FLEnd")
            self.send_messages_to_all(self.weights_global,"FLEnd")
            self.add_message("Sending Last Global Model \n")
            self.numberFogsreceived=0    
            
         
         else :
           self.send_messages_to_all_fogs(self.weights_global,"FL")
           self.send_messages_to_all(self.weights_global,"FL")
         self.numberFogsreceived=0  
         self.receivedLocalModels=0
         receiv=0
         total=0        


#*****************************************************************************************# 

    def aggregate(self,weights_clients,method_name):
        self.method_name=method_name
        self.weights_locals=weights_clients
        #edge = random.randint(0, len(self.active_clients)-1)
        #self.global_model=self.active_clients[edge][3][1] #the complete model
       
        self.i=0
        if (self.method_name=='FedAVG'):
              self.weights_global=FedAvg(self.weights_locals)
        elif (self.method_name=='FedGA'):
             initial_population=self.weights_locals
        
             for d in self.weights_locals: # for each user
                weight=[]
                if isinstance(d, dict):
                     for x in d.items():                    #get weights of each layer
                         array = np.array(x[1], dtype='f')  #1 is a tensor
                         array= array.flatten()
                         weight= np.concatenate((weight, array), axis=0)
                         initial_population[ self.i]= np.array(weight,dtype='f')

                self.i= self.i+1                            #next weight vector (user)
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

        if (self.method_name=='FedAVG'):
             self.global_model.load_state_dict(self.weights_global)
        else :

            self.global_model.classification.load_state_dict(self.weights_global)

        return self.global_model


#*****************************************************************************************#   
    def display(self):
       if (self.receivedLocalModels==len(self.active_clients)):
        Plot_Graphes_for_fog(self.id,len(self.accuracy_locals_train),len(self.active_clients),self.accuracy_locals_train,self.accuracy_locals_test,self.loss_locals_train,self.loss_locals_test)
        self.accuracy_locals_train=[]
        self.accuracy_locals_test=[]
        self.loss_locals_train=[]
        self.loss_locals_test=[]
        self.receivedLocalModels=0
        
   
if __name__ == '__main__':

    args = args_parser()   # ajoute id 
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #print(torch.cuda.is_available())
       # Set server limit
    fog =Fog(args=args,HOST='127.0.0.1',HostCloud='127.0.0.1')
    threading.Thread(target=fog.receive_clients, args=()).start()
    fog.root.mainloop()