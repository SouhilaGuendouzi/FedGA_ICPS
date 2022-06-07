# Import required modules

#from utils.Empty import Empty
from queue import Empty
import socket
import threading
import pickle
from matplotlib.style import use
import tkinter as tk
from utils.FogOptions import args_parser
import torch


# Function to listen for upcoming messages from a client
class Fog:
    def __init__(self,args,HOST,HostCloud):
        self.id=args.id
        self.HOST=HOST
        self.PORT=args.myport
        self.LISTENER_LIMIT=args.LISTENER_LIMIT
        self.HostCloud=HostCloud
        self.PortCloud=args.portCloud
        self.active_clients = []
        self.receivedLocalModels=0
        self.Actuator="Free"
        self.globalModel=None
        self.scoring=0
        self.pyhical_attributes={}      
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.cloud=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.args=args
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
        self.Start_FL = tk.Button(self.root, height = 2,
                 width = 20,
                 text ="Start ",
                 command = lambda:self.send_message_to_cloud("hello","TEst")
                 )
        self.Start_FL.pack(padx=5, pady=20, side=tk.LEFT)
    
        try:
          self.server.bind((self.HOST, self.PORT))
          print(f"Running the server on {self.HOST} {self.PORT}")
          self.server.listen(self.LISTENER_LIMIT)
          self.add_message(f"Running the server on {self.HOST} {self.PORT} \n")
        except:
           print(f"Unable to bind to host {self.HOST} and port {self.PortCloud}")
           self.add_message(f"Unable to bind to host {self.HOST} and port {self.PortCloud} \n")
    
    
    
    
    #***********Cloud Setup*****************************************#


    def connect(self):
       Var= False
       # try except block
       try:

    
           self.cloud.connect((self.HostCloud, self.PortCloud))
           self.add_message("Successfully connected to Cloud server \n")
           self.send_message_to_cloud(self.id,'Connection')
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
                     self.Actuator="FL"
                     
                     self.send_messages_to_all(message.data, "FLstart")

                  elif  (message.subject=='FL'):
                      self.receivedLocalModels=0
                      self.add_message('Cloud server is requesting for an Other  FL Round \n')
                      self.send_messages_to_all(message.data, "FL")
                  elif  (message.subject=='FLEnd'):
                      self.Actuator="Free"
                      self.add_message('Cloud server Compeleted  the Last FL Round \n')
                      self.send_messages_to_all(None, "FLEnd")

                  elif  (message.subject=='TLModel'):
                      self.TransferModelToClient(message)

                  elif  (message.subject=='RequestTLModel'):
                      self.TransferModelToCloud(message)
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
      id_user=message.data[0]  #id
      for usr in self.active_clients:
        if (usr[0]==id_user): 
       
          data=[message.data[0],message.data[2],usr[3][1]]  #id, #address, #complete model
         
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
          id=msg.data
          
          if str(id) != '':
               while(i<len(self.active_clients) and  ExistedClient==False ):
                 if (self.active_clients[i][0]==id): ExistedClient=True
                 else : i=i+1
               if (ExistedClient==False):
                 self.active_clients.append([id, client,address,[None,None,None,None,None]])  #username, adr, data object     
                 print( "SERVER~" + f"Client {id} added to the System")
                 self.add_message(f"Client {id} added to the System \n")
                 
                 
               else :
                 self.active_clients[i][1]=client
                 self.active_clients[i][2]=address
                 print( "SERVER~" + f"Client {id} is reconnected to the System")
                 self.add_message(f"Client {id} reconnected to the System \n")
              
          else:
            print("Client username is empty")
      
  
          threading.Thread(target=self.listen_for_messages_from_Client, args=(client, id, )).start()
   

    
    def receive_clients(self):

      while 1:      
        client, address = self.server.accept()
        threading.Thread(target=self.client_handler, args=(client,address)).start()

    def send_message_to_client(self,client, message,subject): 
          
          obj =Empty()
          obj.data=message
          obj.subject=subject
          message=obj
          
          client.send(pickle.dumps(message)) #sendall    message.ffffffffffhhfgode()

    def send_messages_to_all(self,message,subject):
    
       for user in self.active_clients:

          self.send_message_to_client(user[1], message,subject)

 

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
                   self.active_clients[i][3][0]=message.personnalizedModel ### update model parameters
                   self.active_clients[i][3][1]=message.completeModel
                   self.active_clients[i][3][2]=message.accuracy
                   self.active_clients[i][3][3]=message.domain
                   self.active_clients[i][3][4]=message.task
                   if (self.receivedLocalModels==len(self.active_clients) ): #and self.Actuator=="FL"
                     self.sendLocalModels()
                 elif (message.subject=="RequestTLModel"):
               
                   msg=[self.active_clients[i][0],self.active_clients[i][2],self.active_clients[i][3][3],self.active_clients[i][3][4]] #id, adress, domain, task
                   self.send_message_to_cloud(msg,"RequestTLModel" )
                except Exception as e:
                  print('Exception from listen_for_messages',e)
              i=i+1
            self.send_message_to_client(client,"FOG  ~~ Successful Demand Received ","ACK")
          

        else:
            print(f"The message send from client {username} is empty")


    #**********************************************************************#


    def sendLocalModels(self):
      list= []
      
      for usr in self.active_clients:
        avgAccuracy=(usr[3][2][0]+usr[3][2][1])/2
        list.append([usr[0],usr[2],avgAccuracy,usr[3][0],usr[3][3],usr[3][4]]) #id, address, accuracy, Personnalized Model, domain, task
      
      self.send_message_to_cloud(list,"LocalModels")
      



    #**********************************************************************#
    def TransferModelToClient(self,message):
      model=None
      for usr in self.active_clients:
        if (usr[0]==message.data[0]):  #id
          model =message.data[0] # the model and not wights
      
      self.send_message_to_cloud(model, "TLModel")
    #**********************************************************************#

   


if __name__ == '__main__':

    args = args_parser()   # ajoute id 
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #print(torch.cuda.is_available())
       # Set server limit
    fog =Fog(args=args,HOST='127.0.0.1',HostCloud='127.0.0.1')
    threading.Thread(target=fog.receive_clients, args=()).start()
    fog.root.mainloop()