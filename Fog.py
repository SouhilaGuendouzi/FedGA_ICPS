# Import required modules

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
        self.registry={} #§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§#
        self.Actuator=False
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
                 command = lambda:self.send_message_to_cloud(len(self.active_clients))
                 )
        self.Start_FL.pack(padx=5, pady=20, side=tk.LEFT)
    
        try:
          self.server.bind((self.HOST, self.PORT))
          print(f"Running the server on {self.HOST} {self.PORT}")
          self.server.listen(self.LISTENER_LIMIT)
          self.add_message(f"Running the server on {self.HOST} {self.PORT} \n")
        except:
           print(f"Unable to bind to host {self.HOST} and port {self.PortCloud}")
           self.add_message(f"Unable to bind to host {self.HOST} and port {self.PortCloud}")
    #*****************************************************************************************#
    def listen_for_messages_from_server(self,socket):

       while 1:

        message = socket.recv(1000000)#.decode('utf-8')
        message=pickle.loads(message)
        if message != '':
  
            self.add_message(message)
            
        else:
            print("Error", "Message recevied from Server is empty")
#*****************************************************************************************#
    def connect(self):
       Var= False
       # try except block
       try:

    
           self.cloud.connect((self.HostCloud, self.PortCloud))
           self.send_message_to_cloud(self.id)
           Var= True

       except Exception as e:
        print(e)
        
    
       threading.Thread(target=self.listen_for_messages_from_server, args=(self.cloud, )).start()
    
       return Var
       
#*****************************************************************************************#
    def send_message_to_cloud(self,message):
       try :
   
        if message != '':
           message = pickle.dumps(message)
           self.cloud.send(message)
      
        else:
           print("Empty message", "Message cannot be empty")
       except Exception as e : 
           print(e)
    #*****************************************************************************************#
    def add_message(self,message):
     
       self.inputtxt.config(state=tk.NORMAL)
       self.inputtxt.insert(tk.END, message )
       self.inputtxt.config(state=tk.DISABLED)
#*****************************************************************************************#    
    def send_message_to_client(self,client, message): 
          client.send(pickle.dumps(message)) #sendall    message.ffffffffffhhfgode()

    def send_messages_to_all(self,message):
    
       for user in self.active_clients:

          self.send_message_to_client(user[1], message)

 

    def listen_for_messages(self,client, id):

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
        
        
                 self.active_clients[i][3][0]=message.data ### update model parameters
                 self.active_clients[i][3][1]=message.completeModel
                 self.active_clients[i][3][2]=message.accuracy
                 self.active_clients[i][3][3]=message.domain
                 self.active_clients[i][3][4]=message.task
                except Exception as e:
                  print('Exception from listen_for_messages',e)
              i=i+1
            self.send_message_to_client(client,"FOG  ~~ Successful Demand Received ")
            print(self.active_clients)

        else:
            print(f"The message send from client {username} is empty")


     
    
    def client_handler(self,client,address):  
    # Server will listen for client message that will
    # Contain the username
     
      while 1:
          ExistedClient=False
          i=0
          id = client.recv(1000000)#.decode('utf-8')
          id=pickle.loads(id)
          print(id)
          if str(id) != '':
               while(i<len(self.active_clients) and  ExistedClient==False ):
                 if (self.active_clients[i][0]==id): ExistedClient=True
                 else : i=i+1
               if (ExistedClient==False):
                 self.active_clients.append((id, client,address,[None,None,None,None,None]))  #username, adr, data object
                
                 print( "SERVER~" + f"Client {id} added to the System")
                 self.add_message(f"Client {id} added to the System \n")
                 
                 
               else :
                 
                 print( "SERVER~" + f"Client {id} is reconnected to the System")
                 self.add_message(f"Client {id} reconnected to the System \n")
               

          else:
            print("Client username is empty")
      
  
          threading.Thread(target=self.listen_for_messages, args=(client, id, )).start()
   

    def request_for_models(self):
      for user in self.active_clients:
         self.send_message_to_client(user[1], 'Models')

      

    def receive_clients(self):
    # This while loop will keep listening to client connections
       
      while 1:      
        client, address = self.server.accept()
        threading.Thread(target=self.client_handler, args=(client,address)).start()


if __name__ == '__main__':

    args = args_parser()   # ajoute id 
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(torch.cuda.is_available())
       # Set server limit
    fog =Fog(args=args,HOST='127.0.0.1',HostCloud='127.0.0.1')
    threading.Thread(target=fog.receive_clients, args=()).start()
    fog.root.mainloop()