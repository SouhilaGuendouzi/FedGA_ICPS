# Import required modules

import socket
import threading
import pickle
from matplotlib.style import use
import tkinter as tk
from utils.FogOptions import args_parser
import torch

HOST = '127.0.0.1'
PORT = 12346 # You can use any port between 0 to 65535
LISTENER_LIMIT = 5



# Function to listen for upcoming messages from a client
class Fog:
    def __init__(self,id,HOST,PORT,LISTENER_LIMIT,args,HostCloud,PortCloud):
        self.id=id
        self.HOST=HOST
        self.PORT=PORT
        self.LISTENER_LIMIT=LISTENER_LIMIT
        self.HostCloud=HostCloud
        self.PortCloud=PortCloud
        self.active_clients = []
        self.clients=[]  ###" pour le momennt fihoum weights"
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
          self.server.bind((HOST, PORT))
          print(f"Running the server on {HOST} {PORT}")
          self.add_message(f"Running the server on {HOST} {PORT} \n")
        except:
           print(f"Unable to bind to host {HOST} and port {self.PortCloud}")
           self.add_message(f"Unable to bind to host {HOST} and port {self.PortCloud}")
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

        # Connect to the server
           self.cloud.connect((self.HostCloud, self.PortCloud))
           #print("Successfully connected to Cloud")
           #self.add_message("Successfully connected to Cloud")
           self.send_message_to_cloud('Fog {}'.format(self.id))
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

      # Function to send any new message to all the clients that
      # are currently connected to this server


    def send_messages_to_all(self,message):
    
       for user in self.active_clients:

          self.send_message_to_client(user[1], message)

     # Function to handle client
    def listen_for_messages(self,client, username):

      while 1:

        message = client.recv(1000000)#.decode('utf-8')
        message=pickle.loads(message)
        i=0
        
        if message != '':
            #msg=username+"  "+message
            self.add_message(username+' weights are received \n')
            self.add_message(message=message.subject)
            print(username,message.subject)
            while (i<len(self.active_clients)):
              username="Client "+str(message.id)
              if (self.active_clients[i][0].find(username)): ### search about user 
                try:
                 self.active_clients[i][2].data= message.data ### update model parameters
                except:
                  self.active_clients[i][2]=message
            self.send_message_to_client(client,"FOG  ~~ Successful Weights received ")
            print(self.active_clients)

        else:
            print(f"The message send from client {username} is empty")


     
    
    def client_handler(self,client,address,existedClient):  
    # Server will listen for client message that will
    # Contain the username
      while 1:
        if (existedClient==False):
          username = client.recv(1000000)#.decode('utf-8')
          username=pickle.loads(username)
          if username != '':
            if username.find("Client")!=-1:
               self.active_clients.append((username, address,None))  #username, adr, data object
               #prompt_message = "SERVER~" + f"{username} added to the chat"
               print( "SERVER~" + f"{username} added to the System")
               self.add_message(f"{username} added to the System \n")
               #self.send_messages_to_all(prompt_message)
               break
          else:
            print("Client username is empty")
        else:

           self.add_message(f"{username} reconnected to the System \n")

        
   

      threading.Thread(target=self.listen_for_messages, args=(client, username, )).start()
   

    def request_for_models(self):
      for user in self.active_clients:
         self.send_message_to_client(user[1], 'Models')

      

    def receive_clients(self):
    # This while loop will keep listening to client connections
       existedClient=False
       i=0
       while 1:
            
        client, address = self.server.accept()
        while  (existedClient==False) and i< len(self.active_clients) :
            
            if (self.active_clients[i][1][1]==address[1]):
              existedClient=True
            else :
              i=i+1
        if (existedClient==True):
           print(f"Successfully reconnected to client {address[0]} {address[1]}")
        else :
          print(f"Successfully connected to client {address[0]} {address[1]}")

        threading.Thread(target=self.client_handler, args=(client,address,existedClient )).start()


if __name__ == '__main__':
    args = args_parser()   # ajoute id 
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(torch.cuda.is_available())
       # Set server limit
    fog =Fog(args.id,HOST=HOST, PORT=args.myport, LISTENER_LIMIT=LISTENER_LIMIT,args=args,HostCloud='127.0.0.1',PortCloud=args.portCloud)
    fog.server.listen(LISTENER_LIMIT)
    threading.Thread(target=fog.receive_clients, args=()).start()
    fog.root.mainloop()