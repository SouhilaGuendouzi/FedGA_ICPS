# Import required modules

from queue import Empty
import socket
import threading
import pickle
from matplotlib.style import use
import tkinter as tk
from utils.CloudOptions import args_parser
import torch


HOST = '127.0.0.1'
PORT = 12345 # You can use any port between 0 to 65535
LISTENER_LIMIT = 5



# Function to listen for upcoming messages from a client
class Cloud:
    def __init__(self,HOST,PORT,LISTENER_LIMIT,args):
        self.HOST=HOST
        self.PORT=args.myport
        self.LISTENER_LIMIT=LISTENER_LIMIT
        self.active_fogs = []
        self.FLrounds=args.epochs
        self.aggregation=args.aggr
        self.localmodels=[None for i in self.args.num_users]
        self.registry={} #§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§#
        self.Actuator=False
        self.globalModel=None
        self.numberFogsreceivedForFL=0
        self.scoring=0
        self.pyhical_attributes={}      
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # IPv4 addresses, TCP
        self.args=args
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
          self.add_message(f"Running the server on {HOST} {PORT}")
        except Exception as e:
           print(f"Unable to bind to host {HOST} and port {PORT} because of {e}")

        
    #*****************************************************************************************#
    def add_message(self,message):
     
       self.inputtxt.config(state=tk.NORMAL)
       self.inputtxt.insert(tk.END, message )
       self.inputtxt.config(state=tk.DISABLED)




#*****************************************************************************************#
    def send_message_to_fog(self,fog,message):
       try :
   
        if message != '':
           message = pickle.dumps(message)
           fog.send(message)
      
        else:
           print("Empty message", "Message cannot be empty")
       except Exception as e : 
           print(e)
    #*****************************************************************************************#
  
   

    def send_messages_to_all(self,message,subject):   #in case of broadcasting
    
       for user in self.active_fogs:

          self.send_message_to_fog(user[1], message,subject)

    def listen_for_messages(self,fog, username):

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

                 if (message.subject=="LocalModels"):
                   self.active_fogs[i][3]=message.data
                   
                  
                
              
               
                except Exception as e:
                  print('Exception from listen_for_messages',e)
              i=i+1
          self.send_message_to_fog(client,"FOG  ~~ Successful Demand Received ","ACK")


        else:
            print(f"The message send from Fog {username} is empty")


    #*****************************************************************************************#   

    def main(self):
    
       
       
        self.root.mainloop()

    def receive_fogs(self):

      
       while 1:
        fog, address = self.server.accept()
      
        threading.Thread(target=self.Fog_handler, args=(fog, )).start()

#*****************************************************************************************#    
    def Fog_handler(self,fog):  
  
      existedFog=False
      i=0
      while 1:
          id = fog.recv(1000000)#.decode('utf-8')
          id=pickle.loads(id)
          if str(id) != '':
            while (i<len(self.active_fogs) and existedFog==False):
               if (self.active_fogs[i][0]==id): existedFog=True
               else: i=i+1
            if (existedFog==False):
               self.active_fogs.append((id, fog,address, [None,None,None,None,None,None]))  #id, socket, address, (list of ==> [idEdge,address,accuracy,persoModel,domain,task])
               print( "" + f" Fog {id} added to the System")
               self.add_message(f"Fog {id} added to the System")
               self.send_message_to_fog(fog,"Server ~~ Successfully connected to Cloud Server ")
               break
            else:
               self.active_fogs[i][1]=fog
               self.active_fogs[i][2]=address
               print( "" + f" Fog {id} reconnected to the System")
               self.add_message(f"Fog {id}  reconnected to the System")
          else:
            print("Fog username is empty")
     
          threading.Thread(target=self.listen_for_messages, args=(fog, id, )).start()



#*****************************************************************************************# 

    def start_FL(self):
      self.add_message("Starting FL \n")
      self.send_messages_to_all(None,"FLstart")

 

  #
  #
  #
  #
  #
  #
  #
  #
#*****************************************************************************************# 



if __name__ == '__main__':
  
    args = args_parser()   # ajoute id 
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(torch.cuda.is_available())
    cloud =Cloud(HOST=HOST, PORT=PORT, LISTENER_LIMIT=LISTENER_LIMIT,args=args)

    cloud.server.listen(LISTENER_LIMIT)
    threading.Thread(target=cloud.receive_fogs, args=()).start()
    cloud.main()
    