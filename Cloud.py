# Import required modules

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
        self.models=[]
        self.registry={} #§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§#
        self.Actuator=False
        self.globalModel=None
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
                 text ="Start ",
                 command = lambda:self.start_FL()
                 )
        self.Start_FL.pack(padx=5, pady=20, side=tk.LEFT)
        try:
          self.server.bind((HOST, PORT))
          print(f"Running the server on {HOST} {PORT}")
          self.add_message(f"Running the server on {HOST} {PORT}")
        except Exception as e:
           print(f"Unable to bind to host {HOST} and port {PORT} because of {e}")

        
    #*****************************************************************************************#
    def add_message(self,message):
     
       self.inputtxt.config(state=tk.NORMAL)
       self.inputtxt.insert(tk.END, message + '\n')
       self.inputtxt.config(state=tk.DISABLED)

#*****************************************************************************************#   
    def start_FL(self):


        self.request_for_models()


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
  
   

    def send_messages_to_all(self,message):   #in case of broadcasting
    
       for user in self.active_fogs:

          self.send_message_to_fog(user[1], message)

    def listen_for_messages(self,fog, username):

      while 1:

        message = fog.recv(1000000)#.decode('utf-8')
        message=pickle.loads(message)
        
        if message != '':
            print(username," ",message)
            msg= username+" "+message
            self.add_message(msg)

        else:
            print(f"The message send from Fog {username} is empty")


     
 


#*****************************************************************************************# 
    def request_for_models(self):
      for user in self.active_fogs:
         self.send_message_to_fog(user[1], 'Models')


#*****************************************************************************************# 
    def main(self):
    
       
       
        self.root.mainloop()
# Main function

    def receive_fogs(self):

       existedFog=False
       i=0
       while 1:
        fog, address = self.server.accept()
        while not existedFog and i< len(self.active_fogs) :
             if (self.active_fogs[i]['port']==address.port):
              existedFog=True
             else :
              i=i+1
          
        if (existedFog==False):
           print(f"Successfully reconnected to Fog {address[0]} {address[1]}")
        else :
          print(f"Successfully connected to Fog {address[0]} {address[1]}")


        threading.Thread(target=self.Fog_handler, args=(fog,existedFog )).start()

#*****************************************************************************************#    
    def Fog_handler(self,fog,existedFog):  
  
    
      while 1:
        if (existedFog==False):
          username = fog.recv(1000000)#.decode('utf-8')
          username=pickle.loads(username)
          if username != '':
            if username.find("Fog")!=-1:
               self.active_fogs.append((username, fog,None,None))  #username, adr, model, accuracy
               print( "SERVER~" + f"{username} added to the System")
               self.add_message(f"{username} added to the System")
               self.send_message_to_fog(fog,"Server ~~ Successfully connected to Cloud Server ")
               break
          else:
            print("Fog username is empty")
      print(self.active_fogs)
      threading.Thread(target=self.listen_for_messages, args=(fog, username, )).start()
   
if __name__ == '__main__':
  
    args = args_parser()   # ajoute id 
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(torch.cuda.is_available())
    cloud =Cloud(HOST=HOST, PORT=PORT, LISTENER_LIMIT=LISTENER_LIMIT,args=args)

    cloud.server.listen(LISTENER_LIMIT)
    threading.Thread(target=cloud.receive_fogs, args=()).start()
    cloud.main()
    