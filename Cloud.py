# Import required modules

import socket
import threading
import pickle
from matplotlib.style import use
import tkinter as tk
from utils.Options import args_parser
import torch






HOST = '127.0.0.1'
PORT = 12345 # You can use any port between 0 to 65535
LISTENER_LIMIT = 5


# Function to listen for upcoming messages from a client
class Cloud:
    def __init__(self,HOST,PORT,LISTENER_LIMIT,args):
        self.HOST=HOST
        self.PORT=PORT
        self.LISTENER_LIMIT=LISTENER_LIMIT
        self.active_fogs = []
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
    def send_message_to_fog(self,fog, message):

          
          fog.send(pickle.dumps(message)) #sendall    


#*****************************************************************************************# 
    def send_messages_to_all(self,message):
    
       for user in self.active_fogs:

          self.send_message_to_fog(user[1], message)

     # Function to handle client

#*****************************************************************************************# 
    def listen_for_messages(self,fog, username):

      while 1:

        message = fog.recv(1000000)#.decode('utf-8')
        message=pickle.loads(message)
        
        if message != '':
            print(username,message)

        else:
            print(f"The message send from Fog {username} is empty")


     
 
#*****************************************************************************************#    
    def Fog_handler(self,fog):  
    # Server will listen for client message that will
    # Contain the username
      while 1:

        username = fog.recv(1000000)#.decode('utf-8')
        username=pickle.loads(username)
        if username != '':
            if username.find("Fog")!=-1:
               self.active_fogs.append((username, fog,None,None))  #username, adr, model, accuracy
               #prompt_message = "SERVER~" + f"{username} added to the chat"
               print( "SERVER~" + f"{username} added to the chat")
               #self.send_messages_to_all(prompt_message)
               break
        else:
            print("Fog username is empty")
      print(self.active_fogs)
      threading.Thread(target=self.listen_for_messages, args=(client, username, )).start()
   

#*****************************************************************************************# 
    def request_for_models(self):
      for user in self.active_fogs:
         self.send_message_to_fog(user[1], 'Models')

      

#*****************************************************************************************# 
    def main(self):
    
       
       
        self.root.mainloop()
# Main function

    def receive_fogs(self):

    
    # Set server limit
    
       while 1:

        fog, address = cloud.server.accept()
        print(f"Successfully connected to Fog {address[0]} {address[1]}")

        threading.Thread(target=cloud.Fog_handler, args=(fog, )).start()


if __name__ == '__main__':
  
    args = args_parser()   # ajoute id 
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(torch.cuda.is_available())
    cloud =Cloud(HOST=HOST, PORT=PORT, LISTENER_LIMIT=LISTENER_LIMIT,args=args)

    cloud.server.listen(LISTENER_LIMIT)
    threading.Thread(target=cloud.receive_fogs, args=()).start()
    cloud.main()
    