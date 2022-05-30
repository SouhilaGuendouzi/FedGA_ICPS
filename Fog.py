# Import required modules
import socket
import threading

from matplotlib.style import use

HOST = '127.0.0.1'
PORT = 1234 # You can use any port between 0 to 65535
LISTENER_LIMIT = 5


# Function to listen for upcoming messages from a client
class Fog:
    def __init__(self,HOST,PORT,LISTENER_LIMIT):
        self.HOST=HOST
        self.PORT=PORT
        self.LISTENER_LIMIT=LISTENER_LIMIT
        self.active_clients = []
        self.clients=[]
        self.registry={} #§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§#
        self.Actuator=False
        self.globalModel=None
        self.scoring=0
        self.pyhical_attributes={}      
         # Creating the socket class object
         # AF_INET: we are going to use IPv4 addresses
         # SOCK_STREAM: we are using TCP packets for communication
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Creating a try catch block
        try:
        # Provide the server with an address in the form of
        # host IP and port
          self.server.bind((HOST, PORT))
          print(f"Running the server on {HOST} {PORT}")
        except:
           print(f"Unable to bind to host {HOST} and port {PORT}")
        
    def send_message_to_client(self,client, message):

          client.send(message.encode()) #sendall

      # Function to send any new message to all the clients that
      # are currently connected to this server


    def send_messages_to_all(self,message):
    
       for user in self.active_clients:

          self.send_message_to_client(user[1], message)

     # Function to handle client
    def listen_for_messages(self,client, username):

      while 1:

        message = client.recv(1000000).decode('utf-8')
        
        if message != '':
            print(username,message)
            
            #final_msg = username + '~' + message
            
            #print(final_msg)
            #self.send_messages_to_all(final_msg)

        else:
            print(f"The message send from client {username} is empty")


     # Function to send message to a single client
    
    def client_handler(self,client):  
    # Server will listen for client message that will
    # Contain the username
      while 1:

        username = client.recv(1000000).decode('utf-8')
        if username != '':
            self.active_clients.append((username, client))
            #prompt_message = "SERVER~" + f"{username} added to the chat"
            print( "SERVER~" + f"{username} added to the chat")
            #self.send_messages_to_all(prompt_message)
            break
        else:
            print("Client username is empty")

      threading.Thread(target=self.listen_for_messages, args=(client, username, )).start()
   
# Main function
def main():


    # Set server limit
    fog =Fog(HOST=HOST, PORT=PORT, LISTENER_LIMIT=LISTENER_LIMIT)
    fog.server.listen(LISTENER_LIMIT)

    # This while loop will keep listening to client connections
    while 1:

        client, address = fog.server.accept()
        print(f"Successfully connected to client {address[0]} {address[1]}")

        threading.Thread(target=fog.client_handler, args=(client, )).start()


if __name__ == '__main__':
    main()