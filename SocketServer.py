

# coding: utf-8

import socket
import pickle 
from FedAVG import FedAvg
import select
from _thread import *

list_of_clients = []
def clientthread(conn, addr):
 
    # sends a message to the client whose user object is conn
    conn.send("Welcome to this chatroom!")
 
    while True:
            try:
                message = conn.recv(2048)
                if message:
 
                    """prints the message and address of the
                    user who just sent the message on the server
                    terminal"""
                    print ("<" + addr[0] + "> " + message)
 
                    # Calls broadcast function to send message to all
                    message_to_send = "<" + addr[0] + "> " + message
                    broadcast(message_to_send, conn)
 
                else:
                    """message may have no content if the connection
                    is broken, in this case we remove the connection"""
                    remove(conn)
 
            except:
                continue
def remove(connection):
    if connection in list_of_clients:
        list_of_clients.remove(connection)
 
"""Using the below function, we broadcast the message to all
clients who's object is not the same as the one sending
the message """
def broadcast(message, connection):
    for clients in list_of_clients:
        if clients!=connection:
            try:
                clients.send(message)
            except:
                clients.close()
 
                # if the link is broken, we remove the client
    remove(clients)



socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.bind(('192.168.69.163', 15555))
weights=[]
num_clients=0

socket.listen(100)
while True:
    conn, addr = server.accept()
 
    """Maintains a list of clients for ease of broadcasting
    a message to all available people in the chatroom"""
    list_of_clients.append(conn)
 
    # prints the address of the user that just connected
    print (addr[0] + " connected")
 
    # creates and individual thread for every user
    # that connects
    start_new_thread(clientthread,(conn,addr))   



        #client, address = socket.accept()

        #print ("{} connected".format( address ))

        #response = client.recv(4096)
        
        #print(response.decode())
       # print(response)
        #if response != "":
             #   weights[num_clients]= pickle.loads( response.decode())
              #  print (weights[num_clients])
            
             #   clientO, addressO=  client, address

print ("Close")
client.close()


