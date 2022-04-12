

# coding: utf-8

import socket
import pickle 
from FedAVG import FedAvg



socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.bind(('192.168.69.163', 15555))
weights=[]
num_clients=0
list_of_clients = []
while True:
        socket.listen(5)
        client, address = socket.accept()

        print ("{} connected".format( address ))

        response = client.recv(4096)
        print(response.decode())
        
        if response != "":
                weights[num_clients]= pickle.loads( response.decode())
                print (weights[num_clients])
            
                clientO, addressO=  client, address

print ("Close")
client.close()