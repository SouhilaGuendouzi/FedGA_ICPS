

# coding: utf-8

import socket
import pickle 
socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.bind(('', 15555))
weights=[]
while True:
        socket.listen(5)
        client, address = socket.accept()
        print ("{} connected".format( address ))

        response = client.recv(255)
        weights= pickle.loads( response)
        if response != "":
                print (weights)
                print (weights.decode())

print ("Close")
client.close()