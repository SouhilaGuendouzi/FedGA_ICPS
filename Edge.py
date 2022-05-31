# import required modules
import socket
import threading
import pickle
import tkinter as tk
from Entities.Model import *
from utils.Options import args_parser
from utils.create_MNIST_datasets import get_FashionMNIST
import torch.nn.functional as F
import torch
from torch import nn
import copy
from torch.utils.data import DataLoader


HOST = '127.0.0.1'
PORT = 1234

global i
i=0
root =tk.Tk()
root.geometry("600x600")
root.title(" Client Interface ") 


# Creating a socket object
# AF_INET: we are going to use IPv4 addresses
# SOCK_STREAM: we are using TCP packets for communication


class Edge(object):
     def __init__(self, id,model, dataset, args):#,device
         self.id=id
         self.datasetTrain = dataset[0]
         self.datasetTest = dataset[1]
         self.model=model
        
         self.accuracy=None
         self.loss=None
         self.args=args
         self.socket= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
         #self.inputtxt=None
         if torch.cuda.is_available():
              self.model.cuda()

         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     def connect(self):
   
       Var= False
       # try except block
       try:

        # Connect to the server
           self.socket.connect((HOST, PORT))
           print("Successfully connected to server")
           self.send_message('Client {}'.format(self.id))
           Var= True

       except:
        print("Unable to connect to server", f"Unable to connect to server {HOST} {PORT}")
        

   # username = input('enter your username')
    #if username != '':
    #    client.sendall(username.encode())
    #else:
    #    print("Invalid username", "Username cannot be empty")
    
       threading.Thread(target=self.listen_for_messages_from_server, args=(self.socket, )).start()
    
       return Var


#*****************************************************************************************#
     def send_message(self,message):
       try :
        #message =input("entrer un message")
        print(message,'aaaaaaaaa')
        if message != '':
           message = pickle.dumps(message)
           print('cest fait')
           self.socket.send(message)
      
        else:
           print("Empty message", "Message cannot be empty")
       except Exception as e : 
           print(e)
          

#*****************************************************************************************#
     def add_message(self,message):
       inputtxt.config(state=tk.NORMAL)
       inputtxt.insert(tk.END, message + '\n')
       inputtxt.config(state=tk.DISABLED)

#*****************************************************************************************#
     def listen_for_messages_from_server(self,socket):

       while 1:

        message = socket.recv(1000000)#.decode('utf-8')
        message=pickle.loads(message)
        if message != '':
        
           
            self.add_message("GLobal model recevied from Server")
            
        else:
            print("Error", "Message recevied from Server is empty")
#*****************************************************************************************#
     def Training(self):
   
         
         threading.Thread(target=self.local_update_FedAVG, args=()).start()
        
        
        

 

#*****************************************************************************************#
     def main(self):
        global inputtxt 
        l = tk.Label(text = "Client {}".format(self.id))
        l.pack()
        inputtxt = tk.Text(root, height = 25,
                width = 60,
                bg = "light yellow")

        inputtxt.pack()
        inputtxt.configure(state='disabled')
        Connect =tk.Button(root, height = 2,
                 width = 20,
                 text ="Connect",
                 #command=self.connect
                 command=self.connect
                 )

        Connect.pack(padx=100, pady=10, side=tk.LEFT)


        Train = tk.Button(root, height = 2,
                 width = 20,
                 text ="Train",
                 command = lambda:self.Training()
                 )
        Train.pack(padx=5, pady=20, side=tk.LEFT)
        Upload = tk.Button(root, height = 2,
                 width = 20,
                 text ="Upload",
                 command = lambda:self.send_message(self.model.state_dict())
                 )
        Upload.pack(padx=5, pady=20, side=tk.LEFT)
        root.mainloop()



#*****************************************************************************************#
     def local_update_FedAVG(self):# with its own weights in case of FedAVG, with similar models
         self.add_message('training')
         
         self.model.train()
         self.loss_func = nn.CrossEntropyLoss()
         self.data = self.datasetTrain
         self.w= self.model.state_dict()
         optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
         epoch_loss = []
    
         for iter in range(self.args.local_ep):
            self.add_message('.')
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.data):
               
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                log_probs = self.model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
              
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            self.loss=sum(batch_loss)/len(batch_loss)
         self.weights=copy.deepcopy(self.model.state_dict())  #it contains all layers weights
         
            
         msg= self.test_img('train')
         self.add_message('Accuracy Train'+str(msg[1]))
         msg= self.test_img('test')
         self.add_message('Accuracy Test'+str(msg[1]))
         return self.weights, sum(epoch_loss) / len(epoch_loss)# state_dict(): Returns a dictionary containing a complete state of the module /// , loss_function of model_i
#*****************************************************************************************#
     def local_update_Other(self):# with its own weights
        
         self.model.train()
         self.loss_func = nn.CrossEntropyLoss()
         self.data = self.datasetTrain
         self.w= self.model.state_dict()
         optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
         epoch_loss = []
    
         for iter in range(self.args.local_ep):
           
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.data):
               
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                log_probs = self.model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
              
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            self.loss=sum(batch_loss)/len(batch_loss)
         self.weights=copy.deepcopy(self.model.state_dict())  #it contains all layers weights
         for i in range(10):
          try :
           
           del[self.weights['features.{}.weight'.format(i)]]
      
           del[self.weights['features.{}.bias'.format(i)]]

          except:
             print('')
            
         # Here ==> self.weights contains only classification layers (fully connected layers)
         return self.weights, sum(epoch_loss) / len(epoch_loss)# state_dict(): Returns a dictionary containing a complete state of the module /// , loss_function of model_i

#*****************************************************************************************#
     def local_update(self,weights_global):   #with the global layers weights in case of FedAVG, with similar models
 
         self.model.train()
         self.loss_func = nn.CrossEntropyLoss()
         self.previous_weights=self.model.state_dict()
         self.data = DataLoader(self.datasetTrain, shuffle=True,batch_size=self.args.local_bs)
         self.data = self.datasetTrain
         self.w=weights_global
         self.model.load_state_dict(self.w)

         optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
       
         epoch_loss = []
        
         
         for iter in range(self.args.local_ep):
           
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.data):

                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                log_probs = self.model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
              
             
            if (self.loss>sum(batch_loss)/len(batch_loss)):
                self.loss=sum(batch_loss)/len(batch_loss)
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            else :
                epoch_loss.append(self.loss)
                self.model.load_state_dict(self.previous_weights)
           
        
         return  self.weights, sum(epoch_loss) / len(epoch_loss)# state_dict(): Returns a dictionary containing a complete state of the module /// , loss_function of model_i


 #*****************************************************************************************#       
     def local_updatePer(self,weights_global): #with the global personnalized layers weights in case of FedPer, FedGa ..
         
         loss_func = nn.CrossEntropyLoss()
         self.weights=copy.deepcopy(self.model.state_dict())  #it contains all layers weights
         self.w=weights_global #it contains only fully connected layers
         self.previous_weights=self.model.state_dict()
    
         self.weights.update(self.w)
         self.data = self.datasetTrain
         self.model.load_state_dict(self.weights)
         self.model.train()

        
         optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
         epoch_loss = []

         for iter in range(self.args.local_ep):
          
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.data):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.model.zero_grad()
                log_probs = self.model(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())


            if (self.loss>sum(batch_loss)/len(batch_loss)):
                  self.loss=sum(batch_loss)/len(batch_loss)
                  epoch_loss.append(sum(batch_loss)/len(batch_loss))
            else :
                epoch_loss.append(self.loss)
                self.model.load_state_dict(self.previous_weights)
            
            
         self.weights=copy.deepcopy(self.model.state_dict())  #fih koulchi
         for i in range(10):
          try :
           
           del[self.weights['features.{}.weight'.format(i)]]
      
           del[self.weights['features.{}.bias'.format(i)]]

          except:
             print('')
         
     
         return self.weights, sum(epoch_loss) / len(epoch_loss) # state_dict(): Returns a dictionary containing a complete state of the module /// , loss_function of model_i
    
#*****************************************************************************************#
     def test_img(self,datasetName):


        self.model.eval()
        #self.data = DataLoader(dataset=dataset, shuffle=True)
        if (datasetName=='test'):
           self.data=self.datasetTest
        elif (datasetName=='train'):
            self.data=self.datasetTrain
        self.w=self.model.state_dict()
        
       
    
        test_loss = 0
        correct = 0
    
        
        for idx, (data, target) in enumerate(self.data): #self.data= 4 (number of batches) self.data.dataset=1919 ==> samples in all batch
        
    
           #data, target = data.cuda(), target.cuda() # add this line for GPU 
           log_probs =  self.model(data)
           
           # sum up batch loss
           test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
           # get the index of the max log-probability
           _,y_pred = log_probs.max(1,keepdim=True)
           #==> y_pred = log_probs.data.max(1, keepdim=True)[1]
           correct += torch.sum(y_pred.view(-1,1)==target.view(-1, 1)).item() #==> y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
           

        test_loss /= len(self.data.dataset)
        accuracy = 100.00 * correct / len(self.data.dataset)
       

        if self.args.verbose:
            
            print('\n Client: {}  {} set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(self.id,datasetName, test_loss, correct, len(self.data.dataset), accuracy))
            
           
        return accuracy, test_loss


#*****************************************************************************************#
if __name__ == '__main__':
    args = args_parser()   # ajoute id 
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(torch.cuda.is_available())

    mnist_non_iid_train_dls, mnist_non_iid_test_dls = get_FashionMNIST(args.iid,
    n_samples_train =1500, n_samples_test=250, n_clients =4,  # i have calculated because there are 60000/ 1000
    batch_size =50, shuffle =True)     
    datasetTrain= mnist_non_iid_train_dls[args.id]  
    datasetTest=mnist_non_iid_test_dls[args.id]      #(1500+250) samples for each client / 50 batch size ==num of epochs / and 30 number of batch
    dataset =[datasetTrain,datasetTest]
    model =Model_Fashion()
    edge =Edge(id=args.id,model=model,dataset=dataset,args=args)
   
   
    edge.main()





   #connect()
   #while 1:
    # threading.Thread(target=listen_for_messages_from_server, args=(client, )).start()
     #threading.Thread(target=send_message).start()
          




