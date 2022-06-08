# import required modules
#from utils.Empty import Empty
import socket
import threading
import pickle
import tkinter as tk
from queue import Empty

from Entities.Model import *
from utils.EdgeOptions import args_parser
#from utils.Empty import Empty
from utils.create_MNIST_datasets import get_FashionMNIST
import torch.nn.functional as F
import torch
from torch import import_ir_module, nn
import copy
from torch.utils.data import DataLoader


HOST = '127.0.0.1'
PORT = 12346

global i

root =tk.Tk()
root.geometry("1000x600")
root.title(" Client Interface ") 


class Edge(object):
     def __init__(self,model, dataset, args):#,device
         self.args=args
         self.id=args.id
         self.datasetTrain = dataset[0]
         self.datasetTest = dataset[1]
         self.model=model
         self.GlobalModelWeghts=copy.deepcopy(model.state_dict())
         self.weightsJustforReturn=copy.deepcopy(model.state_dict())
         self.accuracy=[None,None]
         self.loss=None
         self.domain=args.domain
         self.task=args.task
         self.portFog=args.portFog
         self.socket= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
         self.aggregationmethod="FedAVG"
          
         #self.inputtxt=None
         if torch.cuda.is_available():
              self.model.cuda()

         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#### Socket Requests and Responses
     def connect(self):
   
       Var= False
       # try except block
       try:

           # Connect to the server
           self.socket.connect((HOST, self.portFog))
           print("Successfully connected to server")
           self.add_message("Successfully connected to Fog server \n")
           self.send_message(self.id,'Connection')
           Var= True

       except:
        print("Unable to connect to server", f"Unable to connect to server {HOST} {self.portFog}")
        
    
       threading.Thread(target=self.listen_for_messages_from_server, args=(self.socket, )).start()
    
       return Var


#*****************************************************************************************#
     def send_message(self,message,subject):
       try :
        if message != '':
           try:
             if isinstance(message, dict):  #on a fait dict pour le cas des vecteurs de poids
               objectToSend=Empty()
               objectToSend.id=self.id
               objectToSend.subject=subject
               objectToSend.completeModel=copy.deepcopy(self.model.state_dict())
               objectToSend.personnalizedModel=message
               objectToSend.accuracy=self.accuracy
               objectToSend.domain=self.domain
               objectToSend.task=self.task
               message = objectToSend

             else:
                  objectToSend=Empty()
                  objectToSend.data=message
                  objectToSend.subject=subject
                  message=objectToSend
               
           except Exception as e:
                 print(e)
           message = pickle.dumps(message)
           self.socket.send(message)
      
        else:
           print("Empty message", "Message cannot be empty")
       except Exception as e : 
           print(e)
          

#*****************************************************************************************#
     def add_message(self,message):
       inputtxt.config(state=tk.NORMAL)
       inputtxt.insert(tk.END, message )#+ '\n'
       inputtxt.config(state=tk.DISABLED)

#*****************************************************************************************#
     def listen_for_messages_from_server(self,socket):

       while 1:
        message = socket.recv(1000000)#.decode('utf-8')
        message=pickle.loads(message)
        if message != '':
           
            try:
                  self.add_message('Fog Server is requesting For: '+message.subject+"\n")
                  if (message.subject=='FLstart'):
                     self.add_message('Fog server is requesting for starting FL \n')
                     self.aggregationmethod=message.data
                     threading.Thread(target=self.Training, args=(True,)).start() #it returns the whole model

                  elif  (message.subject=='FL'):
                             self.add_message('Fog server is requesting for an Other  FL Round \n')
                    
                             threading.Thread(target=self.Updating, args=(message.data,True,)).start() #message.data is personnalized weights
                  elif  (message.subject=='FLEnd'):
                             self.add_message('Fog server Compeleted  the Last FL Round \n')
                             threading.Thread(target=self.Updating, args=(message.data,False,)).start() 

                  elif  (message.subject=='TLModel'):
                      self.model=message.data #le model tout entier 
                      threading.Thread(target=self.Training, args=(False,)).start() #it returns the whole model
                  
                  else :
                       self.add_message(message.subject+"\n")

                     
            except Exception as e: 
                print('Error from listen_for_messages_from_server', e)
           
        else:
            print("Error", "Message recevied from Server is empty")
#*****************************************************************************************#
     def Training(self,Request):
         if (Request==True):
             self.add_message('Starting FedGA-ICPS \n')
             if (self.aggregationmethod=='FedAVG'):
                threading.Thread(target=self.local_train_FedAVG, args=(Request,)).start() #it returns the whole model
             else :
                threading.Thread(target=self.local_train_Other, args=(Request,)).start() #it returns the whole model
         else :
              threading.Thread(target=self.local_train_FedAVG, args=(Request,)).start() #it returns the whole model


     def Updating(self,data,Request):

         if (self.aggregationmethod=='FedAVG'):
              threading.Thread(target=self.local_update_FedAVG, args=(data,Request,)).start() #it returns the whole model
         else: 
              threading.Thread(target=self.local_update_Other, args=(data,Request,)).start() #it returns the whole model
        
     def TransferLearningRequest(self):

        self.send_message(None,"RequestTLModel")
        

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
                 command=self.connect
                 )

        Connect.pack(padx=50, pady=10, side=tk.LEFT)


        Train = tk.Button(root, height = 2,
                 width = 20,
                 text ="Train",
                 command = lambda:self.Training(False)
                 )
        Train.pack( padx=50,pady=20, side=tk.LEFT)
        Upload = tk.Button(root, height = 2,
                 width = 20,
                 text ="Upload",
                 command = lambda:self.send_message(self.model.state_dict(),'LocalModel')
                 )
        Upload.pack(padx=5, pady=20, side=tk.LEFT)
        TLRequest = tk.Button(root, height = 2,
                 width = 20,
                 text ="Request For Model",
                 command = lambda:self.TransferLearningRequest()
                 )
        TLRequest.pack(padx=5, pady=20, side=tk.LEFT)
        root.mainloop()



#### Learning Processes

     def local_train_FedAVG(self,Request):# with its own weights in case of FedAVG, with similar models
         self.add_message('Training')
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
         
         self.add_message('. \n')   
         self.add_message("Testing")
         acc, loss= self.test_img('train')
         self.add_message('Accuracy Train  \t'+str(acc)+"\n")
         acc, loss= self.test_img('test')
         self.add_message('Accuracy Test   \t'+str(acc)+"\n")  
         self.weightsJustforReturn=self.weights
         if (Request==True):
             self.send_message(self.weightsJustforReturn,'LocalModel')
         return self.weights, sum(epoch_loss) / len(epoch_loss)# state_dict(): Returns a dictionary containing a complete state of the module /// , loss_function of model_i
#*****************************************************************************************#
     def local_train_Other(self,Request):# with its own weights
         self.add_message('Training')
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
         for i in range(10):
          try :
           
           del[self.weights['features.{}.weight'.format(i)]]
      
           del[self.weights['features.{}.bias'.format(i)]]

          except Exception as e:
             print(e)
         self.add_message('. \n')   
         self.add_message("Testing")
         acc, loss= self.test_img('train')
         self.add_message('Accuracy Train  \t'+str(acc)+"\n")
         acc, loss= self.test_img('test')
         self.add_message('Accuracy Test   \t'+str(acc)+"\n")   
         # Here ==> self.weights contains only classification layers (fully connected layers)
         self.weightsJustforReturn=self.weights
         if (Request==True):
             self.send_message(self.weightsJustforReturn,'LocalModel')
         return self.weights, sum(epoch_loss) / len(epoch_loss)# state_dict(): Returns a dictionary containing a complete state of the module /// , loss_function of model_i

#*****************************************************************************************#
     def local_update_FedAVG(self,weights_global,Request):   #with the global layers weights in case of FedAVG, with similar models
         self.add_message('Updating .')
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
              
             
            if (self.loss>sum(batch_loss)/len(batch_loss)):
                self.loss=sum(batch_loss)/len(batch_loss)
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            else :
                epoch_loss.append(self.loss)
                self.model.load_state_dict(self.previous_weights)
         self.add_message('. \n')   
         self.add_message("Testing")
         acc, loss= self.test_img('train')
         self.add_message('Accuracy Train  \t'+str(acc)+"\n")
         acc, loss= self.test_img('test')
         self.add_message('Accuracy Test   \t'+str(acc)+"\n")    
         self.weightsJustforReturn=self.weights
         if (Request==True): self.send_message(self.weightsJustforReturn,'LocalModel')
         return  self.weights, sum(epoch_loss) / len(epoch_loss)# state_dict(): Returns a dictionary containing a complete state of the module /// , loss_function of model_i


 #*****************************************************************************************#       
     def local_update_Other(self,weights_global,Request): #with the global personnalized layers weights in case of FedPer, FedGa ..
         self.add_message('Updating .')
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
            self.add_message('.')
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
             break#('')print
         self.add_message('. \n')   
         self.add_message("Testing")
         acc, loss= self.test_img('train')
         self.add_message('Accuracy Train  \t'+str(acc)+"\n")
         acc, loss= self.test_img('test')
         self.add_message('Accuracy Test   \t'+str(acc)+"\n")  
         self.weightsJustforReturn=self.weights
         if (Request==True): self.send_message(self.weightsJustforReturn,'LocalModel')
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
        
           self.add_message('.')   
           data, target = data.cuda(), target.cuda() # add this line for GPU 
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
       
        self.add_message('. \n')   
        if (datasetName=='test'):
           self.accuracy[0]=accuracy
        elif (datasetName=='train'):
            self.accuracy[1]=accuracy   
        return accuracy, test_loss


#*****************************************************************************************#
if __name__ == '__main__':
    args = args_parser()   
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #print(torch.cuda.is_available())
    list_users=range(4)
    if (args.id in list_users): 
      mnist_non_iid_train_dls, mnist_non_iid_test_dls = get_FashionMNIST(args.iid,
      n_samples_train =1500, n_samples_test=250, n_clients =4,  # i have calculated because there are 60000/ 1000
      batch_size =50, shuffle =True)     
      datasetTrain= mnist_non_iid_train_dls[args.id]  
      datasetTest=mnist_non_iid_test_dls[args.id]      #(1500+250) samples for each client / 50 batch size ==num of epochs / and 30 number of batch
      dataset =[datasetTrain,datasetTest]
    else :
      mnist_non_iid_train_dls, mnist_non_iid_test_dls = get_FashionMNIST(args.iid,
      n_samples_train =1500, n_samples_test=250, n_clients =4,  # i have calculated because there are 60000/ 1000
      batch_size =50, shuffle =True)     
      datasetTrain= mnist_non_iid_train_dls[2]  
      datasetTest=mnist_non_iid_test_dls[2]      #(1500+250) samples for each client / 50 batch size ==num of epochs / and 30 number of batch
      dataset =[datasetTrain,datasetTest]
    print(args.model)
    if (args.model=='default'):
        model =Model_Fashion()
    elif (args.model=='A'):
        model= Model_A()
    elif (args.model=='B'):
        model= Model_B()
    elif (args.model=='C'):
        model= Model_C()
    elif (args.model=='D'):
        model= Model_D()
    edge =Edge(model=model,dataset=dataset,args=args)
   
   
    edge.main()

          




