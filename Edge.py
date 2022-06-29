# import required modules
#from utils.Empty import Empty
from email import message
import socket
import threading
import pickle
import tkinter as tk
from queue import Empty
from turtle import bgcolor, color

from Entities.Model import *
from utils.EdgeOptions import args_parser
#from utils.Empty import Empty
from utils.create_MNIST_datasets import get_FashionMNIST
import torch.nn.functional as F
import torch
from torch import import_ir_module, nn
import copy
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from PIL import Image, ImageTk



import tkinter.font as tkFont


HOST = '127.0.0.1'
PORT = 12346


class Edge(object):
     def __init__(self,model, dataset, args):#,device
         self.args=args
         self.id=args.id
         self.datasetTrain = dataset[0]
         self.datasetTest = dataset[1]
         self.model=model
         self.GlobalModelWeghts=copy.deepcopy(model.state_dict())
         self.weightsJustforReturn=copy.deepcopy(model.state_dict())
         self.accuracy=[0.0,0.0]
         self.lossTable= [0.0, 0.0]
         self.loss=0.0
         self.domain=args.domain
         self.task=args.task
         self.portFog=args.portFog
         self.socket= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
         self.aggregationmethod="FedAVG"
          
         #self.inputtxt=None
         if torch.cuda.is_available():
              self.model.cuda()

         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


         self.accuracy_locals_train=[]
         self.accuracy_locals_test=[]
         self.loss_locals_train=[]
         self.loss_locals_test=[]
         self.roundGraphes=0


     
    
         self.root =tk.Tk()
         self.root.geometry("1100x800")
         self.root.configure(bg='#092C42')
         self.fontExample = tkFont.Font(family="Sitka Text", size=18, weight="bold")
         self.root.title("Edge {} ".format(self.id)) 
         self.image = Image.open("pictures/logoa.png")
        #image= image.resize(200, 100)
         self.resized_image= self.image.resize((230,50), Image.Resampling.LANCZOS)
         self.img = ImageTk.PhotoImage(self.resized_image)
         self.root.iconphoto(False, tk.PhotoImage(file='pictures/industry.png'))

         self.label = tk.Label(self.root, image=self.img)
         self.label.config(bg='#092C42')
        #label.pack(padx=0, pady=0,side=tk.LEFT)
         self.label.place(relx = 0.02, rely =0.02)
         self.fontText= tkFont.Font(family="Sitka Text", size=12)

         self.domainUI= tk.Label(text = "Domain application : {}".format(self.domain),font=self.fontText,fg="white",bg='#092C42')
         self.domainUI.place(relx = 0.02, rely =0.35)

         self.taskUI= tk.Label(text = "Task application : {}".format(self.task),font=self.fontText,fg="white",bg='#092C42')
         self.taskUI.place(relx = 0.02, rely =0.4)
         self.trainACC=tk.Label(text = "Training accuracy : {}%".format(self.accuracy[0]),font=self.fontText,fg="white",bg='#092C42')
         self.trainACC.place(relx = 0.02, rely =0.45)
         self.testACC=tk.Label(text = "Testing accuracy : {}%".format(self.accuracy[1]),font=self.fontText,fg="white",bg='#092C42')
         self.testACC.place(relx = 0.02, rely =0.5)


         self.l = tk.Label(text = "Edge {}".format(self.id),font=self.fontExample,fg="white")
         self.l.config(bg='#092C42')
         self.l.place(relx = 0.02, rely =0.25)
         self.inputtxt = tk.Text(self.root, height = 25,
                width = 70,
                 bg='#DDEBF4',
                )#bg = "light yellow"
         self.font_terminal= tkFont.Font(family="Sitka Text", size=10)
         self.terminal_label=tk.Label(text = "Terminal output",font=self.font_terminal,fg="white")
         self.terminal_label.config(bg='#092C42')
         self.terminal_label.place(relx=0.4, rely=0.165)
         self.inputtxt.place(relx = 0.4, rely =0.20)
         self.inputtxt.configure(state='disabled')
         self.fontButton= tkFont.Font(family="Sitka Text", size=11, weight="bold")
         self.Connect =tk.Button(self.root, height = 2,
                 width = 20,
                 text ="Connect",
                 bg='#DDEBF4',
                 command=self.connect,
                 font=self.fontButton,
                 fg="#092C42"
                 )

         self.Connect.place(relx = 0.05, rely =0.87)

         self.Train = tk.Button(self.root, height = 2,
                 width = 20,
                 text ="Train",
                 bg='#DDEBF4',
                 font=self.fontButton,
                  fg="#092C42",
                 command = lambda:self.Training(False)
                 )
         self.Train.place(relx = 0.30, rely =0.87)
         self.Upload = tk.Button(self.root, height = 2,
                 width = 20,
                 text ="Upload",
                 bg='#DDEBF4',
                 font=self.fontButton,
                 fg="#092C42",
                 command = lambda:self.send_message(self.model.state_dict(),'LocalModel')
                 )
         self.Upload.place(relx = 0.55, rely =0.87)
         self.TLRequest = tk.Button(self.root, height = 2,
                 width = 20,
                 text ="Request \n For a Model",
                 bg='#DDEBF4',
                  font=self.fontButton,
                  fg="#092C42",
                 command = lambda:self.TransferLearningRequest()
                 )
         self.TLRequest.place(relx = 0.80, rely =0.87)

        
        

#### Socket Requests and Responses
     def connect(self):

       # try except block
       try:

           # Connect to the server
           self.socket.connect((HOST, self.portFog))
           print("Successfully connected to server")
           self.add_message(f'Edge{self.id}> Successfully connected to Fog server \n')
         
          
           self.send_message(self.id,'Connection')
           
       except:
        print("Unable to connect to server", f"Unable to connect to server {HOST} {self.portFog}")
      
       threading.Thread(target=self.listen_for_messages_from_server, args=(self.socket, )).start()
       
    
       #return Var


#*****************************************************************************************#
     def send_message(self,message,subject):
       try :
        if message != '':
            objectToSend=Empty()
            if (subject =='Connection'):
                objectToSend.id=self.id
                objectToSend.subject=subject
                message = objectToSend
            elif (subject=='LocalModel'):
               objectToSend.id=self.id
               objectToSend.subject=subject
               objectToSend.completeModel=copy.deepcopy(self.model.state_dict())
               objectToSend.personnalizedModel=message
               objectToSend.accuracy=self.accuracy
               objectToSend.loss=self.lossTable
               objectToSend.domain=self.domain
               objectToSend.task=self.task
               objectToSend.architecture=self.model
               message = objectToSend
               self.add_message(f'Edge{self.id}> Local model is sent to the Fog server \n')
        
            elif (subject=="FinalLocalModel"):
               objectToSend.id=self.id
               objectToSend.subject=subject
               objectToSend.completeModel=copy.deepcopy(self.model.state_dict())
               objectToSend.personnalizedModel=message
               objectToSend.accuracy=self.accuracy
               objectToSend.loss=self.lossTable
               objectToSend.domain=self.domain
               objectToSend.task=self.task
               objectToSend.architecture=self.model
               objectToSend.measures= [self.accuracy_locals_train,self.accuracy_locals_test,self.loss_locals_train,self.loss_locals_test]
               message = objectToSend
               self.add_message(f'Edge{self.id}> Local model is sent to the Fog server \n')
            elif (subject=="RequestTLModel"):
               objectToSend.id=self.id
               objectToSend.data=message
               objectToSend.subject=subject
               message=objectToSend
               self.add_message(f'Edge{self.id}> A request is sent to Fog server for Transfer Learning \n ')
           
            message = pickle.dumps(message)
            self.socket.send(message)
      
        else:
           print("Empty message", "Message cannot be empty")
       except Exception as e : 
           print(e)
          

#*****************************************************************************************#
     def add_message(self,message):
  
       #text_area.text_area('Output', value=st.session_state.edge.text,height=500, disabled=True)
       #st.write(self.text)
      

       #st.session_state["text_area"]=self.text
       self.inputtxt.config(state=tk.NORMAL)
       self.inputtxt.insert(tk.END, message )#+ '\n'
       self.inputtxt.config(state=tk.DISABLED)
       self.inputtxt.see(tk.END) 
       # text.edit_modified(0)

#*****************************************************************************************#
     def listen_for_messages_from_server(self,socket):

       while 1:
        message = socket.recv(1000000)#.decode('utf-8')
        message=pickle.loads(message)
        if message != '':
           
            try:
                  #self.add_message('Fog Server is requesting For: '+message.subject+"\n")
                  if (message.subject=='FLstart'):
                     self.roundGraphes+=1
                     self.add_message(f'Edge{self.id}> There is a request from Fog server to start Federated Learning \n')
                     self.aggregationmethod=message.data
                     threading.Thread(target=self.Training, args=(True,)).start() #it returns the whole model
                     

                  elif  (message.subject=='FL'):
                             self.roundGraphes+=1
                            
                             self.add_message(f'Edge{self.id}> There is a Request from the Fog server to continue Federated Learning process \n')
                    
                             threading.Thread(target=self.Updating, args=(message.data,True,"NotFinal")).start() #message.data is personnalized weights
                  elif  (message.subject=='FLEnd'):
                             self.roundGraphes+=1
                             self.add_message(f'Edge{self.id}> There is a last Request from the Fog server of Federated Learning  \n')
                             threading.Thread(target=self.Updating, args=(message.data,False,"Final")).start() 
                          

                  elif  (message.subject=='TLModel'):
                      self.add_message(f'Edge{self.id}> The appropriate model is received from the Fog server \n ')
                      #print(message.data)
                      self.model=message.data #le model tout entier 
                      threading.Thread(target=self.Training, args=(False,)).start() #it returns the whole model
                  
                  #else :
                   #    self.add_message(message.subject+"\n")

                     
            except Exception as e: 
                print('Error from listen_for_messages_from_server', e)
           
        else:
            print("Error", "Message recevied from Server is empty")
#*****************************************************************************************#
     def Training(self,Request):
      
         if (Request==True):
             self.add_message(f' Edge{self.id}> Starting FedGA-ICPS \n')
             if (self.aggregationmethod=='FedAVG'):
               t= threading.Thread(target=self.local_train_FedAVG, args=(Request,))
      
               t.start() #it returns the whole model
             else :
               
                threading.Thread(target=self.local_train_Other, args=(Request,)).start() #it returns the whole model
         else :
               t= threading.Thread(target=self.local_train_FedAVG, args=(Request,))
             
               t.start() #it returns the whole model
      


     def Updating(self,data,Request,statut):

         if (self.aggregationmethod=='FedAVG'):
              threading.Thread(target=self.local_update_FedAVG, args=(data,Request,statut)).start() #it returns the whole model
         else: 
              threading.Thread(target=self.local_update_Other, args=(data,Request,statut)).start() #it returns the whole model
  
        
     def TransferLearningRequest(self):
        message=[self.domain,self.task]
        self.send_message(message,"RequestTLModel")
        

#*****************************************************************************************#
     def main(self):
        global inputtxt 
        root =tk.Tk()
        root.geometry("1100x800")
        root.configure(bg='#092C42')
        fontExample = tkFont.Font(family="Sitka Text", size=18, weight="bold")
        root.title("Edge {} ".format(self.id)) 
        image = Image.open("pictures/logoa.png")
        #image= image.resize(200, 100)
        resized_image= image .resize((230,50), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(resized_image)
        root.iconphoto(False, tk.PhotoImage(file='pictures/industry.png'))

        label = tk.Label(root, image=img)
        label.config(bg='#092C42')
        #label.pack(padx=0, pady=0,side=tk.LEFT)
        label.place(relx = 0.02, rely =0.02)
        fontText= tkFont.Font(family="Sitka Text", size=12)

        domain= tk.Label(text = "Domain application : {}".format(self.domain),font=fontText,fg="white",bg='#092C42')
        domain.place(relx = 0.02, rely =0.35)

        task= tk.Label(text = "Task application : {}".format(self.task),font=fontText,fg="white",bg='#092C42')
        task.place(relx = 0.02, rely =0.4)
        trainACC=tk.Label(text = "Training accuracy : {}%".format(self.accuracy[0]),font=fontText,fg="white",bg='#092C42')
        trainACC.place(relx = 0.02, rely =0.45)
        testACC=tk.Label(text = "Testing accuracy : {}%".format(self.accuracy[1]),font=fontText,fg="white",bg='#092C42')
        testACC.place(relx = 0.02, rely =0.5)


        l = tk.Label(text = "Edge {}".format(self.id),font=fontExample,fg="white")
        l.config(bg='#092C42')
        l.place(relx = 0.02, rely =0.25)
        inputtxt = tk.Text(root, height = 25,
                width = 70,
                 bg='#DDEBF4',
                )#bg = "light yellow"
        font_terminal= tkFont.Font(family="Sitka Text", size=10)
        terminal_label=tk.Label(text = "Terminal output",font=font_terminal,fg="white")
        terminal_label.config(bg='#092C42')
        terminal_label.place(relx=0.4, rely=0.165)
        inputtxt.place(relx = 0.4, rely =0.20)
        inputtxt.configure(state='disabled')
        fontButton= tkFont.Font(family="Sitka Text", size=11, weight="bold")
        Connect =tk.Button(root, height = 2,
                 width = 20,
                 text ="Connect",
                 bg='#DDEBF4',
                 command=self.connect,
                 font=fontButton,
                 fg="#092C42"
                 )

        Connect.place(relx = 0.05, rely =0.87)
        #.pack(padx=50, pady=20, side=tk.LEFT)


        Train = tk.Button(root, height = 2,
                 width = 20,
                 text ="Train",
                 bg='#DDEBF4',
                 font=fontButton,
                  fg="#092C42",
                 command = lambda:self.Training(False)
                 )
        Train.place(relx = 0.30, rely =0.87)
        Upload = tk.Button(root, height = 2,
                 width = 20,
                 text ="Upload",
                 bg='#DDEBF4',
                 font=fontButton,
                 fg="#092C42",
                 command = lambda:self.send_message(self.model.state_dict(),'LocalModel')
                 )
        Upload.place(relx = 0.55, rely =0.87)
        TLRequest = tk.Button(root, height = 2,
                 width = 20,
                 text ="Request \n For a Model",
                 bg='#DDEBF4',
                  font=fontButton,
                  fg="#092C42",
                 command = lambda:self.TransferLearningRequest()
                 )
        TLRequest.place(relx = 0.80, rely =0.87)
        root.mainloop()



#### Learning Processes

     def local_train_FedAVG(self,Request):# with its own weights in case of FedAVG, with similar models
         self.add_message(f'Edge{self.id}> Training')
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
         self.add_message(f'Edge{self.id}> Testing ')
         acc, loss= self.test_img('train')
         self.add_message(f'Edge{self.id}> Training Accuracy  \t'+str(round(acc,2))+"\n")
         accT, lossT= self.test_img('test')
         self.add_message(f'Edge{self.id}> Testing Accuracy  \t'+str(round(accT,2))+"\n")  
         self.weightsJustforReturn=self.weights
         if (Request==True):
             self.accuracy_locals_train.append(acc)
             self.accuracy_locals_test.append(accT)
             self.loss_locals_train.append(loss)
             self.loss_locals_test.append(lossT)
             self.send_message(self.weightsJustforReturn,'LocalModel')
         print('fin')
         return self.weights, sum(epoch_loss) / len(epoch_loss)# state_dict(): Returns a dictionary containing a complete state of the module /// , loss_function of model_i
#*****************************************************************************************#
     def local_train_Other(self,Request):# with its own weights
         self.add_message(f'Edge{self.id}> Training')
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
      
           del[self.weights[f'features.{i}.bias']]

          except Exception as e:
             print(e)
         self.add_message('. \n')   
         self.add_message(f"Edge{self.id}> Testing")
         acc, loss= self.test_img('train')
         self.add_message(f'Edge{self.id}> Training Accuracy  \t'+str(round(acc,2))+"\n")
         accT, lossT= self.test_img('test')
         self.add_message(f'Edge{self.id}> Testing Accuracy    \t'+str(round(accT,2))+"\n")  
         self.weightsJustforReturn=self.weights
         if (Request==True):
             self.accuracy_locals_train.append(acc)
             self.accuracy_locals_test.append(accT)
             self.loss_locals_train.append(loss)
             self.loss_locals_test.append(lossT)
             self.send_message(self.weightsJustforReturn,'LocalModel')
         #print(len( self.weightsJustforReturn))
         return self.weights, sum(epoch_loss) / len(epoch_loss)# state_dict(): Returns a dictionary containing a complete state of the module /// , loss_function of model_i

#*****************************************************************************************#
     def local_update_FedAVG(self,weights_global,Request,statut):   #with the global layers weights in case of FedAVG, with similar models
         self.add_message(f'Edge{self.id}> Updating .')
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
         self.add_message(f"Edge{self.id}>Testing")
         acc, loss= self.test_img('train')
         self.add_message(f'Edge{self.id}> Training Accuracy   \t'+str(round(acc,2))+"\n")
         accT, lossT= self.test_img('test')
         self.add_message('Testing Accuracy   \t'+str(round(accT,2))+"\n")  
         self.weightsJustforReturn=self.weights
         if (Request==True):
             self.accuracy_locals_train.append(acc)
             self.accuracy_locals_test.append(accT)
             self.loss_locals_train.append(loss)
             self.loss_locals_test.append(lossT)
             self.send_message(self.weightsJustforReturn,'LocalModel')
         if(statut=="Final") : self.send_message(self.weightsJustforReturn,'FinalLocalModel')
         return  self.weights, sum(epoch_loss) / len(epoch_loss)# state_dict(): Returns a dictionary containing a complete state of the module /// , loss_function of model_i


 #*****************************************************************************************#       
     def local_update_Other(self,weights_global,Request,statut): #with the global personnalized layers weights in case of FedPer, FedGa ..
         self.add_message(f'Edge{self.id}> Updating .')
         loss_func = nn.CrossEntropyLoss()
         self.weights=copy.deepcopy(self.model.state_dict())  #it contains all layers weights
         self.w=weights_global #it contains only fully connected layers
         #print("Length global model weights", len(weights_global))
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
            print('')
         self.add_message('. \n')   
         self.add_message(f'Edge{self.id}> Testing')
         acc, loss= self.test_img('train')
         self.add_message(f'Edge{self.id}> Training Accuracy   \t'+str(round(acc,2))+"\n")
         accT, lossT= self.test_img('test')
         self.add_message(f'Edge{self.id}> Testing Accuracy   \t'+str(round(accT,2))+"\n")  
         self.weightsJustforReturn=self.weights
         if (Request==True):
          self.accuracy_locals_train.append(acc)
          self.accuracy_locals_test.append(accT)
          self.loss_locals_train.append(loss)
          self.loss_locals_test.append(lossT)
          self.send_message(self.weightsJustforReturn,'LocalModel')
         if(statut=="Final") : self.send_message(self.weightsJustforReturn,'FinalLocalModel')
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
        
           #self.add_message('.')   
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
           self.lossTable[0]=round(test_loss,2)
           self.accuracy[0]=round(accuracy,2)
           self.testACC['text']="Testing Accuracy: "+str( self.accuracy[0])+"%"
        elif (datasetName=='train'):
            self.lossTable[1]=round(test_loss,2)
            self.accuracy[1]=round(accuracy,2)
            self.trainACC['text']="Training Accuracy: "+str( self.accuracy[1])+"%"
        #print(self.accuracy)
        return accuracy, test_loss

   
     
    

#*****************************************************************************************#
if __name__ == '__main__':
    args = args_parser() 
   
    print("next")
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #print(torch.cuda.is_available())
    list_users=range(5)
    if (args.id in list_users): 
      mnist_non_iid_train_dls, mnist_non_iid_test_dls = get_FashionMNIST(args.iid,
      n_samples_train =1500, n_samples_test=250, n_clients =4,  # i have calculated because there are 60000/ 1000
      batch_size =50, shuffle =True)     
      datasetTrain= mnist_non_iid_train_dls[args.id-1]  
      datasetTest=mnist_non_iid_test_dls[args.id-1]      #(1500+250) samples for each client / 50 batch size ==num of epochs / and 30 number of batch
      dataset =[datasetTrain,datasetTest]
    else :
      mnist_non_iid_train_dls, mnist_non_iid_test_dls = get_FashionMNIST(args.iid,
      n_samples_train =1500, n_samples_test=250, n_clients =4,  # i have calculated because there are 60000/ 1000
      batch_size =50, shuffle =True)     
      datasetTrain= mnist_non_iid_train_dls[2]  
      datasetTest=mnist_non_iid_test_dls[2]      #(1500+250) samples for each client / 50 batch size ==num of epochs / and 30 number of batch
      dataset =[datasetTrain,datasetTest]

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
    edge.root.mainloop()
    #edge.main()

   
    

   
    
   
   
    

          




