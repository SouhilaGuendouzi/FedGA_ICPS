import socket
import torch.nn.functional as F
import torch
from torch import nn
import copy
from torch.utils.data import DataLoader
class Edge(object):

     def __init__(self, id,model, datasetTrain, datasetTest, args):#,device

         self.id=id
         self.datasetTrain = datasetTrain
         self.datasetTest = datasetTest
         self.model=model
         self.Trainaccuracy=0
         self.Testaccuracy=0
         self.loss=None
         self.args=args

         if torch.cuda.is_available():
              self.model.cuda()
         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     
         #self.device=device  
     def local_updateFirst(self):# with its own weights
        
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


        
     def local_updatePer(self,weights_global):
         
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
         for i in range(11):
          try :
           
           del[self.weights['features.{}.weight'.format(i)]]
      
           del[self.weights['features.{}.bias'.format(i)]]

          except:
             print('')
         
          #self.Trainaccuracy , loss = self.test_img('train')
          #self.Testaccuracy, loss =self.test_img('test')
         return self.weights, sum(epoch_loss) / len(epoch_loss) # state_dict(): Returns a dictionary containing a complete state of the module /// , loss_function of model_i
    

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
            
           
        return accuracy, test_loss

