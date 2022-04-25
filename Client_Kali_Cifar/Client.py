# coding: utf-8

import socket
from utils.Options import args_parser
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from Entities.Model import ClientModel
import torch
from torch import nn

class Client(object):

     def __init__(self, id,model, datasetTRain, datasetTest, args):#,device

         self.id=id
         self.datasetTrain = datasetTRain
         self.datasetTest = datasetTest
         self.model=model
         self.accuracy=None
         self.args=args
         #self.device=device,
    
     


     def local_update(self):

         loss_func = nn.CrossEntropyLoss()
         ldr_train = DataLoader(self.datasetTrain, shuffle=True,batch_size=self.args.local_bs)
         
         self.model.train()
         optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
         epoch_loss = []
         
         for iter in range(self.args.local_ep):
            print('epoch',iter)
            batch_loss = []
            
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.model.zero_grad()
                log_probs = self.model(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(ldr_train.dataset),
                               100. * batch_idx / len(ldr_train), loss.item()))
             
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
      
         return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss) # state_dict(): Returns a dictionary containing a complete state of the module /// , loss_function of model_i


     def test_img(self):
        self.model.eval()
        # testing
        test_loss = 0
        correct = 0
        for idx, (data, target) in enumerate(self.datasetTest):
           log_probs =  self.model(data)
           # sum up batch loss
           test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
           # get the index of the max log-probability
           y_pred = log_probs.data.max(1, keepdim=True)[1]
           correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(self.datasetTest.dataset)
        accuracy = 100.00 * correct / len(self.datasetTest.dataset)

        return accuracy, test_loss

