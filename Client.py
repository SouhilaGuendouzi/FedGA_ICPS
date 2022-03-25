import numpy as np
from torchvision import datasets, transforms
import random 


class Client(object):

     def __init__(self, id, dataset,model,accuracy=None):
         self.id=id
         self.dataset=dataset
         self.model=model
         self.accuracy=None
    

     def train(self):

       return self.model.Train()

     def update(self):
         
        return self.model.Update
