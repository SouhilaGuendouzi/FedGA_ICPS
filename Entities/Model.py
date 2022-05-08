from turtle import forward
import torch
from torch import conv2d, nn
import torch.nn.functional as F
import torchvision.models as models
from torch.hub import load_state_dict_from_url


class Model_Fashion(nn.Module):
  def __init__(self):
    super().__init__()

    # define layers
    self.features = nn.Sequential(

        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

    )
    self.classification=nn.Sequential(

        nn.Flatten(),
        nn.Linear(in_features=12*4*4, out_features=120),
        nn.Linear(in_features=120, out_features=60),
        nn.Linear(in_features=60, out_features=10)

    )

  def forward(self, t):

    t = self.features(t)
    t = self.classification(t)

    return t


    
class Model_A(nn.Module):

    
     def __init__(self):
        super(Model_A, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 20, kernel_size=5), #for gray images ==> args.num_channels===1 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d()

            

        )
        
        self.classification=nn.Sequential(
            nn.Flatten(),
             #320=20*4*4
             # 20*6*6
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10)
          
      ) 
      

     def forward(self, x):
         x=self.features(x)
         x=self.classification(x)
         return x

class Model_B(nn.Module):
    
    def __init__(self):
        super(Model_B, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classification = nn.Sequential(
            
            nn.Flatten(),
            nn.Linear(64*6*6, 50),
            nn.Linear(50, 10)


        )
        
    def forward(self, x):
        x = self.features(x)
        x =self.classification(x)
        
        
        return x

class  Model_C(nn.Module):
    def __init__(self):
        super(Model_C, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.classification=nn.Sequential(
               nn.Flatten(),
               nn.Linear(7*7*32, 10)
        )
            
    def forward(self, t):
        t=self.features(t)
        t=self.classification(t)
        return t

class  Model_D(nn.Module):
     def __init__(self):
        super().__init__() # super class constructor
        self.features=nn.Sequential(
            
              nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5)),
              nn.BatchNorm2d(num_features=6),
              nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5,5)),
          
        )

        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=12*20*20, out_features=120),
            nn.BatchNorm1d(num_features=120),
            nn.BatchNorm1d(num_features=120),
            nn.Linear(in_features=120, out_features=60),
            nn.Linear(in_features=60, out_features=10)
        )
      
        
        
        
     def forward(self, t): # implements the forward method (flow of tensors)
        t =self.features(t)
        t = self.classification(t)
    
        
        return t