from turtle import forward
import torch
from torch import conv2d, nn
import torch.nn.functional as F
import torchvision.models as models
from torch.hub import load_state_dict_from_url


class Feature_extractor_Layers(nn.Module):
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
  def forward(self, t):

    t = self.features(t)
  
    return t

class Classification_Layers(nn.Module):
  def __init__(self):
    super().__init__()

    self.classification=nn.Sequential(

            nn.Flatten(),
            nn.Linear(12*4*4, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 10)

    )
  def forward(self, t):

     t = self.classification(t)
  
     return t

class Model_Fashion(nn.Module):
  def __init__(self):
    super().__init__()

    # define layers
    self.features =Feature_extractor_Layers()
    self.classification=Classification_Layers()

  def forward(self, t):

    t = self.features(t)
    t = self.classification(t)

    return t


    
class Model_A(nn.Module):

    
     def __init__(self):
        super(Model_A, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 24, kernel_size=5,padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 12, kernel_size=5,padding=1), 
            nn.ReLU(),
            nn.Dropout2d()

            

        )
        
        self.classification=nn.Sequential(
              nn.Flatten(),
    
            nn.Linear(12*4*4, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 10)
          
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
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        
            nn.Conv2d(32,12,3,padding=2),
            #nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.Dropout2d()
        )
        self.classification = nn.Sequential(
            
           nn.Flatten(),
    
            nn.Linear(12*4*4, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 10)


        )
        
        #self.fc1 = nn.Linear(64*6*6, 50)
        #self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = self.features(x)
    
        x =self.classification(x)
        #x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        #x = self.fc1(x)
        #x = self.fc2(x)
        
        
        return x


class  Model_C(nn.Module):
    def __init__(self):
        super(Model_C, self).__init__()
        self.features = nn.Sequential(
             nn.Conv2d(1, 16, kernel_size=5),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 12, kernel_size=5, padding=1),
            #nn.BatchNorm2d(12),
            nn.Dropout2d(),
            nn.ReLU()
            
            
            
            )
        self.classification=nn.Sequential(
            nn.Flatten(),
    
            nn.Linear(12*4*4, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 10)
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
              #nn.BatchNorm2d(num_features=6),
              nn.MaxPool2d(2, 2),
              nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5,5)),
              nn.MaxPool2d(2, 2),
          
        )

        self.classification=nn.Sequential(
             nn.Flatten(),
    
            nn.Linear(12*4*4, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 10)
        )

     def forward(self, t): # implements the forward method (flow of tensors)
        t =self.features(t)
        t = self.classification(t)
    
        
        return t


  

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

