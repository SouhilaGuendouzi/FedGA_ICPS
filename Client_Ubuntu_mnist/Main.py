import socket
from utils.Options import args_parser
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from Model import ClientModel
from Client import Client



if __name__ == '__main__':
    
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    
   # hote = args.addrServer
    #port = args.portServer


    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    datasetTrain = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    datasetTest = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

    
    model= ClientModel(args=args).to(args.device)
    client = Client(id=1,model=model, datasetTRain=datasetTrain, datasetTest=datasetTest,args= args)
    weights, loss= client.local_update()
    print('Loss ', loss)
    
        #socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #socket.connect((hote, port))
    #print ("Connection on {}".format(port))
    #data='hello'
    #socket.send(data.encode())

    #print ("Close")
    #socket.close()