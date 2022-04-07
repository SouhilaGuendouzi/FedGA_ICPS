import socket
from Options import args_parser
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

from Client import Client
from Model import ClientModel


if __name__ == '__main__':
    
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    
   # hote = args.addrServer
    #port = args.portServer


    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

    
    model= ClientModel(args=args).to(args.device)
    client = Client(id=2,model=model, datasetTRain=dataset_train , datasetTest=dataset_test,args= args)
    weights, loss= client.local_update()
    print('Loss ', loss)
    
        #socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #socket.connect((hote, port))
    #print ("Connection on {}".format(port))
    #data='hello'
    #socket.send(data.encode())

    #print ("Close")
    #socket.close()