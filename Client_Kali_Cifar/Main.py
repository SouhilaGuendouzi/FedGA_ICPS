import socket
from utils.Options import args_parser
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

from Entities.Edge import Client
from Entities.Model import ClientModel
import pickle
import socket
import ipaddress
if __name__ == '__main__':
    
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    hote ='192.168.69.163'   
    port = args.portServer
    client.connect((hote, port))
    print ("Connection on {}".format(port))


    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

    
    model= ClientModel(args=args).to(args.device)
    client = Client(id=2,model=model, datasetTRain=dataset_train , datasetTest=dataset_test,args= args)
    weights, loss= client.local_update()

    dataTosend=pickle.dumps(weights)
    client.send(dataTosend)
    print('Loss ', loss)
    print ("Close")
    socket.close()
    
        #socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #socket.connect((hote, port))
    #print ("Connection on {}".format(port))
    #data='hello'
    #socket.send(data.encode())

    #print ("Close")
    #socket.close()