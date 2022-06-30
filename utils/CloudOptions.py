#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments 
    parser.add_argument('--id', type=int, default=0, help="id of an entity") 
    parser.add_argument('--epochs', type=int, default=3, help="rounds of FL rounds") 
    parser.add_argument('--num_fogs', type=int, default=2, help="number of fogs: K")
    

    



    # FL arguments
    parser.add_argument('--aggr', type=str, default='FedAVG', help="name of aggregation method")


 
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")


     # socket parameters
    parser.add_argument('--LISTENER_LIMIT', type=int, default=10, help="server port")
    parser.add_argument('--myport', type=int, default=0, help="server port")
    parser.add_argument('--myadr', type=str, default='127.0.0.0', help="server port") 
  

    #Election parameters
    parser.add_argument('--priority', type=int, default=99, help="server priority which is fixed")
    parser.add_argument('--capacity', type=int, default=99, help="server capacity which is random")  



     

    args = parser.parse_args()
    return args
