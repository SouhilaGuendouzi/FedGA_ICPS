#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments 
    parser.add_argument('--id', type=int, default=0, help="id of an entity") 
   
    parser.add_argument('--num_users', type=int, default=4, help="number of users: K")
    parser.add_argument('--epochs', type=int, default=4, help="rounds of FL rounds") 

    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
   
    # socket arguments
    
    parser.add_argument('--myport', type=int, default=0, help="server port") 
    parser.add_argument('--myadr', type=str, default='127.0.0.0', help="server port")
    parser.add_argument('--LISTENER_LIMIT', type=int, default=10, help="server port") 
    parser.add_argument('--portCloud', type=int, default=0, help="server port") 

    #Election parameters
    parser.add_argument('--priority', type=int, default=99, help="server priority which is fixed")
    parser.add_argument('--capacity', type=int, default=99, help="server capacity which is random") 

    args = parser.parse_args()
    return args
