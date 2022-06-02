#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments 
    parser.add_argument('--id', type=int, default=0, help="id of an entity") 
   
    parser.add_argument('--num_users', type=int, default=4, help="number of users: K")

    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
   
    # other arguments
  
    parser.add_argument('--myport', type=int, default=0, help="server port") 

    parser.add_argument('--portCloud', type=int, default=0, help="server port") 

    args = parser.parse_args()
    return args
