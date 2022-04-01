
import copy
import torch
from torch import nn
import numpy
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
def cal_pop_fitness(pop, w_size, net_glob, dataset):
    fitness=[]
    Global=net_glob
    print(w_size)
    
    for i in range(w_size)  :
        Global.load_state_dict(pop[i])
        loss = 0
        l = len(dataset)
        for idx, (data, target) in enumerate(dataset):
           #if args.gpu != -1:
           #   data, target = data.cuda(), target.cuda()
           log_probs = Global(data)
           # sum up batch loss
           loss += F.cross_entropy(log_probs, target, reduction='sum').item()
       
        loss /= len(dataset.dataset)
        fitness.append(loss)
    print('end fitness',fitness)     
    
    return fitness
     


def select_mating_pool(pop, fitness, num_parents):
       # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
       #print(num_parents, pop.shape[1]) 
       parents =[] #numpy.empty((num_parents, sub_weights_size))#10 8
       num_parents=5
       for parent_num in range(num_parents):
          print(len(parents))
          min_fitness_idx = numpy.where(fitness == numpy.min(fitness)) 
          print(min_fitness_idx) #tableau des index
          min_fitness_idx = min_fitness_idx[0][0] #le plus petit index
          #print('hahouwa',pop.item(min_fitness_idx,))
          parents.append(pop.item(min_fitness_idx,)) 
          fitness[min_fitness_idx] = 99999999999
          
       print('end select parents')  
       parents = numpy.array(parents)
     
    
       return parents


def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    print('end cross over')  
    return offspring


def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    print('end mutation')  
    return offspring_crossover



def FedGA_c(w,globalM,dataset):
    w_size= len(w)
    initial_popluation= w
    num_generation= 10
    new_population=initial_popluation


    for i in range( num_generation):
        print('Begin',i,'GA generation')  
        print('fitness')
        fitness=cal_pop_fitness(initial_popluation,w_size,globalM,dataset)
        print('parents')
        parents=select_mating_pool(initial_popluation, fitness, len(w))  
        print(' offspring_crossover')
        offspring_crossover = crossover(parents, offspring_size=(len(w[0])-parents.shape[0],len(w)))
        print(' offspring_mutation')
        offspring_mutation = mutation(offspring_crossover)
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

    return new_population[0:, : ]

