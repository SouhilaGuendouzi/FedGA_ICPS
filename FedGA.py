
import copy
import torch
from torch import nn
import numpy
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pygad
import pygad.torchga as tg

loss_function = nn.CrossEntropyLoss()

def fitness(solution, sol_idx):
   
        loss=0.0
        print('solution',solution)
        model_weights_dict = tg.model_weights_as_dict(model=model, weights_vector=solution)
        model.load_state_dict(model_weights_dict)

        for idx, (data, target) in enumerate(dataset):
        
           prediction = model(data)
   
           loss += loss_function(prediction, target).detach().numpy() + 0.00000001
          
        loss /= len(dataset.dataset)

        loss=  1.0 / (loss.item())
        
    
        return loss
        

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

def FedGA(w,modell,datasett):
   print('begin FedGA')
   global  initial_population, w_size, model, dataset
   dataset = datasett
   model   = modell
   initial_population= w 
   initial_population=initial_population.tolist()
   num_generations =3# Number of generations.
   num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.
  
  
   #print(initial_population)
   parent_selection_type = "sss" # Type of parent selection.
   crossover_type = "single_point" # Type of the crossover operator.
   mutation_type = "random" # Type of the mutation operator.
   mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
   keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

# Create an instance of the pygad.GA class
   ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

# Start the genetic algorithm evolution.
   ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
   #ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
   solution, solution_fitness, solution_idx = ga_instance.best_solution()
 
#Fetch the parameters of the best solution.
   best_solution_weights = tg.model_weights_as_dict(model=model,  weights_vector=solution)
                                                 
   model.load_state_dict(best_solution_weights)
   return best_solution_weights
   


 
  
  

   