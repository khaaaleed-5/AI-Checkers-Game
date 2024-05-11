import numpy as np
from game import Game
from model import NeuralNetwork as NN
import random

class GeneticAlgorithm:
    def __init__(self, population_size: int, nn: NN, game: Game, mutation_rate=0.1, k=10, mutation_type='creep', crossover_type='uniform', selection_type='uniform', replacement_type='age_based', random_seed=None)-> None:
        self.population_size = population_size
        self.nn = nn
        self.game = game
        self.mu = mutation_rate
        self.k = k
        self.random_seed = random_seed
        self.best_scores = None
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type
        self.selection_type = selection_type
        self.replacement_type = replacement_type
        self.best_nn = (None, -1) # (best neural network, best score)
        
    def __init_population(self) -> list[NN]:     
        '''
        Initialize population of specified size.
        '''
        pop = [self.nn() for _ in range(self.population_size)]
        return pop
    
    def tournament_selection(self,fitness_scores: np.ndarray, pop: list[NN]):
        mating_pool = []
        current_member = 1
        #k is tournament size
        while current_member < self.k:
            # Pick k individuals randomly, without replacement
            tournament_members = random.sample(self.population, self.k)

            # Compare these k individuals and select the best of them
            best_individual = max(tournament_members, key=lambda x: fitness_scores[pop.index(x)])

            # Add the best individual to the mating pool
            mating_pool.append(best_individual)

            current_member += 1

        return mating_pool
    def select_parents(self, fitness_scores: np.ndarray, pop: list[NN], num_parents=10):
        selected_parents = []
        total_fitness = np.sum(fitness_scores)
        
        # Select parents
        for _ in range(num_parents):
            # Generate a random number between 0 and total_fitness
            spin = random.uniform(0, total_fitness)
            
            # Find the individual whose slot contains the spin
            current_sum = 0
            for individual in pop:
                current_sum += fitness_scores[pop.index(individual)]
                
                if current_sum >= spin:
                    selected_parents.append(individual)
                    break
        
        return selected_parents
    
    def uniform_crossover(self, parent1: NN, parent2: NN)-> tuple[NN, NN]:
        c1 = parent1.copy()
        c2 = parent2.copy()
        for i in range(len(c1.weights)):
            mask = np.random.rand(*c1.weights[i].shape) <= self.cp
            c1.weights[i] = mask * c1.weights[i] + (1 - mask) * c2.weights[i]
            c2.weights[i] = mask * c2.weights[i] + (1 - mask) * c1.weights[i]
        return c1, c2

    def single_point_crossover(parent1, parent2):
        if len(parent1) > 1 and len(parent2) > 1:  
            crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        else:
            return parent1, parent2
        
    def self_adaptive_mutation(self, nn: NN, fitness_scores, mutation_rate_factor=1.1, mutation_rate_decay=0.9):
        mean_fitness = np.mean(fitness_scores)
        
        for i, fitness in enumerate(fitness_scores):
            if fitness > mean_fitness:
                # Increase mutation rate if the player's fitness is above average
                mutation_rate = mutation_rate_factor
            else:
                # Decrease mutation rate if the player's fitness is below average
                mutation_rate = mutation_rate_decay
            
            # Apply Gaussian mutation to the weights of the neural network
            for j in range(len(nn.weights)):
                mask = np.random.rand(*nn.weights[j].shape) <= mutation_rate
                nn.weights[j] += mask * np.random.randn(*nn.weights[j].shape)
        
        return nn
    
    def Gaussian_mutation(self, nn: NN, std: int) -> NN:
        '''
        Mutate the neural network's weights by adding Gaussian noise to the weights.
        '''
        for i in range(len(nn.weights)):
            mask = np.random.rand(*nn.weights[i].shape) <= self.mu
            nn.weights[i] += mask * np.random.uniform(-std, std) # this either adds a random number from weights or leaves it as it is
        return nn

    def fitness_based_replacement(self,pop, offspring, fitness_scores):
        # Replace the weakest individuals in the population with the offspring based on fitness scores
        sorted_indices = np.argsort(fitness_scores)
        num_offspring = len(offspring)
        num_replace = min(num_offspring, len(pop))
        replace_indices = sorted_indices[:num_replace]
        pop[replace_indices] = offspring[:num_replace]
        return pop

    def train(self, epochs=1000, thresh_fitness=35)-> NN:
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        pop = self.__init_population()

        self.best_scores = []
        self.best_nn = (None, -1) # (best neural network, best score)
        for epoch in range(epochs):
            # Evaluate
            fitness_scores = self.game.evaluate_population(pop)
            
            # Select parents
            if self.selection_type == 'uniform':
                parents = self.select_parents(pop, fitness_scores)
            elif self.selection_type == 'tournament':
                parents = self.tournament_selection(pop, fitness_scores)

            # Crossover
            children = []
            for i in range(0, len(parents), 2):
                if self.crossover_type == 'single_point':
                    child1, child2 = self.single_point_crossover(parents[i], parents[i + 1])
                elif self.crossover_type == 'uniform':
                    child1, child2 = self.uniform_crossover(parents[i], parents[i + 1])
                children.append(child1)
                children.append(child2)

            # Mutate
            mutated_children = []
            for child in children:
                if self.mutation_type == 'Gaussian':
                    child = self.Gaussian_mutation(child)
                elif self.mutation_type == 'self_adaptive':
                    child = self.self_adaptive_mutation(child)
                mutated_children.append(child)

            # Replace
          
            pop = self.fitness_based_replacement(pop,fitness_scores,mutated_children)

            # Update best neural network
            best_score = max(fitness_scores)
            if best_score > self.best_nn[1]:
                self.best_nn = (pop[fitness_scores.index(best_score)], best_score)
                self.best_scores.append(best_score)
                if best_score > thresh_fitness:
                    break

            return self.best_nn[0]