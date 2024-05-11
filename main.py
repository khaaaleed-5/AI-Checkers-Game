from game import Game
from GA import GeneticAlgorithm
from model import Player

game = Game()
genetic_algorithm = GeneticAlgorithm(population_size=50,mutation_rate=0.76,nn=Player,game=game)

best_player = genetic_algorithm.train(epochs=100,thresh_fitness=6)