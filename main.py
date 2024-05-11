from game import Game
from GA import GeneticAlgorithm
from model import Player
import tkinter as tk
from tkinter import ttk
import pygame
from checkers.constants import WIDTH, HEIGHT

class MutationPage(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Genetic configuration")
        self.geometry("400x250")
        self.mutation_type = tk.StringVar()
        self.mutation_rate = tk.DoubleVar()
        self.crossover_type = tk.StringVar()

        # Mutation Type Label and ComboBox
        mutation_type_label = ttk.Label(self, text="Mutation Type:")
        mutation_type_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        mutation_type_combo = ttk.Combobox(self, textvariable=self.mutation_type, values=["Self-adaptive mutation", "Non-unifom mutation"])
        mutation_type_combo.grid(row=0, column=1, padx=10, pady=5)

        # Mutation Rate Label and Scale
        mutation_rate_label = ttk.Label(self, text="Mutation Rate:")
        mutation_rate_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        mutation_rate_scale = ttk.Scale(self, from_=0.0, to=1.0, variable=self.mutation_rate, orient="horizontal", length=200)
        mutation_rate_scale.grid(row=1, column=1, padx=10, pady=5)

        # Crossover Type Label and ComboBox
        crossover_type_label = ttk.Label(self, text="Crossover Type:")
        crossover_type_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        crossover_type_combo = ttk.Combobox(self, textvariable=self.crossover_type, values=["Uniform crossover", "Single-point crossover"])
        crossover_type_combo.grid(row=2, column=1, padx=10, pady=5)

        # Start Button
        start_button = ttk.Button(self, text="Start Game", command=self.start_game)
        start_button.grid(row=3, columnspan=2, padx=10, pady=10)

    def start_game(self):
        mutation_type = self.mutation_type.get()
        mutation_rate = self.mutation_rate.get()
        crossover_type = self.crossover_type.get()
        # Start the game with selected configuration
        print("Mutation Type:", mutation_type)
        print("Mutation Rate:", mutation_rate)
        print("Crossover Type:", crossover_type)
        WIN = pygame.display.set_mode((WIDTH, HEIGHT))
        game = Game(WIN)
        genetic_algorithm = GeneticAlgorithm(population_size=50,
                                             mutation_rate=mutation_rate,
                                             nn=Player,
                                             game=game,
                                             mutation_type=mutation_type,
                                             crossover_type=crossover_type)

        best_player = genetic_algorithm.train(epochs=100,thresh_fitness=6)

if __name__ == "__main__":
    app = MutationPage()
    app.mainloop()
