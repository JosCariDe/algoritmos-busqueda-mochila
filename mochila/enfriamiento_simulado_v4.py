import math
import random
import time
import timeit
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class SimAnnealKnapsack(object):
    def __init__(self, items, max_weight, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.items = items
        self.max_weight = max_weight
        self.num_items = len(items)
        self.T = math.sqrt(self.num_items) if T == -1 else T
        self.alpha = 0.80 if alpha == -1 else alpha
        self.stopping_temperature = 0 if stopping_T == -1 else stopping_T
        self.stopping_iter = 2000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1

        self.current_solution = None
        self.current_value = float("-Inf")
        self.current_weight = float("Inf")
        self.best_solution = None
        self.best_weight = float("Inf")
        self.best_value = float("-Inf")
        self.value_list = []

    def initial_solution(self):
        """
        Generate a random initial solution.
        """
        vector = np.zeros(self.num_items)
        
        while self.calc_weight_solution(vector) < self.max_weight:
            casilla_aleatoria = random.randint(0, self.num_items - 1)
            vector[casilla_aleatoria] += 1
            
            if self.items[casilla_aleatoria]["Cantidad"] < vector[casilla_aleatoria]:
                vector[casilla_aleatoria] -= 1
            
            if self.calc_weight_solution(vector) > self.max_weight :
                vector[casilla_aleatoria] -= 1
                break
        
        self.new_best_solution(vector)
        
        return vector
    
    def new_best_solution(self, solution):
        self.best_solution = solution
        self.best_weight = self.calc_weight_solution(solution)
        self.best_value = self.calc_value_solution(solution)
        
    def new_current_solution(self, solution):
        self.current_solution = solution
        self.current_weight = self.calc_weight_solution(solution)
        self.current_value = self.calc_value_solution(solution)
        
            
    def calc_weight_solution(self, solution):
        """
        Calculate the weight of the current solution.
        """
        return sum(solution[i] * self.items[i]['Peso_kg'] for i in range(self.num_items))
    
    def calc_value_solution(self, solution):
        return sum(solution[i] * self.items[i]['Valor'] for i in range(self.num_items))

    def evaluate_solution(self, solution):
        """
        Evaluate the value of the current solution.
        """
        total_weight = self.calc_weight_solution(solution)
        total_value = self.calc_value_solution(solution)
        
        if total_weight <= self.max_weight and total_value > self.best_value:
            self.new_best_solution(solution)


    def p_accept(self, candidate_value):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        return math.exp((candidate_value - self.best_value) / self.T)

    def accept(self, candidate):
        """
        Accept with probability 1 if candidate is better than current.
        Accept with probability p_accept(..) if candidate is worse.
        """
        candidate_value = self.calc_value_solution(candidate)
        # candidate_weight = self.calc_weight_solution(candidate)
        
        if candidate_value > self.current_value:
            self.new_current_solution(candidate)
            if candidate_value > self.best_value :
                self.new_best_solution(candidate)
        else:
            if random.random() < self.p_accept(candidate_value):
                self.current_solution = candidate.copy()

    def anneal(self):
        """
        Execute simulated annealing algorithm.
        """
        start_time = time.time()  # Record start time
        self.current_solution = self.initial_solution()
        self.new_current_solution(self.current_solution)

        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = self.current_solution.copy()
            
            # Generar un nuevo candidato
            index_to_move = random.randint(0, self.num_items - 1)
            new_index = random.randint(0, self.num_items - 1)
            
            # Mover un único elemento de la solución actual a una nueva posición aleatoria
            candidate[new_index] = min(candidate[new_index] + 1, self.items[new_index]['Cantidad'])
            candidate[index_to_move] = max(candidate[index_to_move] - 1, 0)
            
            # Evaluar la nueva solución solo si el peso máximo no se excede
            total_weight = self.calc_weight_solution(candidate)
            if total_weight <= self.max_weight:
                self.evaluate_solution(candidate)
                self.accept(candidate)
            
            self.T *= self.alpha
            self.iteration += 1

            self.value_list.append(self.best_value)
        
        print("Best Weight: ", self.best_weight)
        print("Best value obtained: ", self.best_value)
        print("Time taken: {:.2f} seconds".format(time.time() - start_time))  # Print execution time

        # Plot the learning curve
        plt.plot([i for i in range(len(self.value_list))], self.value_list)
        plt.ylabel("Total Sales Value")
        plt.xlabel("Iteration")
        plt.title("Learning Curve")
        plt.show()

        # Print objects in the knapsack
        print("Solution:")
        print(self.best_solution)

def read_items_from_excel(path):
    df = pd.read_excel(path)
    items = []
    for index, row in df.iterrows():
        item = {
            'Peso_kg': row['Peso_kg'],
            'Valor': row['Valor'],
            'Cantidad': row['Cantidad']
        }
        items.append(item)
    return items

def main():
    items = read_items_from_excel("Mochila_capacidad_maxima_15.9kg.xlsx")
    print(items)
    max_weight = 15.9

    end_times = []
    best_values = []
    
    best_value_iteration = float("-Inf")
    best_wight_search = None
    best_mochila = None

    for i in range(30):
        print('Ejecución No. ', i + 1)
        start_time = timeit.default_timer() 
        sim_anneal = SimAnnealKnapsack(items, max_weight)
        sim_anneal.anneal()
        stop_time = timeit.default_timer()
        end_times.append(stop_time - start_time)
        best_values.append(sim_anneal.best_value)
        
        if best_value_iteration < sim_anneal.best_value:
            best_mochila = sim_anneal.best_solution
            best_wight_search = sim_anneal.best_weight

    print("Tiempo total de ejecución:", sum(end_times))
    print("Mejor resultado: $", max(best_values))
    print("Mochila Mejor Evaluada: " , best_mochila)
    print("Mejor Resultado con el peso de: ", best_wight_search, "kg")
    print("Peor resultado:", min(best_values))
    print("Resultado promedio:", statistics.mean(best_values))
    print("Varianza:", statistics.variance(best_values))

if __name__ == "__main__":
    main()
