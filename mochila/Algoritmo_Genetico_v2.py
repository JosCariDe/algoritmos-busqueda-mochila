import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt

class Item:
    def __init__(self, weight, value, quantity):
        self.weight = weight
        self.value = value
        self.quantity = quantity
    
    def __repr__(self):
        return "Item(w={}, v={}, q={})".format(self.weight, self.value, self.quantity)

# Crear una función para leer los datos del archivo Excel y devolver una lista de objetos Item
def read_items_from_excel(path):
    df = pd.read_excel(path)
    items = []
    for index, row in df.iterrows():
        item = Item(weight=row['Peso_kg'], value=row['Valor'], quantity=row['Cantidad'])
        items.append(item)
    return items

# Crear una función para calcular el valor total de una mochila
    # (Item item, List<Integer> backpack()
def calculate_total_value( items, backpack ):
    total_value = sum(item.value * quantity for item, quantity in zip(items, backpack))
    return total_value

# Crear una función para calcular el peso total de una mochila
def calculate_total_weight(items, backpack):
    total_weight = sum(item.weight * quantity for item, quantity in zip(items, backpack))
    return total_weight

# Crear la clase Fitness para evaluar una mochila
class Fitness:
    def __init__(self, backpack, max_weight):
        self.backpack = backpack
        self.max_weight = max_weight
        self.value = 0
        self.weight = 0
    
    def evaluate(self, items):
        total_value = calculate_total_value(items, self.backpack)
        total_weight = calculate_total_weight(items, self.backpack)
        
        if total_weight <= self.max_weight:
            self.value = total_value
            self.weight = total_weight
        else:
            self.value = 0
            self.weight = self.max_weight + 1  # Penalizar las soluciones que exceden el peso máximo

# Crear una función para generar una mochila aleatoria
def generate_random_backpack(items, max_weight=15.9):
    vector = np.zeros(16, dtype=int)
    
    while calculate_total_weight(items, vector) < max_weight:
        casilla_aleatoria = random.randint(0, 15)
        vector[casilla_aleatoria] += 1
        
        if items[casilla_aleatoria].quantity < vector[casilla_aleatoria]:
            vector[casilla_aleatoria] -= 1
        
        if calculate_total_weight(items, vector) > max_weight:
            vector[casilla_aleatoria] -= 1
            break
        
    return vector
    #return [random.randint(0, item.quantity) for item in items]

# Crear una función para generar la población inicial
 # (Integer pop_size, List<Item> items)
def initial_population(pop_size, items):
    population = []
    for _ in range(pop_size):
        population.append( generate_random_backpack(items) )
    return population

# Crear una función para seleccionar las mejores mochilas
def selection(population, fitness_values, elite_size):
    selected_indexes = np.argsort(fitness_values)[-elite_size:]
    return [population[i] for i in selected_indexes]

# Crear una función para cruzar dos mochilas
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = []

    for i in range(len(parent1)):
        if i < crossover_point:
            child.append(parent1[i])
        else:
            child.append(parent2[i])

    return child

# Crear una función para aplicar mutaciones a una mochila
def mutate(backpack, mutation_rate):
    for i in range(len(backpack)):
        if random.random() < mutation_rate:
            backpack[i] = random.randint(0, backpack[i])
    return backpack

# Crear la función principal del algoritmo genético
def genetic_algorithm(items, max_weight, pop_size, elite_size, mutation_rate, generations):
    population = initial_population(pop_size, items)
    best_values = []
    best_backpack = None
    
    for _ in range(generations):
        fitness_values = []
        for backpack in population:
            fitness = Fitness(backpack, max_weight)
            fitness.evaluate(items)
            fitness_values.append(fitness.value)
        
        elites = selection(population, fitness_values, elite_size)
        next_generation = elites
        
        while len(next_generation) < pop_size:
            parent1, parent2 = random.choices(elites, k=2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_generation.append(child)
        
        population = next_generation
        
        best_backpack = max(population, key=lambda x: calculate_total_value(items, x))
        best_values.append(calculate_total_value(items, best_backpack))
    
    best_fitness = calculate_total_value(items, best_backpack)
    best_weight = calculate_total_weight(items, best_backpack)
    return best_backpack, best_fitness, best_weight, best_values

# Leer los datos del archivo Excel
items = read_items_from_excel("Mochila_capacidad_maxima_15.9kg.xlsx")

# Parámetros del algoritmo genético
max_weight = 15.9
pop_size = 1000  # No subir mucho, explota la pc
elite_size = 500
mutation_rate = 0.05  # desde 0.001 a 0.05
generations = 50

# Ejecutar el algoritmo genético
best_backpack, best_fitness, best_weight, best_values = genetic_algorithm(items, max_weight, pop_size, elite_size, mutation_rate, generations)

# Mostrar la gráfica del progreso del algoritmo
plt.plot(best_values)
plt.xlabel('Generación')
plt.ylabel('Mejor Valor de la Mochila')
plt.title('Progreso del Algoritmo Genético')
plt.show()

print("Mejor mochila encontrada:", best_backpack)
print("Valor de la mejor mochila:", best_fitness)
print("Peso de la mejor mochila:", best_weight)
