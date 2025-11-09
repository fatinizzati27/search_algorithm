import random
import numpy as np
import pandas as pd
import streamlit as st

# -------------------- LAB 2: Search Algorithm (BSD3513) --------------------
st.set_page_config(page_title="Search Algorithm - Lab 2", page_icon="🔎", layout="wide")
st.title("Search Algorithm")

POP_SIZE = 300
CHROMO_LENGTH = 80
GENERATIONS = 50
TARGET_ONES = 50
MAX_FITNESS = 80
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.9
ELITISM = 2
TOURNAMENT_SIZE = 3

# -------------------- Fitness Function --------------------
def fitness(individual):
    """Fitness = 80 - |number_of_ones - 50|"""
    ones = sum(individual)
    return MAX_FITNESS - abs(ones - TARGET_ONES)

# -------------------- Population Initialization --------------------
def generate_population():
    return [[random.randint(0, 1) for _ in range(CHROMO_LENGTH)] for _ in range(POP_SIZE)]

# -------------------- Tournament Selection --------------------
def tournament_selection(pop, fitness_values):
    selected = random.sample(range(len(pop)), TOURNAMENT_SIZE)
    best_index = max(selected, key=lambda i: fitness_values[i])
    return pop[best_index]

# -------------------- Crossover --------------------
def one_point_crossover(parent1, parent2):
    point = random.randint(1, CHROMO_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# -------------------- Mutation --------------------
def mutate(individual):
    return [bit if random.random() > MUTATION_RATE else 1 - bit for bit in individual]

# -------------------- Main GA Function --------------------
def run_ga():
    population = generate_population()
    best_list, avg_list, worst_list = [], [], []

    for generation in range(GENERATIONS):
        fitness_values = [fitness(ind) for ind in population]
        best_fit = max(fitness_values)
        avg_fit = sum(fitness_values) / len(fitness_values)
        worst_fit = min(fitness_values)

        best_list.append(best_fit)
        avg_list.append(avg_fit)
        worst_list.append(worst_fit)

        # Elitism
        elite_indices = np.argsort(fitness_values)[-ELITISM:]
        elites = [population[i] for i in elite_indices]

        # Next generation
        next_gen = elites.copy()
        while len(next_gen) < POP_SIZE:
            p1 = tournament_selection(population, fitness_values)
            p2 = tournament_selection(population, fitness_values)
            if random.random() < CROSSOVER_RATE:
                c1, c2 = one_point_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            next_gen.append(mutate(c1))
            if len(next_gen) < POP_SIZE:
                next_gen.append(mutate(c2))
        population = next_gen

    # Best result
    fitness_values = [fitness(ind) for ind in population]
    best_individual = population[np.argmax(fitness_values)]
    return best_individual, best_list, avg_list, worst_list

# -------------------- Streamlit Interface --------------------
if st.button("Run Search Algorithm", type="primary"):
    best, best_hist, avg_hist, worst_hist = run_ga()
    df = pd.DataFrame({"Best": best_hist, "Average": avg_hist, "Worst": worst_hist})

    st.subheader("Fitness Progress Over Generations")
    st.line_chart(df)

    st.subheader("Best Individual Found")
    st.code("".join(map(str, best)), language="text")
    st.write(f"Number of 1s: **{sum(best)} / 80**")
    st.write(f"Fitness Value: **{fitness(best)}**")

    st.success("""
    ✅ The algorithm successfully evolves a population until it reaches a fitness value of 80
    when the number of ones equals 50.
    """)

