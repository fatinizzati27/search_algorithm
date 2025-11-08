import streamlit as st
import random
import matplotlib.pyplot as plt

# --- App Title ---
st.title("Genetic Algorithm Simulation (Search Optimization)")
st.write("""
This web app demonstrates a simple **Genetic Algorithm (GA)** that searches for an optimal bit pattern 
where the number of 1s equals 50.  
The goal is to achieve a fitness score of **80**, representing the best individual.
""")

# --- Parameters ---
POPULATION_SIZE = 300
CHROMOSOME_LENGTH = 80
GENERATIONS = 50
MUTATION_RATE = 0.01

# --- Fitness Function ---
def fitness(individual):
    ones = sum(individual)
    return 80 - abs(ones - 50)

# --- Generate Initial Population ---
def generate_population():
    return [[random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)] for _ in range(POPULATION_SIZE)]

# --- Selection (Top 50%) ---
def selection(pop):
    return sorted(pop, key=fitness, reverse=True)[:POPULATION_SIZE // 2]

# --- Crossover ---
def crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    return parent1[:point] + parent2[point:]

# --- Mutation ---
def mutate(individual):
    return [bit if random.random() > MUTATION_RATE else 1 - bit for bit in individual]

# --- Genetic Algorithm ---
def run_ga():
    population = generate_population()
    best_fitness_list = []

    for generation in range(GENERATIONS):
        selected = selection(population)
        children = []
        while len(children) < POPULATION_SIZE:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            children.append(child)
        population = children
        best = max(population, key=fitness)
        best_fitness = fitness(best)
        best_fitness_list.append(best_fitness)

    return best, best_fitness_list

# --- Run Button ---
if st.button("Run Genetic Algorithm"):
    best_solution, fitness_progress = run_ga()
    st.success(f"✅ Best Fitness: {fitness(best_solution)}")
    st.write(f"**Best Individual (first 50 bits):** {best_solution[:50]}...")
    st.write(f"**Total 1s in best individual:** {sum(best_solution)}")

    # --- Plot Fitness Progress ---
    plt.figure()
    plt.plot(range(1, GENERATIONS + 1), fitness_progress)
    plt.title("Fitness Progress Across Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    st.pyplot(plt)

    st.info("""
    **Observation:**  
    The algorithm evolves over generations to maximize the fitness score.
    When the number of 1s approaches 50, the fitness score reaches its maximum of 80.
    """)

# --- Footer ---
st.caption("Developed by: Your Name | Student ID | BSD3513 Artificial Intelligence Lab 2")
