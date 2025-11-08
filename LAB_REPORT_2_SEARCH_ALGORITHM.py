def run_ga():
    population = generate_population()
    best_fitness_list = []
    avg_fitness_list = []

    for generation in range(GENERATIONS):
        selected = selection(population)
        children = []
        while len(children) < POPULATION_SIZE:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            children.append(child)
        population = children

        fitness_values = [fitness(ind) for ind in population]
        best_fitness = max(fitness_values)
        avg_fitness = sum(fitness_values) / len(fitness_values)

        best_fitness_list.append(best_fitness)
        avg_fitness_list.append(avg_fitness)

    return best_fitness_list, avg_fitness_list


if st.button("Run Genetic Algorithm"):
    best_list, avg_list = run_ga()
    plt.figure()
    plt.plot(best_list, label="Best Fitness")
    plt.plot(avg_list, label="Average Fitness")
    plt.legend()
    plt.title("Fitness Progress Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")
    st.pyplot(plt)
