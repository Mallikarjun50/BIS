#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// --- Constants ---
#define POP_SIZE 50
#define NUM_GENERATIONS 100
#define CROSSOVER_RATE 0.8
#define MUTATION_RATE 0.1
#define MIN_X 0.0
#define MAX_X 1.0
#define TARGET_VALUE 1.5

// --- Individual Structure ---
typedef struct {
    double gene;
    double fitness;
} Individual;

// --- Fitness Function: Maximize closeness to target ---
double fitnessFunction(double x) {
    double fx = x * sin(10 * M_PI * x) + 1.0;
    return -fabs(fx - TARGET_VALUE); // Closer to target => higher fitness
}

// --- Generate Random Double in Range [min, max] ---
double randomDouble(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// --- Evaluate Fitness ---
void evaluate(Individual* ind) {
    ind->fitness = fitnessFunction(ind->gene);
}

// --- Tournament Selection (size = 2) ---
Individual tournamentSelection(Individual population[]) {
    int i = rand() % POP_SIZE;
    int j = rand() % POP_SIZE;
    return (population[i].fitness > population[j].fitness) ? population[i] : population[j];
}

// --- Arithmetic Crossover ---
void crossover(const Individual* p1, const Individual* p2, Individual* c1, Individual* c2) {
    *c1 = *p1;
    *c2 = *p2;

    if (randomDouble(0, 1) < CROSSOVER_RATE) {
        double alpha = randomDouble(0, 1);
        c1->gene = alpha * p1->gene + (1 - alpha) * p2->gene;
        c2->gene = alpha * p2->gene + (1 - alpha) * p1->gene;
    }
}

// --- Mutation ---
void mutate(Individual* ind) {
    if (randomDouble(0, 1) < MUTATION_RATE) {
        double mutation_strength = 0.1;
        ind->gene += randomDouble(-mutation_strength, mutation_strength);

        // Clamp to valid range
        if (ind->gene < MIN_X) ind->gene = MIN_X;
        if (ind->gene > MAX_X) ind->gene = MAX_X;
    }
}

// --- Main Program ---
int main() {
    srand((unsigned)time(NULL));

    Individual population[POP_SIZE];
    Individual new_population[POP_SIZE];
    Individual best_individual;

    // --- Initialize Population ---
    for (int i = 0; i < POP_SIZE; ++i) {
        population[i].gene = randomDouble(MIN_X, MAX_X);
        evaluate(&population[i]);
    }

    best_individual = population[0];

    // --- Evolution Loop ---
    for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        int idx = 0;

        while (idx < POP_SIZE) {
            // Parent Selection
            Individual p1 = tournamentSelection(population);
            Individual p2 = tournamentSelection(population);

            // Generate Offspring
            Individual c1, c2;
            crossover(&p1, &p2, &c1, &c2);
            mutate(&c1);
            mutate(&c2);
            evaluate(&c1);
            evaluate(&c2);

            // Add to new population
            new_population[idx++] = c1;
            if (idx < POP_SIZE) {
                new_population[idx++] = c2;
            }
        }

        // Replace old population and update best
        for (int i = 0; i < POP_SIZE; ++i) {
            population[i] = new_population[i];
            if (population[i].fitness > best_individual.fitness) {
                best_individual = population[i];
            }
        }

        // Print generation stats
        double fx = best_individual.gene * sin(10 * M_PI * best_individual.gene) + 1.0;
        double error = fabs(fx - TARGET_VALUE);
        printf("Generation %3d - Best Fitness: %.6f (Error: %.6f)\n", gen + 1, best_individual.fitness, error);
    }

    // --- Final Result ---
    double final_fx = best_individual.gene * sin(10 * M_PI * best_individual.gene) + 1.0;
    double final_error = fabs(final_fx - TARGET_VALUE);

    printf("\n=== Closest Solution Found ===\n");
    printf("x = %.6f\n", best_individual.gene);
    printf("f(x) = %.6f\n", final_fx);
    printf("Error = %.6f\n", final_error);

    return 0;
}
