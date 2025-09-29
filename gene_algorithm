#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define POP_SIZE 6
#define CHROM_LENGTH 5
#define GENERATIONS 5  // Reduced from 10 to 5
#define MUTATION_RATE 0.1

int population[POP_SIZE][CHROM_LENGTH];
int fitness[POP_SIZE];

int decode(int chrom[]) {
    int value = 0;
    for (int i = 0; i < CHROM_LENGTH; i++) {
        value += chrom[i] * (1 << (CHROM_LENGTH - i - 1));
    }
    return value;
}

int evaluate(int chrom[]) {
    int x = decode(chrom);
    return x * x;
}

void initialize_population() {
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < CHROM_LENGTH; j++) {
            population[i][j] = rand() % 2;
        }
    }
}

void evaluate_population() {
    for (int i = 0; i < POP_SIZE; i++) {
        fitness[i] = evaluate(population[i]);
    }
}

void select_parents(int *p1, int *p2) {
    int max1 = -1, max2 = -1;
    for (int i = 0; i < POP_SIZE; i++) {
        if (fitness[i] > max1) {
            max2 = max1;
            *p2 = *p1;
            max1 = fitness[i];
            *p1 = i;
        } else if (fitness[i] > max2) {
            max2 = fitness[i];
            *p2 = i;
        }
    }
}

void crossover(int parent1[], int parent2[], int child1[], int child2[]) {
    int point = rand() % CHROM_LENGTH;
    for (int i = 0; i < CHROM_LENGTH; i++) {
        if (i < point) {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        } else {
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
    }
}

void mutate(int chrom[]) {
    for (int i = 0; i < CHROM_LENGTH; i++) {
        if ((rand() / (float)RAND_MAX) < MUTATION_RATE) {
            chrom[i] = 1 - chrom[i];
        }
    }
}

void replace_population(int child1[], int child2[]) {
    int min1 = 0, min2 = 1;
    for (int i = 2; i < POP_SIZE; i++) {
        if (fitness[i] < fitness[min1]) {
            min2 = min1;
            min1 = i;
        } else if (fitness[i] < fitness[min2]) {
            min2 = i;
        }
    }
    for (int i = 0; i < CHROM_LENGTH; i++) {
        population[min1][i] = child1[i];
        population[min2][i] = child2[i];
    }
}

int main() {
    srand(time(NULL));
    initialize_population();

    for (int gen = 0; gen < GENERATIONS; gen++) {
        evaluate_population();

        int p1 = 0, p2 = 1;
        select_parents(&p1, &p2);

        int child1[CHROM_LENGTH], child2[CHROM_LENGTH];
        crossover(population[p1], population[p2], child1, child2);
        mutate(child1);
        mutate(child2);

        replace_population(child1, child2);

        printf("Generation %d:\n", gen + 1);
        for (int i = 0; i < POP_SIZE; i++) {
            printf("Chromosome %d: ", i);
            for (int j = 0; j < CHROM_LENGTH; j++) {
                printf("%d", population[i][j]);
            }
            printf(" -> x = %d, f(x) = %d\n", decode(population[i]), fitness[i]);
        }
        printf("\n");
    }

    return 0;
}
