#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define GRID_SIZE 10
#define NUM_UNITS 5
#define NUM_INCIDENTS 20
#define MAX_ITERATIONS 5

typedef struct {
    int x, y;
} Point;

double distance(Point a, Point b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

double evaluate_fitness(Point units[], Point incidents[]) {
    double total = 0.0;
    for (int i = 0; i < NUM_INCIDENTS; i++) {
        double min_dist = 1e9;
        for (int j = 0; j < NUM_UNITS; j++) {
            double d = distance(units[j], incidents[i]);
            if (d < min_dist) min_dist = d;
        }
        total += min_dist;
    }
    return total / NUM_INCIDENTS;
}

void randomize_units(Point units[]) {
    for (int i = 0; i < NUM_UNITS; i++) {
        units[i].x = rand() % GRID_SIZE;
        units[i].y = rand() % GRID_SIZE;
    }
}

void generate_incidents(Point incidents[]) {
    for (int i = 0; i < NUM_INCIDENTS; i++) {
        incidents[i].x = rand() % GRID_SIZE;
        incidents[i].y = rand() % GRID_SIZE;
    }
}

void mutate_units(Point units[]) {
    for (int i = 0; i < NUM_UNITS; i++) {
        if (rand() % 100 < 30) { // 30% chance to move
            units[i].x = (units[i].x + (rand() % 3 - 1) + GRID_SIZE) % GRID_SIZE;
            units[i].y = (units[i].y + (rand() % 3 - 1) + GRID_SIZE) % GRID_SIZE;
        }
    }
}

int main() {
    srand(time(NULL));

    Point best_units[NUM_UNITS];
    Point current_units[NUM_UNITS];
    Point incidents[NUM_INCIDENTS];

    generate_incidents(incidents);
    randomize_units(best_units);

    double best_fitness = evaluate_fitness(best_units, incidents);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        for (int i = 0; i < NUM_UNITS; i++)
            current_units[i] = best_units[i];

        mutate_units(current_units);
        double current_fitness = evaluate_fitness(current_units, incidents);

        if (current_fitness < best_fitness) {
            best_fitness = current_fitness;
            for (int i = 0; i < NUM_UNITS; i++)
                best_units[i] = current_units[i];
        }

        printf("Iteration %d: Avg Response Time = %.2f\n", iter + 1, best_fitness);
    }

    printf("\nBest Unit Positions:\n");
    for (int i = 0; i < NUM_UNITS; i++) {
        printf("Unit %d: (%d, %d)\n", i + 1, best_units[i].x, best_units[i].y);
    }

    return 0;
}
