#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_CITIES 5
#define NUM_ANTS 10
#define MAX_ITER 5
#define ALPHA 1.0
#define BETA 5.0
#define EVAPORATION 0.5
#define Q 100.0

double distance[NUM_CITIES][NUM_CITIES] = {
    {0, 2, 9, 10, 7},
    {2, 0, 6, 4, 3},
    {9, 6, 0, 8, 5},
    {10, 4, 8, 0, 6},
    {7, 3, 5, 6, 0}
};

double pheromone[NUM_CITIES][NUM_CITIES];
int paths[NUM_ANTS][NUM_CITIES];
double path_length[NUM_ANTS];

void initialize_pheromones() {
    for (int i = 0; i < NUM_CITIES; i++)
        for (int j = 0; j < NUM_CITIES; j++)
            pheromone[i][j] = 1.0;
}

int select_next_city(int ant, int current_city, int visited[]) {
    double probabilities[NUM_CITIES];
    double sum = 0.0;

    for (int i = 0; i < NUM_CITIES; i++) {
        if (!visited[i]) {
            probabilities[i] = pow(pheromone[current_city][i], ALPHA) *
                               pow(1.0 / distance[current_city][i], BETA);
            sum += probabilities[i];
        } else {
            probabilities[i] = 0.0;
        }
    }

    double r = ((double)rand() / RAND_MAX) * sum;
    double total = 0.0;
    for (int i = 0; i < NUM_CITIES; i++) {
        total += probabilities[i];
        if (total >= r) return i;
    }


    for (int i = 0; i < NUM_CITIES; i++)
        if (!visited[i]) return i;

    return -1;
}

void construct_solutions() {
    for (int k = 0; k < NUM_ANTS; k++) {
        int visited[NUM_CITIES] = {0};
        int start = rand() % NUM_CITIES;
        paths[k][0] = start;
        visited[start] = 1;

        for (int step = 1; step < NUM_CITIES; step++) {
            int next = select_next_city(k, paths[k][step - 1], visited);
            paths[k][step] = next;
            visited[next] = 1;
        }
    }
}

void compute_path_lengths() {
    for (int k = 0; k < NUM_ANTS; k++) {
        double length = 0.0;
        for (int i = 0; i < NUM_CITIES - 1; i++) {
            length += distance[paths[k][i]][paths[k][i + 1]];
        }
        length += distance[paths[k][NUM_CITIES - 1]][paths[k][0]];
        path_length[k] = length;
    }
}

void update_pheromones() {
    for (int i = 0; i < NUM_CITIES; i++)
        for (int j = 0; j < NUM_CITIES; j++)
            pheromone[i][j] *= (1.0 - EVAPORATION);

    for (int k = 0; k < NUM_ANTS; k++) {
        for (int i = 0; i < NUM_CITIES - 1; i++) {
            int from = paths[k][i];
            int to = paths[k][i + 1];
            pheromone[from][to] += Q / path_length[k];
            pheromone[to][from] += Q / path_length[k];
        }

        int from = paths[k][NUM_CITIES - 1];
        int to = paths[k][0];
        pheromone[from][to] += Q / path_length[k];
        pheromone[to][from] += Q / path_length[k];
    }
}

void print_best_path() {
    int best_ant = 0;
    for (int k = 1; k < NUM_ANTS; k++) {
        if (path_length[k] < path_length[best_ant])
            best_ant = k;
    }

    printf("Best path (length = %.2f): ", path_length[best_ant]);
    for (int i = 0; i < NUM_CITIES; i++) {
        printf("%d ", paths[best_ant][i]);
    }
    printf("%d\n", paths[best_ant][0]);
}

int main() {
    srand(time(NULL));
    initialize_pheromones();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        construct_solutions();
        compute_path_lengths();
        update_pheromones();
        printf("Iteration %d complete.\n", iter + 1);
    }

    print_best_path();
    return 0;
}
