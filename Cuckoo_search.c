#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_NESTS 15
#define MAX_ITER 5
#define PA 0.25
#define X_MIN -10.0
#define X_MAX 10.0

typedef struct {
    double position;
    double fitness;
} Nest;

double objective_function(double x) {
    return x * x;
}

double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}


double levy_flight(double lambda) {
    double u = rand_double(0, 1);
    double v = rand_double(0, 1);
    double step = pow(u, -1.0 / lambda) * (v - 0.5);
    return step;
}

void initialize_nests(Nest nests[]) {
    for (int i = 0; i < NUM_NESTS; i++) {
        nests[i].position = rand_double(X_MIN, X_MAX);
        nests[i].fitness = objective_function(nests[i].position);
    }
}

void get_best_nest(Nest nests[], Nest *best) {
    *best = nests[0];
    for (int i = 1; i < NUM_NESTS; i++) {
        if (nests[i].fitness < best->fitness) {
            *best = nests[i];
        }
    }
}

void cuckoo_search(Nest nests[]) {
    Nest best;
    get_best_nest(nests, &best);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < NUM_NESTS; i++) {
            double step = levy_flight(1.5);
            double new_pos = nests[i].position + step;


            if (new_pos > X_MAX) new_pos = X_MAX;
            if (new_pos < X_MIN) new_pos = X_MIN;

            double new_fit = objective_function(new_pos);


            if (new_fit < nests[i].fitness) {
                nests[i].position = new_pos;
                nests[i].fitness = new_fit;
            }
        }

        for (int i = 0; i < NUM_NESTS; i++) {
            if (rand_double(0, 1) < PA) {
                nests[i].position = rand_double(X_MIN, X_MAX);
                nests[i].fitness = objective_function(nests[i].position);
            }
        }

        get_best_nest(nests, &best);
        printf("Iteration %d: Best x = %.6f, f(x) = %.6f\n", iter + 1, best.position, best.fitness);
    }

    printf("\nOptimal solution found: x = %.6f, f(x) = %.6f\n", best.position, best.fitness);
}

int main() {
    srand(time(NULL));
    Nest nests[NUM_NESTS];
    initialize_nests(nests);
    cuckoo_search(nests);
    return 0;
}
