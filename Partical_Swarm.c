#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_PARTICLES 10
#define MAX_ITER 5
#define DIM 1
#define W 0.5
#define C1 1.5
#define C2 1.5
#define X_MIN -10.0
#define X_MAX 10.0
#define V_MAX 2.0

typedef struct {
    double position[DIM];
    double velocity[DIM];
    double best_position[DIM];
    double best_value;
} Particle;

double objective_function(double x[]) {
    return x[0] * x[0];  // f(x) = x^2
}

double rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

void initialize_particles(Particle swarm[]) {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        for (int d = 0; d < DIM; d++) {
            swarm[i].position[d] = rand_double(X_MIN, X_MAX);
            swarm[i].velocity[d] = rand_double(-V_MAX, V_MAX);
            swarm[i].best_position[d] = swarm[i].position[d];
        }
        swarm[i].best_value = objective_function(swarm[i].position);
    }
}

void update_velocity_position(Particle *p, double global_best[], double global_best_value) {
    for (int d = 0; d < DIM; d++) {
        double r1 = rand_double(0, 1);
        double r2 = rand_double(0, 1);

        p->velocity[d] = W * p->velocity[d]
                       + C1 * r1 * (p->best_position[d] - p->position[d])
                       + C2 * r2 * (global_best[d] - p->position[d]);

        if (p->velocity[d] > V_MAX) p->velocity[d] = V_MAX;
        if (p->velocity[d] < -V_MAX) p->velocity[d] = -V_MAX;

        p->position[d] += p->velocity[d];

        if (p->position[d] > X_MAX) p->position[d] = X_MAX;
        if (p->position[d] < X_MIN) p->position[d] = X_MIN;
    }
}

int main() {
    srand(time(NULL));
    Particle swarm[NUM_PARTICLES];
    double global_best[DIM];
    double global_best_value = INFINITY;

    initialize_particles(swarm);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < NUM_PARTICLES; i++) {
            double value = objective_function(swarm[i].position);

            if (value < swarm[i].best_value) {
                swarm[i].best_value = value;
                for (int d = 0; d < DIM; d++) {
                    swarm[i].best_position[d] = swarm[i].position[d];
                }
            }

            if (value < global_best_value) {
                global_best_value = value;
                for (int d = 0; d < DIM; d++) {
                    global_best[d] = swarm[i].position[d];
                }
            }
        }

        for (int i = 0; i < NUM_PARTICLES; i++) {
            update_velocity_position(&swarm[i], global_best, global_best_value);
        }

        printf("Iteration %d: Best Value = %.6f at x = %.6f\n", iter + 1, global_best_value, global_best[0]);
    }

    printf("\nOptimal solution found: x = %.6f, f(x) = %.6f\n", global_best[0], global_best_value);
    return 0;
}
