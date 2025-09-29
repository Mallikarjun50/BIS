#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DIM 5
#define N 20
#define MAX_ITER 5
#define LOWER_BOUND -100
#define UPPER_BOUND 100

double wolves[N][DIM];
double alpha[DIM], beta[DIM], delta[DIM];
double alpha_score = INFINITY, beta_score = INFINITY, delta_score = INFINITY;


double fitness(double *x) {
    double sum = 0.0;
    for (int i = 0; i < DIM; i++)
        sum += x[i] * x[i];
    return sum;
}


double rand_01() {
    return (double)rand() / RAND_MAX;
}


void initialize() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < DIM; j++)
            wolves[i][j] = LOWER_BOUND + rand_01() * (UPPER_BOUND - LOWER_BOUND);
}


void update_leaders() {
    for (int i = 0; i < N; i++) {
        double score = fitness(wolves[i]);
        if (score < alpha_score) {
            alpha_score = score;
            for (int j = 0; j < DIM; j++) alpha[j] = wolves[i][j];
        } else if (score < beta_score) {
            beta_score = score;
            for (int j = 0; j < DIM; j++) beta[j] = wolves[i][j];
        } else if (score < delta_score) {
            delta_score = score;
            for (int j = 0; j < DIM; j++) delta[j] = wolves[i][j];
        }
    }
}


void update_positions(int iter) {
    double a = 2.0 - iter * (2.0 / MAX_ITER);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < DIM; j++) {
            double r1 = rand_01(), r2 = rand_01();
            double A1 = 2 * a * r1 - a;
            double C1 = 2 * r2;
            double D_alpha = fabs(C1 * alpha[j] - wolves[i][j]);
            double X1 = alpha[j] - A1 * D_alpha;

            r1 = rand_01(); r2 = rand_01();
            double A2 = 2 * a * r1 - a;
            double C2 = 2 * r2;
            double D_beta = fabs(C2 * beta[j] - wolves[i][j]);
            double X2 = beta[j] - A2 * D_beta;

            r1 = rand_01(); r2 = rand_01();
            double A3 = 2 * a * r1 - a;
            double C3 = 2 * r2;
            double D_delta = fabs(C3 * delta[j] - wolves[i][j]);
            double X3 = delta[j] - A3 * D_delta;

            wolves[i][j] = (X1 + X2 + X3) / 3.0;
        }
    }
}

int main() {
    srand(time(NULL));
    initialize();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        update_leaders();
        update_positions(iter);
        printf("Iteration %d: Best Score = %.6f\n", iter + 1, alpha_score);
    }

    printf("\nOptimal solution found:\n");
    for (int i = 0; i < DIM; i++)
        printf("x[%d] = %.6f\n", i, alpha[i]);
    printf("Minimum value = %.6f\n", alpha_score);

    return 0;
}
