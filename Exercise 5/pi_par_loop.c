#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;
double step;

int main() {
    step = 1.0 / (double) num_steps;

    double sum = 0.0;
    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum) schedule(dynamic, 100)
    for (long i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = step * sum;
    double end = omp_get_wtime();

    printf("Pi = %.10f\n", pi);
    printf("parallel for (dynamic,100) time = %f seconds\n", end - start);

    return 0;
}
