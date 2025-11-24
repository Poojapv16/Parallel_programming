#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;
double step;

int main() {
    step = 1.0 / (double)num_steps;

    // Variant A: critical reduction
    double startA = omp_get_wtime();
    double piA = 0.0;

    #pragma omp parallel
    {
        double local_sum = 0.0;
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        long chunk = num_steps / nthreads;
        long start_i = tid * chunk;
        long end_i = (tid == nthreads - 1) ? num_steps : start_i + chunk;

        for (long i = start_i; i < end_i; i++) {
            double x = (i + 0.5) * step;
            local_sum += 4.0 / (1.0 + x * x);
        }

        #pragma omp critical
        piA += local_sum;
    }

    double endA = omp_get_wtime();
    printf("Variant A (critical block) Pi = %.10f, Time = %f\n", piA * step, endA - startA);

    // Variant B: cyclic decomposition
    double startB = omp_get_wtime();
    double piB = 0.0;

    #pragma omp parallel
    {
        double local_sum = 0.0;
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for (long i = tid; i < num_steps; i += nthreads) {
            double x = (i + 0.5) * step;
            local_sum += 4.0 / (1.0 + x * x);
        }

        #pragma omp critical
        piB += local_sum;
    }

    double endB = omp_get_wtime();
    printf("Variant B (cyclic) Pi = %.10f, Time = %f\n", piB * step, endB - startB);

    return 0;
}
