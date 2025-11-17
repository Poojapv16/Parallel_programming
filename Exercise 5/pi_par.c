#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;
double step;

int main() {
    int i;
    double pi, total_sum = 0.0;
    double start, end;

    step = 1.0 / (double) num_steps;

    start = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int threads = omp_get_num_threads();
        double local_sum = 0.0;

        long chunk = num_steps / threads;
        long start_i = tid * chunk;
        long end_i = (tid == threads - 1) ? num_steps : start_i + chunk;

        for (i = start_i; i < end_i; i++) {
            double x = (i + 0.5) * step;
            local_sum += 4.0 / (1.0 + x * x);
        }

        #pragma omp atomic
        total_sum += local_sum;
    }

    pi = step * total_sum;
    end = omp_get_wtime();

    printf("Approximation of Pi: %.10f\n", pi);
    printf("Manual parallel version time: %f seconds\n", end - start);
    return 0;
}
