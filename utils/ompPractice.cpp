#include <iostream>
#include <omp.h>
#include <vector>
#include <iomanip>
#include <utility>
#include <string>
#include <chrono>

#define NUM_THREADS 4
#define MAX_ELEMENTS 1024

static unsigned long int num_steps = 10000000;
double step;

inline float maxFunc(float A, float B) { return A > B ? A : B; }
inline float minFunc(float A, float B) { return A < B ? A : B; }

int main()
{
    double pi, sum[NUM_THREADS];
    int nthreads;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);

    auto startTime = std::chrono::system_clock::now();
    #pragma omp parallel
    {
        int i, id, nthrds;
        double x, sum;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        /* Only one threads should copy 
           the number of threads to the global value
           to mask sure multiple threads writing to 
           the same address don't conflict. */
        if (id == 0) nthreads = nthrds;

        for (i=id, sum=0.0; i<num_steps; i=i+nthrds) {
            x = (i+0.5)*step;
            sum += 4.0/(1.0+x*x);
        }

        #pragma omp critical
        pi = sum * step;
    }
    auto endTime = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsedTime = endTime - startTime;
    std::cout << "Elepsed time(s): " << elapsedTime.count() << "\n";
    std::cout << ">>>>>>>> " << pi  << " <<<<<<<<<"  << "\n";

    // reduceAve
    float A[MAX_ELEMENTS];
    for (int i=0; i<MAX_ELEMENTS; i++) {
        A[i] = rand() % 100;
    }
    float maxV = A[0], minV = A[0];

    #pragma omp parallel for reduction (max:maxV)
    for (int i=0; i<MAX_ELEMENTS; i++) {
        maxV = maxFunc(A[i], maxV);
        minV = minFunc(A[i], minV);
    }
    std::cout << "Max value: " << maxV << std::endl;
    std::cout << "Min value: " << minV << std::endl; 

    return 0;
}