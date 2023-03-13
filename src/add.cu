#include <math.h>
#include <stdio.h>
#include <math.h>
#include "cuda_tools.h"


#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;

#endif
const real a = 1.23;
const real b = 2.34;
const real c = 3.57;

void __global__ add(const real*x, const real *y, real *z);
void __host__ cpu_add(const real*x, const real *y, real *z, int N);
void check(real *z, const int N);

int main() {
    const int N = 80000000;
    const int M = sizeof(real) * N;
    real *h_x = (real*) malloc(M);
    real *h_y = (real*) malloc(M);
    real *h_z = (real*) malloc(M);

    for(int n = 0; n < N; ++n) {
        h_x[n] = a;
        h_y[n] = b;
    }
    CUDATools::CUDATimeCost cuda_timer;
    real *d_x, *d_y, *d_z;
    cuda_timer.start();
    checkCudaRuntime(cudaMalloc((void**)&d_x, M));
    checkCudaRuntime(cudaMalloc((void**)&d_y, M));
    checkCudaRuntime(cudaMalloc((void**)&d_z, M));

    checkCudaRuntime(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    checkCudaRuntime(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    const int block_size = 1024;
    const int grid_size = N/block_size;

    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

    checkCudaRuntime(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
    cuda_timer.stop();

    CUDATools::CUDATimeCost cpu_timer;
    cpu_timer.start();
    cpu_add(h_x, h_y, h_z, N);
    cpu_timer.stop();
    printf("\n gpu cost time: %g ms, h_x[0]: %lf, h_y[0]: %lf, h_z[0]: %lf\n", cuda_timer.get_elapsed_time(), h_x[0], h_y[0], h_z[0]);
    printf("\n cpu cost time: %g ms, h_x[0]: %lf, h_y[0]: %lf, h_z[0]: %lf\n", cpu_timer.get_elapsed_time(), h_x[0], h_y[0], h_z[0]);
    free(h_x);
    free(h_y);
    free(h_z);
    checkCudaRuntime(cudaFree(d_x));
    checkCudaRuntime(cudaFree(d_y));
    checkCudaRuntime(cudaFree(d_z));
    return 0;
}

void cpu_add(const real*x, const real *y, real *z, int N) {
    for(int i = 0; i < N; ++i) {
        z[i] = sqrt(x[i] + y[i]);
    }
}

void __global__ add(const real*x, const real *y, real *z) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    z[i] = sqrt(x[i] + y[i]);
}

