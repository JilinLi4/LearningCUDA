#include <math.h>
#include <stdio.h>
#include "cuda_tools.h"

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void __global__ add(const double*x, const double *y, double *z);
void check(double *z, const int N);

int main() {
    const int N = 100000;
    const int M = sizeof(double) * N;
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    for(int n = 0; n < N; ++n) {
        h_x[n] = a;
        h_y[n] = b;
    }
    double *d_x, *d_y, *d_z;
    checkCudaRuntime(cudaMalloc((void**)&d_x, M));
    checkCudaRuntime(cudaMalloc((void**)&d_y, M));
    checkCudaRuntime(cudaMalloc((void**)&d_z, M));

    checkCudaRuntime(cudaMemcpy(d_x, h_x, M, cudaMemcpyDeviceToDevice));

    checkCudaRuntime(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    const int block_size = 128;
    const int grid_size = N/block_size;

    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

    checkCudaRuntime(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
    printf("\n h_x[0]: %lf, h_y[0]: %lf, h_z[0]: %lf\n", h_x[0], h_y[0], h_z[0]);
    free(h_x);
    free(h_y);
    free(h_z);
    checkCudaRuntime(cudaFree(d_x));
    checkCudaRuntime(cudaFree(d_y));
    checkCudaRuntime(cudaFree(d_z));
    return 0;
}

void __global__ add(const double*x, const double *y, double *z) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

