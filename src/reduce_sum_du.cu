#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "cuda_tools.h"

// 共享内存
__global__ void sum_kernel(float* array, int n, float* output){
   
    int position = blockIdx.x * blockDim.x + threadIdx.x;

    // 使用 extern声明外部的动态大小共享内存，由启动核函数的第三个参数指定
    extern __shared__ float cache[]; // 这个cache 的大小为 block_size * sizeof(float)
    int block_size = blockDim.x;
    int lane       = threadIdx.x;
    float value    = 0;

    if(position < n)
        value = array[position];

    for(int i = block_size / 2; i > 0; i /= 2){ // 如何理解reduce sum 参考图片：figure/1.reduce_sum.jpg
        cache[lane] = value;
        for(int i = 0; i < block_size; i++) {
            printf("blockIdx: %d threadIdx: %d, cache[%d] = %g \n", blockIdx.x, threadIdx.x, i, cache[i]);
        }
        __syncthreads();  // 等待block内的所有线程储存完毕
        printf("blockIdx: %d epoch: %d ---------------\n", blockIdx.x, i);
        if(lane < i) value += cache[lane + i];
        __syncthreads();  // 等待block内的所有线程读取完毕
    }

    if(lane == 0){
        printf("block %d value = %f\n", blockIdx.x, value);
        atomicAdd(output, value); // 由于可能动用了多个block，所以汇总结果的时候需要用atomicAdd。（注意这里的value仅仅是一个block的threads reduce sum 后的结果）
    }
}

// 全局内存
void __global__ reduce_gpu(float* d_x, int N, float* d_y)
{
    // block
    printf("blockDim.x: %d blockIdx.x: %d threadDim.x: %d threadIdx.x: %d\n", gridDim.x, blockIdx.x, blockDim.x);
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    int tidx = threadIdx.x;
    for(int offset = blockDim.x / 2; offset > 0; offset /= 2) { //loop 1
            if(tidx < offset) {
                d_x[n] += d_x[n+offset];
            }
        __syncthreads();
    }
    if(tidx == 0) {
        printf("++++d_x: %g, dy = %g\n", d_x[n], *d_y);
        atomicAdd(d_y, d_x[n]);
        printf("----d_x: %g, dy = %g\n", d_x[n], *d_y);
    }
}


void launch_reduce_sum(float* array, int n, float* output, int& grid_size){

    const int nthreads = 4;
    int block_size = n < nthreads ? n : nthreads;
    grid_size = (n + block_size - 1) / block_size;

    // 这里要求block_size必须是2的幂次
    float block_sqrt = log2(block_size);
    printf("old block_size = %d, block_sqrt = %.2f\n", block_size, block_sqrt);

    block_sqrt = ceil(block_sqrt);
    block_size = pow(2, block_sqrt);

    printf("block_size = %d, grid_size = %d\n", block_size, grid_size);
    reduce_gpu<<<grid_size, block_size, block_size * sizeof(float), nullptr>>>( // 这里 
        array, n, output
    ); // 这里要开辟 block_size * sizeof(float) 这么大的共享内存，
    // sum_kernel<<<grid_size, block_size, block_size * sizeof(float), nullptr>>>( // 这里 
    //     array, n, output
    // ); // 这里要开辟 block_size * sizeof(float) 这么大的共享内存，
}

int main() 
{
    const int n = 1<<12;
    float* input_host = new float[n];
    float* input_device = nullptr;
    float ground_truth = 0;
    for(int i = 0; i < n; ++i){
        input_host[i] = i;
        ground_truth += i;
    }

    checkCudaRuntime(cudaMalloc(&input_device, n * sizeof(float)));
    checkCudaRuntime(cudaMemcpy(input_device, input_host, n * sizeof(float), cudaMemcpyHostToDevice));

    float output_host = 0;
    float* output_device = nullptr;
    checkCudaRuntime(cudaMalloc(&output_device, sizeof(float)));
    checkCudaRuntime(cudaMemset(output_device, 0,  sizeof(float)));
    int block_size;
    launch_reduce_sum(input_device, n, output_device, block_size);
    checkCudaRuntime(cudaPeekAtLastError());

    checkCudaRuntime(cudaMemcpy(&output_host, output_device, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaDeviceSynchronize());
    printf("output_host = %f, ground_truth = %f\n", output_host, ground_truth);
    if(fabs(output_host - ground_truth) <= 1e-5){ // fabs 求绝对值
        printf("PASS.\n");
    }else{
        printf("ERROR.\n");
    }

    cudaFree(input_device);
    cudaFree(output_device);

    delete [] input_host;
    printf("done\n");
    return 0;
}