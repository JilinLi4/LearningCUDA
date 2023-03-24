#include "cuda_tools.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define __FLT_EPSILON__     1e-5

__global__ void sum_kernel(float* array, int n, float* output) {
    int position = blockIdx.x + blockDim.x * threadIdx.x;

    // 用extern 申明外部的动态大小共享内存，由启动核函数的第三个参数指定
    // cache 的大小为 block_size * sizeof(float)
    extern __shared__ float cache[];

    int block_size  = blockDim.x;
    // block中的线程号
    int lane        = threadIdx.x;
    float value     = 0;

    if(position < n) {
        value = array[position];
    }

    for(int i = block_size / 2; i > 0; i /=2) {
        cache[lane] = value;
        __syncthreads(); // 等待block内的所有线程存储完毕
        if(lane < i ) value += cache[lane + i];
        __syncthreads(); // 等待block内的所有线程读取完毕
    }

    if(lane == 0) {
        printf("block %d value = %f\n", blockIdx.x, value);
        // 由于可能动用了多个block,所以汇总结果的时候需要用atomicAdd
        // 这里的value仅仅是一个仅仅是一个block的所有thread reduce sum 后的结果。
        atomicAdd(output, value); 
    }
}


void __global__ reduce_gpu(float* d_x, int N, float* d_y)
{
    // block
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    int bidx = blockIdx.x;
    if(blockIdx.x ==0 && threadIdx.x == 0) {
        // printf("gridDim.x: %d blockDim.x: %d " gridDim.x,  blockDim.x);
        printf("\nblockIdx.x: %d threadIdx.x: %d \n", blockIdx.x, threadIdx.x);
        printf("gridDim.x: %d blockDim.x: %d \n", gridDim.x, blockDim.x);
    }

    
    for(int offset = N / 2; offset > 0; offset /= 2) { //loop 1
        // for(int n = 0; n< N /2; ++n) { //loop 2 
            if(bidx < offset) {
                d_x[n] += d_x[n + offset];
            }
        // }
        __syncthreads();
    }
    d_y[0] = d_x[0];
}

void reduce_sum_cpu(float *x, int N, float* y) {
    for(int offset = N / 2; offset > 0; offset /= 2) {
        for(int n = 0; n< N /2; ++n) {
            if(n < offset) {
                x[n] += x[n + offset];
            }
        }
    }
    *y = x[0];
}

void launch_reduce_sum(float* array, int n, float* output){

    const int nthreads = 512;
    int block_size = n < nthreads ? n : nthreads;
    int grid_size = (n + block_size - 1) / block_size;

    // 这里要求block_size必须是2的幂次
    float block_sqrt = log2(block_size);
    printf("old block_size = %d, block_sqrt = %.2f\n", block_size, block_sqrt);

    block_sqrt = ceil(block_sqrt);
    block_size = pow(2, block_sqrt);

    printf("block_size = %d, grid_size = %d\n", block_size, grid_size);
    reduce_gpu<<<grid_size, block_size, block_size * sizeof(float), nullptr>>>( // 这里 
        array, n, output
    ); // 这里要开辟 block_size * sizeof(float) 这么大的共享内存
}


void invoke_reduce_sum_global_memory(float* src_array, int n, float& out) {
    float* input_host = new float[n];
    float* input_device = nullptr;

    checkCudaRuntime(cudaMalloc(&input_device, n * sizeof(float)));
    checkCudaRuntime(cudaMemcpy(input_device, input_host, n * sizeof(float),cudaMemcpyHostToDevice));

    float output_host = 0;
    float* output_device = nullptr;
    checkCudaRuntime(cudaMalloc(&output_device, sizeof(float)));
    checkCudaRuntime(cudaMemset(output_device, 0, sizeof(float)));

    launch_reduce_sum(input_device, n, output_device);
    checkCudaRuntime(cudaPeekAtLastError());

    checkCudaRuntime(cudaMemcpy(&output_host, output_device, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaDeviceSynchronize());
    printf("output_host: %g \n", output_host);
    // out = 0.0;
    // for(int i = 0; i < block_size; ++i) {
    //     out += h_out_array[i];
    // }
}

int main() {
    const int n = 2<<10;
    float* input_host = new float[n];
    float* input_device = nullptr;
    float ground_truth = 0;
    for(int i = 0; i < n; ++i){
        input_host[i] = i;
        ground_truth += i;
    }

    // CPU 版本规约求和
    float cpu_rst = 0.0;
    reduce_sum_cpu(input_host, n, &cpu_rst);
    printf("cpu_rst: %g   ground_truth: %g \n", cpu_rst, ground_truth);

    float gpu_global_result = 0.0;
    invoke_reduce_sum_global_memory(input_host, n, gpu_global_result);
    printf(" cpu_global_result: %g   gpu_global_result: %g \n", cpu_rst, gpu_global_result);

    // checkCudaRuntime(cudaMalloc(&input_device, n * sizeof(float)));
    // checkCudaRuntime(cudaMemcpy(input_device, input_host, n * sizeof(float),cudaMemcpyHostToDevice));

    // float output_host = 0;
    // float* output_device = nullptr;
    // checkCudaRuntime(cudaMalloc(&output_device, sizeof(float)));
    // checkCudaRuntime(cudaMemset(output_device, 0, sizeof(float)));

    // launch_reduce_sum(input_device, n, output_device);
    // checkCudaRuntime(cudaPeekAtLastError());

    // checkCudaRuntime(cudaMemcpy(&output_host, output_device, sizeof(float), cudaMemcpyDeviceToHost));
    // checkCudaRuntime(cudaDeviceSynchronize());

    // printf("output_host = %f, ground_truth = %f\n", output_host, ground_truth);
    // if(fabs(output_host - ground_truth) <= __FLT_EPSILON__){ // fabs 求绝对值
    //     printf("result correct.\n");
    // }else{
    //     printf("result error.\n");
    // }

    // cudaFree(input_device);
    // cudaFree(output_device);

    // delete [] input_host;
    // printf("done\n");
    // return 0;

    // return 0;
}