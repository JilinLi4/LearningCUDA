#include "cuda_tools.h"

int main() {
    int device_id = 0;
    checkCudaRuntime(cudaSetDevice(device_id));

    cudaStream_t stream = nullptr;
    checkCudaRuntime(cudaStreamCreate(&stream));

    // 在GPU上开辟一个 100 * float 的空间
    size_t size = 100 * sizeof(float);
    float* device_mem = nullptr;
    checkCudaRuntime(cudaMalloc(&device_mem, size));

    // 在CPU开辟空间，并将数据存入到GPU
    float* host_mem = nullptr;
    checkCudaRuntime(cudaMallocHost(&host_mem, size));
    host_mem[2] = 520.25;
    // 通过流异步操作
    checkCudaRuntime(cudaMemcpyAsync(device_mem, host_mem, size, cudaMemcpyHostToDevice, stream));

    // 在CPU上开辟空间，将数据从GPU拷贝到CPU
    float* pin_mem_host = nullptr;
    checkCudaRuntime(cudaMallocHost(&pin_mem_host, size));
    checkCudaRuntime(cudaMemcpyAsync(pin_mem_host, device_mem, size, cudaMemcpyDeviceToHost));

    // 等待流中的所有任务进行完
    checkCudaRuntime(cudaStreamSynchronize(stream));

    printf("%f\n", pin_mem_host[2]);

    // 释放
    checkCudaRuntime(cudaFree(device_mem));
    checkCudaRuntime(cudaFreeHost(pin_mem_host));
    checkCudaRuntime(cudaFreeHost(host_mem));

    // 释放流
    checkCudaRuntime(cudaStreamDestroy(stream));

    return 0;
}