#include "cuda_tools.h"

int main() {
    int device = 0;
    checkCudaRuntime(cudaSetDevice(device));

    // global memory
    float* memory_device = nullptr;
    checkCudaRuntime(cudaMalloc(&memory_device, 100*sizeof(float)));

    // pageable memory
    float* memory_host = new float[100];
    memory_host[2] = 520.25;
    checkCudaRuntime(cudaMemcpy(memory_device, memory_host, 100*sizeof(float), cudaMemcpyHostToDevice));

    // pinned memory
    // Note: dst, src
    float* memory_page_locked = nullptr;
    checkCudaRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float)));
    checkCudaRuntime(cudaMemcpy(memory_page_locked, memory_device, 100*sizeof(float), cudaMemcpyDeviceToHost));

    printf("%f\n", memory_page_locked[2]);
    delete[] memory_host;
    checkCudaRuntime(cudaFree(memory_device));
    return 0;
}