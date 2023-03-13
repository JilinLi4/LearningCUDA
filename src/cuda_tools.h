#ifndef __CUDA_TOOLS_H__
#define __CUDA_TOOLS_H__
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define INFOE printf
#define checkCudaRuntime(call) CUDATools::checkRuntime(call, #call, __LINE__, __FILE__)

namespace CUDATools
{

bool checkRuntime(cudaError_t e, const char* call, int line, const char *file){
        if (e != cudaSuccess) {
            INFOE("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", 
                call, 
                cudaGetErrorString(e), 
                cudaGetErrorName(e), 
                e, file, line
            );
            return false;
        }
        return true;
}

class CUDATimeCost {
public:
    void start() {
        elapsed_time_ = 0.0;
        // 初始化cudaEvent
        checkCudaRuntime(cudaEventCreate(&start_));
        checkCudaRuntime(cudaEventCreate(&stop_));

        // 记录开始事件
        checkCudaRuntime(cudaEventRecord(start_));
        cudaEventQuery(start_);
    }

    void stop() {
        // 记录结束事件
        checkCudaRuntime(cudaEventRecord(stop_));
        checkCudaRuntime(cudaEventSynchronize(stop_));
        // 计算事件差
        checkCudaRuntime(cudaEventElapsedTime(&elapsed_time_, start_, stop_));
        checkCudaRuntime(cudaEventDestroy(start_));
        checkCudaRuntime(cudaEventDestroy(stop_));
    }

    /**
     * @brief Get the elapsed time ms
     * 
     * @return float 
     */
    float get_elapsed_time() {
        return elapsed_time_;
    }

private:
    cudaEvent_t start_, stop_;
    float elapsed_time_{0.0};
};

} // CUDATools

#endif // __CUDA_TOOLS_H__
