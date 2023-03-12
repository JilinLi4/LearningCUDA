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

} // CUDATools

#endif // __CUDA_TOOLS_H__
