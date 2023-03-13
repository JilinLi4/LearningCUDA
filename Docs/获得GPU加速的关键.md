# CUDA事件计时
CUDA提供了一种基于CUDA事件(CUDA event)的计时方式，可用来给一段CUDA代码(可能包含主机代码和设备代码)计时。
对计时器的封装：
```c
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
```

## 比较单精度和双精度的性能差异
测试的GPU为 MX450 显存带宽为：10GBps
```shell
double : 58.4116 ms
float  : 27.7594 ms
```
### 有效显存带宽
$$ 
\frac{8 \times 10^7 \times 4 B}{2.7 \times 10^{-2}}\tag{1}
$$
# 几个影响GPU加速的关键因素
## 1. 数据传输比
当我们将计时器只计算运算的时间时发现，在向量求和的程序中，大部分的时间都消耗在了数据拷贝上。
- GPU的显存带宽理论值(几百吉比特)远大于PCIex16 (16GB/s)的带宽。相差几十倍
- 在CUDA编程的过程中应该尽可能减少数据在主机和设备之间的拷贝

## 2. 算术强度
一个问题的算术强度指的是其中算术操作的工作量与必要的内存操作的工作量之比。例如在求和操作中，去两次数据、存一次数据，但是只做一次计算，这样的算术强度就不高。在CUDA中，设备内存的读、写都是比较耗时的。

## 3. 并行规模
并行规模可用GPU中的总的线程数目来衡量。从硬件的角度来看，一个GPU由多个流处理器(streaming multiprocessor, SM)构成，而每个SM中有若干CUDA核心。每个SM是相对独立的。一个SM中最多能驻留线程的个数为1024(图灵架构)，开普勒架构到伏特架构最多驻留的线程个数为2048. 一块GPU中一般有几个到几十个SM。所以一块GPU一共可以驻留几万到几十万个线程。所以一个核函数定义的线程数目小于这个数的话，就很难得到很高的加速比。所以对于数据规模很小的问题，用GPU很难得到可观的加速。

# 总结
一个CUDA程序能够获得高性能的必要条件有如下几点：
- 数据传输比例较小
- 核函数的算术强度较高。
- 核函数中定义的线程数目较多

在编写优化CUDA程序时，一定要想方设法做到如下几点：
- 减少主机与设备之间的数据传输
- 提高核函数的算术强度
- 增大核函数的并行规模




