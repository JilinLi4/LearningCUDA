代码： src/add.cu
# 一、 相关API
## 1. 设备CUDARuntime 初始化
在CUDA运行时API中，没有显示地初始化设备的函数。在第一次调用一个和设备管理及版本查询无关的运行时API函数时，设备将自动地初始化。

## 2. 设备内存的分配及释放
申请显存
```c
cudaError_t cudaMalloc(void **address, size_t size);
```
- address 待分配GPU内存的指针。
- size    分配内存的字节数
- 返回错误码，如果成功，返回 **cudaSuccess**

释放显存
```c
cudaError_t cudaFree(void* address);
```

## 3. 主机与设备之间数据的传递
```c
cudaError_t cudaMemcpy(
    void                *dst,
    const void          *src,
    size_t              count,
    enum cudaMemcpyKind kind
)
```
- dst 是目标地址
- src 是源地址
- count 复制数据的字节数
- kind 一个枚举类型的变量，取如下几个值：
    - cudaMemcpyHostToHost 从主机复制到主机
    - cudaMemcpyHostToDevice 从主机复制到设备
    - cudaMemcpyDevicetoHost 从设备复制到主机
    - cudaMemcpyDefault, 根据指针dst和src所指地址自行判断数据传输方向。

# 二、 核函数中数据与线程的对应
## 1. 使用多个线程的核函数
核函数中允许指派很多线程。因为一个GPU中有几千个计算核心，而总的线程数必须至少等于计算核心数才能充分利用GPU中的全部计算资源。
所以根据需要，在调用核函数时可以指定多个线程。比如：
```c
func<<<2, 4>>>();
```
func指定了2个线程块，每个线程块中包含4个线程。总的线程数是 $2 \times 4 = 8$。也就是说，该程序中的核函数中代码执行方式是："单指令-多线程"，即每一个线程都执行同一串指令。

## 2 使用线程索引
线程组织结构是由调用函数配置的。如：
```c
func<<<gridSize, blockSize>>>();
```
一般来说：
- $\text{gridSize} \le 2^{31}-1 $
- $ \text{blockSize} \le 1024$
每个线程在核函数中都有一个唯一的身份表示。由于我们用两个参数指定线程数目。

## 3 核函数中的数据与线程对应
将有关数据从主机传至设备之后，就可以调用核函数在设备中进行计算了。使用以下线程布局:
- $\text{blockSize} = 128 $
- $\text{gridSize} = 10^8 / 128$
在这种情况下，核函数的数据与线程对应关系为：
```c
const int n = blockDim.x * blockIdx.x + threadIdx.x;
```
## 4 是否需要使用 if 判断 n是否越界？
核函数没有使用参数N，当N是blockDim.x的整数倍时，不会引起问题，因为核函数的线程数刚好等于数组元素个数。然而，当N不是 blockDim.x的整数倍时，就有可能发生错误。例如将 N改为： $10^8 + 1$ 而 block_size依然是128,如果griadSize的计算方式不变，那么必将有一个元素无法处理。所以我们应该将原来的计算方式： 
$\text{gridSize} = N / \text{blockSize}$ 改为
$\text{gridSize} = (N + blockSize - 1) / \text{blockSize}$

这样一来也会引入越界的风险，所以需要将线程索引的计算方式改为：
```cpp
const int n = blockDim.x * blockIdx.x + threadIdx.x;
if(n >= N) return;
z[n] = x[n] + y[n];
```
# 三、 自定义核函数
## 1. 核函数的要求
- 返回值必须是void,可以使用return 关键字，但是不可返回任何值
- 必须使用限定符 **__global__** 也可以加上一些其他C++的限定符，如：static 限定符的次序任意
- 函数名支持重载
- 不可以向核函数传递非指针变量
- 除非使用统一内存编程机制，否则传给核函数的指针必须只想设备内存
- 核函数不可成为一个类的成员。通常做法是用一个包装函数调用核函数，将包装函数定义为一个类。
- 核函数之间在算力 > 3.5 开始，可以调用其他核函数，也可以调用自己。
- 核函数必须指定线程布局配置。
## 2. 函数执行空间标识符
- **__global__** 称为核函数，一般由主机调用，也可以被核函数包括自己调用。
- **__device__** 称为设备函数，只能被核函数或其他设备函数调用，在设备中执行。
- **__host__**主机段普通的C++函数，在主机中被调用，在主机执行。可省略，存在的意义在于，可以使用**__host__**和 **__device__**修饰同一个函数，表示即可在设备端调用也可以在主机端调用，减少代码冗余。
- 不能使用**__global__** 和 **__device__** 修饰统一个函数
- 不能使用**__host__** 和 **__device__** 修饰统一个函数
- 编译器决定把设备函数当做内联函数或非内联函数，但可以使用 **__noinline__**建议一个设备函数为非内联函数，也可以用 **__forceinline__** 建议一个设备为内联函数。

## 3. 实例代码
### 3.1 code1 返回值
```cpp
double __device__ add1_device(const double x, const double y) {
    return x + y;
}

void __global__ add1(const double *x, const double *y, double *z, const int N) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < N) {
        z[n] = add1_device(x[n], y[n]);
    }
}
```

### 3.2 code2 传指针
```cpp
void __device__ add2_device(const double x, const double y, double *z) {
    *z = *x + *y;
}

void __global__ add1(const double *x, const double *y, double *z, const int N) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < N) {
        add1_device(x[n], y[n], &z[n]);
    }
}
```

### 3.3  code3 传引用
```cpp
void __device__ add3_device(const double x, const double y, double &z) {
    z = x + y;
}

void __global__ add1(const double *x, const double *y, double *z, const int N) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < N) {
        add1_device(x[n], y[n], z[n]);
    }
}
```

# CUDA 错误检查

```cpp
bool checkRuntime(cudaError_t e, const char* call, int line, const char *file){
        if (e != cudaSuccess) {
            INFOE("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d", 
                call, 
                cudaGetErrorString(e), 
                cudaGetErrorName(e), 
                e, file, line
            );
            return false;
        }
        return true;
    }
```

错误信息实例：
```shell
CUDA Runtime error cudaMemcpy(d_x, h_x, M, cudaMemcpyDeviceToDevice) # invalid argument, code = cudaErrorInvalidValue [ 1 ] in file D:\work_dir\experiment\LearningCUDA\src\add.cu:29
```







