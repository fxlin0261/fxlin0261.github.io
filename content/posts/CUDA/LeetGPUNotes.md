---
title: "LeetGPUNotes"
date: 2026-03-20
---

# LeetGPUNotes
---


## 1D-Convolution

---
最基本写法
```cuda
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx > input_size - kernel_size) return;
    float temp = 0;
    #pragma unroll
    for (int i = 0; i < kernel_size; ++i) {
        temp += input[tidx + i] * kernel[i];
    }
    output[tidx] = temp;
}

extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
```

向量化
```cuda
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx > input_size - kernel_size) return;
    float temp = 0.0f;
    #pragma unroll
    for (int i = 0; i < kernel_size / 4; i++) {
        int p = i * 4;
        float4 tempI = make_float4(input[tidx + p], input[tidx + p + 1], input[tidx + p + 2], input[tidx + p + 3]);
        float4 tempK = make_float4(kernel[p], kernel[p + 1], kernel[p + 2], kernel[p + 3]);
        temp += tempI.x * tempK.x + tempI.y * tempK.y + tempI.z * tempK.z + tempI.w * tempK.w;
    }
    // kernel_size & ~3 = (kernel_size / 4) * 4
    for (int i = kernel_size & ~3; i < kernel_size; i++) {
        float tempI = input[tidx + i];
        float tempK = kernel[i];
        temp += tempI * tempK;
    }
    output[tidx] = temp;
}

extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    if (kernel_size <= 0 || kernel_size > input_size) return;
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
```


## Color-Inversion

---
基础写法
```cuda
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char *image, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= width * height)
        return;
    image[idx * 4] = 255 - image[idx * 4];
    image[idx * 4 + 1] = 255 - image[idx * 4 + 1];
    image[idx * 4 + 2] = 255 - image[idx * 4 + 2];
}

extern "C" void solve(unsigned char *image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
```

位运算
```cuda
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= width * height) return;
    // uint8 255 - x = x ^ 0xFF
    image[idx] ^= 0x00FFFFFFu;
}

extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
}
```


## Leaky-ReLU

---
基础写法
```cuda
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(float * x, int n, const float alpha) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        float v = x[tid];
        x[tid] = v > 0.0f ? v : alpha * v;
    }
}

// x is a device pointer
extern "C" void solve(float* x, int n) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    const float alpha = 0.01f;
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(x, n, alpha);
    cudaDeviceSynchronize();
}
```

向量化
```cuda
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(float * x, int n, const float alpha) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = tid * 4;
    if (idx + 3 < n) {
        float4 v = reinterpret_cast<float4*>(x)[tid];
        reinterpret_cast<float4*>(x)[tid] = make_float4(
            v.x > 0 ? v.x : alpha * v.x,
            v.y > 0 ? v.y : alpha * v.y,
            v.z > 0 ? v.z : alpha * v.z,
            v.w > 0 ? v.w : alpha * v.w
        );
    } else if (idx < n) {
        for (int i = idx; i < n; ++i) {
            x[i] = x[i] > 0 ? x[i] : alpha * x[i];
        }
    }
}

// x is a device pointer
extern "C" void solve(float* x, int n) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = ((n + 3) / 4 + threadsPerBlock - 1) / threadsPerBlock;
    const float alpha = 0.01f;
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(x, n, alpha);
    cudaDeviceSynchronize();
}
```


## Matrix-Addition

---
基础写法
```cuda
#include <cuda_runtime.h>
 
__global__ void matrix_add(const float* __restrict A, const float* __restrict B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N * N) {
        C[tid] = A[tid] + B[tid];
    }
}
 
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;
    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

向量化
```cuda
#include <cuda_runtime.h>

__global__ void matrix_addition(const float *__restrict A, const float *__restrict B, float *C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * 4 + 3 < N * N) {
        const float4 a4 = reinterpret_cast<const float4*>(A)[tid];
        const float4 b4 = reinterpret_cast<const float4*>(B)[tid];
        reinterpret_cast<float4*>(C)[tid] = make_float4(a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w);
    } else if (tid * 4 < N * N) {
        #pragma unroll
        for (int i = tid * 4; i < N * N; ++i) {
            C[i] = A[i] + B[i];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int total_elements = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (((total_elements + 3) / 4) + threadsPerBlock - 1) / threadsPerBlock;
    matrix_addition<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```


## Matrix-Copy

---
基础写法
```cuda
#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (tidx < total) {
        B[tidx] = A[tidx];
    }
}

extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}
```

官方写法
```cuda
#include <cuda_runtime.h>

extern "C" void solve(const float* A, float* B, const int N) {
    cudaMemcpy(B, A, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
}
```


## Matrix-Multiplication

---
最基本写法  
问题：GM多些次数过多
```cuda
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    // MN * NK = MK
    if (tidy < M && tidx < K) {
        float sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += A[tidy * N + i] * B[i * K + tidx];
        }
        C[tidy * K + tidx] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

Tile块  
```cuda
#include <cuda_runtime.h>

template <int TILE_SIZE>
__global__ void matrix_multiplication_kernel(const float *__restrict__ A,
    const float *__restrict__ B, float *__restrict__ C,
    int M, int N, int K) {
    extern __shared__ float shared[];
    float *tileA = shared;
    float *tileB = shared + TILE_SIZE * TILE_SIZE;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    // MN * NK
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        const int aCol = t * TILE_SIZE + threadIdx.x;
        const int bRow = t * TILE_SIZE + threadIdx.y;
        const int localIndex = threadIdx.y * TILE_SIZE + threadIdx.x;
        // 一个block所有thread都去搬数据, 搬满一个tile, 自己负责自己的小格
        tileA[localIndex] = (row < M && aCol < N) ? A[row * N + aCol] : 0.0f;
        tileB[localIndex] = (col < K && bRow < N) ? B[bRow * K + col] : 0.0f;
        // 必须有完整的tile数据才行, 所以block中thread同步
        __syncthreads();
        #pragma unroll
        // 每个thread利用tile中的数据, 计算自己当前小块的单步结果, 并累加
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[threadIdx.y * TILE_SIZE + i] *
                   tileB[i * TILE_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    constexpr int kTileSize = 16;
    dim3 threadsPerBlock(kTileSize, kTileSize);
    dim3 blocksPerGrid(
        (K + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    const size_t sharedMemoryBytes =
        2 * kTileSize * kTileSize * sizeof(float);

    matrix_multiplication_kernel<kTileSize>
        <<<blocksPerGrid, threadsPerBlock, sharedMemoryBytes>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```


## Matrix-Transpose

---
数学公式：不只是 block 之间的位置互换，block 内部的数据访问方向也发生了转置。

输入矩阵：

```text
A  B
C  D
```

输出矩阵：

```text
A^T  C^T
B^T  D^T
```

基础版
```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    if (tidy < rows && tidx < cols) {
        output[tidx * rows + tidy] = input[tidy * cols + tidx];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
```

共享内存 + 访存合并
```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tidy = blockIdx.y * blockDim.y + ty;
    int tidx = blockIdx.x * blockDim.x + tx;

    if (tidy < rows && tidx < cols) {
        tile[ty][tx] = input[tidy * cols + tidx];
    }

    __syncthreads();

    int out_tidy = blockIdx.x * blockDim.x + ty;
    int out_tidx = blockIdx.y * blockDim.y + tx;

    if (out_tidy < cols && out_tidx < rows) {
        output[out_tidy * rows + out_tidx] = tile[tx][ty];
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
```

共享内存 + 访存合并 + Bank Conflict 处理
```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    // +1 padding avoids bank conflicts when reading the tile in transposed order.
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tidy = blockIdx.y * blockDim.y + ty;
    int tidx = blockIdx.x * blockDim.x + tx;

    if (tidy < rows && tidx < cols) {
        tile[ty][tx] = input[tidy * cols + tidx];
    }

    __syncthreads();

    int out_tidy = blockIdx.x * blockDim.x + ty;
    int out_tidx = blockIdx.y * blockDim.y + tx;

    if (out_tidy < cols && out_tidx < rows) {
        output[out_tidy * rows + out_tidx] = tile[tx][ty];
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
```


## Rainbow-Table

---
基础版
```cuda
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned int fnv1a_hash(unsigned int x) {
    // 如果题面已经给了 fnv1a_hash，直接删掉这个实现，保留调用即可
    const unsigned int FNV_OFFSET_BASIS = 2166136261u;
    const unsigned int FNV_PRIME = 16777619u;
    unsigned int h = FNV_OFFSET_BASIS;
    h ^= (x & 0xFFu);
    h *= FNV_PRIME;

    h ^= ((x >> 8) & 0xFFu);
    h *= FNV_PRIME;

    h ^= ((x >> 16) & 0xFFu);
    h *= FNV_PRIME;

    h ^= ((x >> 24) & 0xFFu);
    h *= FNV_PRIME;

    return h;
}

__global__ void rainbow_table_kernel(
    const int *__restrict__ input,
    unsigned int *__restrict__ output,
    int N,
    int R) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N)
        return;
    unsigned int value = static_cast<unsigned int>(input[gid]);
    while (R--) {
        value = fnv1a_hash(value);
    }
    output[gid] = value;
}

extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    constexpr int THREADS = 256;
    int blocks = (N + THREADS - 1) / THREADS;

    rainbow_table_kernel<<<blocks, THREADS>>>(input, output, N, R);
    cudaDeviceSynchronize();
}
```


## ReLU

---
基础写法
```cuda
#include <cuda_runtime.h>

// ReLU(x) = max(0, x)
__global__ void relu_kernel(float* x, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        x[tid] = fmaxf(x[tid], 0.0f);
    }
}

// x is a device pointer
extern "C" void solve(float* x, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(x, n);
    cudaDeviceSynchronize();
}
```

向量化加载
```cuda
#include <cuda_runtime.h>

// ReLU(x) = max(0, x)
__global__ void relu_kernel(float* x, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;
    if (idx + 3 < n) {
        float4* x4 = reinterpret_cast<float4*>(x);
        float4 v = x4[tid];
        v.x = fmaxf(v.x, 0.0f);
        v.y = fmaxf(v.y, 0.0f);
        v.z = fmaxf(v.z, 0.0f);
        v.w = fmaxf(v.w, 0.0f);
        x4[tid] = v;
    } else if (idx < n) {
        for (int i = idx; i < n; ++i) {
            x[i] = x[i] > 0.0f ? x[i] : 0.0f;
        }
    }
}

// x is a device pointer
extern "C" void solve(float* x, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (((n + 3) / 4) + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(x, n);
    cudaDeviceSynchronize();
}
```


## Reduction

---
共享内存 + __shfl_down
```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void reduction(const float *input, float *output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sharedMem[BLOCK_SIZE];

    // 拷到共享内存（越界用 0），不能提前 return，下面有 __syncthreads
    float v = (idx < N) ? input[idx] : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads();

    // 树形二分归约。还剩一个 warp（32 个线程）时停下。
    for (int stride = blockDim.x / 2; stride >= warpSize; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // warp 内归约
    if (threadIdx.x < warpSize) {
        float sum = sharedMem[threadIdx.x];
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0x0000ffff, sum, 8);
        sum += __shfl_down_sync(0x000000ff, sum, 4);
        sum += __shfl_down_sync(0x0000000f, sum, 2);
        sum += __shfl_down_sync(0x00000003, sum, 1);
        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
    reduction<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(input, output, N);
}
```

基础写法（会有确定性问题，不要用！！！）
```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void reduction(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(output, input[idx]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    reduction<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(input, output, N);
}
```


## Reverse-Array

---
基础写法
```cuda
#include <cuda_runtime.h>

__global__ void reverse_array_inplace(float* a, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n / 2) {
        auto temp = a[tid];
        a[tid] = a[n - tid - 1];
        a[n - tid - 1] = temp;
    }
}

extern "C" void solve(float* a, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n / 2 + threadsPerBlock - 1) / threadsPerBlock;
    reverse_array_inplace<<<blocksPerGrid, threadsPerBlock>>>(a, n);
    cudaDeviceSynchronize();
}
```


## Vector-Addition

---
最基本写法
```cuda
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```
向量化加载
```cuda
#include <cuda_runtime.h>

__global__ void vector_add_vec4(const float* A, const float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int vecN = N / 4;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    if (tid < vecN) {
        float4 a = A4[tid];
        float4 b = B4[tid];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C4[tid] = c;
    }

    int idx = vecN * 4 + tid;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int vecN = (N + 3) / 4;
    int blocksPerGrid = (vecN + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_vec4<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```
