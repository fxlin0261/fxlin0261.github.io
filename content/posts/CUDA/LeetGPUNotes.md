---
title: "LeetGPUNotes"
date: 2026-03-20
tags: ["CUDA"]
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

## softmax

---
# 最基础写法
```cuda
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Row-wise softmax
// output[row, col] = exp(input[row, col] - row_max) / sum_j exp(input[row, j] - row_max)
__global__ void softmax_kernel_basic(const float* input,
                                     float* output,
                                     int num_rows,
                                     int num_cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= num_rows || col >= num_cols) return;

    // Step 1: find max value of this row
    float row_max = -FLT_MAX;
    for (int j = 0; j < num_cols; ++j) {
        row_max = fmaxf(row_max, input[row * num_cols + j]);
    }

    // Step 2: compute denominator
    float row_sum = 0.0f;
    for (int j = 0; j < num_cols; ++j) {
        row_sum += expf(input[row * num_cols + j] - row_max);
    }

    // Step 3: normalize
    output[row * num_cols + col] =
        expf(input[row * num_cols + col] - row_max) / row_sum;
}

extern "C" void solve(const float* input, float* output, int num_rows, int num_cols) {
    dim3 block_dim(num_cols);
    dim3 grid_dim(num_rows);
    softmax_kernel_basic<<<grid_dim, block_dim>>>(input, output, num_rows, num_cols);
    cudaDeviceSynchronize();
}


```

```cuda
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#define FULL_MASK 0xffffffffu

__inline__ __device__ float warp_reduce_max(float value) {
    value = fmaxf(value, __shfl_down_sync(FULL_MASK, value, 16));
    value = fmaxf(value, __shfl_down_sync(FULL_MASK, value, 8));
    value = fmaxf(value, __shfl_down_sync(FULL_MASK, value, 4));
    value = fmaxf(value, __shfl_down_sync(FULL_MASK, value, 2));
    value = fmaxf(value, __shfl_down_sync(FULL_MASK, value, 1));
    return value;
}

__inline__ __device__ float warp_reduce_sum(float value) {
    // 从同一个 warp 里“更高编号 lane”的线程那里取一个寄存器值过来
    // warp 里的每个 thread 都会执行这几行代码 lane 0 的结果一定是整个 warp 的总和
    // 为什么不会有thread2先执行在thread0 因为wrap中是同步的 thread2和thread0同步执行
    // 可以理解成两步同时发生：tmp = lane(i+16) 在这一行开始前的 value value = 自己原来的 value + tmp
    value += __shfl_down_sync(FULL_MASK, value, 16);
    value += __shfl_down_sync(FULL_MASK, value, 8);
    value += __shfl_down_sync(FULL_MASK, value, 4);
    value += __shfl_down_sync(FULL_MASK, value, 2);
    value += __shfl_down_sync(FULL_MASK, value, 1);
    return value;
}

__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int num_rows,
                               int num_cols) {
    // 一个block一行 一行 THREADS_PER_BLOCK 个threads
    const int row_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int lane_idx = thread_idx & 31;
    const int warp_idx = thread_idx >> 5;
    const int num_warps = THREADS_PER_BLOCK / 32;

    if (row_idx >= num_rows) return;

    __shared__ float warp_max_buffer[THREADS_PER_BLOCK / 32];
    __shared__ float warp_sum_buffer[THREADS_PER_BLOCK / 32];

    // Step 1: compute row maximum
    float thread_max = -FLT_MAX;
    // 寻找当前thread的最大值
    for (int col_idx = thread_idx; col_idx < num_cols; col_idx += THREADS_PER_BLOCK) {
        thread_max = fmaxf(thread_max, input[row_idx * num_cols + col_idx]);
    }
    // 针对当前wrap的最大值
    thread_max = warp_reduce_max(thread_max);
    if (lane_idx == 0) {
        warp_max_buffer[warp_idx] = thread_max;
    }
    __syncthreads();

    float row_max = -FLT_MAX;
    // 只有第 0 个 warp 的线程参与这一步归约。
    if (warp_idx == 0) {
        // 第 0 个 warp 的前 num_warps 个线程，各自读取一个 warp 的局部最大值
        row_max = (thread_idx < num_warps) ? warp_max_buffer[lane_idx] : -FLT_MAX;
        //  在第 0 个 warp 内继续做一次 warp 级最大值归约
        row_max = warp_reduce_max(row_max);
        // 让 block 中的第 0 号线程把最终结果写回共享内存 warp_max_buffer[0]
        if (thread_idx == 0) {
            warp_max_buffer[0] = row_max;
        }
    }
    __syncthreads();
    // 每个thread读取自己行的最大值
    row_max = warp_max_buffer[0];

    // Step 2: compute denominator
    // 每个线程先初始化自己的局部和。
    float thread_sum = 0.0f;
    // 当前线程以 blockDim.x 为步长，处理这一行中属于自己的若干列。
    for (int col_idx = thread_idx; col_idx < num_cols; col_idx += THREADS_PER_BLOCK) {
        thread_sum += expf(input[row_idx * num_cols + col_idx] - row_max);
    }
    // 先在每个 warp 内部做一次求和归约。
    thread_sum = warp_reduce_sum(thread_sum);
    //  每个 warp 只让 lane 0 把本 warp 的局部和写入共享内存
    if (lane_idx == 0) {
        warp_sum_buffer[warp_idx] = thread_sum;
    }
    __syncthreads();
    // 先把当前行的总和初始化为 0
    float row_sum = 0.0f;
    if (warp_idx == 0) {
        // 只有第 0 个 warp 参与这一步
        // 前面已经算出了“每个 warp 的局部和”，并把它们写到了 warp_sum_buffer[warp_idx] 中
        row_sum = (thread_idx < num_warps) ? warp_sum_buffer[lane_idx] : 0.0f;
        // 在第 0 个 warp 内部继续做一次 warp 级求和归约
        row_sum = warp_reduce_sum(row_sum);
        if (thread_idx == 0) {
            // 让 block 中的第 0 号线程把最终结果写回共享内存 warp_sum_buffer[0]
            warp_sum_buffer[0] = row_sum;
        }
    }
    __syncthreads();
    // 所有线程都从共享内存的 warp_sum_buffer[0] 中读取同一个最终结果
    row_sum = warp_sum_buffer[0];

    // Step 3: normalize
    for (int col_idx = thread_idx; col_idx < num_cols; col_idx += THREADS_PER_BLOCK) {
        // 每个线程继续按“跨步访问”的方式处理自己负责的若干
        // exp(input - row_max) / row_sum
        output[row_idx * num_cols + col_idx] =
            expf(input[row_idx * num_cols + col_idx] - row_max) / row_sum;
    }
}

extern "C" void solve(const float* input, float* output, int num_rows, int num_cols) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(num_rows);
    softmax_kernel<<<grid_dim, block_dim>>>(input, output, num_rows, num_cols);
    cudaDeviceSynchronize();
}
```

## Reduction

---
```cuda
#include <cuda_runtime.h>

template<int BS>
__global__ void reduction_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    __shared__ float smem[BS];
    // 一个block中一次处理 2 * BS个元素 总共开了sm * 8个block
    const int blockStride = gridDim.x * (BS * 2);
    int tidx = blockIdx.x * (BS * 2) + threadIdx.x;
    int threadId = threadIdx.x;
    
    float threadSum = 0;
    for (int i = tidx; i < N; i += blockStride) {
        threadSum += input[i];
        if (i + BS < N) threadSum += input[i + BS];
    }
    smem[threadId] = threadSum;
    __syncthreads();
    #pragma unroll
    for (int stride = BS / 2; stride > 32; stride /= 2) {
        if (threadId < stride) {
            smem[threadId] += smem[threadId + stride];
        }
        __syncthreads();
    }
    if (threadId < 32) {
        float warpSum = smem[threadId];
        unsigned mask = 0xffffffffu;
        // 让当前线程去“拿同一个 warp 里、编号比自己大 offset 的那个线程的 x
        warpSum += __shfl_down_sync(mask, warpSum, 16);
        warpSum += __shfl_down_sync(mask, warpSum, 8);
        warpSum += __shfl_down_sync(mask, warpSum, 4);
        warpSum += __shfl_down_sync(mask, warpSum, 2);
        warpSum += __shfl_down_sync(mask, warpSum, 1);
        if (threadId == 0) atomicAdd(output, warpSum);
    }
}

extern "C" void solve(const float *input, float *output, int N) {
    int dev = 0, sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);

    constexpr int BS = 16;
    int naturalBlocks = (N + (BS * 2) - 1) / (BS * 2);
    if (naturalBlocks > sm * 8) naturalBlocks = sm * 8; 
    dim3 blocksPerGrid(naturalBlocks);
    dim3 threadsPerBlock(BS);
    cudaMemset(output, 0, sizeof(float));
    reduction_kernel<BS><<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}

```
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
