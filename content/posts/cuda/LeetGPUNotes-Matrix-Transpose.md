---
title: "LeetGPUNotes: Matrix Transpose"
date: 2026-03-16
---

# LeetGPUNotes: Matrix Transpose
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