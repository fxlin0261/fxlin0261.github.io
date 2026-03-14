---
title: "Matrix Multiplication"
date: 2026-03-12
---

# Matrix Multiplication
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

__global__ void matrix_multiplication_kernel(
    const float *A, const float *B, float *C,
    int M, int N, int K, int tileSize) {
    extern __shared__ float shared[];
    float *tileA = shared;
    float *tileB = shared + tileSize * tileSize;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    // MN * NK
    float sum = 0.0f;
    for (int t = 0; t < (N + tileSize - 1) / tileSize; ++t) {
        const int aCol = t * tileSize + threadIdx.x;
        const int bRow = t * tileSize + threadIdx.y;
        const int localIndex = threadIdx.y * tileSize + threadIdx.x;
        // 一个block所有thread都去搬数据, 搬满一个tile
        tileA[localIndex] = (row < M && aCol < N) ? A[row * N + aCol] : 0.0f;
        tileB[localIndex] = (bRow < N && col < K) ? B[bRow * K + col] : 0.0f;
        // 必须有完整的tile数据才行, 所以block中thread同步
        __syncthreads();
        #pragma unroll
        // 每个thread利用tile中的数据, 计算自己当前小块的单步结果, 并累加
        for (int i = 0; i < tileSize; ++i) {
            sum += tileA[threadIdx.y * tileSize + i] * tileB[i * tileSize + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

void solve(const float *A, const float *B, float *C, int M, int N, int K) {
    constexpr int tileSize = 16;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 blocksPerGrid(
        (K + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    size_t sharedMemoryBytes = 2 * tileSize * tileSize * sizeof(float);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemoryBytes>>>(
        A, B, C, M, N, K, tileSize);
    cudaDeviceSynchronize();
}
```



