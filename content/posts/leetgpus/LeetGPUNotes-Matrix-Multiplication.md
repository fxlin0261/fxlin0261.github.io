---
title: "LeetGPUNotes: Matrix Multiplication"
date: 2026-03-13
---

# LeetGPUNotes: Matrix Multiplication
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
