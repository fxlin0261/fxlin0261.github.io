---
title: "LeetGPUNotes: Reduction"
date: 2026-03-19
---

# LeetGPUNotes: Reduction
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
