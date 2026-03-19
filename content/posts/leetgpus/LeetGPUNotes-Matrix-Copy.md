---
title: "LeetGPUNotes: Matrix Copy"
date: 2026-03-19
---

# LeetGPUNotes: Matrix Copy
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
