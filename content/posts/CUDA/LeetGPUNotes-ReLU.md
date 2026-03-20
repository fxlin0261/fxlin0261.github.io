---
title: "LeetGPUNotes: ReLU"
date: 2026-03-19
---

# LeetGPUNotes: ReLU
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
