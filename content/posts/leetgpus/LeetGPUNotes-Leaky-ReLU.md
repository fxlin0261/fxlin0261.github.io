---
title: "LeetGPUNotes: Leaky ReLU"
date: 2026-03-19
---

# LeetGPUNotes: Leaky ReLU
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
