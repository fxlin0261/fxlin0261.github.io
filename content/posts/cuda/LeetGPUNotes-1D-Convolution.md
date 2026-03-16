---
title: "LeetGPUNotes: 1D Convolution"
date: 2026-03-16
---

# LeetGPUNotes: 1D Convolution
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