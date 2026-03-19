基础写法
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

# 向量化
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
    int blocksPerGrid = ((total_elements + 4 - 1) / 4 * 4 / 4 + threadsPerBlock - 1) / threadsPerBlock;
    matrix_addition<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}




