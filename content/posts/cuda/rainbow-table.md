基础版
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




