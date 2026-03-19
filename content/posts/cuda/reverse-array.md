基础（这个只有基础的）
#include <cuda_runtime.h>
__global__ void reverse_array_inplace(float* a, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n / 2) {
        auto temp = a[tid];
        a[tid] = a[n - tid - 1];
        a[n - tid - 1] = temp;
    }
}
extern "C" void solve(float* a, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n / 2 + threadsPerBlock - 1) / threadsPerBlock;
    reverse_array_inplace<<<blocksPerGrid, threadsPerBlock>>>(a, n);
    cudaDeviceSynchronize();
}



