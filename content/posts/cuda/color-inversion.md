基础写法

#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char *image, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= width * height)
        return;
    image[idx * 4] = 255 - image[idx * 4];
    image[idx * 4 + 1] = 255 - image[idx * 4 + 1];
    image[idx * 4 + 2] = 255 - image[idx * 4 + 2];
}
extern "C" void solve(unsigned char *image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}

位运算
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= width * height) return;
    // uint8 255 - x = x ^ 0XFF
    image[idx] ^= 0x00FFFFFFu;
}
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
}
