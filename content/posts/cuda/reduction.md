基础
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


共享内存+__shfl_down
#include <cuda_runtime.h>
#define BLOCK_SIZE 1024
__global__ void reduction(const float *input, float *output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sharedMem[BLOCK_SIZE];
    // 拷到共享内存（越界用0）, 不能提前return, 下面有 __syncthreads
    float v = (idx < N) ? input[idx] : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads();

    // 树形二分归纳 还剩一个 warp（32 个线程）时就停下
    // 一个 warp 内线程天然同步，执行效率高 改用 warp-level 原语来做最后归约
    for (int stride = blockDim.x / 2; stride >= warpSize; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // wrap内规约
    if (threadIdx.x < warpSize) {
        // 每个线程先取自己对应的部分和
        float sum = sharedMem[threadIdx.x];
        // warp shuffle 指令，允许一个线程直接读取同一 warp 内另一个线程的寄存器值
        // lane 0 加上 lane 16 lane 1 加上 lane 17
        // mask全写0xffffffff都行
        // shfl_down 从高lane向低 shfl_up相反 shfl_xor 配对交换
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        // lane 0 读取 lane 8 的当前值 lane 1 读取 lane 9 的当前值
        sum += __shfl_down_sync(0x0000ffff, sum, 8);
        sum += __shfl_down_sync(0x000000ff, sum, 4);
        sum += __shfl_down_sync(0x0000000f, sum, 2);
        // lane 0 读取 lane 1 的和 lane 0 最终得到全部 32 个数之和
        sum += __shfl_down_sync(0x00000003, sum, 1);
        if (threadIdx.x == 0)
        {
            atomicAdd(output, sum);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
    reduction<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(input, output, N);
}



