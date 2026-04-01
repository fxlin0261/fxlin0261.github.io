---
title: "Reduce"
date: 2026-04-01
summary: "Reduce 相关笔记。"
tags: ["CUDA"]
---

# Reduce

最基本的写法
每个线程先读 1 个元素到 shared memory
然后按 s=1,2,4,8... 做归约
用 tid % (2*s) == 0 决定谁参与
问题
线程闲置多
warp 分化严重
访存模式也不好

__global__ void reduce_v0(float* g_idata, float* g_odata) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

解决线程限制多
__global__ void reduce_v1(float* g_idata, float* g_odata) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + tid;

    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

解决warp 分化严重
不再让 “间隔线程” 工作，改成 “前半部分线程工作，后半部分线程休息”
__global__ void reduce_v2(float* g_idata, float* g_odata) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + tid;

    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

最后一个 warp 展开
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_v3(float* g_idata, float* g_odata) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + tid;

    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(sdata, tid);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

工程版写法
一个 block 处理固定一段
一次 kernel 出一批中间结果
再二次 launch
grid-stride + 寄存器累加 + warp reduce + 少量 shared memory

#define BLOCK_SIZE 256
#define FULL_MASK 0xffffffff

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__global__ void reduce_better(const float* __restrict__ g_idata,
                              float* __restrict__ g_odata,
                              int N) {
    __shared__ float sdata[BLOCK_SIZE / 32]; // 每个 warp 一个槽

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;
    unsigned int stride = blockDim.x * 2 * gridDim.x;

    float sum = 0.0f;

    // grid-stride loop：每个线程处理多个元素
    while (idx < N) {
        sum += g_idata[idx];
        if (idx + blockDim.x < N) {
            sum += g_idata[idx + blockDim.x];
        }
        idx += stride;
    }

    // 先做 warp 内归约
    sum = warp_reduce_sum(sum);

    // 每个 warp 的 lane0 写入 shared memory
    if ((tid & 31) == 0) {
        sdata[tid >> 5] = sum;
    }
    __syncthreads();

    // 只让第一个 warp 归约所有 warp 的结果
    sum = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
    if (tid < 32) {
        sum = warp_reduce_sum(sum);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sum;
    }
}

