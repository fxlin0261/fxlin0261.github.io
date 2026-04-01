---
title: "Reduce"
date: 2026-04-01
summary: "CUDA Reduce 相关笔记，按常见优化版本整理。"
tags: ["CUDA"]
---

# Reduce

下面按常见的优化顺序简单记一下。默认 `BLOCK_SIZE` 是 2 的幂，每个 block 先算出一个部分和，再把结果写到 `g_odata[blockIdx.x]`。

## v0：最基础的写法

思路：
- 每个线程先读 1 个元素到 shared memory。
- 然后按 `s = 1, 2, 4, 8, ...` 做归约。
- 用 `tid % (2 * s) == 0` 决定谁参与。

问题：
- 线程闲置多。
- warp 分化严重。
- 访存模式也不够好。

```cpp
__global__ void reduce_v0(float* g_idata, float* g_odata) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // s = 1 时，参与的是 tid = 0, 2, 4, 6, ... 
        //  tid = 0 sdata[0] += sdata[1]
        //  tid = 2 sdata[2] += sdata[3]
        //  tid = 4 sdata[4] += sdata[5]
        // s = 2 时，参与的是 tid = 0, 4, 8, 12, ...
        //  tid = 0 sdata[0] += sdata[2]
        //  tid = 4 sdata[4] += sdata[6]
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}
```

## v1：每个线程先处理 2 个元素

目的：
- 让每个线程一开始就多做一点事。
- 减少一部分线程空转。

这一版只是把加载阶段改了，归约阶段还是旧写法。

```cpp
__global__ void reduce_v1(float* g_idata, float* g_odata) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    // 一个 block 不再只覆盖 blockDim.x 个元素, 而是覆盖 blockDim.x * 2 个元素
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
```

## v2：改成顺序归约

核心变化：
- 不再让“隔一个线程工作一次”。
- 改成“前半部分线程工作，后半部分线程休息”。

这样可以明显减少 warp 分化。

```cpp
__global__ void reduce_v2(float* g_idata, float* g_odata) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + tid;

    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();
    // 先把 block 分成前后两半, 每轮把 s 除以 2，继续缩小范围
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        // 只让前半部分线程干活
        if (tid < s) {
            // 半部分线程把“自己”和“后半部分对应位置”的值加起来
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // 最后 sdata[0] 就是这个 block 的总和
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}
```

## v3：展开最后一个 warp

思路：
- 当 block 内只剩最后一个 warp 时，不再每轮都做 `__syncthreads()`。
- 直接把最后几步手动展开，减少同步开销。

```cpp
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
    // 只做到 s > 32 = 归约到只剩 64 个值时就停下
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // 最后一个 warp 的归约不再反复 __syncthreads() 减少了6个 32 16 8 4 2 1
    if (tid < 32) {
        warpReduce(sdata, tid);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}
```

## 工程里更常见的写法

通常会这样做：
- 一个 block 处理一段数据。
- 一次 kernel 先产出一批中间结果。
- 如果还有多个 block 的结果，就再 launch 一轮。
- 常见组合是 `grid-stride loop + 寄存器累加 + warp reduce + 少量 shared memory`。

```cpp
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #define FULL_MASK 0xffffffff
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

    // 1. 每个线程都按“两元素一组”去处理数据
    while (idx < N) {
        sum += g_idata[idx];
        if (idx + blockDim.x < N) {
            sum += g_idata[idx + blockDim.x];
        }
        idx += stride;
    }
    // 2. 同一个 warp 的 32 个线程，把各自的 sum 合成一个值
    sum = warp_reduce_sum(sum);
    // 判断这个线程是不是 warp 的第 0 号线程
    if ((tid & 31) == 0) {
        // 第几个 warp
        sdata[tid >> 5] = sum;
    }
    __syncthreads();
    // 3. 第一个 warp 负责把所有 warp 的结果再归约一次
    // 比如 BLOCK_SIZE=256 时，总共有 8 个 warp，那么只有 tid=0..7 会从 sdata 里拿到有效值，其余线程拿 0.0f
    sum = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
    if (tid < 32) {
        sum = warp_reduce_sum(sum);
    }

    if (tid == 0) {
        // 当前 block 的部分和写入全局最终和
        g_odata[blockIdx.x] = sum;
    }
}
```

对应的 kernel launch ：

```cpp
while (n > 1) {
    #define BLOCK_SIZE 256 // 不能大于32x32

    int blocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    reduce_better<<<blocks, BLOCK_SIZE>>>(src, dst, n);
    n = blocks;
    std::swap(src, dst);
}
```

循环结束后，结果就在 `src[0]`。
