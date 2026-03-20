---
title: "cuda性能测试脚本"
date: 2026-03-20
---

# cuda性能测试脚本
---

下面这份脚本用于对 `torch.scatter_` 做简单的性能分析，同时顺手检查结果是否具备确定性。

主要做了几件事：

- 打开 PyTorch / cuDNN / cuBLAS 的确定性配置
- 构造一组测试数据
- 封装 `scatter_` 操作，方便 profiling
- 使用 `torch.profiler` 采集 CPU / CUDA 时间与内存信息
- 输出 profiling 摘要，并保存完整结果
- 重复执行两次，检查结果是否一致

```python
import torch
import torch.profiler as profiler
import os

# ==============================================
# 1. 开启PyTorch确定性模式（全局）
# ==============================================
# 方式1：通过环境变量（推荐，覆盖所有操作）
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 解决cuBLAS非确定性问题
torch.use_deterministic_algorithms(True)  # 强制所有操作使用确定性算法
torch.backends.cudnn.deterministic = True  # cuDNN确定性
torch.backends.cudnn.benchmark = False     # 关闭benchmark（避免自动选择非确定性算法）

# ==============================================
# 2. 构建测试数据（GPU/CPU可选，推荐GPU看profiling效果）
# ==============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义scatter参数：模拟"按索引分散赋值"场景
# batch_size = 1024
# num_classes = 512
# src = torch.randn(batch_size, num_classes, device=device)  # 源张量
# index = torch.randint(0, num_classes, (batch_size, num_classes), device=device)  # 分散索引
# out = torch.zeros(batch_size, num_classes, device=device)  # 输出张量（初始化为0）

out = torch.zeros((2165666,), device=device, dtype=torch.bfloat16)
src = torch.randn((2165665,), device=device, dtype=torch.bfloat16)
index = torch.randint(low=0, high=1, size=(2165660,), device=device, dtype=torch.int32)

# ==============================================
# 3. 封装scatter操作（便于profiling追踪）
# ==============================================
def scatter_operation(out_tensor, src_tensor, index_tensor, dim=0):
    """
    封装torch.scatter调用
    :param out_tensor: 输出张量
    :param src_tensor: 源张量
    :param index_tensor: 索引张量
    :param dim: 分散维度
    :return: 分散后的张量
    """
    # torch.scatter：将src按index的索引，分散到out的dim维度上
    # 模式：out[i][index[i][j]] = src[i][j]（dim=1时）
    # out=1 index=2 src = 3
    out_tensor.scatter_(
        dim=dim,        # dim 0
        index=index_tensor,
        src=src_tensor
        # reduce="sum"  # 可选：sum/mul，这里用sum模拟累加场景 reduce=none
    )
    return out_tensor

# ==============================================
# 4. 性能分析（Profiling）配置与执行
# ==============================================
# 预热：避免首次执行的初始化开销影响profiling
for _ in range(10):
    _ = scatter_operation(out.clone(), src, index)

# 定义profiler配置：追踪CPU/GPU耗时、算子名称、内存使用
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None
    ],
    record_shapes=True,  # 记录张量形状
    profile_memory=True,  # 记录内存使用
    with_stack=True,  # 记录调用栈（便于定位代码）
    on_trace_ready=profiler.tensorboard_trace_handler("./scatter_profiler_logs")  # 输出TensorBoard日志
) as prof:
    # 执行scatter操作（多次执行放大耗时，便于观察）
    for _ in range(100):
        result = scatter_operation(out.clone(), src, index)
        # 同步GPU（确保CUDA操作完成，避免profiling漏记）
        if torch.cuda.is_available():
            torch.cuda.synchronize()

# ==============================================
# 5. 输出Profiling结果（文本+文件）
# ==============================================
# 打印简要统计信息
print("\n===== Scatter操作Profiling摘要 =====")
print(prof.key_averages().table(
    sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
    row_limit=10  # 仅显示前10个耗时最高的算子
))

# 保存详细profiling结果到文件
with open("scatter_profiling_result.txt", "w") as f:
    f.write(prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=None  # 保存所有算子
    ))
print("\n详细Profiling结果已保存到: scatter_profiling_result.txt")

# ==============================================
# 6. 验证确定性（多次执行结果一致性）
# ==============================================
print("\n===== 验证确定性（多次执行结果是否一致） =====")
# 第一次执行
result1 = scatter_operation(out.clone(), src, index)
# 第二次执行（相同输入）
result2 = scatter_operation(out.clone(), src, index)

# 检查结果是否完全一致
is_deterministic = torch.allclose(result1, result2, atol=1e-8)
print(f"多次执行结果是否一致: {is_deterministic}")
if not is_deterministic:
    print("警告：结果非确定性！请检查确定性开关配置")
else:
    print("确定性验证通过！")

# 输出关键信息
print(f"\nScatter操作输出张量形状: {result1.shape}")
print(f"输出张量均值: {result1.mean().item():.4f}")
```

## 说明

### 1. 确定性配置
这部分主要是尽量消除框架层和底层库带来的非确定性：

- `CUBLAS_WORKSPACE_CONFIG`
- `torch.use_deterministic_algorithms(True)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

如果是在做性能测试或 correctness 对比，建议先把这些配置理顺，否则多次运行的结果和耗时都可能不稳定。

### 2. 预热很重要
第一次运行通常会包含：

- CUDA Context 初始化
- Kernel 首次加载
- 内存分配等额外开销

所以 profiling 前先预热几轮，否则统计结果会偏大。

### 3. profiling 关注什么
这个脚本里重点看：

- `cuda_time_total`
- 算子调用次数
- 张量形状
- 内存使用

如果要进一步分析，可以配合 TensorBoard 打开 `scatter_profiler_logs` 看 trace。

### 4. 确定性检查
最后用相同输入执行两次，检查输出是否一致。

这个步骤比较适合在下面这些场景使用：

- 调试含原子操作的算子
- 排查某些算子是否存在非确定性行为
- 做 benchmark 前确认实验条件一致

## 可继续改进的点

这份脚本现在更偏“快速验证 + 简单 profiling”，后面还可以继续补：

- 加入更规范的 warmup / iteration 配置
- 增加多组 shape 的批量测试
- 统计平均耗时、P50、P95
- 单独测试 `scatter_` 不同 dtype / 不同索引分布下的表现
- 把结果导出成 csv，方便横向对比
