---
title: "PyTorch CUDA Profiling 脚本"
date: 2026-03-21
---

# PyTorch CUDA Profiling 脚本

下面是一份使用 `torch.profiler` 对 `torch.scatter_` 进行性能分析的测试脚本：

```python
from pathlib import Path
import torch
import torch.profiler as profiler

if __name__ == "__main__":
    PROFILE_DIR = "torch_cuda_profiler_logs"
    RESULT_FILE = "torch_cuda_profiling_result.txt"

    # ==== 构造输入 ====
    out = torch.zeros(2165666, device='cpu', dtype=torch.bfloat16)
    src = torch.randn(2165660, device='cpu', dtype=torch.bfloat16)
    index = torch.randint(
        low=0,
        high=2165666,
        size=(2165660,),
        device='cpu',
        dtype=torch.int64,
    )

    # ==== 预热 ====
    WARMUP = 10
    for _ in range(WARMUP):
        warmup_result = out.clone()
        warmup_result.scatter_(dim=0, index=index, src=src)
        torch.cuda.synchronize()

    # ==== 执行 profile ====
    ITERATIONS = 10
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=profiler.tensorboard_trace_handler(PROFILE_DIR),
    ) as prof:
        for _ in range(ITERATIONS):
            result = out.clone()
            result.scatter_(dim=0, index=index, src=src)
            torch.cuda.synchronize()

    # ==== 输出结果 ====
    print("\n===== Profiling Summary =====")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=None)
    Path(RESULT_FILE).write_text(table)
    print(f"\nDetailed profiling results saved to: {RESULT_FILE}")
    print(f"TensorBoard trace directory: {PROFILE_DIR}")
```
