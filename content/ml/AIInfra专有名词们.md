## 写 kernel 的语言

- CUDA C++：最底层、最原生，控制力最强，但代码也最繁琐
- Triton：Python 风格写 GPU kernel
- TileLang：强调 tile-based 编程，底层建立在 TVM 上
- CuTe DSL：NVIDIA CUTLASS 体系里的 Python DSL
- cuTile：NVIDIA 的 tile-based 并行编程模型及其 Python DSL

## 推理框架

- SGLang：偏高性能服务化推理，比较适合复杂生成控制和长上下文场景
- vLLM：偏通用高吞吐推理引擎，最大特点是内存利用率高、吞吐高
- TensorRT-LLM：偏 NVIDIA 生态下的极致性能优化
- llama.cpp：偏轻量、易部署、硬件适配广，既适合本地也能做服务化
- Ollama：本地启动简单、命令行和 API 友好

## 编译器

- LLVM：通用编译基础设施
- MLIR：LLVM 体系里的多层 IR 框架，适合做领域专用编译器
- TVM：面向机器学习的编译框架。强项是把模型/算子编译到不同硬件
- Triton Compiler：Triton 自带的编译器
- nvcc：NVIDIA 的 CUDA 编译驱动
- CUDA Tile IR：NVIDIA 新的 tile-based IR
- tileiras：对应的 Tile IR 编译器 / optimizing assembler

## NVIDIA CUDA 生态

- 开发环境 / SDK：CUDA Toolkit
- 基础库：cuBLAS、cuFFT、cuSPARSE、cuFFT、cuRAND（成品库，直接调）
- 深度学习库：cuDNN
- 多 GPU 通信库：NCCL
- 高性能模板库 / 内核构建库：CUTLASS（模板抽象库，自己配参数就可以用）
- 编译工具：nvcc
- 分析与调试工具：Nsight Systems、Nsight Compute
- 推理框架 / 推理优化：TensorRT、TensorRT-LLM
