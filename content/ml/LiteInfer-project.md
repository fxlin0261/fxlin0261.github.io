## 推理流程

`string -> tokenid[T] -> embeddeding[T, d_model] -> for in embeddeding = vector -> infer -> logits[1, vocab] -> sampler[1] -> tokenid -> word`

两个阶段：

- Prefill: 下一次喂入的 token 是 embeddeding 里取的
- Decode: 上一次生成的 token 作为下一次的输入

## 算子之间是怎么做到数据传递的

- 初始化的时候有些权重 mmap 映射到内存
- 用 Tensor 封装数据，kernel 直接从输入 Tensor 的底层 Buffer 读数据、往输出 Tensor 的底层 Buffer 写数据，模型层面再通过 runtime_tensors_ 统一管理这些中间张量和 KV cache，中间通过指针交换

## 改进

推理层面：

- prefill 阶段，是一个一个 tokenid 喂入的，可以改成一次性喂入，batched prefill
- kvcache 依然是固定大小，极大地限制了上下文，更先进的 kvcache 管理方法，涉及到模型架构

## 算子

- Add
- Embedding
- MatMul
- SwiGLU
- Argmax
- RMSNorm
- RoPE
- Softmax
- MHA
