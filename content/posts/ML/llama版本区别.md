---
title: "llama版本区别"
date: 2026-03-26
summary: "关于llama版本区别的笔记。"
tags: ["ML"]
---
# LLaMA 版本区别

## 1. 总体结论

LLaMA 系列整体仍然属于 Transformer 架构，但在不同代际中持续优化了注意力机制、上下文长度、词表规模和训练配置。  
如果只抓主线，可以概括为：

- LLaMA 1 打基础
- LLaMA 2 在训练数据与上下文长度上增强
- LLaMA 3 在注意力机制、词表和长上下文支持上进一步升级

## 2. LLaMA 系列的通用结构

LLaMA 系列的核心结构特征包括：

- Pre-normalization（前置归一化）
- RoPE（Rotary Position Embedding，旋转位置编码）
- SwiGLU 激活函数

这些设计共同构成了 LLaMA 系列较稳定的“基础骨架”。

## 3. LLaMA 1 与 LLaMA 2 的 7B 结构对比

对于 7B 版本，LLaMA 1 和 LLaMA 2 的主体结构基本一致，关键参数几乎相同：

| 参数 | LLaMA 1 (7B) | LLaMA 2 (7B) |
| :--- | :--- | :--- |
| 层数（Layers） | 32 | 32 |
| 隐藏层维度（Hidden Size） | 4096 | 4096 |
| 注意力头数（Attention Heads） | 32 | 32 |
| 词表大小（Vocab Size） | 32,000 | 32,000 |

这说明两者在 7B 规格上，模型“骨架”变化不大。

### 主要差异

虽然结构接近，但 LLaMA 2 相比 LLaMA 1 仍有几项重要升级：

| 维度 | LLaMA 1 | LLaMA 2 |
| :--- | :--- | :--- |
| 上下文长度 | 2048 tokens | 4096 tokens |
| 训练数据 | 约 1T tokens | 约 2T tokens，且质量过滤更严格 |
| 注意力机制 | 全部使用 MHA | 7B、13B 仍为 MHA；34B、70B 引入 GQA |

### 说明

- MHA：Multi-Head Attention，多头注意力。
- GQA：Grouped-Query Attention，分组查询注意力，主要用于提升大模型推理效率。
- 在 LLaMA 2 中，GQA 并不是全系列标配，而是从大模型版本开始引入。

![](/posts/ml/assets/2026-03-15-15-02-49.png)

## 4. LLaMA 3 相比前代的主要变化

LLaMA 3 的变化可以重点看两部分：注意力机制和词表/上下文能力。

### 4.1 注意力机制

- LLaMA 1 全部使用标准 MHA。
- LLaMA 2 只有大模型版本（34B、70B）使用 GQA。
- LLaMA 3 全系列使用 GQA。

### 4.2 词表与长上下文支持

- LLaMA 3 将词表大小扩展到 128k。
- 同时调整了 RoPE 的 base frequency（基础频率），以支持更长的上下文。

可以理解为：LLaMA 3 不只是参数升级，更是在输入表示和长文本处理能力上做了增强。

## 5. 版本演进脉络

如果按演进思路来总结：

| 版本 | 核心特点 |
| :--- | :--- |
| LLaMA 1 | 奠定基础结构，使用标准 MHA |
| LLaMA 2 | 保持基础骨架稳定，增强训练数据和上下文长度；大模型开始引入 GQA |
| LLaMA 3 | 全系列采用 GQA，扩大词表，并优化 RoPE 以支持更长上下文 |

## 6. 结构示意图

![](/posts/ml/assets/2026-03-15-17-22-29.png)
