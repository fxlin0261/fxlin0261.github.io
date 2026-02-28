---
title: "Philox算法"
date: 2026-02-27
tags: ["OP"]
---

# Philox算法
---

### 1. 初始化 (Initialization)

状态由两个 $32$-bit 的种子和四个 $32$-bit 的计数器组成：  

$$\begin{aligned}
key &= [seed_{low}, \quad seed_{high}] \\
counter &= [0, \quad 0, \quad offset_{low}, \quad offset_{high}]
\end{aligned}$$

### 2. 计算流程 (Execution Flow)

Philox 算法的核心是一个嵌套循环。通常进行 10 轮运算（9 次带 Key 更新的混淆 + 1 次最终混淆）

```cpp
for (int i = 0; i < totalCount / 4; i++) {
    // 1. 创建工作状态副本
    uint32_t state[4] = {counter[0], counter[1], counter[2], counter[3]};
    uint32_t round_key[2] = {key[0], key[1]};

    // 2. 执行 10 轮混淆
    for (int j = 0; j < 9; j++) {
        Action_A(state, round_key); // 对 state 进行乘法拆分和混合
        Action_B(round_key);        // 使用 Weyl 常数更新 round_key
    }
    Action_A(state, round_key);     // 第 10 轮不更新密钥

    // 3. 此时的 state 数组就是生成的 4 个 32-bit 伪随机数
    OutputRandomNumbers(state);

    // 4. Action C：原始计数器自增 (为下一个 block 准备)
    Action_C(counter); 
}
```
A. 单轮运算 (Philox Round)  
输入：

$$
\mathrm{counter}[0..3], \quad \mathrm{key}[0..1]
$$

过程：  
1. 乘法拆分：  
使用常数 $M_0 = \text{0xD2511F53}, \quad M_1 = \text{0xCD9E8D57}$：  

$$\begin{aligned}
prod_0 &= \mathrm{counter}[0] \times M_0 \quad &\text{(64-bit)} \\
Lo_0 &= prod_0 \pmod{2^{32}} \quad &\text{(提取低32位)} \\
Hi_0 &= \lfloor prod_0 / 2^{32} \rfloor \quad &\text{(提取高32位)}
\end{aligned}$$

$$\begin{aligned}
prod_1 &= \mathrm{counter}[2] \times M_1 \quad &\text{(64-bit)} \\
Lo_1 &= prod_1 \pmod{2^{32}} \quad &\text{(提取低32位)} \\
Hi_1 &= \lfloor prod_1 / 2^{32} \rfloor \quad &\text{(提取高32位)}
\end{aligned}$$

2. 混合 (XOR & Swap)：  

$$
\begin{aligned}
\mathrm{counter}[0] &= Hi_1 \oplus \mathrm{counter}[1] \oplus \mathrm{key}[0] \\
\mathrm{counter}[1] &= Lo_1 \\
\mathrm{counter}[2] &= Hi_0 \oplus \mathrm{counter}[3] \oplus \mathrm{key}[1] \\
\mathrm{counter}[3] &= Lo_0
\end{aligned}
$$

B. 密钥更新 (Bump Key) [在多轮迭代之间（除最后一轮外）执行，防止密钥退化]  
使用 Weyl 常数 $W_0 = \text{0x9E3779B9}, \quad W_1 = \text{0xBB67AE85}$：  

$$
\begin{aligned}
key[0] &= (key[0] + W_0) \ \& \ \text{MASK}_{32} \\
key[1] &= (key[1] + W_1) \ \& \ \text{MASK}_{32}
\end{aligned}
$$

C. 计数器自增 (Increment Counter) [将 counter 视为一个 128 位的整数进行 +1 操作（小端序，低位在前）] 
```python
def inc_counter(counter):
    # 遍历 4 个分量 (从低位到高位)
    for i in range(4):
        # 当前位加 1，并强制截断为 32 位
        counter[i] = (counter[i] + 1) & MASK_32
        
        # 判断是否需要进位：
        # 如果当前位不为 0，说明没有发生溢出(wrap around)，
        # 不需要向高位进位，直接结束函数。
        if counter[i] != 0:
            return counter
            
    # 如果循环走完（即 counter[3] 也溢出了），说明整个 128 位数溢出归零
    return counter
```
