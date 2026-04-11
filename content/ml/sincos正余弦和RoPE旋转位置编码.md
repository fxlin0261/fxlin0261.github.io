![[Pasted image 20260405160827.png|442]]

- 为什么需要位置编码：因为计算注意力中，没有区分词的顺序，没办法区分 “你爱我”和“我爱你”，位置编码就来解决这一问题

- 为什么不直接使用顺序编码，比如1 2 3 4 5 6，序列太长影响，并且不能固定序列长度

## Transformer位置编码

$$
Input(pos)=Embedding(token_{pos})+PositionalEncoding(pos)
$$
第 pos 个 token 的输入 = 第 pos 个 token 的词向量 + 第 pos 个位置的位置编码向量

PositionalEncoding是个什么计算：

$$
PE(pos)=
\begin{bmatrix}
\sin(0) & \cos(0) & \sin(0) & \cos(0) \\
\sin(1) & \cos(1) & \sin(0.01) & \cos(0.01) \\
\sin(2) & \cos(2) & \sin(0.02) & \cos(0.02) \\
\sin(3) & \cos(3) & \sin(0.03) & \cos(0.03)
\end{bmatrix}
$$
$$
\begin{array}{l}
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right) \\
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
\end{array}
$$
- $pos$：单词在序列中的位置
- $d_{model}$：词向量的维度（固定值）
- $i$：位置编码向量中的维度索引，范围在 $\left[0, \frac{d_{model}}{2}-1\right]$
位置编码向量的维度是成对设计的：偶数维用 $\sin$，奇数维用 $\cos$。所以 $i$ 只需要遍历前一半的索引，就能生成长度为 $d_{model}$ 的完整位置编码向量。
###  推导过程
目的是寻找到式子满足：pos+k的位置编码可以由于pos的位置线性表示就行
$$
PE(pos+k)=T \times PE(pos)
$$
从论文已知
$$
\alpha=\frac{pos}{10000^{\frac{2i}{d_{model}}}}, \qquad
\beta=\frac{k}{10000^{\frac{2i}{d_{model}}}}
$$
则
$$
\begin{aligned}
PE(pos+k,2i) &= \sin(\alpha+\beta) \\
&= \sin(\alpha)\cos(\beta)+\cos(\alpha)\sin(\beta) \\
&= PE(pos,2i)\cos(\beta)+PE(pos,2i+1)\sin(\beta)
\end{aligned}
$$

$$
\begin{aligned}
PE(pos+k,2i+1) &= \cos(\alpha+\beta) \\
&= \cos(\alpha)\cos(\beta)-\sin(\alpha)\sin(\beta) \\
&= PE(pos,2i+1)\cos(\beta)-PE(pos,2i)\sin(\beta)
\end{aligned}
$$

可以合并为矩阵写法
$$
T \times PE(pos)=PE(pos+k)
$$
$$
\begin{aligned}
\begin{bmatrix}
\cos(\beta) & \sin(\beta) \\
-\sin(\beta) & \cos(\beta)
\end{bmatrix}
\begin{bmatrix}
PE(pos,2i) \\
PE(pos,2i+1)
\end{bmatrix}
&=
\begin{bmatrix}
PE(pos,2i)\cos(\beta)+PE(pos,2i+1)\sin(\beta) \\
PE(pos,2i+1)\cos(\beta)-PE(pos,2i)\sin(\beta)
\end{bmatrix} \\
&=
\begin{bmatrix}
PE(pos+k,2i) \\
PE(pos+k,2i+1)
\end{bmatrix}
\end{aligned}
$$

从数学上看，在固定 $k$ 的情况下，较小的 $i$ 对应更高频的变化，位置编码随位置变化更快，因此更侧重刻画局部位置信息；较大的 $i$ 对应更低频的变化，位置编码变化更平缓，因此更适合表达全局位置信息

### 举个例子
序列长度为 4，词向量维度为 4。

$$
\begin{array}{l}
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right) \\
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
\end{array}
$$

这里 $d_{model}=4$，所以 $i=0,1$。

于是：

- 第 0 维：$\sin(pos/10000^0)=\sin(pos)$
- 第 1 维：$\cos(pos/10000^0)=\cos(pos)$
- 第 2 维：$\sin(pos/10000^{2/4})=\sin(pos/100)$
- 第 3 维：$\cos(pos/10000^{2/4})=\cos(pos/100)$

所以 4 个位置 $pos=0,1,2,3$ 的位置编码矩阵为：

$$
PE=
\begin{bmatrix}
\sin(0) & \cos(0) & \sin(0) & \cos(0) \\
\sin(1) & \cos(1) & \sin(0.01) & \cos(0.01) \\
\sin(2) & \cos(2) & \sin(0.02) & \cos(0.02) \\
\sin(3) & \cos(3) & \sin(0.03) & \cos(0.03)
\end{bmatrix}
$$

近似数值是：

$$
PE \approx
\begin{bmatrix}
0 & 1 & 0 & 1 \\
0.8415 & 0.5403 & 0.0100 & 0.99995 \\
0.9093 & -0.4161 & 0.0200 & 0.99980 \\
0.1411 & -0.9900 & 0.0300 & 0.99955
\end{bmatrix}
$$

## ROPE位置编码
![[Pasted image 20260405234030.png]]

Transformer原生词向量算法的问题：位置权重直接累加在词向量上，污染了语义，分不清是哪一部分
$$
Q = W_q (X_m + P_m), \quad
K = W_k (X_n + P_n)
$$
$$
\text{Score} = (X_m + P_m)(X_n + P_n)^\top
= (X_m + P_m)(X_n^\top + P_n^\top)
= X_m X_n^\top + X_m P_n^\top + P_m X_n^\top + P_m P_n^\top
$$
结果 = 纯语义*纯语义+噪声1+噪声2+纯相对位置

核心思想：位置权重加法改为了旋转
$$
Q = R_m X_m, \quad K = R_n X_n
$$
$$
\text{Score} = (R_m X_m)(R_n X_n)^\top = X_m (R_n - R_m) X_n^\top
$$
很纯净

### 推导过程

已知逆时针旋转矩阵
$$
R(\alpha) = 
\begin{bmatrix}
\cos \alpha & -\sin \alpha \\
\sin \alpha & \cos \alpha
\end{bmatrix}
$$
且如果是第 $m$ 个词，就逆时针旋转 $m\theta$ 个角度；第 $n$ 个词，就逆时针旋转 $n\theta$ 个角度。

假设原始向量是 $q$ 和 $k$，旋转后的向量分别是 $q'$ 和 $k'$。
$$
q' = R(m \theta)\, q
$$
$$
k' = R(n \theta)\, k
$$
已知注意力公式，点积注意力分数（未缩放）
$$
\text{score}(q, k) = q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$
缩放后的注意力分数
$$
\text{score}(q, k) = \frac{q \cdot k}{\sqrt{d_k}}
$$
Softmax 归一化之后的注意力权重矩阵
$$
A = \text{softmax}\Big(\frac{Q K^\top}{\sqrt{d_k}}\Big)
$$
Scaled Dot-Product Attention 输出
$$
\text{Attention}(Q, K, V) = \text{softmax}\Big(\frac{Q K^\top}{\sqrt{d_k}}\Big) V
$$
则有

$$
\text{Score} = (q')^\top \cdot k' = (R(m\theta) \cdot q)^\top \cdot (R(n\theta) \cdot k) = q^\top \cdot R(m\theta)^\top \cdot R(n\theta) \cdot k
$$
假设

$$
\alpha = m\theta, \quad \beta = n\theta
$$
且
$$
\begin{aligned}
R(\alpha)^\top
&=
\begin{pmatrix}
\cos \alpha & -\sin \alpha \\
\sin \alpha & \cos \alpha
\end{pmatrix}^\top \\
&=
\begin{pmatrix}
\cos \alpha & \sin \alpha \\
-\sin \alpha & \cos \alpha
\end{pmatrix} \\
&=
\begin{pmatrix}
\cos(-\alpha) & -\sin(-\alpha) \\
\sin(-\alpha) & \cos(-\alpha)
\end{pmatrix} \\
&= R(-\alpha) \\
&= R(-m\theta)
\end{aligned}
$$
并且
$$
R(-m\theta) \cdot R(n\theta) =
\begin{pmatrix}
\cos \alpha & \sin \alpha \\
-\sin \alpha & \cos \alpha
\end{pmatrix}
\begin{pmatrix}
\cos \beta & -\sin \beta \\
\sin \beta & \cos \beta
\end{pmatrix}
=
\begin{pmatrix}
\cos((n-m)\theta) & -\sin((n-m)\theta) \\
\sin((n-m)\theta) & \cos((n-m)\theta)
\end{pmatrix}
=
R((n-m)\theta)
$$

最终
$$
\text{Score} = (q')^\top \cdot k' 
= q^\top \cdot R(m\theta)^\top \cdot R(n\theta) \cdot k 
= q^\top \cdot R(-m\theta) \cdot R(n\theta) \cdot k
= q^\top \cdot R((n-m)\theta) \cdot k
$$
$$
\theta_i = 10000^{- \frac{2(i-1)}{d_\text{model}}}, \quad
i \in [1, 2, \dots, \frac{d_\text{model}}{2}]
$$
这样一个对焦块的形式
$$
\begin{pmatrix}
\cos \alpha & -\sin \alpha & 0 & 0 & \cdots & 0 \\
\sin \alpha & \cos \alpha & 0 & 0 & \cdots & 0 \\
0 & 0 & \cos \alpha & -\sin \alpha & \cdots & 0 \\
0 & 0 & \sin \alpha & \cos \alpha & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos \alpha & -\sin \alpha \\
0 & 0 & 0 & 0 & \cdots & \sin \alpha & \cos \alpha
\end{pmatrix}
\begin{pmatrix}
0.9 \\ 0.21 \\ \vdots \\ -0.3 \\ 0.23
\end{pmatrix}
$$

### 举个例子

$$
X =
\begin{pmatrix}
1 & 0 & 0.5 & 0.2 \\
0 & 1 & 0.1 & 0.3 \\
0.5 & 0.5 & 0.2 & 0.1
\end{pmatrix}
$$
$$
R(\theta_1) =
\begin{pmatrix}
\cos \theta_1 & -\sin \theta_1 \\
\sin \theta_1 & \cos \theta_1
\end{pmatrix}
\approx
\begin{pmatrix}
0.5403 & -0.8415 \\
0.8415 & 0.5403
\end{pmatrix}
$$
$$
R(\theta_2) =
\begin{pmatrix}
\cos \theta_2 & -\sin \theta_2 \\
\sin \theta_2 & \cos \theta_2
\end{pmatrix}
\approx
\begin{pmatrix}
0.99995 & -0.0099998 \\
0.0099998 & 0.99995
\end{pmatrix}
$$
$$
R_\text{RoPE} =
\begin{pmatrix}
R(\theta_1) & 0 \\
0 & R(\theta_2)
\end{pmatrix}
=
\begin{pmatrix}
0.5403 & -0.8415 & 0 & 0 \\
0.8415 & 0.5403 & 0 & 0 \\
0 & 0 & 0.99995 & -0.0099998 \\
0 & 0 & 0.0099998 & 0.99995
\end{pmatrix}
$$

$$
X^\text{RoPE} \approx
\begin{pmatrix}
0.5403 & -0.8415 & 0.501975 & 0.19499 \\
0.8415 & 0.5403 & 0.103045 & 0.300985 \\
0.6909 & -0.1506 & 0.19899 & 0.101995
\end{pmatrix}
$$
