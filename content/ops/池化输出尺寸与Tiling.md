## 基本知识

### 基本概念
- `kernel_size`：滑动窗口大小
- `stride`：滑动步长
- `padding`：边界填充方式
  - `VALID`：不填充
  - `SAME`：自动填充，使输出长度约为 `L_in / stride`
- `dilation`：空洞大小

### 计算公式

先定义有效窗口大小：

$$
K_{\text{eff}} = \text{dilation} \times (\text{kernel\_size} - 1) + 1
$$

#### 1. 手动指定 padding

如果两侧对称填充 `padding`，则输出长度为：

$$
L_{\text{out}} = \left\lfloor \frac{L_{\text{in}} + 2 \times \text{padding} - K_{\text{eff}}}{\text{stride}} \right\rfloor + 1
$$

展开后也可写成：

$$
L_{\text{out}} = \left\lfloor \frac{L_{\text{in}} + 2 \times \text{padding} - \text{dilation} \times (\text{kernel\_size} - 1) - 1}{\text{stride}} \right\rfloor + 1
$$

#### 2. 自动 padding

`VALID`：不填充

$$
L_{\text{out}} = \left\lfloor \frac{L_{\text{in}} - K_{\text{eff}}}{\text{stride}} \right\rfloor + 1
$$

`SAME`：自动填充

$$
L_{\text{out}} = \left\lceil \frac{L_{\text{in}}}{\text{stride}} \right\rceil
= \left\lfloor \frac{L_{\text{in}} + \text{stride} - 1}{\text{stride}} \right\rfloor
$$

以上公式以 1D 为例；2D/3D 池化时，对每个空间维度分别计算即可

### 常用数学符号

向上取整：
**$\lceil \dots \rceil$**

向下取整：
**$\lfloor \dots \rfloor$**

等价公式：
$$
\left\lfloor \frac{n - 1}{k} \right\rfloor + 1 = \left\lceil \frac{n}{k} \right\rceil
$$

## NPU的通用池化TILING逻辑

### 核心决策流程

简化的决策树，建立全局视角：

1. **第一关：只切 N/C 维度**（最高效，无空间开销）
   - 成功 -> `Return`
   - 失败 -> 进入第二关（并且强制把 NC 设为 1）

2. **第二关：对齐切分 D/H/W**（次高效，仅限无 Pad/无 Overlap 场景）
   - 条件不满足 -> 跳过
   - 尝试切 D -> 成功? Return
   - 尝试切 H -> 成功? Return
   - 尝试切 W -> 成功? Return
   - 全失败 -> 进入第三关

3. **第三关：非对齐通用切分**（保底方案，最复杂）
   - 初始化（根据是否有 Pad 决定起点）
   - 循环收缩（D -> H -> W）直到塞进内存
   - `Return`

### 详细步骤解析

#### 第一阶段：尝试纯 N/C 切分 (`TrySplitNC`)

这是最理想的情况。如果能只把 Batch(N) 或 Channel(C) 切开，就能不动 D/H/W，完全避免处理图像边缘的重叠（Overlap）和填充（Pad）。

- **策略**：二分查找（Binary Search）。
- **目标**：找到满足内存限制的**最大** N*C 块。
- **逻辑**：
  1. 保持 `d/h/w` 为全图大小。
  2. 在 `[1, TotalNC]` 范围内进行二分查找。
  3. 如果能找到一个值，既能塞进 UB 内存，又能保证任务数足够分给所有核心，就直接采用。
- **如果失败**：说明 D/H/W 实在太大了，即使 N*C=1 也塞不进去，必须切空间维度。**此时强制 `NC = 1`，进入下一阶段。**

#### 第二阶段：尝试对齐切分 (`TrySplitAlign D/H/W`)

**前提条件**：`No Pad` 且 `No Overlap` (Kernel <= Stride)。 在这种简单场景下，我们可以按 Stride 的倍数完美切分，效率很高。

- **切分顺序**：**D 优先 -> H 其次 -> W 最后**。
  - *为什么？* 因为 W 维度在内存中是连续的，切 W 会打断连续性，降低向量计算效率。所以只要能切 D 或 H 搞定，就绝不切 W。
- **策略**：同样是**二分查找**。
  - 以 D 维度为例：在 `[1, MaxD]` 范围内找一个满足条件的 `k`，使得 `k * Stride` 是最大的可行块。
- **结果**：一旦在某一级（比如切 D）成功了，就直接返回，不再往下走。如果 D/H/W 都切不开（比如模型太巨大），进入第三阶段。

#### 第三阶段：非对齐通用切分 (`SplitUnalignDHW`)

这是最后的保底手段，适用于有 Pad、有 Overlap 或者前两关都过不去的情况。这里的逻辑不再是二分查找，而是**迭代收缩**。

1. **初始化起点（决定了是“扩张”还是“收缩”）：**

   - **场景 A（简单）：无 Pad/Overlap**
     - 前面第二阶段失败进来的。说明全图放不下。
     - **起点**：直接设为 `Stride` 大小（极小值）。
     - **逻辑**：既然全图放不下，那我直接退到安全线。通常 Stride 大小肯定能放下，不需要循环调整。
   - **场景 B（复杂）：有 Pad/Overlap**
     - **起点**：设为 `Full Size`（全图）。
     - **逻辑**：由于重叠区域（Ghost Area）的存在，切得越碎，重复计算越多。所以我们要**从大往小切**，力求切分份数最少。

2. **迭代调整循环 (`while` loop)：**

   只要当前的配置不满足条件（内存爆了 OR 核心没吃饱），就调用 **`DynamicAdjustmentDWH()`** 砍一刀。

3. **砍一刀的策略（DynamicAdjustmentDWH）：**

   这是一个**严格的降维打击顺序**，目的是保护内部连续性：

   1. **先看 D**：
      - 如果 D 还没被切成 1：**D的份数 + 1**。
      - 重新计算 D 的大小，Check 内存。如果不满足，下次进来继续切 D。
   2. **再看 H**：
      - 只有当 D 已经被切成 1 了：**H的份数 + 1**。
      - 重新计算 H 的大小。
   3. **最后看 W**：
      - 只有当 D=1 且 H=1 了：**W的份数 + 1**。
      - 这是最无奈的选择。
