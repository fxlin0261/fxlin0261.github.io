## 初始化
按实现里的映射，可以直接写成：

```cpp
key[0] = (uint32_t)seed;
key[1] = (uint32_t)(seed >> 32);

counter[0] = 0;
counter[1] = 0;
counter[2] = (uint32_t)offset;
counter[3] = (uint32_t)(offset >> 32);
```

## 整体流程
Philox 的核心是“计数器驱动 + 多轮混淆”。常见写法是做 `10` 轮：

- 前 `9` 轮：单轮运算 + key 更新
- 第 `10` 轮：只做单轮运算，不再更新 key

```cpp
for (int i = 0; i < totalCount / 4; i++) {
    uint32_t state[4] = {counter[0], counter[1], counter[2], counter[3]};
    uint32_t round_key[2] = {key[0], key[1]};

    for (int j = 0; j < 9; j++) {
        Action_A(state, round_key);
        Action_B(round_key);
    }
    Action_A(state, round_key);

    // state 中得到 4 个 32-bit 伪随机数
    OutputRandomNumbers(state);

    // 原始 counter 自增，准备下一组输出
    Action_C(counter);
}
```

## Action_A：单轮运算
### A1. 乘法拆分
使用常数：

$$
M_0 = \text{0xD2511F53}, \quad M_1 = \text{0xCD9E8D57}
$$

分别对 `counter[0]` 和 `counter[2]` 做 `32-bit x 32-bit -> 64-bit` 乘法，再拆成高低 `32-bit`：

```cpp
uint64_t prod_0 = (uint64_t)counter[0] * M_0;
uint32_t Lo_0 = (uint32_t)(prod_0 & 0xFFFFFFFF);
uint32_t Hi_0 = (uint32_t)(prod_0 >> 32);

uint64_t prod_1 = (uint64_t)counter[2] * M_1;
uint32_t Lo_1 = (uint32_t)(prod_1 & 0xFFFFFFFF);
uint32_t Hi_1 = (uint32_t)(prod_1 >> 32);
```

### A2. 混合
将乘法结果的高位、原计数器的相邻分量以及 key 做异或，再把低位写回：

```cpp
uint32_t old_c1 = counter[1];
uint32_t old_c3 = counter[3];

counter[0] = Hi_1 ^ old_c1 ^ key[0]; // ^ 按位异或，相同为0，不同为1
counter[1] = Lo_1;
counter[2] = Hi_0 ^ old_c3 ^ key[1];
counter[3] = Lo_0;
```

## Action_B：Key 更新
在每轮之间更新 key，最后一轮除外。使用 Weyl 常数：

```cpp
uint32_t W_0 = 0x9E3779B9;
uint32_t W_1 = 0xBB67AE85;

key[0] = (key[0] + W_0) & 0xFFFFFFFF;
key[1] = (key[1] + W_1) & 0xFFFFFFFF;
```

## Action_C：Counter 自增
将 `counter` 看成一个低位在前的 `128-bit` 整数做 `+1`：先给 `counter[0]` 加 `1`；如果这一位加完后没有溢出，就在这里停止。只有当前位加完变成 `0` 时，才继续向更高位进位；如果 `counter[3]` 也溢出为 `0`，说明整个 `128-bit counter` 回到 `0`。

```python
def inc_counter(counter):
    for i in range(4):
        counter[i] = (counter[i] + 1) & MASK_32
        if counter[i] != 0:
            return counter
    return counter
```
