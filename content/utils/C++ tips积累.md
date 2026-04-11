## 1. `explicit`

用于禁止单参数构造函数的隐式转换。

```cpp
class A {
public:
    explicit A(int value) : value_(value) {}

private:
    int value_;
};

A a1(1);     // OK
// A a2 = 1; // error: 禁止隐式转换
```

## 2. 宏里的 `do { } while(false)`

多语句、无返回值宏建议用 `do { } while(false)` 包装，避免悬空分号和 `if/else` 配对问题。

```cpp
#define LOG_AND_RETURN_IF_NULL(ptr) \
    do {                            \
        if ((ptr) == nullptr) {     \
            LOG(ERROR) << "null";   \
            return;                 \
        }                           \
    } while (false)
```

优先用函数或 `inline` 替代宏。

## 3. `inline`

核心作用：

1. 作为内联提示，是否展开由编译器决定。
2. 可替代部分宏，具备类型检查和作用域语义。
3. 允许函数定义放在头文件中，避免 ODR 冲突。

补充：

1. 模板实现通常写在头文件；若放到 `.cpp`，通常需要显式实例化。
2. `inline` 中的 `return` 只返回当前函数。

## 4. `[[nodiscard]]`

属性，不是关键字。用于提示返回值不应被忽略。

```cpp
[[nodiscard]] int foo();

foo();         // 可能触发警告
int x = foo(); // OK
```

补充：

1. `[[nodiscard]]` 是 C++17。
2. `[[nodiscard("reason")]]` 是 C++20。

## 5. 一些常用位运算

对于 `w` 位无符号整数 `x`：

```cpp
MAX - x == x ^ MAX == ~x
```

把 `x` 向下对齐到 `2^n` 的倍数：

```cpp
x & ~((1 << n) - 1) == x & ~(2^n - 1) == (x / 2^n) * 2^n
```
