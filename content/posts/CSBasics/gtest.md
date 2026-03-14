---
title: "gtest"
date: 2026-03-14
---

# gtest
---

## 1. gtest 是什么

Google Test，C++ 里最常见的单元测试框架之一。

主要用来做：

1. 写单元测试
2. 写断言
3. 做回归验证

最小例子：

```cpp
TEST(MathTest, Add) {
    EXPECT_EQ(1 + 1, 2);
}
```

---

## 2. Ubuntu 安装

```bash
sudo apt install libgtest-dev
```

注意：

- 有些环境装完能直接用
- 有些环境给的是源码，还要自己编

如果项目比较新，也可以走 vcpkg、FetchContent 之类的方法。

---

## 3. CMake 最小接入

```cmake
cmake_minimum_required(VERSION 3.16)
project(gtest_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

add_executable(unit_tests tests/test_demo.cpp)
target_link_libraries(unit_tests PRIVATE GTest::gtest GTest::gtest_main)

enable_testing()
add_test(NAME unit_tests COMMAND unit_tests)
```

核心是：

```cmake
target_link_libraries(unit_tests PRIVATE GTest::gtest GTest::gtest_main)
```

其中 `GTest::gtest_main` 会提供测试程序入口。

---

## 4. 最小 demo

```cpp
#include <gtest/gtest.h>

int add(int a, int b) {
    return a + b;
}

TEST(MathTest, AddPositiveNumbers) {
    EXPECT_EQ(add(1, 2), 3);
}

TEST(MathTest, AddZero) {
    EXPECT_EQ(add(5, 0), 5);
}
```

---

## 5. 怎么编译和运行

```bash
mkdir -p build
cd build
cmake ..
make -j
./unit_tests
```

如果已经注册给 CTest，也可以：

```bash
ctest --output-on-failure
```

---

## 6. 最常用宏

### `EXPECT_EQ`

检查是否相等：

```cpp
EXPECT_EQ(add(1, 2), 3);
```

### `EXPECT_NE`

检查是否不相等：

```cpp
EXPECT_NE(x, y);
```

### `EXPECT_TRUE` / `EXPECT_FALSE`

```cpp
EXPECT_TRUE(ok);
EXPECT_FALSE(has_error);
```

### 大小比较

```cpp
EXPECT_LT(a, b);
EXPECT_LE(a, b);
EXPECT_GT(a, b);
EXPECT_GE(a, b);
```

---

## 7. ASSERT 和 EXPECT 的区别

这是最重要的一点。

### `EXPECT_*`

失败后，当前测试**继续执行**。

```cpp
EXPECT_EQ(x, 10);
EXPECT_EQ(y, 20);
```

适合：想一次多看几个结果。

### `ASSERT_*`

失败后，当前测试**立刻停止**。

```cpp
ASSERT_NE(ptr, nullptr);
EXPECT_EQ(ptr->size(), 10);
```

适合：前置条件失败后，后面已经没必要测了。

一句话：

- `EXPECT`：错了也继续看
- `ASSERT`：前面错了就别往下跑了

---

## 8. 什么时候用 ASSERT

这几类情况更适合 `ASSERT_*`：

1. 指针不能为空
2. 文件必须成功打开
3. 对象必须创建成功
4. 前置 shape / size 必须合法

比如：

```cpp
ASSERT_NE(model, nullptr);
ASSERT_EQ(tokens.size(), 128);
```

---

## 9. 什么时候用 EXPECT

这几类情况更适合 `EXPECT_*`：

1. 结果值比较
2. 边界条件比较
3. 同一个测试里想多收集几个失败信息

比如：

```cpp
EXPECT_EQ(output.rows(), 32);
EXPECT_EQ(output.cols(), 64);
EXPECT_LT(loss, 1e-3);
```

---

## 10. 一个更像真实项目的例子

```cpp
#include <gtest/gtest.h>

int divide(int a, int b) {
    return a / b;
}

TEST(DivideTest, NormalCase) {
    EXPECT_EQ(divide(10, 2), 5);
}

TEST(DivideTest, NegativeCase) {
    EXPECT_EQ(divide(-10, 2), -5);
}
```

如果后面涉及对象、文件、模型，就会更频繁地用 `ASSERT_*` 做前置检查。

---

## 11. 常见坑

1. `EXPECT_*` 和 `ASSERT_*` 用反了
2. 测试代码能编译，但没有 `enable_testing()` / `add_test()`，所以 `ctest` 看不到
3. 忘了链接 `GTest::gtest_main`
4. 一个测试写太长，失败后很难定位问题

---

## 12. 新项目里最推荐先会的几个

- `TEST`
- `EXPECT_EQ`
- `EXPECT_TRUE`
- `EXPECT_FALSE`
- `ASSERT_EQ`
- `ASSERT_TRUE`

先把这些用顺手，已经够写大部分基础测试。

---

## 13. 一句话总结

`gtest` 最核心的价值不是“写测试框架语法”，而是：

**以后你改代码时，不用纯靠运气判断有没有改坏。**

从零开始的话，先把这几件事掌握就够：

- `TEST`
- `EXPECT_*`
- `ASSERT_*`
- CMake 里 `GTest::gtest` 和 `GTest::gtest_main`

先写起来，比一次学太多高级特性更重要。
