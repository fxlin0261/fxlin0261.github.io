---
title: "glog"
date: 2026-03-14
---

# glog
---

## 1. glog 是什么

Google 的 C++ 日志库，主要干两件事：

1. 打日志：`LOG(INFO)` / `LOG(ERROR)`
2. 做运行时检查：`CHECK` / `CHECK_EQ`

适合新项目快速接入，写法比 `std::cout` 和手写 `if + abort` 顺手很多。

---

## 2. Ubuntu 安装

```bash
sudo apt install libgoogle-glog-dev
```

CMake 里一般直接：

```cmake
find_package(glog REQUIRED)
```

---

## 3. CMake 最小接入

```cmake
cmake_minimum_required(VERSION 3.16)
project(glog_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glog REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE glog::glog)
```

核心就一句：

```cmake
target_link_libraries(main PRIVATE glog::glog)
```

---

## 4. 最小 demo

```cpp
#include <glog/logging.h>

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);

    FLAGS_alsologtostderr = 1;
    FLAGS_colorlogtostderr = 1;

    LOG(INFO) << "program started";
    LOG(WARNING) << "this is a warning";
    LOG(ERROR) << "this is an error";

    int x = 3;
    int y = 3;
    CHECK_EQ(x, y) << "x should equal y";

    google::ShutdownGoogleLogging();
    return 0;
}
```

---

## 5. 初始化和常用配置

### 初始化

```cpp
google::InitGoogleLogging(argv[0]);
```

一般放在 `main` 开头。

### 常见配置

```cpp
FLAGS_alsologtostderr = 1;
FLAGS_colorlogtostderr = 1;
FLAGS_minloglevel = 0;
```

含义：

- `FLAGS_alsologtostderr = 1`：日志同时输出到终端
- `FLAGS_colorlogtostderr = 1`：终端彩色显示
- `FLAGS_minloglevel = 0`：从 `INFO` 开始打印

### 结束

```cpp
google::ShutdownGoogleLogging();
```

不是必须，但写上更完整。

---

## 6. 最常用日志宏

### `LOG(INFO)`

普通信息：

```cpp
LOG(INFO) << "loading model...";
```

### `LOG(WARNING)`

警告：

```cpp
LOG(WARNING) << "config missing, use defaults";
```

### `LOG(ERROR)`

错误，但程序可继续：

```cpp
LOG(ERROR) << "failed to open file";
```

### `LOG(FATAL)`

致命错误，打印后直接退出：

```cpp
LOG(FATAL) << "unrecoverable error";
```

注意：`LOG(FATAL)` 不是普通报错，它会终止程序。

---

## 7. 条件日志

### `LOG_IF`

```cpp
LOG_IF(WARNING, batch_size > 1024) << "batch size too large";
```

条件满足才打印。

---

## 8. CHECK 系列

`CHECK` 的逻辑很简单：

- 条件成立：继续跑
- 条件不成立：打印日志并终止程序

### `CHECK`

```cpp
CHECK(ptr != nullptr) << "ptr must not be null";
```

### `CHECK_EQ`

```cpp
CHECK_EQ(a, b);
```

### 常见同类

```cpp
CHECK_NE(a, b);
CHECK_LT(x, y);
CHECK_LE(x, y);
CHECK_GT(x, y);
CHECK_GE(x, y);
```

适合做前置条件检查、shape 检查、参数合法性检查。

---

## 9. 调试时常见宏

### `DLOG(INFO)`

```cpp
DLOG(INFO) << "debug info";
```

通常只在 debug 场景下有用。

### `VLOG(n)`

```cpp
VLOG(1) << "verbose info";
VLOG(2) << "more verbose info";
```

适合做可控的详细日志。

---

## 10. 常见使用场景

1. 程序入口日志
2. 模型加载/配置打印
3. 关键流程打点
4. 文件、指针、参数、shape 检查

比如：

```cpp
LOG(INFO) << "model path: " << model_path;
CHECK(!model_path.empty()) << "model path must not be empty";
```

---

## 11. 常见坑

1. 忘记 `InitGoogleLogging`
2. 把 `LOG(FATAL)` 当 `LOG(ERROR)` 用
3. 以为 `CHECK` 只是打印，其实它会终止程序
4. 终端没看到日志，其实日志写到文件了，这时加 `FLAGS_alsologtostderr = 1`

---

## 12. 一句话总结

`glog` 最值得先会的就这些：

- `InitGoogleLogging`
- `LOG(INFO/WARNING/ERROR/FATAL)`
- `CHECK / CHECK_EQ`
- `FLAGS_alsologtostderr`

先把这些用顺手，已经够覆盖大部分项目里的基础日志场景。
