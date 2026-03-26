---
title: "tmux基本"
date: 2026-03-19
tags: ["TOOLS"]
---
# tmux基本
---

1）启动 tmux
```bash
tmux
```

2）临时退出 tmux，回到普通终端
```text
Ctrl+b d
```

3）重新进入 tmux
```bash
tmux attach
```

4）左右分屏
```text
Ctrl+b %
```

5）上下分屏
```text
Ctrl+b "
```

6）在分屏之间切换
```text
Ctrl+b 方向键
```

7）关闭当前窗格
```bash
exit
```

或者按：

```text
Ctrl+d
```

8）查看当前有哪些 tmux 会话
```bash
tmux ls
```

你只背这个逻辑就行

- `tmux`：进去
- `Ctrl+b d`：出来，但不关
- `%`：左右分
- `"`：上下分
- `方向键`：切换
- `exit`：关闭当前窗格
- `tmux ls`：看会话
