"""Benchmark task definitions for ApexCoder vs ColonyCoder comparison."""

TASK_2048 = """\
用 Python 写一个 2048 游戏。

## 核心要求
1. 使用 curses 库实现终端 UI
2. 4x4 棋盘，初始随机放置 2 个 2（或偶尔 4）
3. 方向键（或 WASD）控制所有数字方块向四个方向滑动
4. 滑动时相同数字合并（2+2=4, 4+4=8...），每次合并后随机生成新的 2/4
5. 达到 2048 显示胜利；无法移动显示失败

## UI 要求
- 棋盘有边框，每格居中显示数字
- 不同数字用不同颜色（2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048）
- 顶部显示当前分数和最高分
- 底部显示操作提示和退出键（Q）

## 技术要求
- 单文件实现，保存到 /tmp/game_2048/game_2048.py
- 代码结构清晰：Board 类、Game 类、UI 渲染分离
- 可直接 python3 game_2048.py 运行
- 必须能在标准 24x80 终端下正常运行
"""

TASK_TETRIS = """\
用 Python 写一个俄罗斯方块 (Tetris) 游戏。

## 核心要求
1. 使用 curses 库实现终端 UI
2. 10x20 游戏区，7 种标准方块 (I/O/T/S/Z/L/J)，随机生成
3. 左右键移动、上键旋转、下键加速下落、空格硬降落
4. 一行填满自动消除，加分
5. 方块堆到顶部游戏结束

## UI 要求
- 游戏区有边框
- 不同方块用不同颜色
- 右侧显示：分数、等级、消除行数、下一个方块预览
- 底部显示操作提示和退出键（Q）
- 等级越高下落越快（每消 10 行升 1 级）

## 技术要求
- 单文件实现，保存到 /tmp/tetris/tetris.py
- 代码结构清晰：Piece 类、Board 类、Game 类
- 可直接 python3 tetris.py 运行
- 必须能在标准 24x80 终端下正常运行
"""

TASK_MINESWEEPER = """\
用 Python 写一个扫雷 (Minesweeper) 游戏。

## 核心要求
1. 使用 curses 库实现终端 UI
2. 9x9 棋盘，10 个地雷随机分布
3. 方向键移动光标，空格揭示格子，F 键插/去旗
4. 揭示空格时递归展开周围空格（经典扫雷行为）
5. 踩雷失败、揭完非雷格子胜利

## UI 要求
- 棋盘有边框，每格显示状态：? (未揭)、F (旗)、数字 (周围雷数)、* (雷)
- 不同数字用不同颜色（1-8）
- 顶部显示剩余地雷数、计时器
- 底部显示操作提示和退出键（Q）
- 游戏结束（胜/败）显示覆盖层

## 技术要求
- 单文件实现，保存到 /tmp/minesweeper/minesweeper.py
- 代码结构清晰：Board 类、Cell 类、Game 类
- 可直接 python3 minesweeper.py 运行
- 必须能在标准 24x80 终端下正常运行
"""

TASKS = {
    "2048": (TASK_2048, "/tmp/game_2048", "game_2048.py"),
    "tetris": (TASK_TETRIS, "/tmp/tetris", "tetris.py"),
    "minesweeper": (TASK_MINESWEEPER, "/tmp/minesweeper", "minesweeper.py"),
}
