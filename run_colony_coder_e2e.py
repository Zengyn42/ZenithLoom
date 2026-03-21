#!/usr/bin/env python3
"""
Colony Coder 完整 E2E 测试 — 真实 LLM 调用。

流程：plan → execute → integrate
所有 LLM 节点使用真实 API 调用（Claude SDK + Gemini CLI）。
Debug 模式全开，输出完整 state snapshot。
"""

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)

# 注册 state schema
import blueprints.functional_graphs.colony_coder.state  # noqa: F401

from framework.agent_loader import EntityLoader
from framework.debug import set_debug, push_graph_scope, pop_graph_scope
from langchain_core.messages import HumanMessage


async def main():
    set_debug(True)
    print("🔧 Colony Coder 完整 E2E 测试 — Debug ON\n", flush=True)

    loader = EntityLoader(Path("blueprints/functional_graphs/colony_coder"))
    graph = await loader.build_graph(checkpointer=None)

    task = (
        "用 Python 写一个 AI vs AI 双蛇竞技场（AI Snake Arena）。\n"
        "这个游戏的核心目的：让两个不同的 AI 算法各控制一条蛇，在同一竞技场中对战，观察哪个 AI 更强。\n"
        "\n"
        "## 游戏规则\n"
        "1. 使用 curses 库实现终端 UI\n"
        "2. 两条蛇同时出现，全部由 AI 控制（无人类玩家），玩家只是观战者\n"
        "3. 屏幕上同时存在 3 个食物，蛇吃到食物后身体变长\n"
        "4. 蛇撞墙或撞自己则死亡\n"
        "5. 碰撞对战规则：\n"
        "   a. 头对头相撞：长的蛇获胜，短的蛇死亡。等长则同归于尽。\n"
        "   b. 蛇A的头碰到蛇B的身体：比较蛇A总长度与蛇B从被碰位置到尾部的长度。\n"
        "      - 蛇A长度 > 蛇B被碰段长度 → 蛇A获胜，蛇B从被碰位置截断\n"
        "      - 蛇A长度 <= 蛇B被碰段长度 → 蛇A被消灭\n"
        "6. 最后存活的蛇获胜\n"
        "\n"
        "## 两个 AI 算法（必须严格按照以下描述实现）\n"
        "\n"
        "### AI-Alpha（绿蛇 @#，左侧出生）\n"
        "算法特点：BFS寻路 + 全图flood fill空间评估 + 12%随机偏移\n"
        "- 构建障碍物集合时，移除双方蛇尾（预判尾巴会移走）\n"
        "- 用 BFS 寻找到食物的最短路径\n"
        "- 如果找到路径，以 88% 概率走最优方向，12% 概率随机选其他安全方向（增加不可预测性）\n"
        "- 找不到食物路径时，对每个安全方向做完整 flood fill，选择可达空间最大的方向\n"
        "- 没有安全方向时保持当前方向（必死）\n"
        "\n"
        "### AI-Beta（红蛇 &+，右侧出生）\n"
        "算法特点：BFS寻路 + 限深flood fill + 中心偏好 + 主动进攻\n"
        "- 构建障碍物集合时，跳过自身头和尾、对手尾\n"
        "- 用 BFS 寻找最近食物（场上有3个食物，寻最近的）\n"
        "- 找到路径后，验证该方向是否安全（不会立即死亡 + 下一步至少有1个逃生路径）\n"
        "- 安全评分机制：对每个方向计算 flood fill（限深30）可达格数作为基础分\n"
        "- 中心偏好：离棋盘中心越近得分越高（减少曼哈顿距离 * 0.1）\n"
        "- 主动进攻：如果碰到对方身体且自身长度 > 对方尾段长度，该方向加5分\n"
        "- 不能180度掉头\n"
        "\n"
        "## UI 要求\n"
        "- 顶部状态栏显示：AI-Alpha 分数/长度 | AI-Beta 分数/长度 | 当前帧数\n"
        "- 游戏区域有边框\n"
        "- 食物用 * 显示（黄色）\n"
        "- AI-Alpha 绿色（头@身体#），AI-Beta 红色（头&身体+）\n"
        "- 游戏结束显示获胜者、双方最终分数、存活帧数\n"
        "- 按 Q 退出，按 R 重新开始\n"
        "- 按 +/- 调节游戏速度\n"
        "- 帧率默认 ~10 FPS\n"
        "\n"
        "## 技术要求\n"
        "- 单文件实现，保存到 /tmp/snake_arena_v2/snake_arena.py\n"
        "- 代码结构清晰，两个 AI 分别是独立的类（AIAlpha 和 AIBeta）\n"
        "- 可直接 python3 snake_arena.py 运行\n"
        "- 碰撞检测在移动前用 snapshot 判定，避免顺序偏差\n"
        "\n"
        "## ⚠️ 关键注意事项（必须遵守）\n"
        "1. 安全方向检查必须包含自咬检测：判断一个方向是否安全时，必须检查下一步是否会撞到自己的身体。\n"
        "   不能因为排除了自身蛇就完全跳过自身碰撞检查。正确做法：排除自身蛇尾（因为尾巴会移走），但自身其余身体必须作为障碍物。\n"
        "2. BFS 寻路必须返回第一步方向：BFS 从蛇头出发寻食物时，返回的方向必须是从蛇头出发的第一步方向（不是到达食物时的最后一步方向）。\n"
        "   正确做法：BFS 队列中每个元素记录 (当前位置, 第一步方向)，找到食物后返回第一步方向。\n"
        "3. 障碍物集合要用 set 预计算，不要在 BFS/flood fill 的每一步都调用遍历全蛇体的函数，否则性能极差。\n"
        "4. 两条蛇都必须能存活 100+ tick，长度增长到 10+。如果某条蛇在 30 tick 内就自咬死亡，说明代码有 bug。"
    )

    init_state = {
        "messages": [HumanMessage(content=task)],
    }

    push_graph_scope("colony_coder_e2e")
    try:
        print("🚀 Starting colony_coder (plan → execute → integrate)\n")
        print("=" * 70, flush=True)

        last_state = {}
        async for event in graph.astream(init_state, stream_mode="updates"):
            for node_id, update in event.items():
                if node_id in ("__start__", "__end__"):
                    continue
                last_state.update(update)

                # 打印节点输出摘要
                msgs = update.get("messages", [])
                for msg in msgs:
                    content = getattr(msg, "content", "")
                    if not content:
                        continue
                    lines = content.split("\n")
                    print(f"\n┌─ [{node_id}]", flush=True)
                    for line in lines[:30]:
                        print(f"│ {line}", flush=True)
                    if len(lines) > 30:
                        print(f"│ ... ({len(content)} chars total)", flush=True)
                    print(f"└─", flush=True)

                # 关键 state 变化
                tasks = update.get("tasks")
                if tasks:
                    print(f"  📦 tasks: {len(tasks)} items", flush=True)
                    for t in tasks:
                        if isinstance(t, dict):
                            print(f"     {t.get('id','?')}: {str(t.get('description',''))[:80]}", flush=True)

                eo = update.get("execution_order")
                if eo:
                    print(f"  📋 execution_order: {eo}", flush=True)

                completed = update.get("completed_tasks")
                if completed:
                    print(f"  ✅ completed_tasks: {completed}", flush=True)

                files = update.get("final_files")
                if files:
                    print(f"  📁 final_files: {files}", flush=True)

                success = update.get("success")
                if success is not None:
                    print(f"  {'✅' if success else '❌'} success: {success}", flush=True)

                abort = update.get("abort_reason", "")
                if abort:
                    print(f"\n  🛑 ABORT: {abort}", flush=True)

        print("\n" + "=" * 70, flush=True)

        # ── 最终结果 ──
        print(f"\n📋 Final Results:", flush=True)
        print(f"   success:         {last_state.get('success')}", flush=True)
        print(f"   abort_reason:    {last_state.get('abort_reason') or '(none)'}", flush=True)
        print(f"   tasks:           {len(last_state.get('tasks', []))} items", flush=True)
        print(f"   completed_tasks: {last_state.get('completed_tasks', [])}", flush=True)
        print(f"   final_files:     {last_state.get('final_files', [])}", flush=True)

        # ── 检查生成的文件 ──
        working_dir = last_state.get("working_directory", "/tmp/snake_arena_v2")
        # Find any .py files (game might be snake_battle.py or similar)
        wd = Path(working_dir)
        py_files = list(wd.glob("*.py")) if wd.exists() else []
        snake_path = py_files[0] if py_files else wd / "snake_battle.py"
        if snake_path.exists():
            content = snake_path.read_text(encoding="utf-8")
            print(f"\n🐍 {snake_path.name} exists ({len(content)} chars, {len(content.splitlines())} lines)")
            # 语法检查
            try:
                compile(content, str(snake_path), "exec")
                print(f"   ✅ Syntax check: PASS", flush=True)
            except SyntaxError as e:
                print(f"   ❌ Syntax check: FAIL — {e}", flush=True)
        else:
            print(f"\n❌ snake.py not found at {snake_path}", flush=True)
            # 看看 working_dir 下有什么
            wd = Path(working_dir)
            if wd.exists():
                files = list(wd.rglob("*"))
                print(f"   Files in {working_dir}: {[str(f.relative_to(wd)) for f in files]}", flush=True)

    finally:
        pop_graph_scope()

    # ── Debug logs ──
    from datetime import datetime
    date = datetime.now().strftime("%Y-%m-%d")
    log_base = Path("logs") / date / "colony_coder_e2e"
    print(f"\n📁 Debug logs: {log_base}/", flush=True)
    if log_base.exists():
        for p in sorted(log_base.rglob("*.md")):
            rel = p.relative_to(log_base)
            size = p.stat().st_size
            print(f"   {rel} ({size:,} bytes)", flush=True)
    else:
        print("   (no logs generated)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
