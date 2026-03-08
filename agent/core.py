"""
无垠智穹 Hani - LangGraph 核心状态机 (v5)

架构原则：
  - SSOT: AgentState 只存文件路径，不存内容。claude_node 每次实时读磁盘。
  - Provider 抽象: 通过 get_claude_provider() / get_gemini_provider() 注入，
    未来可无缝切换 API key 模式。
  - consult_gemini: 3轮内部对抗（Claude挑刺 × 2），有进度指示器。
  - JSON 解析: 使用 extract_json() 两段式防线，LLM 自纠正兜底。
  - SOUL + IDENTITY: 每次调用前注入 agent/SOUL.md + agent/IDENTITY.md，
    确立不可覆盖的价值铁律和角色身份。
  - Git 时间机器: claude_node 前自动 snapshot，验证失败时 git reset --hard 回退。
"""

import datetime
import os
import py_compile
import re
import sqlite3
import subprocess
import tempfile
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

import logging
from agent.cli_wrapper import extract_json, heartbeat_probe
from agent.git_ops import ensure_repo, snapshot, rollback

logger = logging.getLogger(__name__)
from agent.providers import get_claude_provider, get_gemini_provider

load_dotenv()

# ==========================================
# 配置常量
# ==========================================
SESSION_THREAD_ID = os.getenv("SESSION_THREAD_ID", "boss_bootstrap_session_01")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "cyber_bootstrap.db")
MAX_MESSAGES = 20        # 滑动窗口：防止 prompt 过长
PLAN_MAX_CHARS = 3000    # 架构文件：头部截取（首部最重要）
TASKS_MAX_CHARS = 2000   # 任务文件：尾部截取（最新任务最重要）
MAX_RETRIES = 2          # git 回退重试上限（超过后放行，避免死循环）
_TOMBSTONE_FILE = ".DO_NOT_REPEAT.md"  # 耻辱柱：不受 Git 控制的跨时空记忆

# consult_gemini 信号正则
_CONSULT_SIGNAL_RE = re.compile(r'\{"action"\s*:\s*"consult_gemini".*?\}', re.DOTALL)

# ==========================================
# 动态工具路由（Lazy Tool Loading）
# ==========================================
# 基础工具集：所有任务都带的最小工具
_BASE_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]

# 按关键词按需追加的工具
_TOOL_RULES: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"论文|paper|arXiv|arxiv|研究|publication", re.IGNORECASE), ["WebFetch", "WebSearch"]),
    (re.compile(r"搜索|search|网络|internet|url|http", re.IGNORECASE),       ["WebFetch", "WebSearch"]),
    (re.compile(r"notebook|jupyter|\.ipynb",               re.IGNORECASE), ["NotebookEdit"]),
]


def _select_tools(user_input: str) -> list[str]:
    """根据用户输入动态组装工具列表，默认只用基础工具集。"""
    tools = list(_BASE_TOOLS)
    for pattern, extra in _TOOL_RULES:
        if pattern.search(user_input):
            for t in extra:
                if t not in tools:
                    tools.append(t)
    return tools

# ==========================================
# Provider 单例（模块级，所有节点共用）
# ==========================================
_claude_provider = get_claude_provider()
_gemini_provider = get_gemini_provider()

# ==========================================
# SOUL + IDENTITY 加载（模块级单例）
# ==========================================
_AGENT_DIR = os.path.dirname(__file__)


def _load_persona_file(filename: str) -> str:
    """读取 agent/ 目录下的 persona 文件，找不到时返回空字符串。"""
    path = os.path.join(_AGENT_DIR, filename)
    if not os.path.exists(path):
        logger.warning(f"[Persona] 找不到 {filename}，跳过注入。")
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


_SOUL = _load_persona_file("SOUL.md")
_IDENTITY = _load_persona_file("IDENTITY.md")

# 合并为一个 persona 头部块，注入到每次 prompt 的最前面
_PERSONA_HEADER = "\n\n---\n\n".join(filter(None, [_SOUL, _IDENTITY]))


# ==========================================
# 1. 状态空间 (SSOT 版)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    gemini_context: str   # Gemini 3轮建议结果，或 __PENDING__topic|ctx 信号
    project_root: str     # Claude subprocess 的 cwd（多项目 session 隔离）
    project_meta: dict    # {"plan": "PLAN.md", "tasks": "TASKS.md"} — 只存路径！
    consult_count: int    # 每次用户 turn 内的 Gemini 咨询次数，防止死循环
    last_stable_commit: str  # claude_node 运行前的 git snapshot hash
    retry_count: int         # 当前 turn 已触发回退重试的次数
    rollback_reason: str     # 验证失败原因；非空时 claude_node 注入警告


# ==========================================
# 2. 操作规则提示词（与 SOUL/IDENTITY 分离）
# ==========================================
# 注意：SOUL 和 IDENTITY 在 claude_node 里动态拼接在最前面，
# 此处只保留"如何工作"的操作层规则。
_OPERATIONAL_PROMPT = """## 运行时环境（你的自我认知）

你正在运行于无垠智穹的 LangGraph 状态机中。这不是裸 Claude，这是一个完整的 Agent 框架：
- 你的每条回复都经过中间件处理，不是直接发给老板
- consult_gemini 信号管道已就绪：你输出 JSON → 系统自动路由给 Gemini → Gemini 3轮对抗后结果注入回你的下一轮 prompt
- 当你看到 [Gemini 首席架构师建议] 段落时，说明管道已完成，你直接基于建议回复老板即可
- 老板也可以用 @Gemini 关键词绕过你直接触发咨询

## 操作规则

1. 回答简明扼要，直接输出操作结果或请老板 Approve。
2. 遇到宏大架构规划或物理隔离问题，必须咨询 Gemini（首席架构师）。
   咨询方式：回复的**第一行**单独输出以下 JSON，其余什么都不写：
   {"action": "consult_gemini", "topic": "<问题>", "context": "<当前状态>"}
   系统会自动接管，无需你解释这个机制。
3. 用中文回复，代码和命令用英文。"""


# ==========================================
# 3. P0-B 文件读取（SSOT + 尾部截断）
# ==========================================
_IN_PROGRESS_RE = re.compile(r"^## In Progress", re.MULTILINE)


def _read_project_file(path: str, is_tasks: bool = False, max_chars: int = 3000) -> str:
    """
    实时读取磁盘文件，不信任 state 缓存。

    tasks 文件（is_tasks=True）：
      1. 用正则找 '## In Progress' 锚点，从锚点开始读
      2. 找不到则从尾部截取 max_chars（最新任务在文件末尾）

    plan 文件：从头截取 max_chars（架构首部最稳定、最重要）
    """
    if not path:
        return ""
    full_path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    if not os.path.exists(full_path):
        return ""

    with open(full_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    if is_tasks:
        match = _IN_PROGRESS_RE.search(content)
        if match:
            content = content[match.start():]
        # 尾部截取
        if len(content) > max_chars:
            content = "...(older tasks truncated)\n" + content[-max_chars:]
    else:
        # 头部截取
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...(rest truncated)"

    return content.strip()


def _build_project_section(state: AgentState) -> str:
    """根据 project_meta 实时读取文件，构建注入 prompt 的项目上下文段落。"""
    meta = state.get("project_meta") or {}
    root = state.get("project_root") or ""
    sections = []

    plan_file = meta.get("plan", "")
    if plan_file:
        path = os.path.join(root, plan_file) if root else plan_file
        content = _read_project_file(path, is_tasks=False, max_chars=PLAN_MAX_CHARS)
        if content:
            sections.append(f"[当前项目架构 — {plan_file}]\n{content}")

    tasks_file = meta.get("tasks", "")
    if tasks_file:
        path = os.path.join(root, tasks_file) if root else tasks_file
        content = _read_project_file(path, is_tasks=True, max_chars=TASKS_MAX_CHARS)
        if content:
            sections.append(f"[当前任务列表 — {tasks_file}（尾部/In Progress 截取）]\n{content}")

    return "\n\n".join(sections)


def _format_history(messages: list[BaseMessage], max_n: int) -> str:
    recent = messages[-max_n:] if len(messages) > max_n else messages
    lines = []
    for msg in recent:
        role = "老板" if isinstance(msg, HumanMessage) else "Hani"
        content = msg.content
        # 把历史里的 consult_gemini JSON 替换成摘要，防止 Claude 看到后模仿输出
        if isinstance(msg, AIMessage) and '"action"' in content and "consult_gemini" in content:
            content = "[Hani 触发了 Gemini 架构咨询]"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ==========================================
# 4. Claude 主控节点
# ==========================================
_GEMINI_MENTION_RE = re.compile(r"@[Gg]emini\s*", re.IGNORECASE)


def claude_node(state: AgentState) -> dict:
    latest_input = state["messages"][-1].content

    # ── @Gemini 组装路由 ──────────────────────────────────────────────
    # 老板消息含 @Gemini 时，不跳过 Claude，而是注入强制指令让 Hani 组装问题。
    # 这样 Hani 能带入完整项目上下文，Gemini 收到的是专业提问，不是原始话。
    if _GEMINI_MENTION_RE.search(latest_input):
        raw_topic = _GEMINI_MENTION_RE.sub("", latest_input).strip() or latest_input
        logger.info(f"[claude_node] @Gemini 触发，强制 Hani 组装咨询: topic={raw_topic!r}")
        # 让 latest_input 保持原样传入历史，但 prompt 末尾注入组装指令
        # （fallthrough 到下面的正常 Claude 调用流程）
        _gemini_override_topic = raw_topic
    else:
        _gemini_override_topic = None

    history_text = _format_history(state["messages"][:-1], MAX_MESSAGES)
    gemini_ctx = state.get("gemini_context", "")
    project_root = state.get("project_root", "") or None

    project_section = _build_project_section(state)

    gemini_section = (
        f"\n[Gemini 首席架构师建议（经3轮对抗验证）]\n{gemini_ctx}\n[建议结束]\n"
        f"⚠️ 你已收到 Gemini 的建议，请直接基于以上建议回复老板。"
        f"严禁再次输出 consult_gemini JSON，否则造成死循环。\n"
        if gemini_ctx and not gemini_ctx.startswith("__PENDING__")
        else ""
    )

    # ── Git 回退警告（上一次操作触发了熔断）────────────────────────────
    rollback_reason = state.get("rollback_reason", "")
    last_commit = state.get("last_stable_commit", "")
    rollback_warning = (
        f"\n⚠️ 【系统警告·时光倒流】\n"
        f"你上一次的操作触发了验证失败（原因：{rollback_reason}）。\n"
        f"系统已执行 git reset --hard，文件已恢复到 {last_commit[:8] if last_commit else '上一个稳定版本'}。\n"
        f"现在的工作目录是干净的。请换一种思路，不要重复同样的错误。\n"
        if rollback_reason
        else ""
    )

    # 耻辱柱注入：有历史失败案例时强行警告，防止时光机后失忆重蹈覆辙
    tombstone_raw = _read_tombstone(project_root or "")
    tombstone_section = (
        f"\n⛔ 【跨时空耻辱柱·绝对禁止重蹈】\n"
        f"以下是系统已经证明失败的方案，git 时光机已回滚它们，但这段记忆必须保留：\n"
        f"{tombstone_raw}\n"
        f"绝对不要重复以上任何模式。\n"
        if tombstone_raw
        else ""
    )

    # @Gemini 触发时，注入强制组装指令（覆盖正常的"当前指令"）
    if _gemini_override_topic:
        current_instruction = (
            f"老板: {latest_input}\n\n"
            f"【系统指令·强制咨询】老板明确要求咨询 Gemini 首席架构师，关于：{_gemini_override_topic}\n"
            f"你的任务：结合当前项目上下文，把这个问题组装成专业提问，"
            f"**第一行且只有第一行**输出以下 JSON，不要任何前缀或解释：\n"
            f'{{"action": "consult_gemini", "topic": "<提炼后的问题>", "context": "<相关项目状态>"}}'
        )
    else:
        current_instruction = f"老板: {latest_input}"

    prompt = f"""{_PERSONA_HEADER}

---

{_OPERATIONAL_PROMPT}
{rollback_warning}{tombstone_section}
[历史对话]
{history_text}

{project_section}
{gemini_section}
[当前指令]
{current_instruction}

Hani:"""

    debug = os.getenv("DEBUG", "").lower() in ("1", "true")
    if debug:
        logger.debug(f"[claude_node] prompt length={len(prompt)} chars")

    dynamic_tools = _select_tools(latest_input)
    if debug:
        logger.debug(f"[claude_node] tools={dynamic_tools}")
    raw_output = _claude_provider.complete(prompt, cwd=project_root, tools=dynamic_tools)

    if debug:
        logger.debug(f"[claude_node] raw_output={raw_output[:300]!r}")

    # 信号检测：consult_gemini JSON 必须出现在输出的最前面（第一行）。
    # 全文搜索会导致 Claude 解释自己行为时（引用 JSON 格式）误触发。
    first_line = raw_output.lstrip().split("\n")[0].strip()
    if first_line.startswith("{") and "consult_gemini" in first_line:
        signal = extract_json(first_line, llm_provider=_claude_provider)
        if signal and signal.get("action") == "consult_gemini":
            topic = signal.get("topic", "")
            context = signal.get("context", "")
            if debug:
                logger.debug(f"[claude_node] → consult_gemini: topic={topic!r}")
            return {
                "messages": [AIMessage(content=raw_output)],
                "gemini_context": f"__PENDING__{topic}|{context}",
            }

    return {
        "messages": [AIMessage(content=raw_output)],
        "gemini_context": "",
        "consult_count": 0,   # 完成本轮 turn，重置计数器
        "rollback_reason": "", # 回复成功后清零（下一轮不再注入警告）
        "retry_count": 0,      # 重置重试计数器
    }


# ==========================================
# 5. Gemini 战略顾问节点（3轮对抗）
# ==========================================
def gemini_node(state: AgentState) -> dict:
    """
    3轮内部对抗性咨询（不新增 LangGraph 节点，内部同步循环）：
      Round 1: Gemini 首次回答
      Round 2: Claude 挑刺 → Gemini 修订
      Round 3: Claude 深度挑刺 → Gemini 最终建议
    全程打印进度，缓解"系统假死焦虑"。
    """
    pending = state.get("gemini_context", "")
    if pending.startswith("__PENDING__"):
        rest = pending[len("__PENDING__"):]
        parts = rest.split("|", 1)
        topic = parts[0]
        context = parts[1] if len(parts) > 1 else ""
    else:
        topic = pending
        context = ""

    project_root = state.get("project_root", "") or None

    gemini_system = (
        "你是无垠智穹的首席架构师（Gemini）。Hani向你咨询架构问题。\n"
        "请根据\"5090显存绝对清场\"和\"物理隔离\"铁律，给出极致架构建议。\n"
        "直接输出建议，不需要客套。"
    )

    # ── Round 1: Gemini 首次回答 ──────────────────
    print(f"\n[咨询 1/3] 🔄 Gemini 首次回答中...", flush=True)
    g1 = _gemini_provider.complete(
        f"{gemini_system}\n\n问题：{topic}\n当前上下文：{context}"
    )

    # ── Round 2: Claude 挑刺 → Gemini 修订 ────────
    print(f"[咨询 2/3] 🔍 Claude 挑刺中...", flush=True)
    critique = _claude_provider.complete(
        f"以下是 Gemini 架构师对「{topic}」的建议，请找出其中的逻辑漏洞、"
        f"遗漏的边界情况或过于理想化的假设（简明扼要，3点以内）：\n\n{g1}",
        cwd=project_root,
    )

    print(f"[咨询 2/3] 🔄 Gemini 修订中...", flush=True)
    g2 = _gemini_provider.complete(
        f"{gemini_system}\n\n原始问题：{topic}\n"
        f"你的第一轮建议：\n{g1}\n\n"
        f"Hani的质疑：\n{critique}\n\n"
        "请针对上述质疑修订你的建议："
    )

    # ── Round 3: Claude 深度挑刺 → Gemini 最终 ────
    print(f"[咨询 3/3] ⚡ Claude 深度挑刺...", flush=True)
    nitpick = _claude_provider.complete(
        f"对以下修订后的架构建议进行最后一轮深度审查。\n"
        f"重点关注：实施复杂度、潜在的单点故障、与现有 WSL/5090 环境的兼容性。\n\n"
        f"{g2}",
        cwd=project_root,
    )

    print(f"[咨询 3/3] 🏁 Gemini 最终建议...", flush=True)
    g_final = _gemini_provider.complete(
        f"{gemini_system}\n\n原始问题：{topic}\n"
        f"经过两轮修订后的建议：\n{g2}\n\n"
        f"Hani 的最终审查意见：\n{nitpick}\n\n"
        "请给出你的最终建议（这将直接被 Hani 采纳执行）："
    )

    print(f"[咨询完成] ✅ Gemini 3轮建议已就绪\n", flush=True)
    return {
        "gemini_context": g_final,
        "consult_count": state.get("consult_count", 0) + 1,
    }


# ==========================================
# 6. Git 时间机器节点
# ==========================================

def git_snapshot_node(state: AgentState) -> dict:
    """
    在 claude_node 运行前执行：对 project_root 做 git snapshot，
    记录 commit hash 到 last_stable_commit。
    project_root 为空时静默跳过。
    """
    root = state.get("project_root") or ""
    if not root or not os.path.isdir(root):
        return {}
    ensure_repo(root)
    h = snapshot(root, "Auto-snapshot before Hani task")
    if h:
        logger.info(f"[git_snapshot] {h[:8]} @ {root}")
    return {"last_stable_commit": h or ""}


# ==========================================
# VRAM 大清洗节点（硬件级显存释放）
# ==========================================
def vram_flush_node(state: AgentState) -> dict:
    """
    物理沙盒执行后（无论成功或失败）都必须经过此节点。
    扫描所有仍占用 GPU 的进程（包括 Docker 僵尸），强制物理级杀死。

    为什么用 fuser -k -9 /dev/nvidia* 而不是 docker kill？
    CUDA Illegal Memory Access 可以把 NVIDIA 驱动卡死，此时容器虽然被
    LangGraph 超时 kill，但 VRAM 依然被 <defunct> 僵尸进程占住。
    fuser 直接对设备文件下令，绕过所有软件层，是唯一可靠的清洗手段。

    前提：宿主机需配置 sudo 免密（仅限此命令）。
    """
    logger.info("[vram_flush] 开始扫描 GPU 占用...")
    try:
        # 查询当前占用 GPU 的所有 PID
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            # nvidia-smi 不可用（无 GPU 环境），静默跳过
            logger.debug("[vram_flush] nvidia-smi 不可用，跳过")
            return {}

        pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not pids:
            logger.info("[vram_flush] ✅ GPU 干净，无残留进程")
            return {}

        logger.warning(f"[vram_flush] ⚠️  检测到 {len(pids)} 个 GPU 残留进程: {pids}")

        # 物理级强杀：对 /dev/nvidia* 设备文件执行 fuser -k -9
        kill_result = subprocess.run(
            ["sudo", "fuser", "-k", "-9", "/dev/nvidia*"],
            capture_output=True, text=True, timeout=15,
        )
        if kill_result.returncode == 0:
            logger.info("[vram_flush] ✅ GPU 大清洗完成，5090 归零")
        else:
            logger.error(f"[vram_flush] ❌ 大清洗失败: {kill_result.stderr.strip()}")

    except FileNotFoundError:
        logger.debug("[vram_flush] nvidia-smi / fuser 不存在，跳过（非 GPU 环境）")
    except subprocess.TimeoutExpired:
        logger.error("[vram_flush] nvidia-smi 超时，跳过本次清洗")

    return {}


# ==========================================
# 耻辱柱（Tombstone）：跨时空记忆，不受 Git 控制
# ==========================================
def _write_tombstone(project_root: str, reason: str, bad_output: str) -> None:
    """
    git reset --hard 之前调用。
    把失败的输出片段和原因追加写到 .DO_NOT_REPEAT.md，
    该文件不受 Git 管控，时光机不会抹掉它，确保下次开局能读到前车之鉴。
    """
    if not project_root or not os.path.isdir(project_root):
        return
    tombstone_path = os.path.join(project_root, _TOMBSTONE_FILE)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 截取坏输出的前 500 字符，避免文件无限膨胀
    snippet = bad_output[:500].strip() if bad_output else "(空输出)"
    entry = (
        f"\n---\n"
        f"## [{ts}] 失败案例（已回滚）\n"
        f"**失败原因：** {reason}\n\n"
        f"**问题输出片段（前500字）：**\n```\n{snippet}\n```\n"
    )
    with open(tombstone_path, "a", encoding="utf-8") as f:
        f.write(entry)
    logger.info(f"[tombstone] 已写入耻辱柱: {tombstone_path}")


def _read_tombstone(project_root: str) -> str:
    """
    读取耻辱柱内容，注入 claude_node 的 prompt。
    文件不存在则返回空字符串。
    """
    if not project_root:
        return ""
    tombstone_path = os.path.join(project_root, _TOMBSTONE_FILE)
    if not os.path.exists(tombstone_path):
        return ""
    with open(tombstone_path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return ""
    # 只取最近 2000 字符，防止 prompt 被历史失败案例撑爆
    return content[-2000:]


def _check_failure(last_output: str, project_root: str) -> str:
    """
    简单规则验证 claude_node 输出是否可接受。
    返回失败原因字符串（非空 = 失败），或空字符串（通过）。
    后续可替换为蓝皮书中的 Level 2/3 熔断标准。
    """
    if not last_output or len(last_output.strip()) < 10:
        return "输出为空或过短"
    if last_output.lstrip().startswith("[错误]"):
        return f"输出包含错误前缀: {last_output[:80]}"
    if "CLI 超时" in last_output or "已强制终止" in last_output:
        return f"CLI 超时: {last_output[:80]}"

    # 检查 project_root 里被修改的 .py 文件是否有语法错误
    if project_root and os.path.isdir(project_root):
        for dirpath, _, filenames in os.walk(project_root):
            # 跳过 .git、node_modules、__pycache__
            if any(skip in dirpath for skip in (".git", "node_modules", "__pycache__")):
                continue
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    py_compile.compile(fpath, doraise=True)
                except py_compile.PyCompileError as e:
                    return f"Python 语法错误: {e}"

    return ""


def validate_node(state: AgentState) -> dict:
    """
    在 claude_node 之后运行：检查输出质量。
    通过 → rollback_reason = ""
    失败 → rollback_reason = 原因字符串，触发 git_rollback_node
    retry_count 达到上限时强制放行（避免死循环）。
    """
    retry = state.get("retry_count", 0)
    if retry >= MAX_RETRIES:
        logger.warning(f"[validate] retry_count={retry} 已达上限，放行")
        return {"rollback_reason": ""}

    last_output = state["messages"][-1].content
    root = state.get("project_root") or ""
    reason = _check_failure(last_output, root)
    if reason:
        logger.warning(f"[validate] 验证失败: {reason}")
    return {"rollback_reason": reason}


def git_rollback_node(state: AgentState) -> dict:
    """
    验证失败时执行：git reset --hard 回到 last_stable_commit，
    递增 retry_count，清空 rollback_reason（警告已写入 state，
    claude_node 下次运行时会读取并注入 prompt）。
    """
    root = state.get("project_root") or ""
    commit = state.get("last_stable_commit", "")
    reason = state.get("rollback_reason", "")

    if root and commit:
        # 回滚前先写耻辱柱——时光机不能抹掉教训
        bad_output = state["messages"][-1].content if state.get("messages") else ""
        _write_tombstone(root, reason, bad_output)

        ok = rollback(root, commit)
        if not ok:
            logger.error(f"[git_rollback] 回退失败，commit={commit[:8]!r}")
    else:
        logger.warning("[git_rollback] project_root 或 commit hash 为空，跳过 git 操作")

    return {
        "retry_count": state.get("retry_count", 0) + 1,
        # rollback_reason 保留给 claude_node 读取，claude_node 回复后清零
    }


# ==========================================
# 7. 路由逻辑
# ==========================================
_MAX_CONSULTS_PER_TURN = 1  # 每次用户 turn 最多咨询 Gemini 1 次


def _validate_route(state: AgentState) -> str:
    """
    validate_node 之后的路由：
      - 验证失败 → rollback（git 回退 → 重试 claude_agent）
      - 需要咨询 Gemini → consult_gemini
      - 其他 → end
    """
    if state.get("rollback_reason"):
        return "rollback"
    ctx = state.get("gemini_context", "")
    count = state.get("consult_count", 0)
    if ctx.startswith("__PENDING__"):
        if count >= _MAX_CONSULTS_PER_TURN:
            logger.warning(f"[route] consult_count={count} 已达上限，强制结束，防止死循环")
            return "end"
        return "consult_gemini"
    return "end"


# ==========================================
# 8. 构建工作流图
# ==========================================
def _build_graph() -> StateGraph:
    """
    新图拓扑（含 Git 时间机器）：

    入口
      ↓
    git_snapshot   ← 执行前快照
      ↓
    claude_agent   ← 主逻辑（含 @Gemini 直通 + 回退警告注入）
      ↓
    validate       ← 质检
      ├── rollback → git_rollback → claude_agent（带警告重试）
      ├── consult_gemini → gemini_advisor → claude_agent
      └── end
    """
    g = StateGraph(AgentState)
    g.add_node("git_snapshot", git_snapshot_node)
    g.add_node("claude_agent", claude_node)
    g.add_node("validate", validate_node)
    g.add_node("gemini_advisor", gemini_node)
    g.add_node("git_rollback", git_rollback_node)
    g.add_node("vram_flush", vram_flush_node)  # 硬件级清洗，沙盒完成后必经

    g.set_entry_point("git_snapshot")
    g.add_edge("git_snapshot", "claude_agent")
    g.add_edge("claude_agent", "validate")
    g.add_conditional_edges(
        "validate",
        _validate_route,
        # end 路径也经过 vram_flush，确保 GPU 僵尸进程被清理后才真正结束
        {"rollback": "git_rollback", "consult_gemini": "gemini_advisor", "end": "vram_flush"},
    )
    g.add_edge("vram_flush", END)
    g.add_edge("git_rollback", "claude_agent")
    g.add_edge("gemini_advisor", "claude_agent")
    return g


# ==========================================
# 8. 单例引擎
# ==========================================
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        heartbeat_probe()
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()
        memory = SqliteSaver(conn)
        _engine = _build_graph().compile(checkpointer=memory)
    return _engine


def get_config() -> dict:
    return {"configurable": {"thread_id": SESSION_THREAD_ID}}


# ==========================================
# 9. Session 管理工具
# ==========================================
def session_stats(thread_id: str | None = None) -> dict:
    """
    返回指定 thread_id（默认当前）的 session 统计。
    {"thread_id": ..., "message_count": ..., "db_size_kb": ...}
    """
    tid = thread_id or SESSION_THREAD_ID
    db = os.path.abspath(DB_PATH)
    stats = {"thread_id": tid, "message_count": 0, "db_size_kb": 0}
    if not os.path.exists(db):
        return stats
    stats["db_size_kb"] = round(os.path.getsize(db) / 1024, 1)
    try:
        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (tid,)
        ).fetchone()
        conn.close()
        stats["message_count"] = rows[0] if rows else 0
    except Exception:
        pass
    return stats


def session_compact(thread_id: str | None = None, keep_last: int = 20) -> int:
    """
    Compact：只保留最近 keep_last 条 checkpoint，删除更早的记录。
    返回删除的行数。
    """
    tid = thread_id or SESSION_THREAD_ID
    db = os.path.abspath(DB_PATH)
    if not os.path.exists(db):
        return 0
    try:
        conn = sqlite3.connect(db, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        # checkpoint_id 是递增的，保留最大的 keep_last 个
        deleted = conn.execute(
            """
            DELETE FROM checkpoints
            WHERE thread_id = ?
              AND checkpoint_id NOT IN (
                  SELECT checkpoint_id FROM checkpoints
                  WHERE thread_id = ?
                  ORDER BY checkpoint_id DESC
                  LIMIT ?
              )
            """,
            (tid, tid, keep_last),
        ).rowcount
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
        conn.close()
        # 引擎缓存失效，下次 get_engine() 重建
        global _engine
        _engine = None
        logger.info(f"[session_compact] thread={tid!r} 删除 {deleted} 条，保留最近 {keep_last} 条")
        return deleted
    except Exception as e:
        logger.error(f"[session_compact] 失败: {e}")
        return 0


def session_reset(thread_id: str | None = None) -> int:
    """
    Reset：删除指定 thread_id 的全部 checkpoint，从零开始。
    返回删除的行数。
    """
    tid = thread_id or SESSION_THREAD_ID
    db = os.path.abspath(DB_PATH)
    if not os.path.exists(db):
        return 0
    try:
        conn = sqlite3.connect(db, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")  # 等待锁最多 10 秒
        deleted = conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (tid,)
        ).rowcount
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
        conn.close()
        global _engine
        _engine = None
        logger.info(f"[session_reset] thread={tid!r} 已清空，删除 {deleted} 条")
        return deleted
    except Exception as e:
        logger.error(f"[session_reset] 失败: {e}")
        return 0
