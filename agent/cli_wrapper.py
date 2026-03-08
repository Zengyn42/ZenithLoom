"""
无垠智穹 0号管家 - CLI 劫持核心层 (v2)

P0 防御机制：
  1. ANSI 清洗器        — 剥离颜色代码、控制字符
  2. Stream Watchdog    — 线程实时扫描交互提示词，匹配即 kill
  3. Timeout + Kill     — Popen + communicate(timeout) + 显式 proc.kill()
  4. stdin=DEVNULL      — 彻底断开 stdin，防进程等待人类输入
  5. 心跳探针           — 定期 ping，检测 CLI 是否存活
"""

import io
import logging
import os
import random
import re
import subprocess
import time
import threading
from subprocess import DEVNULL, PIPE

logger = logging.getLogger(__name__)

# ==========================================
# P0 防御 ①：ANSI 清洗器
# ==========================================
_ANSI_ESCAPE = re.compile(
    r"""
    \x1B
    (?:
        [@-Z\\-_]
        | \[[0-?]*[ -/]*[@-~]
        | \][^\x07]*\x07
    )
    """,
    re.VERBOSE,
)
_MARKDOWN_FENCE = re.compile(r"```[\w]*\n?|```")


def clean_output(raw: str) -> str:
    return _ANSI_ESCAPE.sub("", raw).strip()


def strip_json_fence(text: str) -> str:
    return _MARKDOWN_FENCE.sub("", text).strip()


# ==========================================
# P0 防御 ②：Stream Watchdog 交互词黑名单
# ==========================================
_INTERACTIVE_KEYWORDS = re.compile(
    r"\[y/N\]|\[Y/n\]|\(y/n\)|password:|Press Enter|Agree\?|proceed\?|"
    r"Do you want to|Continue\?|Are you sure|Enter passphrase",
    re.IGNORECASE,
)


class SubprocessInteractiveError(RuntimeError):
    """CLI 子进程输出了需要人类交互的提示词，已被强制终止。"""


def _stream_reader(proc: subprocess.Popen, buf: io.StringIO, kill_event: threading.Event):
    """
    独立线程：逐行读取 stdout+stderr，实时扫描交互提示词。
    发现危险词 → 设置 kill_event，主线程负责 kill 进程。
    """
    for line in proc.stdout:
        buf.write(line)
        if _INTERACTIVE_KEYWORDS.search(line):
            logger.warning(f"[Watchdog] 检测到交互提示词: {line.strip()!r}")
            kill_event.set()
            return
    # 也消费 stderr（防止 PIPE buffer 满导致阻塞）
    for line in proc.stderr:
        buf.write("")  # stderr 不计入结果，只消费


# ==========================================
# P0 防御 ⑥：拟人化退避抖动（Anti-Bot Jitter）
# ==========================================
# 每次拉起 CLI 前强制随机等待，模拟人类反应时间，防触发 WAF / Cloudflare 速率限制。
_JITTER_MIN = 3.0   # 秒，正常调用下限
_JITTER_MAX = 7.0   # 秒，正常调用上限
_BACKOFF_BASE = 15.0  # 秒，出错后第一次等待
_BACKOFF_MULT = 2.0   # 指数退避乘数


def _human_jitter(backoff_attempt: int = 0) -> None:
    """
    调用 CLI 前的拟人化等待。
    backoff_attempt=0: 正常调用，随机 3~7 秒
    backoff_attempt=N: 出错重试，等待 base × mult^(N-1) 秒（指数退避）
    """
    if backoff_attempt <= 0:
        delay = random.uniform(_JITTER_MIN, _JITTER_MAX)
        logger.debug(f"[Jitter] 拟人等待 {delay:.1f}s")
    else:
        delay = _BACKOFF_BASE * (_BACKOFF_MULT ** (backoff_attempt - 1))
        logger.warning(f"[Jitter] 退避等待 {delay:.0f}s（第{backoff_attempt}次重试）")
    time.sleep(delay)


def _run_watched(cmd: list[str], env: dict, cwd: str | None, timeout: int,
                 _backoff: int = 0) -> str:
    """
    核心执行器：Popen + Stream Watchdog + Timeout Kill。
    返回清洗后的 stdout 文本，或在异常情况下返回错误字符串。
    _backoff: 内部退避层级（0=正常, 1+=重试）
    """
    _human_jitter(backoff_attempt=_backoff)
    kill_event = threading.Event()
    buf = io.StringIO()

    proc = subprocess.Popen(
        cmd,
        stdout=PIPE,
        stderr=PIPE,
        stdin=DEVNULL,          # P0: 彻底断开 stdin
        text=True,
        env=env,
        cwd=cwd or None,
    )

    reader = threading.Thread(target=_stream_reader, args=(proc, buf, kill_event), daemon=True)
    reader.start()

    try:
        reader.join(timeout=timeout)
    finally:
        pass

    if kill_event.is_set():
        proc.kill()
        proc.wait()
        raise SubprocessInteractiveError(
            "CLI 子进程输出了交互提示词（如 [y/N], password: 等），已强制终止。"
            "可能是依赖包更新触发了人工确认。"
        )

    if reader.is_alive():
        # 超时：进程还在跑，强制杀掉
        proc.kill()
        proc.wait()
        reader.join(timeout=2)
        logger.error(f"[Watchdog] 超时（>{timeout}s），已强制 kill")
        return f"[错误] CLI 超时（{timeout}s），已强制终止。"

    proc.wait()
    return clean_output(buf.getvalue())


# ==========================================
# Token 计数器（进程级累计）
# ==========================================
import json as _json

_token_stats = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "calls": 0,
}


def get_token_stats() -> dict:
    """返回当前进程的累计 token 使用统计。"""
    return dict(_token_stats)


def reset_token_stats() -> None:
    for k in _token_stats:
        _token_stats[k] = 0


_last_session_id: str = ""


def get_last_session_id() -> str:
    return _last_session_id


def _parse_claude_json(raw: str) -> tuple[str, dict, str]:
    """
    解析 --output-format json 的响应。
    返回 (text_result, usage_dict, session_id)。
    解析失败时 fallback 到原始文本。
    """
    global _last_session_id
    try:
        data = _json.loads(raw)
        text = data.get("result", "") or data.get("content", "") or raw
        usage = data.get("usage", {})
        sid = data.get("session_id", "")
        if sid:
            _last_session_id = sid
        return clean_output(str(text)), usage, sid
    except (_json.JSONDecodeError, Exception):
        return clean_output(raw), {}, ""


# ==========================================
# Claude CLI 调用
# ==========================================
def call_claude(
    prompt: str,
    cwd: str | None = None,
    tools: list[str] | None = None,
    skip_permissions: bool = False,
    timeout: int = 120,
    resume_session_id: str = "",
) -> str:
    """
    调用 `claude -p <prompt>`。
    - resume_session_id: 非空时用 --resume 续接已有 session，历史由 Claude 管理
    """
    cmd = ["claude", "-p", prompt, "--output-format", "json"]
    if resume_session_id:
        cmd += ["--resume", resume_session_id]
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    if tools:
        cmd += ["--allowedTools", ",".join(tools)]

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_SESSION", None)

    try:
        raw = _run_watched(cmd, env, cwd, timeout)
    except SubprocessInteractiveError as e:
        logger.error(f"[claude CLI] {e}")
        return f"[错误] Claude 被交互提示中断: {e}"
    except FileNotFoundError:
        return "[错误] claude 命令不存在，请确认已安装 Claude Code CLI。"

    text, usage, _ = _parse_claude_json(raw)

    # 累计 token 统计
    _token_stats["calls"] += 1
    for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        _token_stats[key] += usage.get(key, 0)

    if usage:
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        cache_r = usage.get("cache_read_input_tokens", 0)
        cache_c = usage.get("cache_creation_input_tokens", 0)
        sid_short = _last_session_id[:8] if _last_session_id else "new"
        logger.info(f"[tokens] sid={sid_short} in={inp} out={out} cache_read={cache_r} cache_create={cache_c} | 累计 calls={_token_stats['calls']}")

    return text


# ==========================================
# Gemini CLI 调用
# ==========================================
def call_gemini(
    prompt: str,
    extensions: list[str] | None = None,
    yolo: bool = False,
    timeout: int = 60,
) -> str:
    """
    调用 `gemini -p <prompt>`。
    - extensions: 传递给 -e（可多个）
    - yolo: --approval-mode yolo（自动 approve 所有工具调用）
    """
    cmd = ["gemini", "-p", prompt]
    if yolo:
        cmd += ["--approval-mode", "yolo"]
    for ext in (extensions or []):
        cmd += ["-e", ext]

    env = os.environ.copy()

    try:
        return _run_watched(cmd, env, cwd=None, timeout=timeout)
    except SubprocessInteractiveError as e:
        logger.warning(f"[gemini CLI] {e}")
        return f"[Gemini 被交互提示中断，无法获取建议: {e}]"
    except FileNotFoundError:
        return "[Gemini CLI 不可用。]"


# ==========================================
# P0-C: JSON 两段式自纠正提取
# ==========================================
_JSON_BROAD_RE = re.compile(r"\{.*?\}", re.DOTALL)
import json


def extract_json(text: str, llm_provider=None, max_retries: int = 3) -> dict | None:
    """
    两段式 JSON 提取：
    Step 1 — 宽松正则抠出 {...}，尝试 json.loads()
    Step 2 — 解析失败时，把 traceback 甩回给 llm_provider 要求自纠正（≤3次）

    llm_provider: LLMProvider 实例（用于自纠正），None 则跳过 Step 2。
    返回 dict 或 None（彻底失败）。
    """
    # Step 1: 宽松抠取
    candidate = strip_json_fence(text)
    match = _JSON_BROAD_RE.search(candidate)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # 也试一次全文
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as first_err:
        pass

    if llm_provider is None:
        logger.warning("[extract_json] JSON 解析失败，无 llm_provider 可用于自纠正。")
        return None

    # Step 2: 魔法打败魔法 — 把报错甩回 LLM 自纠正
    current_text = text
    for attempt in range(1, max_retries + 1):
        try:
            probe = json.loads(strip_json_fence(current_text))
            return probe
        except json.JSONDecodeError as err:
            logger.info(f"[extract_json] 自纠正第 {attempt}/{max_retries} 次...")
            _human_jitter(backoff_attempt=attempt)  # 解析失败=重试，指数退避
            fix_prompt = (
                f"你之前的输出无法被 Python json.loads() 解析，报错如下：\n"
                f"  {type(err).__name__}: {err}\n\n"
                f"原始输出：\n{current_text}\n\n"
                "请只输出合法的 JSON 对象（不要任何额外文字、不要 Markdown 代码块）："
            )
            current_text = llm_provider.complete(fix_prompt)

    logger.error(f"[extract_json] {max_retries} 次自纠正后仍失败，放弃解析。")
    return None


# ==========================================
# P0 防御 ③：心跳探针
# ==========================================
_last_heartbeat: float = 0.0
_HEARTBEAT_INTERVAL = 23 * 3600


def heartbeat_probe(force: bool = False) -> bool:
    global _last_heartbeat
    now = time.monotonic()

    if not force and (now - _last_heartbeat) < _HEARTBEAT_INTERVAL:
        return True

    logger.info("[心跳探针] 开始检测...")
    claude_ok = "ok" in call_claude("请回复单词 OK，不要其他内容。").lower()
    gemini_ok = "ok" in call_gemini("Reply with just OK.").lower()

    if claude_ok and gemini_ok:
        _last_heartbeat = now
        logger.info("[心跳探针] ✅ 均存活")
    else:
        logger.warning(
            f"[心跳探针] ⚠️  Claude={'OK' if claude_ok else 'DEAD'}, "
            f"Gemini={'OK' if gemini_ok else 'DEAD'}"
        )

    return claude_ok and gemini_ok
