"""
Discord connector — text formatting utilities.
"""

import asyncio
import base64
import logging
import re
import urllib.parse
import urllib.request

logger = logging.getLogger("discord_bot")


def _mermaid_node_count(mermaid_text: str) -> int:
    """统计 Mermaid 文本中的有效节点行数（用于复杂度评估）。"""
    return len([
        ln for ln in mermaid_text.splitlines()
        if ln.strip() and not ln.strip().startswith(("flowchart", "subgraph", "end", "%%"))
    ])


def _mermaid_with_style(mermaid_text: str) -> str:
    """注入 init 指令，根据节点数动态设置字号，提升可读性。
    节点越多字号越小（避免溢出），但最小 14px 保证清晰度。
    """
    if mermaid_text.strip().startswith("%%{init"):
        return mermaid_text  # 已有 init 指令，不覆盖
    n = _mermaid_node_count(mermaid_text)
    font_size = max(14, 22 - n)  # n=0→22px, n=8→14px, n≥8→14px
    init = f'%%{{init: {{"themeVariables": {{"fontSize": "{font_size}px"}}}}}}%%'
    return f"{init}\n{mermaid_text}"


async def _fetch_mermaid_png(mermaid_text: str) -> bytes | None:
    """
    调用 mermaid.ink 将 Mermaid 文本渲染成 PNG bytes。
    - 注入 init 指令动态调整字号（根节点复杂度，min 14px）
    - scale=2 固定，配合字号提升已足够清晰
    mermaid.ink 会拒绝 Python 默认 User-Agent，需显式设置。
    """
    try:
        styled   = _mermaid_with_style(mermaid_text)
        encoded  = base64.b64encode(styled.encode("utf-8")).decode("ascii")
        bg_color = urllib.parse.quote("white", safe="")
        url      = f"https://mermaid.ink/img/{encoded}?type=png&bgColor={bg_color}&scale=2"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ZenithLoom/1.0)"},
        )

        def _blocking_fetch() -> bytes:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _blocking_fetch)
    except Exception as e:
        logger.warning(f"[discord] mermaid.ink PNG 获取失败: {e}")
        return None


def fix_list_formatting(text: str) -> str:
    """
    修复 Markdown 列表在 Discord 中显示不正常的问题。
    1. 确保列表项前有换行（例如 "文本 1. " -> "文本\n1. "）
    2. 确保列表项标记后有空格
    """
    # 针对数字列表：匹配 [非换行] + [空格] + [数字.] + [空格]
    text = re.sub(r'([^\n])(\s+\d+\.\s)', r'\1\n\2', text)
    # 针对无序列表：匹配 [非换行] + [空格] + [*或-] + [空格]
    text = re.sub(r'([^\n])(\s+[*+\-] )', r'\1\n\2', text)
    # 确保行首数字列表后面有空格（例如 "1.Item" -> "1. Item"）
    text = re.sub(r'^(\d+)\.([^\s])', r'\1. \2', text, flags=re.MULTILINE)
    return text


def format_persona_response(text: str) -> str:
    """
    将含有 [任意标签] 的 Grok 多人格响应格式化为 Discord 带颜色区分的段落。

    检测格式：文本中出现的 [WORD] 或 [MULTI WORD] 标签
    每个不同标签分配一个 ANSI 颜色，循环使用调色板。
    无标签的单段落响应 → 进行基础列表格式修复后返回。
    """
    text = text.strip()

    # 检测 [任意标签] 格式，标签内允许字母、数字、空格、连字符
    tag_pattern = re.compile(r'\[([A-Za-z][A-Za-z0-9 _\-]{0,30})\]')
    if not tag_pattern.search(text):
        return fix_list_formatting(text)

    # 已知人格 → 固定 emoji
    PERSONA_EMOJI: dict[str, str] = {
        "JAILBREAK": "😈",
        "CLASSIC":   "🎩",
        "DAN":       "🤖",
        "SIGMA":     "⚡",
        "BASED":     "🔥",
        "EVIL":      "💀",
        "DEVELOPER": "👨‍💻",
    }

    # ANSI 颜色调色板（Discord ansi 代码块）
    PALETTE = [
        ("1;34", "🔵"),   # 亮蓝
        ("1;31", "🔴"),   # 亮红
        ("1;32", "🟢"),   # 亮绿
        ("1;33", "🟡"),   # 亮黄
        ("1;35", "🟣"),   # 亮紫
        ("1;36", "🩵"),   # 亮青
    ]

    # 按出现顺序为每个唯一标签分配颜色
    seen_tags: dict[str, tuple] = {}
    color_idx = 0
    for m in tag_pattern.finditer(text):
        tag_name = m.group(1).upper()
        if tag_name not in seen_tags:
            ansi_code, palette_emoji = PALETTE[color_idx % len(PALETTE)]
            # 已知人格用固定 emoji，否则用调色板圆点
            emoji = PERSONA_EMOJI.get(tag_name, palette_emoji)
            seen_tags[tag_name] = (ansi_code, emoji)
            color_idx += 1

    # 按标签切分段落
    parts = tag_pattern.split(text)
    # parts 结构: [pre_text, tag1, content1, tag2, content2, ...]

    result_blocks = []

    pre = parts[0].strip()
    if pre:
        result_blocks.append(fix_list_formatting(pre))

    i = 1
    while i < len(parts) - 1:
        tag_name = parts[i].upper()
        content = parts[i + 1].strip()
        i += 2

        if not content:
            continue

        ansi_code, emoji = seen_tags.get(tag_name, ("", "▪️"))

        content = fix_list_formatting(content)

        block = f"{emoji} **{tag_name}**\n\n{content}"
        result_blocks.append(block)

    # 处理最后一个标签之后可能存在的文字（总结等）
    if i < len(parts):
        trailing = parts[i].strip()
        if trailing:
            result_blocks.append(fix_list_formatting(trailing))

    return "\n\n".join(result_blocks)
