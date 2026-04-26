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


async def _fetch_mermaid_png(mermaid_text: str) -> bytes | None:
    """
    调用 mermaid.ink /svg 端点获取 SVG，用 Chrome headless 渲染为 PNG。

    流程：
    1. mermaid.ink /svg 端点（稳定，支持复杂图）
    2. 写入临时文件
    3. google-chrome --headless 渲染（正确处理 foreignObject/HTML 文字）
    4. PIL 裁掉空白边距，缩放到合理尺寸
    """
    import re, subprocess, tempfile, os, io as _io
    try:
        # 1. 获取 SVG
        encoded = base64.urlsafe_b64encode(mermaid_text.encode("utf-8")).decode("ascii").rstrip("=")
        url     = f"https://mermaid.ink/svg/{encoded}"
        req     = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ZenithLoom/1.0)"},
        )

        def _render() -> bytes:
            with urllib.request.urlopen(req, timeout=15) as resp:
                svg_bytes = resp.read()

            # 2. 写入临时 SVG 文件
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
                f.write(svg_bytes)
                svg_path = f.name

            png_path = svg_path.replace(".svg", ".png")
            try:
                # 从 SVG viewBox 估算渲染尺寸
                svg_txt = svg_bytes.decode(errors="replace")
                vb = re.search(r'viewBox=["\'][\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)', svg_txt)
                win_w = int(float(vb.group(1))) * 2 + 200 if vb else 2400
                win_h = int(float(vb.group(2))) * 2 + 200 if vb else 1600

                # 3. Chrome headless 渲染（支持 foreignObject HTML 文字）
                subprocess.run([
                    "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
                    f"--screenshot={png_path}", f"--window-size={win_w},{win_h}",
                    "--force-device-scale-factor=2",
                    f"file://{svg_path}",
                ], capture_output=True, timeout=30, check=True)

                # 4. PIL 裁空白 + 缩小到合理尺寸
                from PIL import Image
                import numpy as np
                img = Image.open(png_path).convert("RGB")
                arr = np.array(img)
                mask = (arr < 245).any(axis=2)
                rows = np.where(mask.any(axis=1))[0]
                cols = np.where(mask.any(axis=0))[0]
                if len(rows) and len(cols):
                    pad = 40
                    r0 = max(0, rows[0] - pad)
                    r1 = min(arr.shape[0], rows[-1] + pad)
                    c0 = max(0, cols[0] - pad)
                    c1 = min(arr.shape[1], cols[-1] + pad)
                    img = img.crop((c0, r0, c1, r1))

                # 超过 3000px 宽则缩小一半
                if img.width > 3000:
                    img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)

                buf = _io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                return buf.getvalue()
            finally:
                os.unlink(svg_path)
                if os.path.exists(png_path):
                    os.unlink(png_path)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _render)
    except Exception as e:
        logger.warning(f"[discord] mermaid PNG 生成失败: {e}")
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
