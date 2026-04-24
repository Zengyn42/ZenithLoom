"""
Discord Tool Client — framework/discord_tool_client.py

EXTERNAL_TOOL 节点调用的 CLI 客户端。
routing_context 格式（空格分隔 key=value）：

  history [limit=N]              — 读取当前频道最近 N 条消息（默认 20）
  search q=<keyword> [limit=N]   — 搜索关键词（默认搜最近 100 条）
  channels                       — 列出服务器所有文字频道
  user id=<USER_ID>              — 查询用户信息

用法：
  python3 -m framework.discord_tool_client "history limit=20"
  python3 -m framework.discord_tool_client "search q=视频生成 limit=50"
  python3 -m framework.discord_tool_client "channels"
  python3 -m framework.discord_tool_client "user id=286003878997262337"
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def _parse_args(context: str) -> tuple[str, dict]:
    """解析 'cmd key=val key=val ...' 格式。"""
    parts = context.strip().split()
    if not parts:
        return "", {}
    cmd = parts[0].lower()
    params: dict = {}
    for part in parts[1:]:
        if "=" in part:
            k, _, v = part.partition("=")
            params[k.strip()] = v.strip()
        else:
            # 裸词作为 q 参数（方便 search keyword 写法）
            params.setdefault("q", part)
    return cmd, params


def _call(endpoint: str, params: dict) -> dict | list:
    port = os.environ.get("DISCORD_TOOL_PORT", "")
    if not port:
        raise RuntimeError("DISCORD_TOOL_PORT not set — Discord tool server not running")

    url = f"http://127.0.0.1:{port}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _format_history(msgs: list) -> str:
    if not msgs:
        return "（无消息记录）"
    lines = [f"[{m['ts']}] {m['author']}: {m['content']}" for m in msgs]
    return "\n".join(lines)


def _format_search(msgs: list, query: str) -> str:
    if not msgs:
        return f"（未找到包含 '{query}' 的消息）"
    lines = [f"[{m['ts']}] {m['author']}: {m['content']}" for m in msgs]
    return f"搜索 '{query}' 共找到 {len(msgs)} 条：\n" + "\n".join(lines)


def _format_channels(channels: list) -> str:
    lines = []
    for ch in channels:
        marker = " ← 当前频道" if ch.get("current") else ""
        cat = f"[{ch['category']}] " if ch.get("category") else ""
        lines.append(f"  {cat}#{ch['name']}{marker}")
    return "服务器频道列表：\n" + "\n".join(lines)


def _format_user(user: dict) -> str:
    roles = "、".join(user.get("roles", [])) or "无"
    return (
        f"用户：{user['display_name']} (@{user['name']})\n"
        f"ID：{user['id']}\n"
        f"身份组：{roles}\n"
        f"加入时间：{user.get('joined_at', '未知')}"
    )


def main():
    if len(sys.argv) < 2:
        print("用法: python3 -m framework.discord_tool_client '<routing_context>'", file=sys.stderr)
        sys.exit(1)

    context = " ".join(sys.argv[1:])
    cmd, params = _parse_args(context)

    try:
        if cmd == "history":
            data = _call("history", {"limit": params.get("limit", 20)})
            print(_format_history(data))

        elif cmd == "search":
            query = params.get("q", "")
            if not query:
                print("错误：search 需要 q=<keyword>", file=sys.stderr)
                sys.exit(1)
            api_params = {"q": query}
            if "limit" in params:
                api_params["limit"] = params["limit"]
            data = _call("search", api_params)
            print(_format_search(data, query))

        elif cmd == "channels":
            data = _call("channels", {})
            print(_format_channels(data))

        elif cmd == "user":
            user_id = params.get("id", "")
            if not user_id:
                print("错误：user 需要 id=<USER_ID>", file=sys.stderr)
                sys.exit(1)
            data = _call("user", {"id": user_id})
            print(_format_user(data))

        else:
            print(f"未知命令: {cmd!r}。可用：history / search / channels / user", file=sys.stderr)
            sys.exit(1)

    except RuntimeError as exc:
        print(f"Discord tool error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
