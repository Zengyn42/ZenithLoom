"""
框架级信号解析器协议 — framework/signal_parser.py

LlmNode 通过 SignalParser 检测 LLM 输出中的路由信号，
避免在 llm_node.py 中硬编码特定格式的正则。

内置解析器：
  json_line   — 检测首行 JSON（Claude 格式，默认）
  regex_xml   — 检测 <signal>...</signal> XML 标签（Llama 风格）

node_config 配置：
  "signal_parser": "json_line"   # 默认，省略可不写
  "signal_parser": "regex_xml"   # 用于 Llama 等非标准格式

自定义解析器：
  from framework.signal_parser import register_parser
  class MyParser:
      def parse(self, text: str) -> dict | None: ...
  register_parser("my_format", MyParser())
"""

import json
import logging
import re
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class SignalParser(Protocol):
    """路由信号解析器协议。"""

    def parse(self, text: str) -> dict | None:
        """
        从 LLM 输出文本中解析路由信号。

        返回：
          dict  — 解析成功，包含 "action" 等字段
          None  — 未检测到信号，继续正常路由
        """
        ...


class JsonLineParser:
    """
    默认解析器：检测输出首行是否为 JSON 对象（Claude 格式）。

    Claude 的路由信号格式：
      {"action": "consult_gemini", "topic": "...", "context": "..."}
    """

    def parse(self, text: str) -> dict | None:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in lines:
            if not line.startswith("{"):
                continue
            try:
                result = json.loads(line)
                if isinstance(result, dict) and "route" in result:
                    logger.debug(f"[signal_parser.json_line] signal={result.get('route')!r}")
                    return result
            except json.JSONDecodeError:
                # 常见问题：LLM 在 JSON 字符串值中生成未转义的双引号
                # 尝试修复：提取 route 字段，context 做宽松处理
                if '"route"' in line:
                    repaired = self._try_repair(line)
                    if repaired:
                        logger.warning(
                            f"[signal_parser.json_line] repaired invalid JSON, "
                            f"route={repaired.get('route')!r}"
                        )
                        return repaired
                    logger.warning(
                        f"[signal_parser.json_line] JSON parse failed on line "
                        f"that looks like a routing signal: {line[:120]}..."
                    )
        return None

    @staticmethod
    def _try_repair(line: str) -> dict | None:
        """尝试修复 LLM 常见的 JSON 错误（未转义引号）。"""
        import re
        # 提取 route 值（可靠，通常是简短的 node_id）
        m_route = re.search(r'"route"\s*:\s*"([^"]+)"', line)
        if not m_route:
            return None
        route = m_route.group(1)
        # 提取 context 值（宽松：取最后一个 "} 之前的所有内容）
        m_ctx = re.search(r'"context"\s*:\s*"(.*)"[^"]*$', line, re.DOTALL)
        context = m_ctx.group(1) if m_ctx else ""
        return {"route": route, "context": context}


class RegexXmlParser:
    """
    备选解析器：检测 <signal>...</signal> XML 标签（Llama/Qwen 风格）。

    Llama 的路由信号格式：
      <signal>{"action": "consult_gemini", "topic": "..."}</signal>
    """

    _RE = re.compile(r"<signal>(.*?)</signal>", re.DOTALL)

    def parse(self, text: str) -> dict | None:
        m = self._RE.search(text)
        if not m:
            return None
        try:
            result = json.loads(m.group(1).strip())
            if isinstance(result, dict):
                logger.debug(f"[signal_parser.regex_xml] signal={result.get('action')!r}")
                return result
        except json.JSONDecodeError:
            pass
        return None


# 内置解析器注册表（单例，模块级）
_PARSERS: dict[str, SignalParser] = {
    "json_line": JsonLineParser(),
    "regex_xml": RegexXmlParser(),
}


def register_parser(name: str, parser: SignalParser) -> None:
    """注册自定义信号解析器。"""
    _PARSERS[name] = parser
    logger.debug(f"[signal_parser] registered {name!r}")


def get_signal_parser(name: str) -> SignalParser:
    """
    按名称获取信号解析器。
    未找到时回退到 json_line（不抛异常，避免启动失败）。
    """
    parser = _PARSERS.get(name)
    if parser is None:
        logger.warning(
            f"[signal_parser] unknown parser {name!r}, falling back to json_line"
        )
        return _PARSERS["json_line"]
    return parser
