"""
框架级 Gemini 顾问节点 — 向后兼容导出

实现已移至 framework/gemini/node.py
"""
from framework.gemini.node import GeminiNode, GeminiQuotaError  # noqa: F401

__all__ = ["GeminiNode", "GeminiQuotaError"]
