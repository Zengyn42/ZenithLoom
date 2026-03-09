"""
框架导入与配置验证测试

不做实际 API 调用，只验证：
  - AgentLoader 能正确加载 Hani / Asa 配置
  - ClaudeNode / LlamaNode / GeminiNode 可实例化
  - call_llm 接口存在
  - db_path / sessions_file 路径在 agent 目录内

运行：python3 test_cli.py
"""

import asyncio
from pathlib import Path


def test_agent_loader_hani():
    print("--- AgentLoader: Hani ---")
    from framework.agent_loader import AgentLoader
    loader = AgentLoader(Path("agents/hani"))
    assert loader.name == "hani"
    assert loader.json["llm"] == "claude"

    cfg = loader.load_config()
    assert "agents/hani" in cfg.db_path, f"db_path 应在 agents/hani 内: {cfg.db_path}"
    assert "agents/hani" in cfg.sessions_file

    prompt = loader.load_system_prompt()
    assert len(prompt) > 100, "system_prompt 为空"
    print(f"   db_path: {cfg.db_path}")
    print(f"   system_prompt: {len(prompt)} chars")
    print("✅ AgentLoader Hani OK\n")


def test_agent_loader_asa():
    print("--- AgentLoader: Asa ---")
    from framework.agent_loader import AgentLoader
    loader = AgentLoader(Path("agents/asa"))
    assert loader.name == "asa"
    assert loader.json["llm"] == "llama"
    cfg = loader.load_config()
    assert "agents/asa" in cfg.db_path
    print(f"   llm: {loader.json['llm']} model: {loader.json['llama_model']}")
    print("✅ AgentLoader Asa OK\n")


def test_claude_node_interface():
    print("--- ClaudeNode 接口 ---")
    from framework.claude.node import ClaudeNode
    from framework.config import AgentConfig
    node = ClaudeNode(AgentConfig(), "test system prompt")
    assert hasattr(node, "call_claude"), "缺少 call_claude"
    assert hasattr(node, "call_llm"), "缺少 call_llm 标准接口"
    assert ClaudeNode.call_llm is ClaudeNode.call_claude, "call_llm 应是 call_claude 的别名"
    print("✅ ClaudeNode 接口 OK\n")


def test_llama_node_interface():
    print("--- LlamaNode 接口 ---")
    from framework.llama.node import LlamaNode
    from framework.config import AgentConfig
    node = LlamaNode(AgentConfig(), "test")
    assert hasattr(node, "call_llm"), "缺少 call_llm"
    print("✅ LlamaNode 接口 OK\n")


def test_agent_node():
    print("--- AgentNode (通用节点) ---")
    from framework.nodes.agent_node import AgentNode, AgentClaudeNode
    assert AgentClaudeNode is AgentNode, "AgentClaudeNode 向后兼容别名失效"
    print("✅ AgentNode + AgentClaudeNode 别名 OK\n")


def test_tool_rules():
    print("--- tool_rules 关键词匹配 ---")
    from framework.agent_loader import AgentLoader
    from framework.nodes.agent_node import AgentNode
    from framework.config import AgentConfig
    from framework.claude.node import ClaudeNode
    from framework.gemini.node import GeminiNode

    loader = AgentLoader(Path("agents/hani"))
    cfg = loader.load_config()
    claude = ClaudeNode(cfg, "")
    gemini = GeminiNode(cfg, claude)
    node = AgentNode(claude, gemini, node_config=loader.json)

    # 中文搜索关键词应触发 WebSearch/WebFetch
    tools = node._select_tools("帮我搜索一下最新的论文")
    assert "WebFetch" in tools or "WebSearch" in tools, f"搜索关键词应追加 WebFetch/WebSearch: {tools}"

    # 普通指令不应追加额外工具
    tools_normal = node._select_tools("帮我修改这个文件")
    assert "WebFetch" not in tools_normal, f"普通指令不应有 WebFetch: {tools_normal}"
    print(f"   搜索触发工具: {tools}")
    print("✅ tool_rules OK\n")


def test_hani_graph_exports():
    print("--- agents/hani/graph.py 导出 ---")
    from agents.hani.graph import get_engine, invalidate_engine
    from framework.graph import get_config, switch_session, new_session
    assert callable(get_engine)
    assert callable(invalidate_engine)
    assert callable(get_config)
    print("✅ hani/graph.py 导出 OK\n")


def test_main_arg_parsing():
    print("--- main.py --agent 解析 ---")
    import sys, importlib.util
    # 模拟 sys.argv
    orig = sys.argv[:]
    sys.argv = ["main.py", "--agent", "hani", "cli"]
    spec = importlib.util.spec_from_file_location("main", "main.py")
    main_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_mod)
    agent_name, mode = main_mod._parse_args()
    sys.argv = orig
    assert agent_name == "hani", f"agent_name={agent_name}"
    assert mode == "cli", f"mode={mode}"
    print(f"   parsed: agent={agent_name} mode={mode}")
    print("✅ main.py 参数解析 OK\n")


if __name__ == "__main__":
    test_agent_loader_hani()
    test_agent_loader_asa()
    test_claude_node_interface()
    test_llama_node_interface()
    test_agent_node()
    test_tool_rules()
    test_hani_graph_exports()
    test_main_arg_parsing()
    print("🎉 全部通过")
