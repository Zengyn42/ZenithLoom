"""
快速测试：验证 Claude CLI 和 Gemini CLI 是否可用
运行：python3 test_cli.py
"""

from agent.cli_wrapper import call_claude, call_gemini, clean_output

def test_claude():
    print("--- Claude CLI ---")
    response = call_claude("请回复：我是Claude，已就位。不要多余内容。")
    print(f"输出: {response!r}")
    assert response and len(response) > 3, "Claude 输出为空或过短"
    print("✅ Claude OK\n")

def test_gemini():
    print("--- Gemini CLI ---")
    response = call_gemini("Reply with exactly: Gemini online.")
    print(f"输出: {response!r}")
    assert response and len(response) > 3, "Gemini 输出为空或过短"
    print("✅ Gemini OK\n")

def test_ansi_cleaner():
    print("--- ANSI 清洗器 ---")
    dirty = "\x1b[32mHello\x1b[0m \x1b[1mWorld\x1b[0m"
    clean = clean_output(dirty)
    print(f"原始: {dirty!r}")
    print(f"清洗后: {clean!r}")
    assert "\x1b" not in clean, "ANSI 码未清除干净"
    print("✅ ANSI 清洗器 OK\n")

if __name__ == "__main__":
    test_ansi_cleaner()
    test_claude()
    test_gemini()
    print("🎉 全部通过")
