"""
Test script for GrokNode
"""
import asyncio
import sys
from pathlib import Path

from framework.config import AgentConfig
from framework.nodes.llm.grok import GrokNode
from framework.nodes.llm.llm_node import set_stream_callback

async def main():
    print("--- Testing GrokNode ---")
    cfg = AgentConfig()
    node_config = {
        "id": "grok_test",
        "type": "GROK",
    }
    
    node = GrokNode(cfg, node_config, system_prompt="You are a helpful assistant. Reply concisely.")
    
    def on_stream(text, is_thinking):
        print(text, end="", flush=True)
        
    set_stream_callback(on_stream)
    
    print("\nCalling Grok:")
    try:
        text, sid = await node.call_llm("Please reply with the exact text: GROK_IS_WORKING", session_id="")
        print(f"\n\n--- Result ---\n{text}")
        print(f"Session ID: {sid}")
        
        if "GROK_IS_WORKING" in text:
            print("✅ GrokNode test passed!")
        else:
            print("❌ GrokNode test failed: Output does not contain expected text.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ GrokNode test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
