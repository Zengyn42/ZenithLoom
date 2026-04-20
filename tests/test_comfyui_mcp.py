"""Tests for ComfyUI MCP infrastructure.

Covers:
  - WorkflowManager: template loading, parameter injection, txt2vid support
  - ComfyUIClient: health check, upload, submit (mocked HTTP)
  - MCPManager dependency lifecycle
  - MCP Server tool registration
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# WorkflowManager tests
# ---------------------------------------------------------------------------

class TestWorkflowManager(unittest.TestCase):
    """Test workflow template management."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "workflow_manager",
            _ROOT / "mcp_servers" / "comfyui" / "workflow_manager.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.WorkflowManager = mod.WorkflowManager
        cls.NODE_IDS = mod.NODE_IDS
        cls.TEMPLATE_FILES = mod.TEMPLATE_FILES

    def test_all_five_workflows_registered(self):
        """5 workflow types must be registered."""
        expected = {"txt2vid", "img2vid", "keyframe_2", "keyframe_3", "digital_human"}
        self.assertEqual(set(self.TEMPLATE_FILES.keys()), expected)
        self.assertEqual(set(self.NODE_IDS.keys()), expected)

    def test_all_templates_exist(self):
        """Every registered template file must exist on disk."""
        wm = self.WorkflowManager()
        for wf in wm.list_workflows():
            self.assertTrue(wf["available"], f"Template missing: {wf['template_file']}")

    def test_list_workflows(self):
        """list_workflows returns correct metadata."""
        wm = self.WorkflowManager()
        workflows = wm.list_workflows()
        self.assertEqual(len(workflows), 5)
        types = {w["type"] for w in workflows}
        self.assertIn("txt2vid", types)

    def test_load_template_valid(self):
        """Loading a valid template returns a dict."""
        wm = self.WorkflowManager()
        template = wm.load_template("img2vid")
        self.assertIsInstance(template, dict)
        self.assertGreater(len(template), 0)

    def test_load_template_invalid(self):
        """Loading an unknown workflow type raises ValueError."""
        wm = self.WorkflowManager()
        with self.assertRaises(ValueError):
            wm.load_template("nonexistent")

    def test_prepare_txt2vid_no_image_required(self):
        """txt2vid workflow should work with empty uploaded_files."""
        wm = self.WorkflowManager()
        workflow = wm.prepare_workflow(
            "txt2vid",
            prompt="A sunset over the ocean",
            uploaded_files={},
            width=1280,
            height=720,
        )
        self.assertIsInstance(workflow, dict)
        # txt2vid template should not have LoadImage node (269)
        self.assertNotIn("269", workflow)

    def test_prepare_img2vid_sets_prompt(self):
        """img2vid workflow injects the prompt correctly."""
        wm = self.WorkflowManager()
        test_prompt = "TEST_PROMPT_12345"
        workflow = wm.prepare_workflow(
            "img2vid",
            prompt=test_prompt,
            uploaded_files={"image": "test.png"},
            width=640,
            height=480,
        )
        # Prompt node for img2vid is 303
        prompt_node = workflow.get("303", {})
        inputs = prompt_node.get("inputs", {})
        self.assertEqual(inputs.get("value"), test_prompt)

    def test_prepare_sets_dimensions(self):
        """Workflow dimensions are correctly injected."""
        wm = self.WorkflowManager()
        workflow = wm.prepare_workflow(
            "txt2vid",
            prompt="test",
            uploaded_files={},
            width=640,
            height=480,
            frame_rate=30,
            num_frames=121,
        )
        # Width node is 314, Height node is 299
        self.assertEqual(workflow["314"]["inputs"]["value"], 640)
        self.assertEqual(workflow["299"]["inputs"]["value"], 480)
        self.assertEqual(workflow["300"]["inputs"]["value"], 30)
        self.assertEqual(workflow["301"]["inputs"]["value"], 121)

    def test_prepare_with_fixed_seed(self):
        """Fixed seed is applied to all RandomNoise nodes."""
        wm = self.WorkflowManager()
        workflow = wm.prepare_workflow(
            "txt2vid",
            prompt="test",
            uploaded_files={},
            seed=42,
        )
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == "RandomNoise":
                self.assertEqual(node_data["inputs"]["noise_seed"], 42)

    def test_txt2vid_text_mode_true(self):
        """txt2vid template has text_mode=True (node 302)."""
        wm = self.WorkflowManager()
        template = wm.load_template("txt2vid")
        self.assertTrue(template["302"]["inputs"]["value"])


# ---------------------------------------------------------------------------
# ComfyUIClient tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestComfyUIClient(unittest.TestCase):
    """Test ComfyUI HTTP client with mocked responses."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "comfyui_client",
            _ROOT / "mcp_servers" / "comfyui" / "comfyui_client.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.ComfyUIClient = mod.ComfyUIClient

    def test_init_defaults(self):
        """Default host/port are localhost:8188."""
        client = self.ComfyUIClient()
        self.assertEqual(client.host, "localhost")
        self.assertEqual(client.port, 8188)

    def test_init_custom(self):
        """Custom host/port are preserved."""
        client = self.ComfyUIClient(host="10.0.0.1", port=9999)
        self.assertEqual(client.host, "10.0.0.1")
        self.assertEqual(client.port, 9999)

    @patch("aiohttp.ClientSession")
    def test_health_check_success(self, mock_session_cls):
        """health_check returns system stats on success."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "system": {"os": "win32"},
            "devices": [{"name": "RTX 5090"}],
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        client = self.ComfyUIClient()
        result = asyncio.run(client.health_check())
        self.assertIn("devices", result)


# ---------------------------------------------------------------------------
# MCPManager dependency tests
# ---------------------------------------------------------------------------

class TestMCPManagerDependency(unittest.TestCase):
    """Test MCPManager dependency lifecycle (no real processes)."""

    def setUp(self):
        from framework.mcp_manager import MCPManager
        # Reset singleton
        MCPManager._instance = None
        self.mgr = MCPManager.get_instance()

    def tearDown(self):
        from framework.mcp_manager import MCPManager
        MCPManager._instance = None

    def test_dependency_entry_structure(self):
        """_DependencyEntry has correct fields."""
        from framework.mcp_manager import _DependencyEntry
        dep = _DependencyEntry(name="test", check_url="http://localhost:1234")
        self.assertEqual(dep.name, "test")
        self.assertEqual(dep.refs, 0)
        self.assertIsNone(dep.proc)

    def test_server_entry_has_dependency_name(self):
        """_ServerEntry tracks its dependency."""
        from framework.mcp_manager import _ServerEntry
        entry = _ServerEntry(name="test-server", url="http://localhost:8103/sse",
                             dependency_name="test-dep")
        self.assertEqual(entry.dependency_name, "test-dep")

    @patch.object(
        __import__("framework.mcp_manager", fromlist=["MCPManager"]).MCPManager,
        "_is_reachable",
        return_value=True,
    )
    def test_acquire_reuses_external_dependency(self, mock_reachable):
        """If dependency is already reachable, no process is spawned."""
        spec = {
            "name": "test-mcp",
            "url": "http://localhost:9999/sse",
            "dependency": {
                "name": "test-dep",
                "check_url": "http://localhost:8888/health",
            },
        }
        result = asyncio.run(self.mgr.acquire(spec, agent_name="test"))
        self.assertTrue(result)
        self.assertIn("test-dep", self.mgr._deps)
        self.assertEqual(self.mgr._deps["test-dep"].refs, 1)
        self.assertIsNone(self.mgr._deps["test-dep"].proc)  # external, no process

    def test_is_reachable_raw_mode(self):
        """raw=True doesn't append slash to URL."""
        # This just tests the URL manipulation logic, not actual HTTP
        # We can't easily test actual HTTP without a server
        from framework.mcp_manager import MCPManager
        mgr = MCPManager.get_instance()
        # With a non-existent URL, both modes should return False
        self.assertFalse(mgr._is_reachable("http://127.0.0.1:59999/nonexist", raw=True))
        self.assertFalse(mgr._is_reachable("http://127.0.0.1:59999/sse", raw=False))


# ---------------------------------------------------------------------------
# MCP Server tool count test
# ---------------------------------------------------------------------------

class TestMCPServerTools(unittest.TestCase):
    """Verify MCP server registers all expected tools."""

    def test_server_has_seven_tools(self):
        """MCP server must register exactly 7 tools."""
        # Read server.py and count @mcp.tool() decorators
        server_path = _ROOT / "mcp_servers" / "comfyui" / "server.py"
        content = server_path.read_text()
        tool_count = content.count("@mcp.tool()")
        self.assertEqual(tool_count, 7, f"Expected 7 MCP tools, found {tool_count}")

    def test_all_tool_names_present(self):
        """All expected tool function names exist in server.py."""
        server_path = _ROOT / "mcp_servers" / "comfyui" / "server.py"
        content = server_path.read_text()
        expected_tools = [
            "ltx_txt2vid", "ltx_img2vid", "ltx_keyframe_2",
            "ltx_keyframe_3", "ltx_digital_human",
            "comfyui_status", "comfyui_job_status",
        ]
        for tool in expected_tools:
            self.assertIn(f"async def {tool}(", content, f"Tool {tool} not found")


# ---------------------------------------------------------------------------
# Gradio UI coverage test
# ---------------------------------------------------------------------------

class TestGradioUI(unittest.TestCase):
    """Verify Gradio UI has all required tabs and handlers."""

    @classmethod
    def setUpClass(cls):
        cls.content = (_ROOT / "tools" / "ltx_debug_ui.py").read_text()

    def test_all_handlers_present(self):
        """All workflow handler functions exist."""
        handlers = [
            "run_txt2vid", "run_img2vid", "run_keyframe2",
            "run_keyframe3", "run_digital_human", "query_job_status",
        ]
        for h in handlers:
            self.assertIn(f"def {h}(", self.content, f"Handler {h} not found")

    def test_all_tabs_present(self):
        """All tabs are declared in the UI."""
        tabs = ["txt2vid", "img2vid", "keyframe_2", "keyframe_3", "digital_human", "Job Status"]
        for tab in tabs:
            self.assertIn(tab, self.content, f"Tab '{tab}' not found in UI")

    def test_workflow_inspector_includes_txt2vid(self):
        """Workflow inspector dropdown includes txt2vid."""
        self.assertIn('"txt2vid"', self.content)


# ---------------------------------------------------------------------------
# Old nodes removed test
# ---------------------------------------------------------------------------

class TestOldNodesRemoved(unittest.TestCase):
    """Verify old ComfyUI/LTX nodes have been cleaned up."""

    def test_comfyui_node_dir_removed(self):
        """framework/nodes/comfyui/ should not exist."""
        path = _ROOT / "framework" / "nodes" / "comfyui"
        self.assertFalse(path.exists(), "Old comfyui node directory should be removed")

    def test_builtins_no_old_nodes(self):
        """builtins.py should not register COMFYUI or LTX_VIDEO node types."""
        builtins_path = _ROOT / "framework" / "builtins.py"
        content = builtins_path.read_text()
        self.assertNotIn("ComfyUINode", content)
        self.assertNotIn("LTXVideoNode", content)


if __name__ == "__main__":
    unittest.main()
