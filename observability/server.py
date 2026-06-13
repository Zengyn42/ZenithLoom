"""
ZenithLoom Observability Server — observability/server.py

FastAPI application exposing:
  WS  /ingest            ← agent clients push JSON-line events
  WS  /ws/events         → frontend subscribes to real-time broadcast
  WS  /ws               → single-port viewer proxy (→ viewer :8766/ws)
  GET /api/agents        → list agents + current node states
  GET /api/graph/{name}  → AgentGraph topology (reads entity.json via config)

Run:
    uvicorn observability.server:app --port 8765 --host 127.0.0.1

Or via the systemd unit: zl-observability.service

Environment variables
─────────────────────
  ZL_OBSERV_HOST    Host uvicorn binds to (default 127.0.0.1).
                    Set to 0.0.0.0 to accept external connections.
  ZL_OBSERV_TOKEN   Optional shared secret.  When set, /ws and /ws/events
                    require ?token=<secret> query param or
                    Authorization: Bearer <secret> header.
                    Empty string (default) disables auth.
  ZL_VIEWER_PORT    Port the viewer process listens on (default 8766).
                    The /ws proxy endpoint forwards to this port.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import websockets as _ws_lib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime configuration (read once at import time)
# ---------------------------------------------------------------------------

_OBSERV_HOST  = os.environ.get("ZL_OBSERV_HOST",  "127.0.0.1")
_OBSERV_TOKEN = os.environ.get("ZL_OBSERV_TOKEN", "").strip()
_VIEWER_PORT  = int(os.environ.get("ZL_VIEWER_PORT", "8766"))
_VIEWER_URL   = f"ws://127.0.0.1:{_VIEWER_PORT}/ws"

# Print effective config so operators can confirm settings at a glance.
print(
    f"[observability] host={_OBSERV_HOST}  viewer={_VIEWER_URL}  "
    f"token={'SET' if _OBSERV_TOKEN else 'disabled'}",
    flush=True,
)


def _check_token(ws: WebSocket) -> bool:
    """Return True if the request carries a valid token (or auth is disabled)."""
    if not _OBSERV_TOKEN:
        return True
    # 1. ?token= query param
    if ws.query_params.get("token") == _OBSERV_TOKEN:
        return True
    # 2. Authorization: Bearer <token>
    auth = ws.headers.get("authorization", "")
    if auth.lower().startswith("bearer ") and auth[7:].strip() == _OBSERV_TOKEN:
        return True
    return False

# ---------------------------------------------------------------------------
# Config loader — reads observability/config.toml (or env OBSERV_CONFIG_PATH)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.toml"


def _load_config() -> dict:
    """Load config.toml if available; return empty dict on missing/parse errors."""
    path = Path(os.environ.get("OBSERV_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH)))
    if not path.exists():
        logger.warning(f"[server] config not found at {path}, using defaults")
        return {}
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            # Fallback: manual TOML-less parse (only [agents.*] section needed)
            return _parse_simple_toml(path)
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error(f"[server] config parse error: {exc}")
        return {}


def _parse_simple_toml(path: Path) -> dict:
    """
    Ultra-minimal parser for the subset of TOML we use:
        [agents.hani]
        entity_json = "/path/..."
        db_path = "/path/..."
    """
    result: dict = {"agents": {}}
    current_agent: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("[agents.") and line.endswith("]"):
            current_agent = line[8:-1]
            result["agents"].setdefault(current_agent, {})
        elif current_agent and "=" in line:
            k, _, v = line.partition("=")
            result["agents"][current_agent][k.strip()] = v.strip().strip('"').strip("'")
    return result


_config = _load_config()
_agent_cfg: dict[str, dict] = _config.get("agents", {})


def _get_entity_json(agent_name: str) -> dict | None:
    """
    Load entity.json for the given agent using the config mapping.
    Returns None if not configured or file missing.
    """
    cfg = _agent_cfg.get(agent_name, {})
    path_str = cfg.get("entity_json", "")
    if not path_str:
        # Auto-discover: try VoidDraft/blueprints/role_agents/* and functional_graphs/*
        base = Path("/home/kingy/Foundation/VoidDraft/blueprints")
        candidates = [
            base / "role_agents" / "*" / "entity.json",
            base / "functional_graphs" / "*" / "entity.json",
        ]
        import glob
        for pattern in candidates:
            for p in glob.glob(str(pattern)):
                try:
                    d = json.loads(Path(p).read_text(encoding="utf-8"))
                    # Match by name field inside entity.json (if present)
                    # OR by directory name heuristic
                    return d
                except Exception:
                    continue
        return None
    p = Path(path_str)
    if not p.exists():
        logger.warning(f"[server] entity_json not found: {p}")
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error(f"[server] entity_json parse error for {agent_name}: {exc}")
        return None


def _expand_subgraph_nodes(graph_dict: dict, parent_entity_dir: Path) -> dict:
    """
    Expand SUBGRAPH_REF nodes (those with agent_dir) by loading their child graph
    and attaching it as node["subgraph"]. Only one level deep.
    Returns a (possibly mutated) copy of graph_dict.
    """
    nodes = graph_dict.get("nodes", [])
    expanded_nodes = []
    for node in nodes:
        node = dict(node)  # shallow copy
        agent_dir = node.get("agent_dir")
        if agent_dir:
            try:
                # agent_dir is relative to parent entity.json's directory
                child_base = parent_entity_dir / agent_dir
                # Try both entity.json and agent.json
                child_data: dict | None = None
                for fname in ("entity.json", "agent.json"):
                    cp = child_base / fname
                    if cp.exists():
                        child_data = json.loads(cp.read_text(encoding="utf-8"))
                        break
                if child_data is not None:
                    child_graph = child_data.get("graph")
                    if child_graph and isinstance(child_graph, dict):
                        node["subgraph"] = {
                            "nodes": child_graph.get("nodes", []),
                            "edges": child_graph.get("edges", []),
                        }
            except Exception as exc:
                logger.debug(f"[server] subgraph expand error for node {node.get('id')}: {exc}")
        expanded_nodes.append(node)
    return {**graph_dict, "nodes": expanded_nodes}


def _get_graph_data(agent_name: str) -> dict | None:
    """Return the graph dict from entity.json for the given agent."""
    cfg = _agent_cfg.get(agent_name, {})
    entity_json_path = cfg.get("entity_json", "")

    if entity_json_path:
        p = Path(entity_json_path)
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                graph = d.get("graph")
                if graph:
                    graph = _expand_subgraph_nodes(graph, p.parent)
                return graph
            except Exception as exc:
                logger.error(f"[server] graph read error for {agent_name}: {exc}")
                return None

    # Auto-discover from identity.json
    identity_path = Path(f"/home/kingy/Foundation/EdenGateway/agents/{agent_name}/identity.json")
    if identity_path.exists():
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            blueprint_dir = identity.get("blueprint", "")
            if blueprint_dir:
                ep = Path(blueprint_dir) / "entity.json"
                if ep.exists():
                    d = json.loads(ep.read_text(encoding="utf-8"))
                    graph = d.get("graph")
                    if graph:
                        graph = _expand_subgraph_nodes(graph, ep.parent)
                    return graph
        except Exception as exc:
            logger.error(f"[server] auto-discover graph error for {agent_name}: {exc}")
    return None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ZenithLoom Observability", version="1.0.0")


@app.on_event("startup")
async def _startup() -> None:
    logger.info(
        f"[server] ZenithLoom Observability Server started  "
        f"host={_OBSERV_HOST}  viewer={_VIEWER_URL}  "
        f"token={'SET' if _OBSERV_TOKEN else 'disabled'}"
    )


# ── WebSocket: agent → server (ingest) ────────────────────────────────────────

@app.websocket("/ingest")
async def ingest(ws: WebSocket) -> None:
    """
    Receive JSON-line events from an agent process.
    Parses each line, passes to hub.ingest().
    """
    from observability.hub import get_hub
    from observability.schema import ObservEvent

    hub = get_hub()
    await ws.accept()
    agent_name = "unknown"

    try:
        while True:
            raw = await ws.receive_text()
            for line in raw.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    evt = ObservEvent.from_dict(d)
                    agent_name = evt.agent_name
                    hub.ingest(evt)
                except Exception as exc:
                    logger.warning(f"[server] ingest parse error from {agent_name}: {exc}")
    except WebSocketDisconnect:
        logger.info(f"[server] agent {agent_name!r} disconnected")
    except Exception as exc:
        logger.warning(f"[server] ingest error ({agent_name}): {exc}")


# ── WebSocket: server → frontend (broadcast) ──────────────────────────────────

@app.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    """
    Stream real-time events to a frontend subscriber.
    On connect: send initial state snapshot, then stream events as they arrive.
    """
    import asyncio
    from observability.hub import get_hub

    hub = get_hub()
    await ws.accept()

    # Send initial snapshot so frontend can render immediately
    snapshot = {
        "type": "init",
        "agents": hub.agents_snapshot(),
    }
    try:
        await ws.send_text(json.dumps(snapshot))
    except Exception:
        return

    q = hub.subscribe()
    try:
        while True:
            try:
                line = await asyncio.wait_for(q.get(), timeout=30)
                await ws.send_text(line)
            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await ws.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug(f"[server] ws_events closed: {exc}")
    finally:
        hub.unsubscribe(q)


# ── WebSocket: single-port viewer proxy (/ws) ─────────────────────────────────

@app.websocket("/ws")
async def ws_proxy(ws: WebSocket) -> None:
    """
    Proxy incoming WS connections to the viewer process on _VIEWER_URL.

    This lets the frontend connect to a single origin (same host:port as the
    HTTP API) instead of needing a separate port for the viewer.  Any tunnel
    (ngrok, Cloudflare Tunnel, etc.) or direct IP access therefore only
    requires one address.

    Auth (when ZL_OBSERV_TOKEN is set):
      • Query param:  ws://<host>/ws?token=<secret>
      • HTTP header:  Authorization: Bearer <secret>
    """
    await ws.accept()

    if not _check_token(ws):
        logger.warning("[server] /ws rejected: invalid or missing token")
        await ws.close(code=4403)
        return

    try:
        async with _ws_lib.connect(_VIEWER_URL) as viewer_ws:

            async def relay_viewer_to_client() -> None:
                """Forward every message from viewer → browser client."""
                async for msg in viewer_ws:
                    text = msg if isinstance(msg, str) else msg.decode()
                    try:
                        await ws.send_text(text)
                    except Exception:
                        return

            async def relay_client_to_viewer() -> None:
                """Forward messages from browser client → viewer (e.g. custom cmds)."""
                try:
                    while True:
                        data = await ws.receive_text()
                        await viewer_ws.send(data)
                except (WebSocketDisconnect, Exception):
                    return

            tasks = [
                asyncio.create_task(relay_viewer_to_client()),
                asyncio.create_task(relay_client_to_viewer()),
            ]
            _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                try:
                    await t
                except Exception:
                    pass

    except OSError as exc:
        # Viewer not reachable — send a meaningful error then close
        logger.warning(f"[server] /ws proxy: viewer unreachable at {_VIEWER_URL}: {exc}")
        try:
            await ws.send_text(json.dumps({"type": "error", "message": "viewer unavailable"}))
            await ws.close(code=1011)
        except Exception:
            pass
    except Exception as exc:
        logger.debug(f"[server] /ws proxy closed: {exc}")


# ── REST: agents list ──────────────────────────────────────────────────────────

@app.get("/api/agents")
async def api_agents() -> dict:
    """Return current state of all known agents."""
    from observability.hub import get_hub
    return {"agents": get_hub().agents_snapshot()}


# ── REST: graph topology ───────────────────────────────────────────────────────

@app.get("/api/graph/{agent_name}")
async def api_graph(agent_name: str) -> dict:
    """
    Return the AgentGraph topology for the given agent.
    Reads entity.json from the configured blueprint path (or auto-discovers from identity.json).
    Returns {"graph": {...}} or {"error": "..."}.
    """
    graph_data = _get_graph_data(agent_name)
    if graph_data is None:
        return {"error": f"Graph data not found for agent '{agent_name}'"}
    return {"agent": agent_name, "graph": graph_data}


# ── REST: recent events ────────────────────────────────────────────────────────

@app.get("/api/events/{agent_name}")
async def api_events(agent_name: str, limit: int = 100) -> dict:
    """Return the most recent observability events for the given agent."""
    from observability.hub import get_hub
    hub = get_hub()
    events = hub.get_recent_events(agent_name, limit)
    return {"agent": agent_name, "events": events}


# ── Static frontend ────────────────────────────────────────────────────────────

_DIST = Path(__file__).parent / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
else:
    @app.get("/")
    async def root() -> dict:
        return {
            "status": "ZenithLoom Observability Server running",
            "note": "Frontend not built yet. Run: cd observability/frontend && npm install && npm run build",
            "endpoints": ["/api/agents", "/api/graph/{agent_name}", "/ws/events", "/ingest"],
        }


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
