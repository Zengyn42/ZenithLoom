"""
Tool Discovery — DETERMINISTIC node functions.

DeterministicNode 按 node id 查找同名函数：
  - search_aggregate(state) → dict   GitHub API + Web 搜索聚合
  - sandbox_eval(state) → dict        沙箱评估（Docker / venv / 静态分析降级）

每个函数签名: (state: dict) -> dict
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
import urllib.parse
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"


def _github_headers() -> dict:
    """Build GitHub API headers with optional token."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "BootstrapBuilder-ToolDiscovery/1.0",
    }
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_search(query: str, per_page: int = 10) -> list[dict]:
    """Search GitHub repositories. Returns list of repo dicts."""
    params = urllib.parse.urlencode({
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
    })
    url = f"{GITHUB_API}/search/repositories?{params}"
    req = urllib.request.Request(url, headers=_github_headers())

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data.get("items", [])
    except Exception as e:
        logger.warning(f"[search_aggregate] GitHub search failed for '{query}': {e}")
        return []


def _normalize_candidate(item: dict) -> dict:
    """Normalize a GitHub API repo item to our CandidateRepo schema."""
    return {
        "repo": item.get("full_name", ""),
        "url": item.get("html_url", ""),
        "stars": item.get("stargazers_count", 0),
        "forks": item.get("forks_count", 0),
        "last_commit": item.get("pushed_at", ""),
        "license": (item.get("license") or {}).get("spdx_id", "unknown"),
        "language": item.get("language", "unknown"),
        "description": item.get("description", "") or "",
        "topics": item.get("topics", []),
        "open_issues": item.get("open_issues_count", 0),
        "archived": item.get("archived", False),
    }


def _dedup_candidates(candidates: list[dict]) -> list[dict]:
    """Deduplicate by repo full_name, keeping first occurrence (highest stars)."""
    seen = set()
    result = []
    for c in candidates:
        key = c["repo"].lower()
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


def _extract_json_from_message(state: dict) -> str:
    """Extract the last AI message content from state messages."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content:
            return content
    return ""


def _parse_json_from_llm(text: str) -> dict | list | None:
    """Parse JSON from LLM output, handling markdown code fences."""
    # Try to find JSON in code fence
    match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if match:
        text = match.group(1)

    # Direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# search_aggregate — DETERMINISTIC node
# ---------------------------------------------------------------------------

def search_aggregate(state: dict) -> dict:
    """
    GitHub API 搜索聚合节点。

    读取上游 query_expand 的 JSON 输出（messages 中最后一条 AI 消息），
    执行多个 GitHub 搜索查询，聚合去重，写入 raw_candidates。
    """
    config_str = state.get("discovery_config", "")
    config = {}
    if config_str:
        try:
            config = json.loads(config_str)
        except json.JSONDecodeError:
            pass
    depth = config.get("depth", 5)  # Max candidates per query

    errors = []

    # Parse query_expand output from last message
    last_content = _extract_json_from_message(state)
    search_intent = _parse_json_from_llm(last_content)

    if not search_intent or not isinstance(search_intent, dict):
        errors.append(f"Failed to parse search_intent from query_expand output: {last_content[:200]}")
        return {
            "search_intent": last_content,
            "raw_candidates": "[]",
            "discovery_errors": json.dumps(errors, ensure_ascii=False),
        }

    # Execute GitHub searches
    all_candidates = []

    github_queries = search_intent.get("github_queries", [])
    keywords = search_intent.get("keywords", [])

    # GitHub structured queries
    for query in github_queries[:5]:  # Cap at 5 queries
        items = _github_search(query, per_page=depth)
        for item in items:
            if not item.get("archived", False):
                all_candidates.append(_normalize_candidate(item))

    # Keyword fallback search
    for kw in keywords[:3]:
        if not all_candidates:  # Only if structured queries yielded nothing
            items = _github_search(kw, per_page=depth)
            for item in items:
                if not item.get("archived", False):
                    all_candidates.append(_normalize_candidate(item))

    # Deduplicate
    unique = _dedup_candidates(all_candidates)

    logger.info(
        f"[search_aggregate] {len(github_queries)} queries → "
        f"{len(all_candidates)} raw → {len(unique)} unique candidates"
    )

    from langchain_core.messages import AIMessage

    # Build context message for candidate_filter
    user_query = state.get("user_query", "") or state.get("routing_context", "")
    filter_prompt = (
        f"用户需求：{user_query}\n\n"
        f"搜索意图：{json.dumps(search_intent, ensure_ascii=False)}\n\n"
        f"候选仓库（共 {len(unique)} 个）：\n"
        f"{json.dumps(unique, ensure_ascii=False, indent=2)}"
    )

    return {
        "messages": [AIMessage(content=filter_prompt)],
        "search_intent": json.dumps(search_intent, ensure_ascii=False),
        "raw_candidates": json.dumps(unique, ensure_ascii=False),
        "discovery_errors": json.dumps(errors, ensure_ascii=False) if errors else "",
    }


# ---------------------------------------------------------------------------
# sandbox_eval — DETERMINISTIC node
# ---------------------------------------------------------------------------

def _detect_project_type(repo_dir: Path) -> str:
    """Detect project type from files in cloned repo."""
    if (repo_dir / "pyproject.toml").exists() or (repo_dir / "setup.py").exists():
        return "python"
    if (repo_dir / "package.json").exists():
        return "node"
    if (repo_dir / "Cargo.toml").exists():
        return "rust"
    if (repo_dir / "go.mod").exists():
        return "go"
    return "unknown"


def _count_code_lines(repo_dir: Path, lang: str) -> int:
    """Rough line count for primary language files."""
    exts = {
        "python": [".py"],
        "node": [".js", ".ts", ".mjs"],
        "rust": [".rs"],
        "go": [".go"],
    }.get(lang, [".py", ".js"])
    count = 0
    for ext in exts:
        for f in repo_dir.rglob(f"*{ext}"):
            try:
                count += sum(1 for _ in f.open())
            except Exception:
                pass
    return count


def _static_scan_suspicious(repo_dir: Path) -> list[str]:
    """
    Static scan for suspicious patterns in setup/build files.
    Returns list of warnings.
    """
    warnings = []
    suspicious_patterns = [
        r"\bos\.system\b",
        r"\bsubprocess\b",
        r"\beval\b\(",
        r"\bexec\b\(",
        r"\b__import__\b",
    ]
    scan_files = [
        "setup.py", "pyproject.toml", "setup.cfg",
        "Makefile", "install.sh", "postinstall.sh",
    ]
    for fname in scan_files:
        fpath = repo_dir / fname
        if fpath.exists():
            try:
                content = fpath.read_text(errors="replace")
                for pattern in suspicious_patterns:
                    if re.search(pattern, content):
                        warnings.append(f"Suspicious pattern '{pattern}' in {fname}")
            except Exception:
                pass
    return warnings


def _eval_single_repo_venv(repo_url: str, repo_name: str, tmpdir: str, timeout: int = 120) -> dict:
    """
    Evaluate a single repo using venv (降级方案: no Docker).

    Steps: clone → detect type → static scan → venv install → run tests
    """
    result = {
        "repo": repo_name,
        "clone_ok": False,
        "install_ok": False,
        "install_time_s": 0.0,
        "test_count": 0,
        "test_pass_rate": 0.0,
        "dependency_count": 0,
        "code_lines": 0,
        "has_docs": False,
        "has_examples": False,
        "project_type": "unknown",
        "eval_mode": "venv",
        "errors": [],
        "security_warnings": [],
    }

    repo_dir = Path(tmpdir) / repo_name.replace("/", "_")

    # 1. Clone (shallow)
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(repo_dir)],
            capture_output=True, text=True, timeout=60,
        )
        if not repo_dir.exists():
            result["errors"].append("Clone produced no directory")
            return result
        result["clone_ok"] = True
    except subprocess.TimeoutExpired:
        result["errors"].append("Clone timed out (60s)")
        return result
    except Exception as e:
        result["errors"].append(f"Clone failed: {e}")
        return result

    # 2. Detect project type
    project_type = _detect_project_type(repo_dir)
    result["project_type"] = project_type

    # 3. Static security scan
    warnings = _static_scan_suspicious(repo_dir)
    result["security_warnings"] = warnings

    # 4. Metadata
    result["code_lines"] = _count_code_lines(repo_dir, project_type)
    result["has_docs"] = (repo_dir / "docs").is_dir() or (repo_dir / "README.md").exists()
    result["has_examples"] = (repo_dir / "examples").is_dir() or (repo_dir / "example").is_dir()

    # 5. Install (Python only for now)
    if project_type == "python":
        venv_dir = repo_dir / ".venv"
        try:
            # Create venv
            subprocess.run(
                ["python3", "-m", "venv", str(venv_dir)],
                capture_output=True, text=True, timeout=30,
            )
            pip = str(venv_dir / "bin" / "pip")

            # Install
            import time
            t0 = time.time()
            install_result = subprocess.run(
                [pip, "install", "-e", ".", "--no-deps", "-q"],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(repo_dir),
            )
            result["install_time_s"] = round(time.time() - t0, 1)
            result["install_ok"] = install_result.returncode == 0
            if install_result.returncode != 0:
                result["errors"].append(
                    f"Install failed (exit={install_result.returncode}): "
                    f"{install_result.stderr[:300]}"
                )

            # Count dependencies
            try:
                freeze = subprocess.run(
                    [pip, "freeze"], capture_output=True, text=True, timeout=10,
                )
                result["dependency_count"] = len(freeze.stdout.strip().splitlines())
            except Exception:
                pass

            # 6. Run tests (best-effort)
            if result["install_ok"]:
                python = str(venv_dir / "bin" / "python")
                try:
                    test_result = subprocess.run(
                        [python, "-m", "pytest", "--tb=no", "-q", "--no-header"],
                        capture_output=True, text=True, timeout=60,
                        cwd=str(repo_dir),
                    )
                    # Parse pytest output: "X passed, Y failed"
                    out = test_result.stdout
                    passed = 0
                    failed = 0
                    m = re.search(r"(\d+) passed", out)
                    if m:
                        passed = int(m.group(1))
                    m = re.search(r"(\d+) failed", out)
                    if m:
                        failed = int(m.group(1))
                    total = passed + failed
                    result["test_count"] = total
                    result["test_pass_rate"] = round(passed / total, 2) if total > 0 else 0.0
                except subprocess.TimeoutExpired:
                    result["errors"].append("Tests timed out (60s)")
                except Exception as e:
                    result["errors"].append(f"Test run error: {e}")

        except subprocess.TimeoutExpired:
            result["errors"].append("Venv creation timed out")
        except Exception as e:
            result["errors"].append(f"Venv setup error: {e}")

    elif project_type == "node":
        # Node.js: npm install + npm test
        try:
            import time
            t0 = time.time()
            install_result = subprocess.run(
                ["npm", "install", "--ignore-scripts"],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(repo_dir),
            )
            result["install_time_s"] = round(time.time() - t0, 1)
            result["install_ok"] = install_result.returncode == 0
            if install_result.returncode != 0:
                result["errors"].append(
                    f"npm install failed: {install_result.stderr[:300]}"
                )

            # Count deps
            try:
                pkg = json.loads((repo_dir / "package.json").read_text())
                deps = pkg.get("dependencies", {})
                dev_deps = pkg.get("devDependencies", {})
                result["dependency_count"] = len(deps) + len(dev_deps)
            except Exception:
                pass

        except subprocess.TimeoutExpired:
            result["errors"].append("npm install timed out")
        except Exception as e:
            result["errors"].append(f"Node setup error: {e}")

    else:
        # Static analysis only for other languages
        result["eval_mode"] = "static"

    return result


def _eval_single_repo_docker(repo_url: str, repo_name: str, project_type: str, timeout: int = 300) -> dict:
    """
    Evaluate a single repo using Docker sandbox.

    Uses pre-built sandbox images with resource limits.
    """
    result = {
        "repo": repo_name,
        "clone_ok": False,
        "install_ok": False,
        "install_time_s": 0.0,
        "test_count": 0,
        "test_pass_rate": 0.0,
        "dependency_count": 0,
        "code_lines": 0,
        "has_docs": False,
        "has_examples": False,
        "project_type": project_type,
        "eval_mode": "docker",
        "errors": [],
        "security_warnings": [],
    }

    # Choose Docker image
    image_map = {
        "python": "bb-sandbox-python:latest",
        "node": "bb-sandbox-node:latest",
    }
    image = image_map.get(project_type)
    if not image:
        result["eval_mode"] = "static"
        result["errors"].append(f"No Docker sandbox for {project_type}, static analysis only")
        return result

    # Check if image exists
    check = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        result["eval_mode"] = "venv"
        result["errors"].append(f"Docker image {image} not found, falling back to venv")
        return result

    # Run in Docker
    docker_cmd = [
        "docker", "run", "--rm",
        "--memory=1g", "--cpus=1", "--pids-limit=256",
        "--read-only",
        "--tmpfs", "/workspace:rw,size=512m",
        "--tmpfs", "/tmp:rw,size=256m",
        "--network=bridge",
        image,
        "/bin/bash", "-c",
        f"cd /workspace && git clone --depth=1 {repo_url} repo && cd repo && "
        f"if [ -f requirements.txt ]; then pip install -r requirements.txt -q 2>&1; "
        f"elif [ -f pyproject.toml ] || [ -f setup.py ]; then pip install -e . -q 2>&1; "
        f"elif [ -f package.json ]; then npm install --ignore-scripts 2>&1; fi && "
        f"echo '===INSTALL_OK===' && "
        f"python -m pytest --tb=no -q --no-header 2>&1 || npm test 2>&1 || echo '===NO_TESTS==='"
    ]

    try:
        proc = subprocess.run(
            docker_cmd,
            capture_output=True, text=True, timeout=timeout,
        )
        output = proc.stdout
        result["clone_ok"] = "===INSTALL_OK===" in output or proc.returncode == 0
        result["install_ok"] = "===INSTALL_OK===" in output

        # Parse test results from output
        m = re.search(r"(\d+) passed", output)
        if m:
            passed = int(m.group(1))
            failed_m = re.search(r"(\d+) failed", output)
            failed = int(failed_m.group(1)) if failed_m else 0
            total = passed + failed
            result["test_count"] = total
            result["test_pass_rate"] = round(passed / total, 2) if total > 0 else 0.0

    except subprocess.TimeoutExpired:
        result["errors"].append(f"Docker evaluation timed out ({timeout}s)")
    except Exception as e:
        result["errors"].append(f"Docker evaluation error: {e}")

    return result


def _is_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def sandbox_eval(state: dict) -> dict:
    """
    沙箱评估节点。

    读取上游 candidate_filter 的 JSON 输出（筛选后的候选列表），
    对每个候选执行：克隆 → 安装 → 测试。

    Docker 可用时使用 Docker 沙箱，否则降级到 venv + 静态分析。
    """
    errors = json.loads(state.get("discovery_errors", "[]") or "[]")
    config_str = state.get("discovery_config", "")
    config = {}
    if config_str:
        try:
            config = json.loads(config_str)
        except json.JSONDecodeError:
            pass
    timeout = config.get("timeout", 120)

    # Parse filtered candidates from last AI message
    last_content = _extract_json_from_message(state)
    filtered = _parse_json_from_llm(last_content)

    if not filtered or not isinstance(filtered, list):
        errors.append(f"Failed to parse filtered_candidates: {last_content[:200]}")
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(content=f"评估失败：无法解析候选列表。\n错误: {errors}")],
            "filtered_candidates": last_content,
            "evaluation_results": "[]",
            "discovery_errors": json.dumps(errors, ensure_ascii=False),
        }

    use_docker = _is_docker_available()
    eval_results = []

    with tempfile.TemporaryDirectory(prefix="bb_discovery_") as tmpdir:
        for candidate in filtered[:5]:  # Max 5 repos
            repo_name = candidate.get("repo", "")
            repo_url = candidate.get("url", "")
            if not repo_url:
                repo_url = f"https://github.com/{repo_name}.git"
            elif not repo_url.endswith(".git"):
                repo_url = repo_url + ".git"

            logger.info(f"[sandbox_eval] evaluating {repo_name} (docker={use_docker})")

            if use_docker:
                project_type = candidate.get("language", "unknown").lower()
                if project_type in ("python", "javascript", "typescript"):
                    project_type = {"javascript": "node", "typescript": "node"}.get(
                        project_type, project_type
                    )
                result = _eval_single_repo_docker(repo_url, repo_name, project_type, timeout)
                # If Docker fallback to venv
                if result.get("eval_mode") == "venv":
                    result = _eval_single_repo_venv(repo_url, repo_name, tmpdir, timeout)
            else:
                result = _eval_single_repo_venv(repo_url, repo_name, tmpdir, timeout)

            eval_results.append(result)

    logger.info(f"[sandbox_eval] evaluated {len(eval_results)} repos")

    # Build context message for report_gen
    user_query = state.get("user_query", "") or state.get("routing_context", "")
    from langchain_core.messages import AIMessage

    report_context = (
        f"用户需求：{user_query}\n\n"
        f"筛选后的候选工具：\n{json.dumps(filtered, ensure_ascii=False, indent=2)}\n\n"
        f"沙箱评估结果：\n{json.dumps(eval_results, ensure_ascii=False, indent=2)}"
    )

    return {
        "messages": [AIMessage(content=report_context)],
        "filtered_candidates": json.dumps(filtered, ensure_ascii=False),
        "evaluation_results": json.dumps(eval_results, ensure_ascii=False),
        "discovery_errors": json.dumps(errors, ensure_ascii=False) if errors else "",
    }
