"""
test_gemini_session_migrate — _find_session_file fallback + migration

When workspace (project folder) changes, the Gemini session file lives under
the old project_id directory.  _find_session_file should:
  1. Find it via fallback scan of all project directories
  2. Migrate (rename) it to the new project_id directory
  3. Return the new path
"""

import json
import uuid
from pathlib import Path

import pytest

from framework.nodes.llm import gemini_session as gs


@pytest.fixture()
def gemini_tmp(tmp_path, monkeypatch):
    """Redirect _GEMINI_DIR to a temp directory."""
    monkeypatch.setattr(gs, "_GEMINI_DIR", tmp_path)
    return tmp_path


def _create_session_file(gemini_tmp: Path, project_id: str, session_id: str) -> Path:
    """Helper: create a minimal session JSON file under the given project_id."""
    chats = gemini_tmp / project_id / "chats"
    chats.mkdir(parents=True, exist_ok=True)
    filename = f"session-2026-03-25-12-00-{session_id[:8]}.json"
    path = chats / filename
    data = {
        "sessionId": session_id,
        "projectHash": project_id,
        "startTime": "2026-03-25T12:00:00+00:00",
        "lastUpdated": "2026-03-25T12:00:00+00:00",
        "messages": [],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestFindSessionFileFallback:
    """Test _find_session_file with cross-project fallback and migration."""

    def test_find_in_same_project(self, gemini_tmp):
        """Session in the correct project dir — found directly, no migration."""
        sid = str(uuid.uuid4())
        original = _create_session_file(gemini_tmp, "myproject", sid)

        result = gs._find_session_file(sid, "myproject")

        assert result is not None
        assert result == original
        assert result.exists()

    def test_fallback_and_migrate(self, gemini_tmp):
        """Session in old project dir — found via fallback, migrated to new dir."""
        sid = str(uuid.uuid4())
        old_path = _create_session_file(gemini_tmp, "old-project", sid)

        result = gs._find_session_file(sid, "new-project")

        # Should return path under new project
        assert result is not None
        assert "new-project" in str(result)
        assert result.exists()

        # Old file should no longer exist
        assert not old_path.exists()

        # Verify content integrity after migration
        data = json.loads(result.read_text(encoding="utf-8"))
        assert data["sessionId"] == sid

    def test_not_found_anywhere(self, gemini_tmp):
        """Session doesn't exist at all — returns None."""
        sid = str(uuid.uuid4())

        result = gs._find_session_file(sid, "some-project")

        assert result is None

    def test_no_gemini_dir(self, gemini_tmp):
        """_GEMINI_DIR doesn't exist — returns None without error."""
        import shutil
        shutil.rmtree(gemini_tmp)

        sid = str(uuid.uuid4())
        result = gs._find_session_file(sid, "any-project")

        assert result is None

    def test_multiple_projects_finds_correct_one(self, gemini_tmp):
        """Multiple projects with different sessions — finds the right one."""
        sid_a = str(uuid.uuid4())
        sid_b = str(uuid.uuid4())
        _create_session_file(gemini_tmp, "project-a", sid_a)
        _create_session_file(gemini_tmp, "project-b", sid_b)

        # Look for sid_a from project-c (neither a nor b)
        result = gs._find_session_file(sid_a, "project-c")

        assert result is not None
        assert "project-c" in str(result)
        data = json.loads(result.read_text(encoding="utf-8"))
        assert data["sessionId"] == sid_a

        # sid_b should still be in project-b (untouched)
        b_chats = gemini_tmp / "project-b" / "chats"
        b_files = list(b_chats.glob(f"session-*-{sid_b[:8]}.json"))
        assert len(b_files) == 1
