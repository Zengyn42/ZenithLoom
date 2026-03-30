# Vault Sync MCP — Usage Guide

## Overview

The Vault Sync MCP server wraps `rsync` to synchronise the Obsidian Vault between Windows (`/mnt/d/KnowledgeBase/Vault/`) and WSL (`/home/kingy/Foundation/Vault/`). It provides three tools for pull, push, and status operations.

## Tools

### `vault_sync_pull`
Sync the vault **from Windows to WSL**. This is the standard "start of session" operation — pull the latest vault state from Windows before reading or editing notes.

- Command: `rsync -a --delete <win_path> <wsl_path>`
- Use case: Before working with notes in WSL, pull the latest from Windows/Obsidian.

### `vault_sync_push`
Sync the vault **from WSL to Windows**. This is the standard "end of session" operation — push changes back so Obsidian on Windows sees them.

- Command: `rsync -a --delete <wsl_path> <win_path>`
- Use case: After editing notes in WSL, push changes back to Windows/Obsidian.

### `vault_sync_status`
Dry-run both directions to preview what would change **without making any changes**. Useful for checking whether a sync is needed and what files differ.

- Returns pull and push previews with file lists.

## Typical Workflow

1. **Pull** before reading/editing: `vault_sync_pull`
2. Work with notes via the Obsidian MCP or other tools
3. **Push** after changes: `vault_sync_push`

## Configuration

Paths are configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `VAULT_WSL_PATH` | `/home/kingy/Foundation/Vault/` | WSL-side vault directory |
| `VAULT_WIN_PATH` | `/mnt/d/KnowledgeBase/Vault/` | Windows-side vault directory |

## Response Format

All tools return a structured dict:

```json
{
  "success": true,
  "return_code": 0,
  "files_changed": 5,
  "changed": ["notes/example.md", "attachments/image.png"],
  "summary": "5 file(s) changed",
  "direction": "pull (Windows -> WSL)",
  "source": "/mnt/d/KnowledgeBase/Vault/",
  "destination": "/home/kingy/Foundation/Vault/",
  "stderr": null
}
```

## Startup

```bash
# stdio (default)
python -m mcp_servers.vault_sync.server

# SSE on port 8105
python -m mcp_servers.vault_sync.server --transport sse --port 8105
```
