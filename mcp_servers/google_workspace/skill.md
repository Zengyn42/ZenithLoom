# Google Workspace MCP Server

Provides tools to interact with Google Workspace services (Gmail, Drive, Slides, Docs) via the `gws` CLI.

## Available Tools

- **gws_gmail_read** - Read Gmail messages. Pass a Gmail search query (default: `is:unread`).
- **gws_drive_list** - List files in Google Drive. Optionally pass a folder name/ID to scope the listing.
- **gws_slides_exec** - Execute a Google Slides command. The `command` parameter must be a full `gws slides ...` command string.
- **gws_docs_exec** - Execute a Google Docs command. The `command` parameter must be a full `gws docs ...` command string.

## Usage Notes

- All commands are validated for shell injection safety. Shell metacharacters (`$ \` | ; & > <`) are rejected.
- Commands are executed via `subprocess` with argument lists (no `shell=True`).
- The server wraps the existing `gws` CLI tool; ensure it is installed and authenticated before use.

## Transport

Default SSE endpoint: `http://127.0.0.1:8103/sse`
