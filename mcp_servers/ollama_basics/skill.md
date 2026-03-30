# Ollama Basics MCP — Usage Guide

This MCP server provides basic operating capabilities for Ollama LLM nodes, giving them file operations, bash execution, search, and web fetching abilities that Claude and Gemini have natively.

## Available Tools

### read_file
Read a file's contents from the filesystem.
- **path** (str, required): Absolute or relative file path.
- Returns: `content`, `line_count`, `truncated` flag. Limited to 50,000 characters.

### write_file
Write content to a file, creating parent directories if needed.
- **path** (str, required): File path to write.
- **content** (str, required): Text content to write.
- Returns: `bytes_written`.

### list_directory
List files and directories at a given path.
- **path** (str, optional, default "."): Directory path.
- Returns: `entries` list with `name`, `type` (file/dir), `size`.

### run_command
Execute a bash command with timeout enforcement.
- **command** (str, required): Shell command to run.
- **timeout** (int, optional, default 30): Max seconds (capped at 120).
- Returns: `stdout`, `stderr`, `exit_code`.

### search_files
Search for files by glob pattern.
- **pattern** (str, required): Glob pattern (e.g. `**/*.py`).
- **path** (str, optional, default "."): Root directory.
- **max_results** (int, optional, default 20): Result limit.
- Returns: `matches` list of file paths.

### grep_content
Search file contents using regex.
- **pattern** (str, required): Regular expression to search for.
- **path** (str, optional, default "."): Root directory.
- **file_glob** (str, optional, default "*"): Filter which files to search.
- **max_results** (int, optional, default 20): Result limit.
- Returns: `results` list with `file`, `line_number`, `line`.

### web_fetch
Fetch a URL and return plain text content (HTML tags stripped for web pages).
- **url** (str, required): HTTP or HTTPS URL.
- Returns: `content`, `content_type`, `length`, `truncated`. Limited to 50,000 characters.

## Response Format

All tools return a dict with an `ok` boolean field:
- `ok: true` — success, plus tool-specific fields
- `ok: false` — failure, plus `error` string describing the problem

## Connection

- **Transport**: SSE
- **Endpoint**: `http://127.0.0.1:8102/sse`
- **Startup**: Auto (shared across agents)

## Examples

```
# Read a Python file
read_file(path="/home/user/project/main.py")

# Write a config file
write_file(path="/tmp/config.json", content='{"key": "value"}')

# List project root
list_directory(path="/home/user/project")

# Run a git command
run_command(command="git status", timeout=10)

# Find all Python files
search_files(pattern="**/*.py", path="/home/user/project")

# Search for a function definition
grep_content(pattern="def main\\(", path="/home/user/project", file_glob="*.py")

# Fetch a web page
web_fetch(url="https://example.com")
```
