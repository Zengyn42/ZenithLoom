# Document Render MCP Server

Provides tools to render Markdown content into PDF slides (via Presenton) and DOCX documents (via Pandoc).

## Available Tools

- **render_slides** - Render content to PDF slides via the Presenton API. Pass the slide content as a string. The Presenton Docker container is started automatically if not running and stopped after use.
- **render_docs** - Render Markdown content to DOCX (or other Pandoc-supported formats). Pass the Markdown content as a string and optionally a filename and output format.

## Usage Notes

- `render_slides` calls the Presenton API (`/api/v1/ppt/presentation/generate`) running in a Docker container. The container is auto-started if stopped, and auto-stopped after rendering to conserve resources.
- `render_docs` uses `pandoc` via subprocess. If a reference template (`skills/pandoc/templates/professional.docx`) exists, it is applied automatically.
- Output files are written to `/tmp/` and the full path is returned in the response.
- Both tools accept an optional `filename` parameter; if omitted, a timestamped default name is generated.

## Transport

Default SSE endpoint: `http://127.0.0.1:8104/sse`
