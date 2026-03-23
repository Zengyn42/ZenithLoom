#!/bin/bash
# render_docs.sh — 调用 Pandoc 生成专业的 DOCX 文档
#
# Usage: render_docs.sh <markdown_file> [output_path]
#
# Arguments:
#   markdown_file — Markdown 内容文件路径
#   output_path   — 输出 DOCX 文件路径 (默认: /tmp/document.docx)

set -euo pipefail

MARKDOWN_FILE="${1:?Usage: render_docs.sh <markdown_file> [output_path]}"
OUTPUT="${2:-/tmp/document.docx}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_DIR="${SCRIPT_DIR}/../templates"
REFERENCE_DOC="${TEMPLATE_DIR}/professional.docx"

if [ ! -f "$MARKDOWN_FILE" ]; then
    echo "Error: Markdown file not found: $MARKDOWN_FILE" >&2
    exit 1
fi

if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc not installed. Run: sudo apt install pandoc" >&2
    exit 1
fi

echo "Generating document with Pandoc..."
echo "  Input: $MARKDOWN_FILE"

PANDOC_ARGS=("$MARKDOWN_FILE" -o "$OUTPUT")

if [ -f "$REFERENCE_DOC" ]; then
    PANDOC_ARGS+=(--reference-doc="$REFERENCE_DOC")
    echo "  Template: $REFERENCE_DOC"
else
    echo "  Template: (none, using Pandoc default)"
fi

pandoc "${PANDOC_ARGS[@]}"

if [ -f "$OUTPUT" ]; then
    FILE_SIZE=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT" 2>/dev/null)
    echo "Generated: $OUTPUT ($FILE_SIZE bytes)"
else
    echo "Error: Failed to generate DOCX file" >&2
    exit 1
fi
