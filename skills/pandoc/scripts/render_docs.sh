#!/bin/bash
# render_docs.sh — 调用 Pandoc 生成专业的 DOCX 文档
#
# Usage: render_docs.sh <markdown_content>
#
# Arguments:
#   markdown_content — Markdown 内容文本（直接传入，非文件路径）
#
# 由 EXTERNAL_TOOL 节点调用，markdown_content 来自 routing_context。

set -euo pipefail

MARKDOWN_CONTENT="${1:?Usage: render_docs.sh <markdown_content>}"
OUTPUT="/tmp/document_$(date +%s).docx"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_DIR="${SCRIPT_DIR}/../templates"
REFERENCE_DOC="${TEMPLATE_DIR}/professional.docx"

if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc not installed. Run: sudo apt install pandoc" >&2
    exit 1
fi

# 写入临时 Markdown 文件
TMPFILE=$(mktemp /tmp/doc_content_XXXXXX.md)
echo "$MARKDOWN_CONTENT" > "$TMPFILE"
trap "rm -f '$TMPFILE'" EXIT

PANDOC_ARGS=("$TMPFILE" -o "$OUTPUT")

if [ -f "$REFERENCE_DOC" ]; then
    PANDOC_ARGS+=(--reference-doc="$REFERENCE_DOC")
fi

pandoc "${PANDOC_ARGS[@]}"

if [ -f "$OUTPUT" ]; then
    FILE_SIZE=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT" 2>/dev/null)
    echo "Generated: $OUTPUT ($FILE_SIZE bytes)"
else
    echo "Error: Failed to generate DOCX file" >&2
    exit 1
fi
