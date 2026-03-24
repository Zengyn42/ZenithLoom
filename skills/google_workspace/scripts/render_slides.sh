#!/bin/bash
# render_slides.sh — 调用 Presenton API 生成漂亮的 PDF slides
#
# Usage: render_slides.sh <content_text>
#
# Arguments:
#   content_text — slides 内容文本（直接传入，非文件路径）
#
# 由 EXTERNAL_TOOL 节点调用，content_text 来自 routing_context。
# 注意：Presenton PPTX 导出有 bug (#442)，暂用 PDF 导出。

set -euo pipefail

CONTENT="${1:?Usage: render_slides.sh <content_text>}"
OUTPUT="/tmp/presentation_$(date +%s).pdf"
N_SLIDES=10
TEMPLATE="general"

PRESENTON_URL="${PRESENTON_API_URL:-http://localhost:5000}"
PRESENTON_CONTAINER="${PRESENTON_CONTAINER_NAME:-presenton}"
API_URL="${PRESENTON_URL}/api/v1/ppt/presentation/generate"
API_KEY="${PRESENTON_API_KEY:-}"

# --- 按需启动 Presenton Docker 容器 ---
ensure_presenton_running() {
    if curl -s --max-time 3 "$PRESENTON_URL" > /dev/null 2>&1; then
        return 0
    fi

    echo "Presenton not running. Starting Docker container..." >&2

    if ! command -v docker &> /dev/null; then
        echo "Error: docker not installed" >&2
        exit 1
    fi

    if docker ps -a --format '{{.Names}}' | grep -q "^${PRESENTON_CONTAINER}$"; then
        docker start "$PRESENTON_CONTAINER"
    else
        echo "Error: Presenton container '$PRESENTON_CONTAINER' not found." >&2
        exit 1
    fi

    echo -n "Waiting for Presenton to be ready" >&2
    for i in $(seq 1 30); do
        if curl -s --max-time 2 "$PRESENTON_URL" > /dev/null 2>&1; then
            echo " ready!" >&2
            return 0
        fi
        echo -n "." >&2
        sleep 2
    done

    echo " timeout!" >&2
    exit 1
}

# 用完自动停止容器
stop_presenton() {
    if command -v docker &> /dev/null && \
       docker ps --format '{{.Names}}' | grep -q "^${PRESENTON_CONTAINER}$"; then
        docker stop "$PRESENTON_CONTAINER" > /dev/null 2>&1
    fi
}
trap stop_presenton EXIT

ensure_presenton_running

AUTH_HEADER=()
if [ -n "$API_KEY" ]; then
    AUTH_HEADER=(-H "Authorization: Bearer $API_KEY")
fi

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL" \
    "${AUTH_HEADER[@]}" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
        --arg content "$CONTENT" \
        --argjson n_slides "$N_SLIDES" \
        --arg template "$TEMPLATE" \
        '{
            content: $content,
            n_slides: $n_slides,
            language: "Chinese",
            template: $template,
            export_as: "pdf"
        }')")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -ne 200 ]; then
    echo "Error: Presenton API returned HTTP $HTTP_CODE" >&2
    echo "$BODY" >&2
    exit 1
fi

DOWNLOAD_URL=$(echo "$BODY" | jq -r '.path // empty')

if [ -z "$DOWNLOAD_URL" ]; then
    echo "Error: No download URL in response" >&2
    echo "$BODY" >&2
    exit 1
fi

if [[ "$DOWNLOAD_URL" == /* ]]; then
    DOWNLOAD_URL="${PRESENTON_URL}${DOWNLOAD_URL}"
fi

curl -s -o "$OUTPUT" "$DOWNLOAD_URL"

if [ -f "$OUTPUT" ]; then
    FILE_SIZE=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT" 2>/dev/null)
    echo "Generated: $OUTPUT ($FILE_SIZE bytes)"
else
    echo "Error: Failed to download PPTX file" >&2
    exit 1
fi
