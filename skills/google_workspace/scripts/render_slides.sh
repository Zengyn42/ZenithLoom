#!/bin/bash
# render_slides.sh — 调用 Presenton API 生成漂亮的 PPTX slides
#
# Usage: render_slides.sh <content_file> [output_path] [n_slides] [template]
#
# Arguments:
#   content_file  — 文本/Markdown 内容文件路径
#   output_path   — 输出 PPTX 文件路径 (默认: /tmp/presentation.pptx)
#   n_slides      — 生成 slides 数量 (默认: 10)
#   template      — 模板名: general, modern, swift (默认: modern)

set -euo pipefail

CONTENT_FILE="${1:?Usage: render_slides.sh <content_file> [output_path] [n_slides] [template]}"
OUTPUT="${2:-/tmp/presentation.pptx}"
N_SLIDES="${3:-10}"
TEMPLATE="${4:-modern}"

PRESENTON_URL="${PRESENTON_API_URL:-http://localhost:5000}"
PRESENTON_CONTAINER="${PRESENTON_CONTAINER_NAME:-presenton}"
API_URL="${PRESENTON_URL}/api/v1/ppt/presentation/generate"
API_KEY="${PRESENTON_API_KEY:?Error: PRESENTON_API_KEY environment variable not set}"

# --- 按需启动 Presenton Docker 容器 ---
ensure_presenton_running() {
    # 检查服务是否已在运行
    if curl -s --max-time 3 "$PRESENTON_URL" > /dev/null 2>&1; then
        return 0
    fi

    echo "Presenton not running. Starting Docker container..."

    if ! command -v docker &> /dev/null; then
        echo "Error: docker not installed" >&2
        exit 1
    fi

    # 容器存在但停止了 → 启动
    if docker ps -a --format '{{.Names}}' | grep -q "^${PRESENTON_CONTAINER}$"; then
        docker start "$PRESENTON_CONTAINER"
    else
        echo "Error: Presenton container '$PRESENTON_CONTAINER' not found." >&2
        echo "Create it first with:" >&2
        echo "  docker run -d --name presenton -p 5000:80 \\" >&2
        echo "    -e LLM=google -e GOOGLE_API_KEY=your-key \\" >&2
        echo "    -v \"\$HOME/Foundation/Tools/presenton_data:/app_data\" \\" >&2
        echo "    ghcr.io/presenton/presenton:latest" >&2
        exit 1
    fi

    # 等待服务就绪（最多 60 秒）
    echo -n "Waiting for Presenton to be ready"
    for i in $(seq 1 30); do
        if curl -s --max-time 2 "$PRESENTON_URL" > /dev/null 2>&1; then
            echo " ready!"
            return 0
        fi
        echo -n "."
        sleep 2
    done

    echo " timeout!" >&2
    echo "Error: Presenton failed to start within 60 seconds" >&2
    exit 1
}

ensure_presenton_running

if [ ! -f "$CONTENT_FILE" ]; then
    echo "Error: Content file not found: $CONTENT_FILE" >&2
    exit 1
fi

CONTENT=$(cat "$CONTENT_FILE")

echo "Generating slides with Presenton..."
echo "  Content: $CONTENT_FILE"
echo "  Slides: $N_SLIDES"
echo "  Template: $TEMPLATE"

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL" \
    -H "Authorization: Bearer $API_KEY" \
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
            export_as: "pptx"
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

# 如果是相对路径，补全为完整 URL
if [[ "$DOWNLOAD_URL" == /* ]]; then
    DOWNLOAD_URL="${PRESENTON_URL}${DOWNLOAD_URL}"
fi

curl -s -o "$OUTPUT" "$DOWNLOAD_URL"

# 用完自动停止容器（释放资源）
stop_presenton() {
    if command -v docker &> /dev/null && \
       docker ps --format '{{.Names}}' | grep -q "^${PRESENTON_CONTAINER}$"; then
        echo "Stopping Presenton container..."
        docker stop "$PRESENTON_CONTAINER" > /dev/null 2>&1
    fi
}
trap stop_presenton EXIT

if [ -f "$OUTPUT" ]; then
    FILE_SIZE=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT" 2>/dev/null)
    echo "Generated: $OUTPUT ($FILE_SIZE bytes)"
else
    echo "Error: Failed to download PPTX file" >&2
    exit 1
fi
