#!/bin/bash
# render_slides.sh — 调用 Presenton API 生成漂亮的 PDF slides
#
# Usage: echo "slides content" | render_slides.sh
#    or: render_slides.sh <content_text>
#
# 内容来源（优先级）：
#   1. stdin（推荐，支持大文本/含特殊字符的内容）
#   2. $1 命令行参数（向后兼容，但受 ARG_MAX 限制）
#
# 由 EXTERNAL_TOOL 节点调用，content 来自 routing_context。
# 注意：Presenton PPTX 导出有 bug (#442)，暂用 PDF 导出。

set -euo pipefail

# --- 依赖检查 ---
for dep in curl jq docker; do
    if ! command -v "$dep" &> /dev/null; then
        echo "Error: $dep not installed" >&2
        exit 1
    fi
done

# --- 读取内容：$1 优先，stdin 兜底 ---
if [ $# -ge 1 ]; then
    CONTENT="$1"
elif [ ! -t 0 ]; then
    CONTENT=$(cat)
else
    echo "Usage: echo 'slides content' | render_slides.sh" >&2
    echo "   or: render_slides.sh <content_text>" >&2
    exit 1
fi

if [ -z "$CONTENT" ]; then
    echo "Error: empty content" >&2
    exit 1
fi

OUTPUT="/tmp/presentation_$(date +%s).pdf"
N_SLIDES=10
TEMPLATE="general"

PRESENTON_URL="${PRESENTON_API_URL:-http://localhost:5000}"
PRESENTON_CONTAINER="${PRESENTON_CONTAINER_NAME:-presenton}"
API_URL="${PRESENTON_URL}/api/v1/ppt/presentation/generate"
API_KEY="${PRESENTON_API_KEY:-}"

# --- 按需启动 Presenton Docker 容器 ---
_presenton_ready() {
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "${PRESENTON_URL}/api/v1/ppt/presentation/all" 2>/dev/null)
    [ "$code" = "200" ]
}

ensure_presenton_running() {
    if _presenton_ready; then
        return 0
    fi

    echo "Presenton not running. Starting Docker container..." >&2

    if docker ps -a --format '{{.Names}}' | grep -q "^${PRESENTON_CONTAINER}$"; then
        docker start "$PRESENTON_CONTAINER" > /dev/null
    else
        echo "Error: Presenton container '$PRESENTON_CONTAINER' not found." >&2
        echo "Hint: docker run -d --name $PRESENTON_CONTAINER -p 5000:5000 presenton/presenton" >&2
        exit 1
    fi

    echo -n "Waiting for Presenton to be ready" >&2
    for i in $(seq 1 30); do
        if _presenton_ready; then
            echo " ready!" >&2
            return 0
        fi
        echo -n "." >&2
        sleep 2
    done

    echo " timeout!" >&2
    exit 1
}

# 用完自动停止容器（节省资源）
stop_presenton() {
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${PRESENTON_CONTAINER}$"; then
        docker stop "$PRESENTON_CONTAINER" > /dev/null 2>&1 || true
    fi
}
trap stop_presenton EXIT

ensure_presenton_running

# --- 确保 Presenton 配置正确（DISABLE_IMAGE_GENERATION 必须为 true，否则 PDF 导出渲染设置页） ---
ensure_presenton_configured() {
    local cfg
    cfg=$(curl -s --max-time 3 "${PRESENTON_URL}/api/user-config" 2>/dev/null)
    if echo "$cfg" | python3 -c "import json,sys; d=json.load(sys.stdin); exit(0 if d.get('DISABLE_IMAGE_GENERATION') else 1)" 2>/dev/null; then
        return 0
    fi
    # 补写配置：保留原有字段，仅强制 DISABLE_IMAGE_GENERATION=true
    local new_cfg
    new_cfg=$(echo "$cfg" | python3 -c "
import json,sys
d=json.load(sys.stdin)
d['DISABLE_IMAGE_GENERATION']=True
d['IMAGE_PROVIDER']=d.get('IMAGE_PROVIDER','')
print(json.dumps(d))
" 2>/dev/null)
    curl -s -X POST "${PRESENTON_URL}/api/user-config" \
        -H "Content-Type: application/json" \
        -d "$new_cfg" > /dev/null 2>&1
}

ensure_presenton_configured

# --- 构建请求 JSON（通过 jq，安全处理特殊字符） ---
REQUEST_JSON=$(jq -n \
    --arg content "$CONTENT" \
    --argjson n_slides "$N_SLIDES" \
    --arg template "$TEMPLATE" \
    '{
        content: $content,
        n_slides: $n_slides,
        language: "Chinese",
        template: $template,
        export_as: "pdf"
    }')

# --- 调用 API ---
AUTH_HEADER=()
if [ -n "${API_KEY:-}" ]; then
    AUTH_HEADER=(-H "Authorization: Bearer $API_KEY")
fi

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL" \
    ${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"} \
    -H "Content-Type: application/json" \
    -d "$REQUEST_JSON")

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
    ENCODED_PATH=$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1]))" "$DOWNLOAD_URL")
    DOWNLOAD_URL="${PRESENTON_URL}${ENCODED_PATH}"
fi

curl -s -o "$OUTPUT" "$DOWNLOAD_URL"

if [ -f "$OUTPUT" ]; then
    FILE_SIZE=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT" 2>/dev/null)
    echo "Generated: $OUTPUT ($FILE_SIZE bytes)"
else
    echo "Error: Failed to download PDF file" >&2
    exit 1
fi
