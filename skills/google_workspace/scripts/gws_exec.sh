#!/bin/bash
# gws_exec.sh — 安全执行 gws 命令
#
# Usage: gws_exec.sh <gws_command_string>
#
# 只允许 gws 开头的命令，拒绝其他任何命令。
# 由 EXTERNAL_TOOL 节点调用，command_string 来自 routing_context。

set -euo pipefail

CMD_STRING="${1:?Usage: gws_exec.sh <gws_command_string>}"

# 安全检查：只允许 gws 命令
if [[ ! "$CMD_STRING" =~ ^[[:space:]]*gws[[:space:]] ]]; then
    echo "Error: only 'gws' commands are allowed. Got: ${CMD_STRING:0:50}" >&2
    exit 1
fi

# 禁止 shell 元字符（$, `, |, ;, &, >）
if [[ "$CMD_STRING" =~ [\$\`\|\;\&\>\<] ]]; then
    echo "Error: shell metacharacters are not allowed in gws commands" >&2
    exit 1
fi

eval "$CMD_STRING"
