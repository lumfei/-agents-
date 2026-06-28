#!/bin/bash
# =============================================================================
# 多 Agent 客服分流系统 — Docker 一键启动脚本
# =============================================================================
# 用法:
#   bash scripts/run_docker.sh              # 启动所有服务
#   bash scripts/run_docker.sh --build      # 重新构建并启动
#   bash scripts/run_docker.sh --down       # 停止并清理
# =============================================================================

set -e

cd "$(dirname "$0")/.."

case "${1:-}" in
    --down)
        echo "🛑 停止所有服务..."
        docker compose down
        echo "✅ 服务已停止"
        exit 0
        ;;
    --build)
        echo "🔨 重新构建镜像..."
        docker compose build --no-cache
        shift
        ;;
esac

echo "🚀 启动多 Agent 客服分流系统..."
docker compose up -d

echo ""
echo "⏳ 等待服务就绪..."

# 等待健康检查通过（最多 60 秒）
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "✅ 服务启动成功！"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  访问地址:"
        echo "    🤖 AI 聊天界面:    http://localhost:8000/api/v1/agent/chat/ui"
        echo "    📋 API 文档:       http://localhost:8000/docs"
        echo "    💚 健康检查:       http://localhost:8000/health"
        echo "    🔔 人工审批面板:   http://localhost:8000/api/v1/approval/ui"
        echo "    📊 可观测性面板:   http://localhost:8000/api/v1/observability/ui"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "⚠️  健康检查超时，请检查日志: docker compose logs app"
exit 1
