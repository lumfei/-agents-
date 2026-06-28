#!/bin/bash
# 本地开发启动脚本
#
# 功能：
#   - 检查环境变量
#   - 启动依赖服务（Redis、Qdrant 等）
#   - 启动 FastAPI 开发服务器
#   - 启动 Worker 进程（可选）
#
# 用法：
#   ./scripts/run_local.sh              # 启动全部
#   ./scripts/run_local.sh --no-worker  # 仅启动 API
#   ./scripts/run_local.sh --docker     # 使用 Docker Compose
#
# 各阶段演进：
#   Phase 1: 单进程启动 API 服务
#   Phase 2: 增加 Agent Worker 进程
#   Phase 3: 增加 HITL WebSocket 服务
#   Phase 4: 增加可观测性服务

echo "多 Agent 客服分流系统 — 本地启动脚本"
echo "======================================"

# TODO Phase 1: 实现基础启动逻辑
#   - 加载 .env 文件
#   - 检查 Python 依赖
#   - uvicorn app.main:app --reload
#
# TODO Phase 2: 增加 Worker 进程管理
# TODO Phase 3: 增加依赖服务健康检查
# TODO Phase 4: 增加生产模式启动选项
