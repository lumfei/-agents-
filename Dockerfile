# =============================================================================
# 多 Agent 客服分流系统 — Docker 镜像
# =============================================================================
# 构建:
#   docker build -t multi-agent-cs:latest .
# 运行:
#   docker run -p 8000:8000 --env-file .env multi-agent-cs:latest
# 或使用 docker compose:
#   docker compose up -d
# =============================================================================

FROM python:3.11-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── 系统依赖 ───────────────────────────────────────────────
# 使用 Python 自带 urllib 做健康检查，无需额外安装 curl（避免 Debian 源被墙）

# ── Python 依赖（利用 Docker layer cache，代码变但依赖不变时不重装） ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 应用代码 ───────────────────────────────────────────────
COPY . .

# ── 创建数据目录 ───────────────────────────────────────────
RUN mkdir -p /app/data/qdrant_local /app/data/audit_logs /app/data/notifications /app/data/drift

EXPOSE 8000

# 健康检查（Python 自带 urllib，无需 curl）
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
