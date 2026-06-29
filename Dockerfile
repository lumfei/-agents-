# =============================================================================
# 多 Agent 客服分流系统 — Docker 镜像（多阶段构建）
# =============================================================================
# 构建:
#   docker build -t multi-agent-cs:latest .
# 运行:
#   docker run -p 8000:8000 --env-file .env multi-agent-cs:latest
# 或:
#   docker compose up -d
# =============================================================================

# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Builder —— 安装依赖、构建 wheel
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

# 仅 builder 阶段需要的编译工具（装完依赖后丢弃）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境（隔离系统 Python，方便复制到 runtime）
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# 先复制依赖文件（利用 Docker layer cache）
COPY requirements.txt .

# 安装全部依赖到虚拟环境
RUN pip install --no-cache-dir -r requirements.txt

# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Runtime —— 只包含运行时代码 + venv，不含编译工具
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

# 安全基线：创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 从 builder 复制预装好的虚拟环境（不复制 gcc/g++ 等编译工具）
COPY --from=builder /opt/venv /opt/venv

# 只复制运行时需要的代码（.dockerignore 已排除 tests/scripts/.git/data/ 等）
COPY . .

# 创建运行时数据目录
RUN mkdir -p /app/data/qdrant_local \
             /app/data/audit_logs \
             /app/data/notifications \
             /app/data/drift \
    && chown -R appuser:appuser /app

# 切换到非 root 用户
USER appuser

EXPOSE 8000

# 健康检查（Python 自带 urllib，无需额外安装 curl）
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
