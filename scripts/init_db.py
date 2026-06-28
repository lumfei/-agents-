"""
数据库初始化脚本

职责：
  - 初始化 PostgreSQL 数据库表
  - 初始化 Qdrant Collection
  - 创建初始索引

初始化内容：
  PostgreSQL:
    - user_memories: 长期记忆镜像表（与 Qdrant 双写）
    - sessions: 会话表
    - audit_logs: 审计日志表
    - approval_requests: 审批请求表
    - feedbacks: 用户反馈表

  Qdrant:
    - user_memories: 用户记忆向量集合（1024 维，COSINE 距离）

运行方式：
  python scripts/init_db.py
  python scripts/init_db.py --reset  # 删除并重建所有表
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  PostgreSQL 初始化
# ═══════════════════════════════════════════════════════════════

def get_pg_config() -> dict:
    """从环境变量读取 PostgreSQL 配置"""
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "dbname": os.environ.get("POSTGRES_DB", "agent_cs"),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    }


def init_postgresql(reset: bool = False):
    """
    初始化 PostgreSQL 数据库表。

    创建以下表：
      - user_memories: 长期记忆结构化存储
      - sessions: 会话记录
      - audit_logs: 审计日志
      - approval_requests: HITL 审批请求
      - feedbacks: 用户反馈
    """
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 未安装，跳过 PostgreSQL 初始化。安装: pip install psycopg2-binary")
        return False

    cfg = get_pg_config()

    try:
        conn = psycopg2.connect(**cfg)
        conn.autocommit = True
        cur = conn.cursor()
        logger.info("PostgreSQL 已连接: %s:%s/%s", cfg["host"], cfg["port"], cfg["dbname"])
    except Exception as e:
        logger.warning("PostgreSQL 连接失败（服务可能未启动）: %s", e)
        logger.warning("跳过 PostgreSQL 初始化，Qdrant 仍可正常使用")
        return False

    try:
        if reset:
            logger.info("重置模式：删除现有表...")
            cur.execute("DROP TABLE IF EXISTS feedbacks CASCADE")
            cur.execute("DROP TABLE IF EXISTS approval_requests CASCADE")
            cur.execute("DROP TABLE IF EXISTS audit_logs CASCADE")
            cur.execute("DROP TABLE IF EXISTS sessions CASCADE")
            cur.execute("DROP TABLE IF EXISTS user_memories CASCADE")

        # ── 1. user_memories：长期记忆镜像表 ──────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_memories (
                mem_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                key TEXT NOT NULL DEFAULT '',
                value TEXT DEFAULT '',
                content TEXT DEFAULT '',
                weight REAL DEFAULT 1.0,
                tags JSONB DEFAULT '[]'::jsonb,
                is_deleted BOOLEAN DEFAULT FALSE,
                sync_status TEXT DEFAULT 'synced',
                timestamp DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW()),
                last_accessed DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW()),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(user_id, category, key)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_mem_user ON user_memories(user_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_mem_category ON user_memories(user_id, category)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_mem_timestamp ON user_memories(timestamp DESC)
        """)
        # 全文搜索索引（用于结构化查询的降级方案）
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_mem_content_gin
            ON user_memories USING gin(to_tsvector('simple', content))
        """)
        # 软删除过滤索引
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_mem_not_deleted
            ON user_memories(user_id, timestamp DESC) WHERE is_deleted = FALSE
        """)
        logger.info("  ✓ user_memories 表")

        # ── 2. sessions：会话表 ──────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                archived_at TIMESTAMP WITH TIME ZONE
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)
        """)
        logger.info("  ✓ sessions 表")

        # ── 3. audit_logs：审计日志表 ────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT NOT NULL DEFAULT '',
                user_id TEXT NOT NULL DEFAULT '',
                agent_name TEXT NOT NULL DEFAULT '',
                event_type TEXT NOT NULL,
                event_data JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_logs(session_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_logs(created_at DESC)
        """)
        logger.info("  ✓ audit_logs 表")

        # ── 4. approval_requests：审批请求表 ──────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS approval_requests (
                request_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL DEFAULT '',
                user_id TEXT NOT NULL DEFAULT '',
                agent_name TEXT NOT NULL DEFAULT '',
                request_type TEXT NOT NULL,
                request_data JSONB DEFAULT '{}'::jsonb,
                status TEXT NOT NULL DEFAULT 'pending',
                approved_by TEXT DEFAULT '',
                approved_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_approval_status ON approval_requests(status)
        """)
        logger.info("  ✓ approval_requests 表")

        # ── 5. feedbacks：用户反馈表 ──────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedbacks (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT NOT NULL DEFAULT '',
                user_id TEXT NOT NULL DEFAULT '',
                rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                comment TEXT DEFAULT '',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedbacks(user_id)
        """)
        logger.info("  ✓ feedbacks 表")

        cur.close()
        conn.close()
        logger.info("PostgreSQL 初始化完成")
        return True

    except Exception as e:
        logger.error("PostgreSQL 初始化失败: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════
#  Qdrant 初始化
# ═══════════════════════════════════════════════════════════════

def get_qdrant_config() -> dict:
    """从环境变量读取 Qdrant 配置"""
    return {
        "host": os.environ.get("QDRANT_HOST", "localhost"),
        "port": int(os.environ.get("QDRANT_PORT", "6333")),
        "api_key": os.environ.get("QDRANT_API_KEY", ""),
    }


def init_qdrant(reset: bool = False):
    """
    初始化 Qdrant Collection。

    创建：
      - user_memories: 用户记忆（1024 维向量，COSINE 距离）
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
    except ImportError:
        logger.warning("qdrant-client 未安装，跳过 Qdrant 初始化")
        return False

    cfg = get_qdrant_config()

    try:
        client = QdrantClient(host=cfg["host"], port=cfg["port"], api_key=cfg.get("api_key") or None)
        # 测试连接
        client.get_collections()
        logger.info("Qdrant 已连接: %s:%s", cfg["host"], cfg["port"])
    except Exception as e:
        logger.warning("Qdrant 连接失败（服务可能未启动）: %s", e)
        return False

    try:
        collection_name = "user_memories"
        collections = [c.name for c in client.get_collections().collections]

        if reset and collection_name in collections:
            client.delete_collection(collection_name)
            logger.info("已删除现有 collection: %s", collection_name)
            collections.remove(collection_name)

        if collection_name not in collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1024,  # text-embedding-v4 维度
                    distance=Distance.COSINE,
                ),
            )
            logger.info("  ✓ Qdrant collection 已创建: %s (dim=1024, distance=cosine)",
                         collection_name)
        else:
            logger.info("  ✓ Qdrant collection 已存在: %s", collection_name)

        # 创建 payload 索引（加速过滤查询）
        # 使用 Qdrant 的 create_payload_index 方法
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.user_id",
                field_schema="keyword",
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.category",
                field_schema="keyword",
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.key",
                field_schema="keyword",
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.timestamp",
                field_schema="float",
            )
            logger.info("  ✓ Qdrant payload 索引已创建")
        except Exception as e:
            # payload 索引创建失败不影响基本功能
            logger.debug("Payload 索引创建（可能已存在）: %s", e)

        logger.info("Qdrant 初始化完成")
        return True

    except Exception as e:
        logger.error("Qdrant 初始化失败: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="初始化数据库和向量存储")
    parser.add_argument("--reset", action="store_true", help="删除并重建所有表/collection")
    parser.add_argument("--pg-only", action="store_true", help="仅初始化 PostgreSQL")
    parser.add_argument("--qdrant-only", action="store_true", help="仅初始化 Qdrant")
    args = parser.parse_args()

    logger.info("=== 数据库初始化开始 ===")
    start = time.time()

    if args.qdrant_only:
        pg_ok = True
        qdrant_ok = init_qdrant(reset=args.reset)
    elif args.pg_only:
        pg_ok = init_postgresql(reset=args.reset)
        qdrant_ok = True
    else:
        pg_ok = init_postgresql(reset=args.reset)
        qdrant_ok = init_qdrant(reset=args.reset)

    elapsed = time.time() - start
    logger.info("=== 初始化完成 (%.1fs) ===", elapsed)

    if not pg_ok:
        logger.warning("⚠ PostgreSQL 初始化未完成（Qdrant 仍可正常使用）")
    if not qdrant_ok:
        logger.warning("⚠ Qdrant 初始化未完成（长期记忆功能将不可用）")

    if pg_ok and qdrant_ok:
        logger.info("✓ 所有数据库初始化成功！")

    return 0 if qdrant_ok else 1


if __name__ == "__main__":
    sys.exit(main())
