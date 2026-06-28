"""
长期记忆模块 — 跨会话持久化的用户信息、偏好、历史决策

存储架构（v2.0）：
  Qdrant（向量数据库，仅服务器/Docker模式）+ PostgreSQL（结构化存储）+ LangChain VectorStore（统一接口）

  向量语义搜索（QdrantVectorStore）
    └→ 将记忆内容向量化，按语义相似度检索

  结构化元数据（Qdrant Payload + PostgreSQL）
    └→ user_id、category、key、weight、timestamp 等字段
    └→ Qdrant Payload 作为热数据（搜索时直接返回）
    └→ PostgreSQL 作为冷数据备份 + 复杂 SQL 查询

  LangChain VectorStore 统一接口
    └→ add_documents() → 自动向量化 + 存储
    └→ similarity_search() → 语义检索 + 过滤

嵌入模型：
  阿里云百炼 text-embedding-v4（1024 维）
  通过 OpenAI 兼容 API 调用

零外部依赖升级：
  - 旧版 SQLite FTS5 → 新版 Qdrant + PostgreSQL
  - 外部 API 完全兼容（memory_manager.py 无需改动）
  - 配置从 app.config.settings 惰性读取，仅支持 Qdrant 服务器/Docker 模式
"""

from __future__ import annotations

import json
import math
import logging
import time
import uuid
import os
import threading
from enum import Enum
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    Distance,
    VectorParams,
    PointStruct,
)
import httpx  # 用于创建绕过系统代理的 HTTP 客户端

try:
    import psycopg2
    import psycopg2.extras
    _PG_AVAILABLE = True
except ImportError:
    psycopg2 = None  # type: ignore
    _PG_AVAILABLE = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════

DEFAULT_SEARCH_TOP_K = 5
DECAY_RATE = 0.01  # 时间衰减率（每天衰减 1%）

_QDRANT_COLLECTION = "user_memories"


def _get_qdrant_config() -> dict:
    """惰性读取 Qdrant 配置（延迟到 settings 加载后，避免导入顺序问题）。"""
    try:
        from app.config import settings
        port = settings.QDRANT_PORT
        host = settings.QDRANT_HOST or ""
        return {
            "host": host,
            "port": port if port else 6333,
            "collection": _QDRANT_COLLECTION,
            "api_key": settings.QDRANT_API_KEY or "",
        }
    except Exception:
        return {
            "host": "127.0.0.1",
            "port": 6333,
            "collection": _QDRANT_COLLECTION,
            "api_key": "",
        }

# Embedding 维度（阿里云百炼 text-embedding-v4 = 1024）
EMBEDDING_DIM = 1024


# ═══════════════════════════════════════════════════════════════
#  数据模型（与旧版完全兼容）
# ═══════════════════════════════════════════════════════════════

class MemoryCategory(str, Enum):
    """记忆类别"""
    USER_PROFILE = "user_profile"
    KEY_FACT = "key_fact"
    PREFERENCE = "preference"
    CONVERSATION = "conversation"
    DECISION = "decision"


class MemoryItem:
    """一条长期记忆（从 Qdrant Payload 构造）"""

    def __init__(self, **kwargs):
        self.id = kwargs.get("id", "")
        self.user_id = kwargs.get("user_id", "")
        cat = kwargs.get("category", "")
        self.category = MemoryCategory(cat) if cat else MemoryCategory.USER_PROFILE
        self.content = kwargs.get("content", "")
        self.key = kwargs.get("key", "")
        self.value = kwargs.get("value", "")
        tags = kwargs.get("tags", [])
        self.tags = json.loads(tags) if isinstance(tags, str) else (tags or [])
        self.weight = float(kwargs.get("weight", 1.0))
        self.timestamp = float(kwargs.get("timestamp", time.time()))
        self.last_accessed = float(kwargs.get("last_accessed", time.time()))

    def get_decayed_weight(self) -> float:
        """时间衰减后的权重（越久远的记忆权重越低）"""
        days = (time.time() - self.timestamp) / 86400.0
        return max(0.0, self.weight * math.exp(-DECAY_RATE * days))

    def touch(self):
        """更新最后访问时间"""
        self.last_accessed = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "user_id": self.user_id,
            "category": self.category.value, "content": self.content,
            "key": self.key, "value": self.value, "tags": self.tags,
            "weight": self.weight, "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "decayed_weight": round(self.get_decayed_weight(), 4),
        }

    def __repr__(self):
        return f"MemoryItem(id={self.id[:12]}, user={self.user_id}, cat={self.category.value})"


class SearchResult:
    """检索结果"""
    def __init__(self, memory: MemoryItem, score: float):
        self.memory = memory
        self.score = score

    def to_dict(self) -> dict:
        return {"memory": self.memory.to_dict(), "score": round(self.score, 4)}

    def __repr__(self):
        return f"SearchResult(score={self.score:.3f}, {self.memory})"


# ═══════════════════════════════════════════════════════════════
#  嵌入模型工厂（延迟加载 + 全局缓存）
# ═══════════════════════════════════════════════════════════════

_embeddings: OpenAIEmbeddings | None = None
_embeddings_lock = threading.Lock()


def _get_embeddings() -> OpenAIEmbeddings:
    """
    获取 Embedding 客户端（全局唯一，延迟加载）。

    首次调用时从环境变量/配置创建，后续调用返回缓存实例。
    使用线程锁保证线程安全。
    自动处理 Windows 代理问题（NO_PROXY）。
    """
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    with _embeddings_lock:
        if _embeddings is not None:
            return _embeddings

        # ── 处理代理问题 ──────────────────────────────────
        # Windows 代理/VPN 会影响 HTTPS 请求，设置 NO_PROXY 让 API 直连
        _ensure_no_proxy()

        api_key = os.environ.get("EMBEDDING_API_KEY", "")
        model = os.environ.get("EMBEDDING_MODEL", "text-embedding-v4")
        base_url = os.environ.get(
            "EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        if not api_key:
            try:
                from app.config import settings
                api_key = settings.EMBEDDING_API_KEY
                model = settings.EMBEDDING_MODEL
                base_url = settings.EMBEDDING_BASE_URL
            except Exception:
                pass

        if not api_key:
            raise RuntimeError(
                "EMBEDDING_API_KEY 未配置。请设置环境变量或在 .env 文件中配置。\n"
                "阿里云百炼 API Key 获取: https://bailian.console.aliyun.com/"
            )

        _embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
            check_embedding_ctx_length=False,
            http_client=httpx.Client(trust_env=False),  # 绕过 Windows 系统代理
        )
        logger.info("Embedding 客户端已初始化: model=%s, base_url=%s", model, base_url)
        return _embeddings


def _ensure_no_proxy():
    """设置 NO_PROXY 环境变量，让 LLM/Embedding API 绕过系统代理"""
    key = "NO_PROXY"
    no_proxy = os.environ.get(key, "")
    hosts_to_add = [
        "dashscope.aliyuncs.com",
        "api.deepseek.com",
        "localhost",
        "127.0.0.1",
        "openaipublic.blob.core.windows.net",
        "raw.githubusercontent.com",
    ]
    existing = set(no_proxy.split(",")) if no_proxy else set()
    updated = False
    for h in hosts_to_add:
        if h not in existing:
            existing.add(h)
            updated = True
    if updated:
        os.environ[key] = ",".join(filter(None, existing))
        logger.debug("NO_PROXY 已更新: %s", os.environ[key])


# ═══════════════════════════════════════════════════════════════
#  Qdrant 客户端工厂
# ═══════════════════════════════════════════════════════════════

_qdrant_client: QdrantClient | None = None
_qdrant_store: QdrantVectorStore | None = None
_qdrant_lock = threading.RLock()  # 可重入锁（避免 _get_qdrant_store → _get_qdrant_client 死锁）


def set_qdrant_config(
    host: str = "localhost",
    port: int = 6333,
    collection: str = "user_memories",
    api_key: str = "",
):
    """
    手动覆盖 Qdrant 连接参数（在首次使用前调用）。

    已废弃：配置现在直接从 app.config.settings 读取。
    保留此函数仅为向后兼容——调用后会 reset 连接，下次连接时使用手动指定的值。
    """
    global _qdrant_client, _qdrant_store
    # 设置环境变量供 _get_qdrant_config 读取
    import os as _os
    _os.environ["QDRANT_HOST"] = host
    _os.environ["QDRANT_PORT"] = str(port)
    reset_qdrant()
    logger.info("Qdrant 配置已更新: %s:%s/%s", host, port, collection)


def _get_qdrant_client() -> QdrantClient:
    """获取 Qdrant 客户端（全局唯一，延迟加载，支持本地文件模式）。"""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    with _qdrant_lock:
        if _qdrant_client is not None:
            return _qdrant_client

        _ensure_no_proxy()

        cfg = _get_qdrant_config()

        # 本地文件模式：host 为空时使用本地路径
        if not cfg["host"]:
            local_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "qdrant_local"
            )
            local_path = os.path.abspath(local_path)
            os.makedirs(local_path, exist_ok=True)
            _qdrant_client = QdrantClient(path=local_path)
            logger.info("Qdrant 本地模式: %s (collection=%s)", local_path, cfg["collection"])
        else:
            _qdrant_client = QdrantClient(
                host=cfg["host"],
                port=cfg["port"],
                api_key=cfg.get("api_key") or None,
                timeout=10,
            )
            _qdrant_client.get_collections()  # 验证连通性
            logger.info("Qdrant 已连接: %s:%s (collection=%s)",
                         cfg["host"], cfg["port"], cfg["collection"])
        return _qdrant_client


def _get_qdrant_store() -> QdrantVectorStore:
    """
    获取 QdrantVectorStore（LangChain 统一接口）。

    自动创建 collection（如果不存在），配置向量维度。
    """
    global _qdrant_store
    if _qdrant_store is not None:
        return _qdrant_store

    with _qdrant_lock:
        if _qdrant_store is not None:
            return _qdrant_store

        client = _get_qdrant_client()
        embeddings = _get_embeddings()
        collection_name = _QDRANT_COLLECTION

        # 确保 collection 存在
        _ensure_collection(client, collection_name)

        _qdrant_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        logger.info("QdrantVectorStore 已就绪: collection=%s", collection_name)
        return _qdrant_store


def _ensure_collection(client: QdrantClient, collection_name: str):
    """确保 Qdrant collection 存在，不存在则创建"""
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Qdrant collection 已创建: %s (dim=%d, distance=cosine)",
                     collection_name, EMBEDDING_DIM)


def reset_qdrant():
    """重置 Qdrant 连接（配置变更后调用）"""
    global _qdrant_client, _qdrant_store
    _qdrant_client = None
    _qdrant_store = None
    logger.info("Qdrant 连接已重置")


# ═══════════════════════════════════════════════════════════════
#  长期记忆模块（Qdrant + LangChain VectorStore）
# ═══════════════════════════════════════════════════════════════

class LongTermMemory:
    """
    长期记忆——Qdrant 向量存储 + LangChain VectorStore 统一接口。

    与旧版 SQLite 实现保持完全相同的公开 API，
    memory_manager.py 无需任何修改即可使用。

    搜索能力：
      - 语义向量检索（基于 text-embedding-v4）
      - 结构化过滤（user_id、category）
      - 时间衰减权重
      - 组合：语义 + 过滤 + 权重排序

    使用方式：
      ltm = LongTermMemory()
      ltm.store(user_id="USR-001", category=MemoryCategory.PREFERENCE,
                content="用户喜欢微信沟通")
      results = ltm.search(user_id="USR-001", query="联系方式偏好")
    """

    def __init__(self):
        self._store = _get_qdrant_store()
        self._client = _get_qdrant_client()
        self._counter = 0
        self._pg_conn = None  # PostgreSQL 连接（延迟初始化）
        self._pending_writes: list[threading.Thread] = []  # 跟踪后台写入线程

    def flush(self, timeout: float = 30.0):
        """
        等待所有后台 Qdrant 写入完成。

        用于测试环境确保写入同步完成后再查询。
        生产环境不需要调用（后台写入不阻塞主流程）。
        """
        threads = self._pending_writes.copy()
        for t in threads:
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning("Qdrant 写入线程超时: %s", t.name)
        self._pending_writes.clear()

    def _next_id(self) -> str:
        self._counter += 1
        return str(uuid.uuid4())

    # ── PostgreSQL 辅助方法 ────────────────────────────────

    def _get_pg_config(self) -> dict:
        """从环境变量读取 PostgreSQL 配置（localhost → 127.0.0.1 避免 DNS 超时）。"""
        host = os.environ.get("POSTGRES_HOST", "localhost")
        if host == "localhost":
            host = "127.0.0.1"
        return {
            "host": host,
            "port": int(os.environ.get("POSTGRES_PORT", "5432")),
            "dbname": os.environ.get("POSTGRES_DB", "agent_cs"),
            "user": os.environ.get("POSTGRES_USER", "postgres"),
            "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
            "connect_timeout": 5,
        }

    def _pg_get_conn(self):
        """获取或创建 PostgreSQL 连接（延迟初始化，自动重连）。"""
        if not _PG_AVAILABLE:
            return None
        try:
            if self._pg_conn is None or self._pg_conn.closed:
                cfg = self._get_pg_config()
                self._pg_conn = psycopg2.connect(**cfg)
                self._pg_conn.autocommit = True
                logger.debug("PostgreSQL 已连接: %s:%s/%s",
                             cfg["host"], cfg["port"], cfg["dbname"])
            return self._pg_conn
        except Exception as e:
            logger.debug("PostgreSQL 连接失败（将仅使用 Qdrant）: %s", e)
            return None

    def _pg_store(self, mem_id: str, user_id: str, category: str,
                  key: str, value: str, content: str, weight: float,
                  tags_str: str, timestamp: float, last_accessed: float,
                  is_deleted: bool = False, sync_status: str = "synced"):
        """写入 PostgreSQL（UPSERT；PG 为 Source of Truth，先于 Qdrant 写入）"""
        conn = self._pg_get_conn()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO user_memories
                    (mem_id, user_id, category, key, value, content,
                     weight, tags, is_deleted, sync_status, timestamp, last_accessed)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, category, key) WHERE key != ''
                DO UPDATE SET
                    mem_id = EXCLUDED.mem_id,
                    value = EXCLUDED.value,
                    content = EXCLUDED.content,
                    weight = EXCLUDED.weight,
                    tags = EXCLUDED.tags,
                    is_deleted = EXCLUDED.is_deleted,
                    sync_status = EXCLUDED.sync_status,
                    timestamp = EXCLUDED.timestamp,
                    last_accessed = EXCLUDED.last_accessed,
                    created_at = NOW()
            """, (
                mem_id, user_id, category, key, value, content,
                weight, tags_str, is_deleted, sync_status, timestamp, last_accessed,
            ))
            cur.close()
            logger.debug("PostgreSQL 已存储: mem_id=%s, user=%s, sync=%s", mem_id, user_id, sync_status)
        except Exception as e:
            logger.debug("PostgreSQL 写入失败（Qdrant 数据不受影响）: %s", e)

    def _pg_delete_user(self, user_id: str):
        """
        PostgreSQL 软删除：标记用户的所有记忆为已删除。

        不物理删除行，保留数据用于审计和恢复。
        """
        conn = self._pg_get_conn()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE user_memories SET is_deleted = TRUE WHERE user_id = %s",
                (user_id,),
            )
            cur.close()
        except Exception as e:
            logger.debug("PostgreSQL 软删除用户失败: %s", e)

    def _pg_mark_pending(self, mem_id: str):
        """标记 PG 中的记录为 pending（Qdrant 写入失败时调用）"""
        conn = self._pg_get_conn()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE user_memories SET sync_status = 'pending' WHERE mem_id = %s",
                (mem_id,),
            )
            cur.close()
        except Exception as e:
            logger.debug("标记 pending 失败: %s", e)

    def _pg_clear_all(self):
        """清空 PostgreSQL user_memories 表"""
        conn = self._pg_get_conn()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM user_memories")
            cur.close()
        except Exception as e:
            logger.debug("PostgreSQL 清空失败: %s", e)

    def _soft_delete_qdrant_points(self, filter_conditions: list[FieldCondition]):
        """
        Qdrant 软删除：将匹配点的 is_deleted 设为 True（不物理删除向量）。

        原理：HNSW 索引对物理删除不友好，频繁删除会导致索引退化。
        改为更新 payload 中的 is_deleted 标记，搜索时用 must_not 过滤。
        """
        try:
            # scroll 找到所有匹配点
            all_point_ids = []
            offset = None
            while True:
                points, next_offset = self._client.scroll(
                    collection_name=_QDRANT_COLLECTION,
                    scroll_filter=Filter(must=filter_conditions),
                    limit=100,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                all_point_ids.extend(p.id for p in points)
                if next_offset is None:
                    break
                offset = next_offset

            if all_point_ids:
                # 更新 payload 标记为已删除
                self._client.set_payload(
                    collection_name=_QDRANT_COLLECTION,
                    payload={"metadata": {"is_deleted": True}},
                    points=all_point_ids,
                )
                logger.debug("Qdrant 软删除: %d 个点已标记", len(all_point_ids))
        except Exception as e:
            logger.debug("Qdrant 软删除失败（将在 cleanup 时处理）: %s", e)

    def _pg_search(
        self,
        user_id: str,
        query: str = "",
        category=None,
        top_k: int = DEFAULT_SEARCH_TOP_K,
        min_weight: float = 0.0,
    ) -> list[SearchResult]:
        """
        PostgreSQL 纯搜索 — Qdrant 不可用时的降级方案。

        不做语义搜索，只用 ILIKE 做简单文本匹配 + 时间排序。
        保证系统在向量库宕机时仍然可用（牺牲语义能力，保留基本功能）。
        """
        conn = self._pg_get_conn()
        if conn is None:
            return []

        try:
            cur = conn.cursor()
            sql = """SELECT mem_id, user_id, category, key, value, content,
                            weight, tags, timestamp, last_accessed
                     FROM user_memories
                     WHERE user_id = %s AND is_deleted = FALSE"""
            params: list = [user_id]

            if category is not None:
                cat_val = category.value if hasattr(category, "value") else str(category)
                sql += " AND category = %s"
                params.append(cat_val)

            if query:
                sql += " AND (content ILIKE %s OR key ILIKE %s)"
                like = f"%{query}%"
                params.extend([like, like])

            sql += " ORDER BY timestamp DESC LIMIT %s"
            params.append(top_k * 2)

            cur.execute(sql, params)
            rows = cur.fetchall()
            cur.close()

            results: list[SearchResult] = []
            seen: set[str] = set()
            for row in rows:
                mem_id = row[0]
                if mem_id in seen:
                    continue
                seen.add(mem_id)

                mem = MemoryItem(
                    id=mem_id, user_id=row[1], category=row[2],
                    key=row[3] or "", value=row[4] or "",
                    content=row[5] or "",
                    weight=float(row[6] or 1.0),
                    tags=row[7] or "[]",
                    timestamp=float(row[8] or time.time()),
                    last_accessed=float(row[9] or time.time()),
                )
                decayed = mem.get_decayed_weight()
                if decayed < min_weight:
                    continue

                results.append(SearchResult(memory=mem, score=round(decayed, 4)))
                if len(results) >= top_k:
                    break

            return results
        except Exception as e:
            logger.warning("PostgreSQL 搜索失败: %s", e)
            return []

    def cleanup_soft_deleted(self) -> int:
        """
        定期清理软删除的向量（释放 Qdrant 存储空间）。

        建议通过 cron / 定时任务调用，例如每天凌晨执行一次。
        返回清理的向量数量。
        """
        try:
            result = self._client.delete(
                collection_name=_QDRANT_COLLECTION,
                points_selector=Filter(
                    must=[FieldCondition(
                        key="metadata.is_deleted",
                        match=MatchValue(value=True),
                    )]
                ),
            )
            count = getattr(result, "deleted", 0) if hasattr(result, "deleted") else 0
            if count > 0:
                logger.info("Qdrant 清理完成: %d 个软删除向量已物理删除", count)
            return count
        except Exception as e:
            logger.warning("Qdrant 清理失败: %s", e)
            return 0

    # ── 存储 ────────────────────────────────────────────────

    def store(
        self,
        user_id: str,
        category: MemoryCategory,
        content: str,
        key: str = "",
        value: Any = None,
        tags: Optional[list[str]] = None,
        weight: float = 1.0,
    ) -> MemoryItem:
        """
        存储一条长期记忆。

        流程：
          1. 生成唯一 ID
          2. 构建 Document（content + metadata payload）
          3. 通过 QdrantVectorStore.add_documents() 自动向量化并存储
          4. 返回 MemoryItem 对象

        参数：
          user_id:  用户 ID
          category: 记忆类别
          content:  记忆内容（会被向量化用于语义搜索）
          key:      记忆键（同一用户+类别下唯一，用于去重）
          value:    附加结构化值（可选）
          tags:     标签列表（可选）
          weight:   权重（0.0-1.0，越高越重要）

        返回：
          MemoryItem 对象
        """
        mem_id = self._next_id()
        now = time.time()
        tags_list = tags or []
        str_value = str(value) if value is not None else ""

        # ── 处理去重：如果同 user_id + category + key 已存在，先删除旧的 ──
        if key:
            self._delete_by_key(user_id, category, key)

        # ── 构建 Document ─────────────────────────────────────
        metadata = {
            "mem_id": mem_id,
            "user_id": user_id,
            "category": category.value,
            "key": key,
            "value": str_value,
            "weight": weight,
            "tags": json.dumps(tags_list, ensure_ascii=False),
            "is_deleted": False,  # 软删除标记
            "timestamp": now,
            "last_accessed": now,
        }

        doc = Document(page_content=content, metadata=metadata)

        # ── ① 先写 PostgreSQL（Source of Truth，拿到自增 ID 后再写 Qdrant）──
        self._pg_store(
            mem_id=mem_id, user_id=user_id, category=category.value,
            key=key, value=str_value, content=content, weight=weight,
            tags_str=json.dumps(tags_list, ensure_ascii=False),
            timestamp=now, last_accessed=now,
            is_deleted=False, sync_status="synced",
        )

        # ── ② 后台线程写入 Qdrant（embedding API 慢，不阻塞主流程）──
        t = threading.Thread(
            target=self._bg_qdrant_write,
            args=(doc, mem_id),
            daemon=True,
            name=f"qdrant-write-{mem_id[:8]}",
        )
        self._pending_writes.append(t)
        t.start()
        # 清理已完成的线程，避免列表无限增长
        self._pending_writes = [t for t in self._pending_writes if t.is_alive()]

        logger.debug("记忆已存储: user=%s, cat=%s, key=%s, id=%s",
                      user_id, category.value, key, mem_id)

        return MemoryItem(
            id=mem_id, user_id=user_id, category=category.value,
            content=content, key=key, value=str_value,
            tags=tags_list, weight=weight, timestamp=now,
            last_accessed=now,
        )

    def _bg_qdrant_write(self, doc: Document, mem_id: str):
        """后台线程：embedding 向量化 + Qdrant 写入。"""
        try:
            self._store.add_documents([doc])
            logger.debug("Qdrant 后台写入成功: %s", mem_id)
        except Exception as e:
            logger.warning("Qdrant 后台写入失败（PG 数据不受影响）: %s — %s", mem_id, e)
            self._pg_mark_pending(mem_id)

    def store_batch(self, items: list[dict]) -> list[MemoryItem]:
        """批量存储记忆"""
        results = []
        for item in items:
            result = self.store(**item)
            results.append(result)
        return results

    def _delete_by_key(self, user_id: str, category: MemoryCategory, key: str):
        """
        软删除指定 user_id + category + key 的旧记忆。

        PG 立即软删除（快速），Qdrant 软删除放到后台线程（不阻塞）。
        """
        conditions = [
            FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="metadata.category", match=MatchValue(value=category.value)),
            FieldCondition(key="metadata.key", match=MatchValue(value=key)),
        ]

        # ── PostgreSQL 软删除（快速）──
        conn = self._pg_get_conn()
        if conn is not None:
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE user_memories SET is_deleted = TRUE WHERE "
                    "user_id = %s AND category = %s AND key = %s",
                    (user_id, category.value, key),
                )
                cur.close()
            except Exception as e:
                logger.debug("PG 软删除失败: %s", e)

        # ── Qdrant 软删除（后台线程，不阻塞）──
        t = threading.Thread(
            target=self._soft_delete_qdrant_points,
            args=(conditions,),
            daemon=True,
            name=f"qdrant-del-{user_id}-{key}",
        )
        self._pending_writes.append(t)
        t.start()

    # ── 语义搜索（核心方法） ───────────────────────────────

    def search(
        self,
        user_id: str,
        query: str = "",
        category: Optional[MemoryCategory] = None,
        tags: Optional[list[str]] = None,
        top_k: int = DEFAULT_SEARCH_TOP_K,
        min_weight: float = 0.0,
    ) -> list[SearchResult]:
        """
        两阶段语义搜索：① Qdrant 粗筛（语义） → ② PG 精排（时间/权重/软删除）。

        降级策略：Qdrant 不可用时自动切换到 PG 纯文本搜索（牺牲语义，保证可用）。

        参数：
          user_id:    用户 ID（必填，双重校验）
          query:      搜索查询（会被向量化）
          category:   限定类别（可选）
          top_k:      返回最多几条
          min_weight: 最低衰减权重阈值

        返回：
          SearchResult 列表（按相似度排序）
        """
        if not query:
            # 无查询时走 PG（可靠，带软删除过滤）
            return self._pg_search(user_id, "", category, top_k, min_weight)

        # ═══════════════════════════════════════════════════════
        #  Stage 1: Qdrant 语义粗筛（带超时 + 软删除过滤）
        # ═══════════════════════════════════════════════════════
        must_conditions = [
            FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
        ]
        must_not_conditions = [
            FieldCondition(key="metadata.is_deleted", match=MatchValue(value=True)),
        ]
        if category:
            must_conditions.append(
                FieldCondition(key="metadata.category", match=MatchValue(value=category.value))
            )

        qdrant_filter = Filter(must=must_conditions, must_not=must_not_conditions)

        # ── Stage 1: Qdrant 语义搜索（带超时，embedding API 慢时不阻塞）──
        docs = self._similarity_search_with_timeout(
            query=query, k=top_k * 2, qdrant_filter=qdrant_filter,
            timeout_seconds=5.0,
        )
        if docs is None:
            # 超时或异常 → 降级 PG
            return self._pg_search(user_id, query, category, top_k, min_weight)

        # ═══════════════════════════════════════════════════════
        #  Stage 2: PG 二次校验 + 衰减权重排序
        # ═══════════════════════════════════════════════════════
        # 从 Qdrant 结果中提取 mem_id，保持语义排序
        mem_ids = []
        id_order: dict[str, int] = {}
        for i, doc in enumerate(docs):
            mid = doc.metadata.get("mem_id", "")
            if mid and mid not in id_order:
                mem_ids.append(mid)
                id_order[mid] = i

        if not mem_ids:
            return []

        # 从 PG 获取精确数据（过滤软删除、验证权限）
        conn = self._pg_get_conn()
        if conn is None:
            # PG 不可用时直接用 Qdrant 结果（降级）
            return self._results_from_docs(docs, user_id, top_k, min_weight)

        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT mem_id, user_id, category, key, value, content,
                          weight, tags, timestamp, last_accessed
                   FROM user_memories
                   WHERE mem_id = ANY(%s)
                     AND user_id = %s
                     AND is_deleted = FALSE""",
                (mem_ids, user_id),
            )
            pg_rows = {row[0]: row for row in cur.fetchall()}
            cur.close()
        except Exception as e:
            logger.debug("PG 二次校验失败，降级为纯 Qdrant 结果: %s", e)
            return self._results_from_docs(docs, user_id, top_k, min_weight)

        # ── 按 Qdrant 原有顺序重排（ORDER BY CASE 语义）──
        results: list[SearchResult] = []
        seen: set[str] = set()
        for mid in mem_ids:
            if mid in seen:
                continue
            seen.add(mid)

            row = pg_rows.get(mid)
            if row is None:
                # PG 中没有（可能被软删除），跳过
                continue

            mem = MemoryItem(
                id=row[0], user_id=row[1], category=row[2],
                key=row[3] or "", value=row[4] or "", content=row[5] or "",
                weight=float(row[6] or 1.0), tags=row[7] or "[]",
                timestamp=float(row[8] or time.time()),
                last_accessed=float(row[9] or time.time()),
            )

            decayed = mem.get_decayed_weight()
            if decayed < min_weight:
                continue

            # 评分 = 语义顺序分 × 衰减权重（Qdrant 顺序为隐式相似度分）
            semantic_score = 1.0 - (id_order[mid] / max(len(mem_ids), 1))
            combined = round(semantic_score * decayed, 4)

            results.append(SearchResult(memory=mem, score=combined))

            if len(results) >= top_k:
                break

        return results

    def _similarity_search_with_timeout(
        self, query: str, k: int,
        qdrant_filter: Filter, timeout_seconds: float = 5.0,
    ) -> list[Document] | None:
        """
        在独立线程中执行 Qdrant 相似度搜索，带超时控制。

        embedding API 调用可能耗时 20+ 秒——此方法限制等待时间，
        超时则返回 None，由调用方降级到 PG 纯文本搜索。

        返回：
          搜索结果列表，超时/异常返回 None
        """
        import concurrent.futures

        def _do_search():
            return self._store.similarity_search(
                query=query, k=k, filter=qdrant_filter,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_search)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "Qdrant 语义搜索超时 (%.1fs)，降级为 PostgreSQL 搜索",
                    timeout_seconds,
                )
                return None
            except Exception as e:
                logger.warning(
                    "Qdrant 搜索异常: %s，降级为 PostgreSQL 搜索", e,
                )
                return None

    def _results_from_docs(
        self, docs: list[Document], user_id: str,
        top_k: int, min_weight: float,
    ) -> list[SearchResult]:
        """从 Qdrant 返回的 Document 列表直接构建 SearchResult（PG 不可用时的降级）"""
        results = []
        seen = set()
        for doc in docs:
            meta = doc.metadata
            mem_id = meta.get("mem_id", "")
            if mem_id in seen:
                continue
            seen.add(mem_id)

            mem = MemoryItem(
                id=mem_id, user_id=meta.get("user_id", user_id),
                category=meta.get("category", ""), content=doc.page_content,
                key=meta.get("key", ""), value=meta.get("value", ""),
                tags=meta.get("tags", "[]"),
                weight=float(meta.get("weight", 1.0)),
                timestamp=float(meta.get("timestamp", time.time())),
                last_accessed=float(meta.get("last_accessed", time.time())),
            )
            decayed = mem.get_decayed_weight()
            if decayed < min_weight:
                continue
            results.append(SearchResult(memory=mem, score=round(decayed, 4)))
            if len(results) >= top_k:
                break
        return results

    def _get_recent(
        self,
        user_id: str,
        category: Optional[MemoryCategory] = None,
        top_k: int = DEFAULT_SEARCH_TOP_K,
        min_weight: float = 0.0,
    ) -> list[SearchResult]:
        """
        获取用户最近的记忆（无查询时的降级方案）。

        优先查 PG（可靠，带软删除过滤），PG 不可用时回退到 Qdrant scroll。
        """
        pg_results = self._pg_search(user_id, "", category, top_k, min_weight)
        if pg_results:
            return pg_results

        # PG 不可用 → Qdrant scroll 作为最后防线
        must_conditions = [
            FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
        ]
        must_not_conditions = [
            FieldCondition(key="metadata.is_deleted", match=MatchValue(value=True)),
        ]
        if category:
            must_conditions.append(
                FieldCondition(key="metadata.category", match=MatchValue(value=category.value))
            )

        try:
            points, _ = self._client.scroll(
                collection_name=_QDRANT_COLLECTION,
                scroll_filter=Filter(must=must_conditions, must_not=must_not_conditions),
                limit=top_k * 2,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.warning("Qdrant scroll 也失败: %s", e)
            return []

        results = []
        seen = set()
        for point in points:
            raw_payload = point.payload or {}
            payload = raw_payload.get("metadata", raw_payload)
            mem_id = payload.get("mem_id", str(point.id))
            if mem_id in seen:
                continue
            seen.add(mem_id)

            mem = MemoryItem(
                id=mem_id,
                user_id=payload.get("user_id", user_id),
                category=payload.get("category", ""),
                content=raw_payload.get("page_content", payload.get("content", "")) or "",
                key=payload.get("key", ""),
                value=payload.get("value", ""),
                tags=payload.get("tags", "[]"),
                weight=float(payload.get("weight", 1.0)),
                timestamp=float(payload.get("timestamp", time.time())),
                last_accessed=float(payload.get("last_accessed", time.time())),
            )

            decayed = mem.get_decayed_weight()
            if decayed < min_weight:
                continue

            results.append(SearchResult(memory=mem, score=round(decayed, 4)))

            if len(results) >= top_k:
                break

        return results

    # ── 用户画像 ────────────────────────────────────────────

    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """
        获取用户完整画像。

        从 Qdrant 中检索该用户的所有记忆，按类别分组。
        """
        profile = {
            "user_id": user_id,
            "preferences": [],
            "key_facts": [],
            "decisions": [],
            "recent_conversations": [],
            "summary": "",
        }

        try:
            points, _ = self._client.scroll(
                collection_name=_QDRANT_COLLECTION,
                scroll_filter=Filter(
                    must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))],
                    must_not=[FieldCondition(key="metadata.is_deleted", match=MatchValue(value=True))],
                ),
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.warning("获取用户画像失败: %s", e)
            return profile

        for point in points:
            raw_payload = point.payload or {}
            # QdrantVectorStore 存储格式: {"page_content": "...", "metadata": {...}}
            payload = raw_payload.get("metadata", raw_payload)
            entry = {
                "content": raw_payload.get("page_content", payload.get("content", "")) or "",
                "key": payload.get("key", ""),
                "value": payload.get("value", ""),
                "weight": float(payload.get("weight", 1.0)),
                "time": float(payload.get("timestamp", time.time())),
            }
            cat = payload.get("category", "")
            if cat == MemoryCategory.PREFERENCE.value:
                profile["preferences"].append(entry)
            elif cat == MemoryCategory.KEY_FACT.value:
                profile["key_facts"].append(entry)
            elif cat == MemoryCategory.DECISION.value:
                profile["decisions"].append(entry)
            elif cat == MemoryCategory.CONVERSATION.value:
                profile["recent_conversations"].append(entry)

        # 生成摘要
        parts = []
        if profile["preferences"]:
            top = max(profile["preferences"], key=lambda x: x["weight"])
            parts.append(f"偏好: {top['content']}")
        if profile["key_facts"]:
            parts.append(f"关键: {'; '.join(f['content'] for f in profile['key_facts'][:3])}")
        profile["summary"] = " | ".join(parts)
        return profile

    def update_preference(self, user_id: str, key: str, value: Any, content: str):
        """更新用户偏好（便捷方法）"""
        self.store(user_id, MemoryCategory.PREFERENCE, content, key=key, value=value, weight=0.9)

    # ── 维护 ────────────────────────────────────────────────

    def forget_user(self, user_id: str):
        """
        软删除用户的所有记忆。

        PG 和 Qdrant 都做软删除，不物理删除向量（HNSW 索引友好）。
        向量在 cleanup_soft_deleted() 定时任务中统一清理。
        """
        # ── PostgreSQL 软删除（快速）──
        self._pg_delete_user(user_id)

        # ── Qdrant 软删除（后台线程）──
        conditions = [
            FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
        ]
        t = threading.Thread(
            target=self._soft_delete_qdrant_points,
            args=(conditions,),
            daemon=True,
            name=f"qdrant-forget-{user_id}",
        )
        self._pending_writes.append(t)
        t.start()
        # 清理已完成的线程
        self._pending_writes = [t for t in self._pending_writes if t.is_alive()]

        logger.info("用户记忆已软删除: user=%s", user_id)

    def count_users(self) -> int:
        """
        统计不同用户的数量（仅统计有未删除记忆的用户）。
        """
        try:
            seen: set[str] = set()
            offset = None
            while True:
                points, next_offset = self._client.scroll(
                    collection_name=_QDRANT_COLLECTION,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                    scroll_filter=Filter(
                        must_not=[FieldCondition(
                            key="metadata.is_deleted",
                            match=MatchValue(value=True),
                        )]
                    ),
                )
                for point in points:
                    raw_payload = point.payload or {}
                    payload = raw_payload.get("metadata", raw_payload)
                    uid = payload.get("user_id", "")
                    if uid:
                        seen.add(uid)
                if next_offset is None:
                    break
                offset = next_offset
            return len(seen)
        except Exception as e:
            logger.warning("统计用户数失败: %s", e)
            return 0

    def count(self, user_id: Optional[str] = None) -> int:
        """
        统计记忆数量（不含软删除）。

        参数：
          user_id: 可选，限定用户

        返回：
          未删除的记忆总数
        """
        try:
            not_deleted = FieldCondition(
                key="metadata.is_deleted", match=MatchValue(value=True),
            )
            if user_id:
                filter_obj = Filter(
                    must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))],
                    must_not=[not_deleted],
                )
            else:
                filter_obj = Filter(must_not=[not_deleted])

            result = self._client.count(
                collection_name=_QDRANT_COLLECTION,
                count_filter=filter_obj,
                exact=True,
            )
            return result.count
        except Exception as e:
            logger.warning("统计记忆数量失败: %s", e)
            return 0

    def clear(self):
        """
        清空整个 collection（危险操作，仅用于测试/调试）。

        删除 collection 后立即重建，确保后续操作正常。
        """
        try:
            self._client.delete_collection(_QDRANT_COLLECTION)
            _ensure_collection(self._client, _QDRANT_COLLECTION)
            logger.warning("Collection 已清空并重建: %s", _QDRANT_COLLECTION)
        except Exception as e:
            logger.error("清空 collection 失败: %s", e)

        # ── 同步清空 PostgreSQL 表 ──
        self._pg_clear_all()

    def __repr__(self):
        cfg = _get_qdrant_config()
        return f"LongTermMemory(qdrant={cfg['host']}:{cfg['port']}/{_QDRANT_COLLECTION}, items={self.count()})"
