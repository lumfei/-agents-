"""
知识库向量化检索模块

使用 Qdrant + Embedding 做语义检索，替代原来的关键词匹配。
启动时一次性将 KB 文章向量化写入 Qdrant（持久化），
后续启动检测到已有数据则跳过，不重复消耗 Token。

复用项目已有的 Qdrant 客户端和阿里云百炼 Embedding 基础设施。
如果 Qdrant 不可用，自动回退到 DataLoader 的关键词搜索。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)

KB_COLLECTION = "kb_articles"
EMBEDDING_DIM = 1024

_store: QdrantVectorStore | None = None


def _get_kb_store() -> QdrantVectorStore | None:
    """获取 KB 专用 VectorStore（全局单例，延迟初始化）。

    Returns:
        QdrantVectorStore 实例，如果 Qdrant/Embedding 不可用则返回 None。
    """
    global _store
    if _store is not None:
        return _store

    try:
        from app.memory.long_term import _get_qdrant_client, _get_embeddings
        client = _get_qdrant_client()
        embeddings = _get_embeddings()
    except Exception as e:
        logger.warning("无法获取 Qdrant/Embedding 客户端: %s，KB 语义搜索不可用", e)
        return None

    # 确保 collection 存在
    try:
        from qdrant_client.models import Distance, VectorParams
        if not client.collection_exists(KB_COLLECTION):
            client.create_collection(
                collection_name=KB_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "KB collection 已创建: %s (dim=%d, distance=cosine)",
                KB_COLLECTION, EMBEDDING_DIM,
            )
    except Exception as e:
        logger.warning("创建 KB collection 失败: %s", e)
        return None

    _store = QdrantVectorStore(
        client=client,
        collection_name=KB_COLLECTION,
        embedding=embeddings,
    )
    logger.info("KB VectorStore 已就绪: collection=%s", KB_COLLECTION)
    return _store


def index_kb_articles(force: bool = False) -> int:
    """将知识库文章向量化并写入 Qdrant。

    首次调用时 embed 所有 KB 文章并写入 Qdrant（约 15 篇）。
    后续启动检测到 collection 中已有数据则跳过，不再消耗 embedding token。

    Args:
        force: 为 True 时强制重新索引（即使已有数据）。

    Returns:
        本次索引的文章数量。0 表示跳过（已有数据）或索引失败。
    """
    store = _get_kb_store()
    if store is None:
        return 0

    try:
        from app.memory.long_term import _get_qdrant_client
        client = _get_qdrant_client()
        info = client.get_collection(KB_COLLECTION)
        if info.points_count > 0 and not force:
            logger.info(
                "KB collection 已有 %d 条向量，跳过索引。"
                "（如需重建请调用 index_kb_articles(force=True)）",
                info.points_count,
            )
            return 0
    except Exception as e:
        logger.warning("检查 KB collection 状态失败: %s", e)
        return 0

    # ── 从 DataLoader 读取 KB 文章并向量化 ────────────────────
    try:
        from app.data.loader import get_loader
        loader = get_loader()
        docs = []
        for article in loader.kb_articles:
            doc = Document(
                # 标题 + 正文一起 embedding，搜索时标题命中也会被召回
                page_content=f"标题：{article['title']}\n\n{article['content']}",
                metadata={
                    "kb_id": article["kb_id"],
                    "title": article["title"],
                    "category": article.get("category", ""),
                },
            )
            docs.append(doc)

        # 阿里云百炼 Embedding API 单次最多 10 条，分批写入
        batch_size = 8
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            store.add_documents(batch)
            logger.debug("KB 批次 %d/%d: %d 篇已写入", i // batch_size + 1, (len(docs) + batch_size - 1) // batch_size, len(batch))

        logger.info("KB 向量化完成: %d 篇文章已写入 Qdrant", len(docs))
        return len(docs)
    except Exception as e:
        logger.warning("KB 文章索引失败: %s，将回退到关键词搜索", e)
        return 0


def search_kb_semantic(
    query: str,
    category: str = "",
    top_k: int = 3,
) -> dict[str, Any]:
    """语义搜索知识库。

    参数和返回格式与 DataLoader.search_kb() 完全一致，
    Agent 工具无需任何改动。

    Args:
        query: 搜索查询（自然语言问题）
        category: 限定搜索类别（空字符串 = 搜索全部）
        top_k: 返回最相关的几条结果

    Returns:
        {"query": ..., "total_found": ..., "results": [
            {"title": ..., "content": ..., "category": ..., "score": ...}
        ]}
    """
    store = _get_kb_store()
    if store is None:
        logger.debug("KB VectorStore 不可用，回退到关键词搜索")
        from app.data.loader import get_loader
        return get_loader().search_kb(query, category, top_k)

    try:
        # ── 构建 Qdrant 过滤器 ────────────────────────────────
        qdrant_filter = None
        if category:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.category",
                        match=MatchValue(value=category),
                    )
                ]
            )

        docs = store.similarity_search(
            query, k=min(top_k, 10), filter=qdrant_filter,
        )

        results = []
        for doc in docs:
            results.append({
                "title": doc.metadata.get("title", ""),
                "content": doc.page_content,
                "category": doc.metadata.get("category", ""),
                "score": 0.0,  # LangChain VectorStore 包装不直接暴露相似度分数
            })

        return {
            "query": query,
            "total_found": len(results),
            "results": results,
        }
    except Exception as e:
        logger.warning("KB 语义搜索异常: %s，回退到关键词搜索", e)
        from app.data.loader import get_loader
        return get_loader().search_kb(query, category, top_k)
