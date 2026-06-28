"""
知识图谱增强检索 — 向量搜索 + 图扩展的混合检索

search_kb_hybrid(query, category, top_k):
  1. 向量语义检索 (现有 search_kb_semantic) → 获取候选文章
  2. 图谱扩展: 从候选文章的图邻居中查找相关文章
  3. 合并去重 → 返回 top_k 结果

降级策略:
  - KG 未构建 → 降级到向量搜索
  - 向量搜索失败 → 降级到关键词搜索
"""

from __future__ import annotations

import logging
from typing import Any

from app.data.kb_vector import search_kb_semantic
from app.data.kg_builder import get_knowledge_graph
from app.data.loader import get_loader

logger = logging.getLogger(__name__)


def search_kb_hybrid(
    query: str,
    category: str = "",
    top_k: int = 3,
    graph_expand: bool = True,
) -> dict[str, Any]:
    """
    混合检索：向量语义搜索 + 知识图谱扩展。

    参数:
      query:         搜索查询
      category:      类别过滤（空 = 不限）
      top_k:         返回结果数量
      graph_expand:  是否启用图谱扩展（默认 True）

    返回:
      {"total_found": int, "results": [...], "method": "hybrid"|"vector"|"keyword"}
    """
    # Step 1: 向量语义检索
    try:
        vector_results = search_kb_semantic(
            query=query, category=category, top_k=max(top_k, 5),
        )
    except Exception as e:
        logger.warning("向量搜索失败，降级到关键词搜索: %s", e)
        loader = get_loader()
        return loader.search_kb(query, category=category, top_k=top_k)

    if not vector_results.get("results"):
        return vector_results

    # 收集种子文章 ID
    seed_ids: list[str] = []
    for r in vector_results["results"]:
        # 从 result 中尝试提取 kb_id（如果有）
        pass  # 向量搜索结果中通常没有 kb_id，只有 title/content/category

    # Step 2: 图谱扩展（如果启用）
    kg = get_knowledge_graph()
    if graph_expand and kg.is_built():
        try:
            # 从向量搜索结果匹配 KG 中的文章
            loader = get_loader()
            seed_articles: list[str] = []
            result_titles = {r.get("title", "") for r in vector_results["results"]}

            for kb_id, article in kg._articles.items():
                if article.get("title", "") in result_titles:
                    seed_articles.append(kb_id)

            if seed_articles:
                # 图扩展：找相关文章
                expanded_ids = kg.expand_from_articles(
                    seed_articles, max_depth=1, max_results=top_k,
                )
                # 合并结果
                expanded_titles: set[str] = set()
                for kb_id in expanded_ids:
                    article = kg._articles.get(kb_id, {})
                    title = article.get("title", "")
                    if title:
                        expanded_titles.add(title)

                # 从 loader 获取扩展文章的详细信息
                if expanded_titles:
                    loader = get_loader()
                    all_kb = {a["title"]: a for a in loader.kb_articles}

                    existing_titles = {r.get("title") for r in vector_results["results"]}
                    new_results = []
                    for title in expanded_titles:
                        if title not in existing_titles and title in all_kb:
                            a = all_kb[title]
                            new_results.append({
                                "title": a["title"],
                                "content": a["content"][:200] + "..." if len(a["content"]) > 200 else a["content"],
                                "category": a.get("category", ""),
                                "score": 0.0,  # 图扩展结果无相似度分数
                                "source": "graph",
                            })

                    # 合并：原结果 + 图扩展结果
                    merged = list(vector_results["results"]) + new_results
                    return {
                        "query": query,
                        "total_found": min(len(merged), top_k),
                        "results": merged[:top_k],
                        "method": "hybrid",
                        "graph_expanded": len(new_results),
                    }
        except Exception as e:
            logger.warning("图谱扩展失败，降级到纯向量搜索: %s", e)

    # 降级：纯向量搜索结果
    vector_results["method"] = "vector"
    return vector_results
