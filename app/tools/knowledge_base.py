"""
知识库检索工具

功能：语义搜索知识库中的解决方案和文档。
检索方式：Qdrant 向量相似度搜索（阿里云百炼 text-embedding-v4），
         Qdrant 不可用时自动回退到关键词匹配。
数据来源：data/seed/knowledge_base.json（15 篇 KB 文章）
"""

from __future__ import annotations

from typing import Any, Annotated

from langchain_core.tools import tool

from app.data.kb_vector import search_kb_semantic
from app.data.kg_search import search_kb_hybrid


@tool
def search_knowledge_base(
    query: Annotated[str, "搜索关键词或问题描述，如'电脑蓝屏'、'退款到账时间'（必填）"],
    category: Annotated[str, "限定搜索类别：政策/物流/使用指南/支付/售后/安全/账户/财务/内部（可选，不传则搜索全部）"] = "",
    top_k: Annotated[int, "返回最相关的几条结果（可选，默认 3，最大 10）"] = 3,
) -> dict[str, Any]:
    """从知识库中搜索解决方案和技术文档。

使用场景：
  - 用户遇到问题需要解决方案
  - 需要查找业务规则或政策（如"退换货政策"、"退款多久到账"）
  - Agent 需要引用标准答案，不依赖模型自身知识
  - 咨询保修流程、会员权益、发票开具等问题

返回字段：
  - total_found: 匹配的文档数量
  - results: 文档列表（title=标题, content=正文, category=类别, score=匹配度）

边界条件：
  - 如果没有匹配结果，total_found 为 0"""
    return search_kb_semantic(query, category, top_k)


@tool
def search_knowledge_base_graph(
    query: Annotated[str, "搜索关键词或问题描述，如'电脑蓝屏'、'退款到账时间'（必填）"],
    category: Annotated[str, "限定搜索类别：政策/物流/使用指南/支付/售后/安全/账户/财务/内部（可选，不传则搜索全部）"] = "",
    top_k: Annotated[int, "返回最相关的几条结果（可选，默认 3，最大 10）"] = 3,
) -> dict[str, Any]:
    """从知识库中搜索解决方案（图谱增强版）。

与 search_knowledge_base 的区别：
  - 在向量语义搜索的基础上，利用知识图谱扩展相关文章
  - 例如：搜索"退款"时，也会找到"退货"、"保修"等关联政策的文章
  - 降级策略：图谱不可用时自动回退到普通向量搜索

使用场景：
  - 需要全面了解某个主题的所有相关政策
  - 复杂问题可能涉及多篇关联文档
  - 客服需要补充了解相关业务流程

返回字段：
  - total_found: 匹配的文档数量
  - results: 文档列表
  - method: 检索方式（hybrid/vector/keyword）
  - graph_expanded: 图谱扩展带来的额外结果数"""
    return search_kb_hybrid(query, category, top_k)
